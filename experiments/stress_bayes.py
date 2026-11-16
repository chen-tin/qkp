"""Does nonstationary noise break the Bayesian decoder?

For every E9-E11 schedule (simulator/nonstationary.py) the fixed-model Bayes
filter, which assumes the stationary nominal model (sigma_q = 0.2L,
sigma_m = 0.05L, rho = 0.9), is compared on paired noise with:
  * the oracle Bayes filter (knows the true schedule; optimal),
  * nearest lattice (no memory),
  * the adaptive IMM filter (regime bank with jumps).

"Breaks" is checked four ways:
  1. numerical: belief collapse (filter resets) or non-finite output;
  2. memory turns harmful: fixed Bayes worse than NL (paired, > 2 SE);
  3. large excess: fixed Bayes flip rate > 2x the oracle's;
  4. no recovery: after an unannounced step, the excess over the oracle in
     the last quarter of the post-step window is still > 1.5x.

Dev random streams only. Run from the repo root:
    MPLBACKEND=Agg python -m experiments.stress_bayes [--trials 2000 --rounds 1000]
Writes results/benchmarks/stress_bayes.json and stress_bayes.png.
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from decoders.bayesian.adaptive import OracleBayesDecoder, RegimeBankDecoder
from decoders.bayesian.grid_filter import BayesGridDecoder
from decoders.fast import FastGridDecoder, FastOracle, run as fast_run
from experiments.common import (
    NOMINAL, eps_fit, flips_per_trial, get_mode, log, make_rng, out_dir, paired_pl, torch_device,
)
from experiments.run_nonstationary import ADAPTIVE_GRID, IMM_HAZARD, POINTS
from experiments.runlog import RunLog
from simulator import nonstationary as ns
from simulator.fast import syndromes_and_nl

OUT = out_dir() / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)


def decoders(device):
    nom = dict(NOMINAL)
    ref_fixed = lambda: BayesGridDecoder(**nom, n_x=96, device=device)
    return {
        "Bayes (fixed nominal)": FastGridDecoder([nom], device=device, reference=ref_fixed,
                                                 name="Bayes (fixed nominal)"),
        "Bayes (oracle)": FastOracle(device=device,
                                     reference=lambda: OracleBayesDecoder(n_x=96, device=device)),
        "Bayes-IMM": FastGridDecoder(
            ADAPTIVE_GRID, device=device, joint=True, hazard=IMM_HAZARD,
            reference=lambda: RegimeBankDecoder(ADAPTIVE_GRID, hazard=IMM_HAZARD, n_x=96,
                                                device=device), name="Bayes-IMM"),
    }


def flip_stats(rec):
    per = flips_per_trial(rec) / rec.shape[1]
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=400 if get_mode() == "quick" else 2000)
    parser.add_argument("--rounds", type=int, default=200 if get_mode() == "quick" else 1000)
    parser.add_argument("--imm-trials", type=int, default=None)
    args = parser.parse_args()
    imm_trials = args.imm_trials or args.trials // 2
    device = torch_device()
    decs = decoders(device)
    results = {}
    index = 0
    with RunLog("stress_bayes", args=vars(args)) as rl:
        for exp, points in POINTS.items():
            for pid, spec, relation in points:
                index += 1
                with rl.phase("point", exp=exp, point=pid):
                    rng = make_rng("dev", "stress_bayes", index)
                    d, n, sch = ns.sample(spec, rng, args.trials, args.rounds)
                    _, K = syndromes_and_nl(d, n)
                    recs = {"NL": K}
                    for name, dec in decs.items():
                        m = imm_trials if name == "Bayes-IMM" else args.trials
                        recs[name] = fast_run(dec, d[:m], n[:m], schedule=sch.slice(0, m),
                                              block=m)
                        dec_resets = getattr(dec, "resets", 0)
                        dec.resets = 0
                        recs[name + " resets"] = dec_resets
                    results[f"{exp}/{pid}"] = summarize(exp, pid, spec, recs, sch, args.rounds,
                                                        imm_trials)
                    r = results[f"{exp}/{pid}"]
                    e = r["eps"]
                    fmt = lambda v: "sat." if v is None else f"{v:.2e}"
                    log(f"{exp}/{pid:<18} eps/round: oracle {fmt(e['Bayes (oracle)'])} "
                        f"fixed {fmt(e['Bayes (fixed nominal)'])} IMM {fmt(e['Bayes-IMM'])} "
                        f"NL {fmt(e['NL'])} | P_L({r['horizon']}) fixed-NL "
                        f"{r['fixed_minus_nl'][0]:+.3f} | breaks={r['breaks'] or 'no'}")
        rl.note(points=len(results))
    (OUT / "stress_bayes.json").write_text(json.dumps(results, indent=1, default=float))
    plot(results)
    log(f"stress_bayes: wrote {OUT / 'stress_bayes.json'}")


def paired_at(a, b, h):
    return paired_pl(a, b, h)


def summarize(exp, pid, spec, recs, sch, rounds, imm_trials):
    flip = {k: flip_stats(v) for k, v in recs.items() if not k.endswith("resets")}
    fixed, orc, nl = recs["Bayes (fixed nominal)"], recs["Bayes (oracle)"], recs["NL"]
    imm = recs["Bayes-IMM"]
    m = imm.shape[0]
    pl = {k: v.mean(axis=0) for k, v in recs.items() if not k.endswith("resets")}
    eps = {k: eps_fit(v) for k, v in pl.items()}
    # common horizon: last round at which the oracle keeps P_L below 0.4
    below = np.flatnonzero(pl["Bayes (oracle)"] < 0.4)
    h = int(below[-1] + 1) if len(below) else 1
    h = max(1, min(h, rounds))
    out = dict(exp=exp, point=pid, family=spec["family"], flip_rate_raw=flip,
               resets={k[:-7]: v for k, v in recs.items() if k.endswith("resets")},
               eps={k: v[0] for k, v in eps.items()}, eps_rounds_used={k: v[1] for k, v in eps.items()},
               horizon=h,
               pl_at_horizon={k: float(v[h - 1]) for k, v in pl.items()},
               fixed_minus_nl=paired_at(fixed, nl, h), fixed_minus_oracle=paired_at(fixed, orc, h),
               imm_minus_oracle=paired_at(imm, orc[:m], h),
               pl_curves={k: v[:: max(1, rounds // 100)].round(4).tolist() for k, v in pl.items()})
    e_or = out["eps"]["Bayes (oracle)"]
    out["excess_fixed"] = (out["eps"]["Bayes (fixed nominal)"] / e_or
                           if e_or and out["eps"]["Bayes (fixed nominal)"] else None)
    out["excess_imm"] = (out["eps"]["Bayes-IMM"] / e_or if e_or and out["eps"]["Bayes-IMM"] else None)
    # after an unannounced event: decay of 1 - 2 P_L over the next 100 rounds, aligned on tau
    if spec["family"] in ("step", "burst"):
        tau = sch.tau
        post = {}
        for name, rec in (("NL", nl), ("Bayes (oracle)", orc), ("Bayes (fixed nominal)", fixed),
                          ("Bayes-IMM", imm)):
            tt = tau[:rec.shape[0]]
            ok = tt + 100 < rounds
            idx = np.flatnonzero(ok)
            curve = np.array([rec[idx, tt[idx] + k].mean() for k in range(0, 101)])
            y0 = 1 - 2 * curve[0]
            y = 1 - 2 * curve
            usable = np.flatnonzero(y >= 0.2 * max(y0, 1e-9))
            k_end = int(usable[-1]) if len(usable) else 0
            if k_end >= 5 and y0 > 0.1:
                post[name] = float(0.5 * (1 - (y[k_end] / y0) ** (1 / k_end)))
            else:
                post[name] = None
        out["post_event_eps"] = post
        if post.get("Bayes (oracle)") and post.get("Bayes (fixed nominal)"):
            out["post_event_excess_fixed"] = post["Bayes (fixed nominal)"] / post["Bayes (oracle)"]
        if post.get("Bayes (oracle)") and post.get("Bayes-IMM"):
            out["post_event_excess_imm"] = post["Bayes-IMM"] / post["Bayes (oracle)"]
    breaks = []
    if any(v for v in out["resets"].values()):
        breaks.append("numerical (resets)")
    d, se = out["fixed_minus_nl"]
    if d > 2 * se:
        breaks.append(f"worse than NL at round {h}")
    if out["excess_fixed"] and out["excess_fixed"] > 2:
        breaks.append(f"{out['excess_fixed']:.1f}x oracle error/round")
    if out.get("post_event_excess_fixed") and out["post_event_excess_fixed"] > 1.5:
        breaks.append(f"{out['post_event_excess_fixed']:.1f}x oracle after the event")
    out["breaks"] = breaks
    return out


def plot(results):
    keys = list(results)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(keys) + 1.5))
    for i, k in enumerate(keys):
        r = results[k]
        for name, c, mk in (("NL", "0.5", "s"), ("Bayes (fixed nominal)", "C3", "o"),
                            ("Bayes-IMM", "C2", "D"), ("Bayes (oracle)", "C0", "o")):
            if r["eps"][name] is None:
                continue
            ax.scatter(r["eps"][name], i, color=c, marker=mk, s=30,
                       label=name if i == 0 else None, zorder=3)
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("logical error per round (from decay of 1 - 2 P_L)")
    ax.grid(alpha=0.3, axis="x", which="both")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Fixed-model Bayes vs oracle, NL and IMM on nonstationary noise")
    fig.tight_layout()
    fig.savefig(OUT / "stress_bayes.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
