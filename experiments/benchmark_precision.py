"""Numerical-precision benchmark of the Bayes grid filters.

The grid filters store the belief in float32 and run the AR(1) prediction
GEMM with TF32 allowed (decoders/bayesian/grid_filter.py, adaptive.py). This
script reruns the same noise with three arithmetic settings and compares:

    fp32+tf32   the default used in every experiment
    fp32        float32 belief, TF32 disabled
    fp64        float64 belief and kernels, TF32 disabled (reference)

The filters are switched to float64 at runtime by converting their tensors
after construction; no decoder source file is changed. Metrics per point and
decoder: fraction of per-round decisions j that differ from fp64, fraction of
trial-rounds whose logical state differs, logical error / flip rate for each
setting with the paired difference vs fp64 and its SE, max |p_one - p_one_fp64|
and filter resets. Random streams: dev split.

Run from the repo root:
    MPLBACKEND=Agg python -m experiments.benchmark_precision
"""

import json
import math

import numpy as np
import torch

from decoders.bayesian.adaptive import OracleBayesDecoder, RegimeBankDecoder, parameter_grid
from decoders.bayesian.grid_filter import BayesGridDecoder
from experiments.common import flips_per_trial, get_mode, log, make_rng, out_dir, torch_device
from simulator import nonstationary as ns
from simulator.noise import L, NoiseModel
from simulator.sim import Simulator

OUT = out_dir() / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = get_mode() == "quick"
SETTINGS = ("fp32+tf32", "fp32", "fp64")


def set_precision(flt, setting):
    """Switch a filter (GridFilter, ScheduleGridFilter, RegimeBankFilter) in place."""
    flt.tf32 = setting == "fp32+tf32"
    if setting != "fp64":
        return
    flt.dtype = torch.float64
    for name in ("kernels", "prior_eps", "transition", "pi_reg"):
        if hasattr(flt, name):
            setattr(flt, name, getattr(flt, name).to(torch.float64))


def run(make, setting, d, n, schedule=None):
    dec = make()
    dec.record = True
    set_precision(dec.filter, setting)
    if schedule is not None:
        dec.set_schedule(schedule)
    res = Simulator(trials=d.shape[0], rounds=d.shape[1]).run(dec, d, n)
    j, p = dec.recorded()
    return dict(j=j, p=p.astype(np.float64), err=res.logical_errors, resets=dec.resets)


def compare(case, decoder, make, d, n, schedule=None):
    out = {s: run(make, s, d, n, schedule) for s in SETTINGS}
    ref = out["fp64"]
    trials, rounds = d.shape
    ref_flip = flips_per_trial(ref["err"]) / rounds
    rows = []
    for s in SETTINGS:
        r = out[s]
        flip = flips_per_trial(r["err"]) / rounds
        diff = flip - ref_flip
        rows.append(dict(
            case=case, decoder=decoder, setting=s, trials=trials, rounds=rounds,
            decisions_differ=float((r["j"] != ref["j"]).mean()),
            logical_state_differs=float((r["err"] != ref["err"]).mean()),
            ler_final=float(r["err"][:, -1].mean()),
            mean_ler=float(r["err"].mean()),
            flip_rate=float(flip.mean()),
            flip_diff_vs_fp64=float(diff.mean()),
            flip_diff_se=float(diff.std(ddof=1) / math.sqrt(trials)),
            max_abs_p_diff=float(np.abs(r["p"] - ref["p"]).max()),
            resets=int(r["resets"]),
        ))
    worst = max(rows, key=lambda x: x["decisions_differ"])
    log(f"{case:<22} {decoder:<16} worst setting {worst['setting']}: "
        f"decisions differ {worst['decisions_differ']:.2e}, "
        f"max|dp| {max(x['max_abs_p_diff'] for x in rows):.2e}")
    return rows


def main():
    device = torch_device()
    log(f"benchmark_precision: mode={get_mode()} device={device}")
    N = 1000 if QUICK else 4000
    rows = []

    stationary = [
        ("nominal", dict(sigma_q=0.20 * L, sigma_m=0.05 * L, rho=0.90), 50),
        ("sm0.02_rho0.99", dict(sigma_q=0.20 * L, sigma_m=0.02 * L, rho=0.99), 50),
        ("sq0.30", dict(sigma_q=0.30 * L, sigma_m=0.05 * L, rho=0.90), 50),
        ("rho0", dict(sigma_q=0.20 * L, sigma_m=0.05 * L, rho=0.0), 50),
        ("nominal_500rounds", dict(sigma_q=0.20 * L, sigma_m=0.05 * L, rho=0.90), 500),
    ]
    for k, (case, p, rounds) in enumerate(stationary):
        trials = N if rounds <= 50 else N // 4
        d, n = NoiseModel(**p).sample(make_rng("dev", "benchmark_precision", k), trials, rounds)
        n_x = 192 if p["sigma_m"] < 0.035 * L else 96
        make = lambda p=p, n_x=n_x: BayesGridDecoder(p["sigma_q"], p["sigma_m"], p["rho"],
                                                     n_x=n_x, device=device)
        rows += compare(case, "Bayes (matched)", make, d, n)

    # nonstationary decoders on a step in sigma_q
    sp = ns.spec("step", target="sigma_q", before=0.20 * L, after=0.26 * L)
    T = 100 if QUICK else 200
    Nn = 300 if QUICK else 1000
    d, n, sch = ns.sample(sp, make_rng("dev", "benchmark_precision", 10), Nn, T)
    rows += compare("step_sq_200rounds", "Bayes (oracle)",
                    lambda: OracleBayesDecoder(device=device), d, n, schedule=sch)
    grid = parameter_grid((0.15 * L, 0.22 * L, 0.30 * L), (0.3, 0.7, 0.9, 0.97), (0.05 * L,))
    Ni = 150 if QUICK else 500
    rows += compare("step_sq_200rounds", "Bayes-IMM",
                    lambda: RegimeBankDecoder(grid, hazard=0.005, device=device),
                    d[:Ni], n[:Ni])

    keys = list(rows[0])
    with open(OUT / "precision.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    (OUT / "precision.json").write_text(json.dumps(dict(settings=SETTINGS, rows=rows), indent=1))
    log(f"benchmark_precision: wrote {OUT / 'precision.json'}")


if __name__ == "__main__":
    main()
