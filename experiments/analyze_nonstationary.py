"""Tables and figures for the nonstationary experiments E9-E11.

Run from the repo root after experiments/run_nonstationary.py:
    MPLBACKEND=Agg python -m experiments.analyze_nonstationary

Metrics. Sequences are long, so P_L(t) saturates at 1/2 for the weaker
decoders. Once that happens a decoder that revises its estimate of the
logical parity dithers, so the raw logical flip rate is *not* a logical error
rate and is reported only for reference. The headline metrics are:
    eps              logical error per round, from the decay of 1 - 2 P_L(t)
                     before saturation (experiments.common.eps_fit)
    excess_vs_oracle eps / eps of the oracle (same noise)
    pl_h             P_L at the common horizon h, the last round at which the
                     oracle keeps P_L below 0.4
    diff_vs_oracle   paired P_L difference at h with its standard error
Figures: P_L(t) per point, and P_L aligned on each trial's unannounced event
(steps and bursts), from the regenerated schedules.
"""

import json
import math
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.analyze import load_point, write_csv
from experiments.common import (
    common_horizon, eps_fit, flips_per_trial, get_budget, get_mode, log, out_dir, paired_pl,
)
from experiments.run_all import noise_chunk_with_schedule, point_dir, point_seed
from experiments.run_nonstationary import POINTS as _NS_POINTS, noise_spec
from experiments.run_hardware import HW_POINTS

POINTS = dict(_NS_POINTS, E12=HW_POINTS)     # E9-E11 and the hardware-scale E12
from simulator.nonstationary import Schedule

RES = out_dir()
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REF = "Bayes (oracle)"

COLORS = {
    "Bayes (oracle)": "C0", "NL": "0.45", "Bayes (nominal)": "C9", "Bayes-broad": "C2",
    "Bayes-IMM": "C5", "Bayes-window 50": "C6", "Bayes-window 200": "C8",
    "LSTM-nominal": "C3", "LSTM-broad": "C1", "LSTM-drift": "C4",
    "Bayes (regime-aware)": "C7", "LSTM-hw": "C4",
}
for _name, _c in list(COLORS.items()):       # E12 suffixed names share colours
    for _suffix in (" (hw grid)", " (hw anchor)"):
        COLORS.setdefault(_name + _suffix, _c)


def family(name):
    """'LSTM-drift s1' -> 'LSTM-drift'."""
    return re.sub(r" s\d+$", "", name)


def decoder_rows(exp, pid, data, relation):
    ref = data.get(REF, {}).get("errors")
    h = common_horizon(ref.mean(axis=0)) if ref is not None else None
    eps_ref = eps_fit(ref.mean(axis=0))[0] if ref is not None else None
    rows = []
    for name, d in data.items():
        rec = d["errors"]
        trials, rounds = rec.shape
        eps, used = eps_fit(rec.mean(axis=0))
        flips = flips_per_trial(rec) / rounds
        row = dict(exp=exp, point=pid, decoder=name, family=family(name),
                   lstm_drift_relation=relation, trials=trials, rounds=rounds,
                   eps=eps, eps_rounds_used=used, ler_final=float(rec[:, -1].mean()),
                   raw_flip_rate_not_a_logical_error_rate=float(flips.mean()),
                   seconds=d["seconds"], resets=d["resets"])
        if ref is not None and ref.shape == rec.shape:
            row["horizon"] = h
            row["pl_h"] = float(rec[:, h - 1].mean())
            row["diff_vs_oracle"], row["diff_vs_oracle_se"] = paired_pl(rec, ref, h)
            row["excess_vs_oracle"] = (eps / eps_ref) if (eps is not None and eps_ref) else None
        rows.append(row)
    return rows


def aggregate(rows):
    """One row per LSTM family (mean over seeds, with the seed range)."""
    groups = defaultdict(list)
    for r in rows:
        if r["family"] != r["decoder"]:
            groups[(r["exp"], r["point"], r["family"])].append(r)
    out = []
    for (exp, pid, fam), rs in groups.items():
        agg = dict(rs[0], decoder=fam, seeds=len(rs))
        for key in ("eps", "pl_h", "diff_vs_oracle", "excess_vs_oracle", "ler_final"):
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                agg[key] = float(np.mean(vals))
                agg[key + "_seed_min"] = float(np.min(vals))
                agg[key + "_seed_max"] = float(np.max(vals))
        ses = [r["diff_vs_oracle_se"] for r in rs if r.get("diff_vs_oracle_se") is not None]
        agg["diff_vs_oracle_se"] = float(np.mean(ses)) if ses else None
        out.append(agg)
    return out


def event_rounds(exp, pid, point_spec, trials, rounds):
    """Regenerate each test trial's event round tau (same streams as the run)."""
    n_file = point_dir(exp, pid) / "n_chunks.json"
    if not n_file.exists():
        return None
    n_chunks = json.loads(n_file.read_text())["n_chunks"]
    chunk = get_budget().ns_chunk_trials
    index = point_seed(exp, pid)
    parts = [noise_chunk_with_schedule(noise_spec(point_spec), exp, index, c, chunk, rounds)[2]
             for c in range(n_chunks)]
    return Schedule.concatenate(parts).tau[:trials]


def _families(data):
    fams = defaultdict(list)
    for name, d in data.items():
        fams[family(name)].append(d["errors"])
    return fams


def plot_pl_curves(ax, data, title):
    for fam, recs in _families(data).items():
        pl = np.mean([r.mean(axis=0) for r in recs], axis=0)
        t = np.arange(1, len(pl) + 1)
        ax.plot(t, pl, label=fam, color=COLORS.get(fam), lw=2.4 if fam == REF else 1.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("round")
    ax.set_ylim(0, 0.55)
    ax.grid(alpha=0.3)


def plot_aligned(ax, data, tau, title, before=50, after=200):
    rounds = next(iter(data.values()))["errors"].shape[1]
    before, after = min(before, rounds // 4), min(after, rounds // 2)
    offsets = np.arange(-before, after)
    for fam, recs in _families(data).items():
        curves = []
        for rec in recs:
            ok = (tau >= before) & (tau + after < rounds)
            idx = np.flatnonzero(ok)
            curves.append([rec[idx, tau[idx] + k].mean() for k in offsets])
        ax.plot(offsets, np.mean(curves, axis=0), label=fam, color=COLORS.get(fam),
                lw=2.4 if fam == REF else 1.4)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("rounds after the unannounced event")
    ax.grid(alpha=0.3)


def summary_dotplot(exp, rows, fname):
    fam_rows = [r for r in rows if r["family"] == r["decoder"] and r.get("eps")]
    points = [p for p, _, _ in POINTS[exp]]
    decs = sorted({r["decoder"] for r in fam_rows},
                  key=lambda n: list(COLORS).index(n) if n in COLORS else 99)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(points) + 1.6))
    for k, dec in enumerate(decs):
        xs, ys = [], []
        for i, p in enumerate(points):
            r = next((r for r in fam_rows if r["point"] == p and r["decoder"] == dec), None)
            if r is not None and r["eps"] > 0:
                xs.append(r["eps"])
                ys.append(i + (k - len(decs) / 2) * 0.06)
        ax.scatter(xs, ys, label=dec, color=COLORS.get(dec), s=36 if dec == REF else 22,
                   zorder=3 if dec == REF else 2)
    ax.set_yticks(range(len(points)), points)
    ax.set_xscale("log")
    ax.set_xlabel("logical error per round, from the decay of 1 - 2 P_L (lower is better)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, which="both", axis="x")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"{exp}: logical error per round (oracle = optimal with the true schedule)")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160)
    plt.close(fig)


def analyze(exp):
    all_rows = []
    loaded = []
    for pid, point_spec, relation in POINTS[exp]:
        data = load_point(exp, pid)
        if not data:
            log(f"{exp}/{pid}: no results yet")
            continue
        rows = decoder_rows(exp, pid, data, relation)
        all_rows += rows + aggregate(rows)
        loaded.append((pid, point_spec, data))
    if not loaded:
        return []
    write_csv(RES / f"{exp}_summary.csv", all_rows)
    summary_dotplot(exp, all_rows, f"{exp}_eps.png")

    n = len(loaded)
    cols = min(3, n)
    fig, axes = plt.subplots(math.ceil(n / cols), cols,
                             figsize=(5 * cols, 3.6 * math.ceil(n / cols)), squeeze=False)
    for ax, (pid, point_spec, data) in zip(axes.flat, loaded):
        plot_pl_curves(ax, data, pid)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    axes.flat[0].set_ylabel("P_L(t)")
    fig.tight_layout()
    fig.savefig(FIG / f"{exp}_pl_over_time.png", dpi=150)
    plt.close(fig)

    events = [(pid, s, d) for pid, s, d in loaded if s["family"] in ("step", "burst")]
    if events:
        fig, axes = plt.subplots(1, len(events), figsize=(5 * len(events), 3.6), squeeze=False)
        for ax, (pid, point_spec, data) in zip(axes.flat, events):
            rec = next(iter(data.values()))["errors"]
            tau = event_rounds(exp, pid, point_spec, rec.shape[0], rec.shape[1])
            if tau is not None:
                plot_aligned(ax, data, tau, pid)
        axes.flat[0].legend(fontsize=7)
        axes.flat[0].set_ylabel("P_L, aligned on the event")
        fig.tight_layout()
        fig.savefig(FIG / f"{exp}_aligned_on_event.png", dpi=150)
        plt.close(fig)
    return all_rows


def main():
    log(f"analyze_nonstationary: mode={get_mode()}")
    summary = {}
    for exp in POINTS:
        try:
            rows = analyze(exp)
            summary[exp] = [r for r in rows if r["family"] == r["decoder"]]
        except Exception:
            import traceback
            summary[exp] = None
            log(f"analyze_nonstationary: {exp} FAILED\n{traceback.format_exc()}")
    (RES / "nonstationary_summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: (
        o.item() if isinstance(o, np.generic) else str(o))))
    log(f"analyze_nonstationary: wrote {RES / 'nonstationary_summary.json'} and figures in {FIG}")


if __name__ == "__main__":
    main()
