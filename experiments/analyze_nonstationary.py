"""Tables and figures for the nonstationary experiments E9-E11.

Run from the repo root after experiments/run_nonstationary.py:
    MPLBACKEND=Agg python -m experiments.analyze_nonstationary

Metrics (sequences are long, so P_L at the last round saturates for most
decoders and is reported only for completeness):
    flip_rate        logical-state changes per round, averaged over the run
    excess_vs_oracle flip_rate / oracle flip_rate (paired, same noise)
    diff_vs_oracle   paired per-trial difference of flip rates, with its SE
    windowed curves  flip rate in consecutive windows (tracking over time)
    aligned curves   flip rate around each trial's unannounced event round
                     (steps and bursts), from the regenerated schedules
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
    aligned_flip_rate, flips_per_trial, get_budget, get_mode, log, out_dir, windowed_flip_rate,
)
from experiments.run_all import noise_chunk_with_schedule, point_dir, point_seed
from experiments.run_nonstationary import POINTS, noise_spec
from simulator.nonstationary import Schedule

RES = out_dir()
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REF = "Bayes (oracle)"

COLORS = {
    "Bayes (oracle)": "C0", "NL": "0.45", "Bayes (nominal)": "C9", "Bayes-broad": "C2",
    "Bayes-IMM": "C5", "Bayes-window 50": "C6", "Bayes-window 200": "C8",
    "LSTM-nominal": "C3", "LSTM-broad": "C1", "LSTM-drift": "C4",
}


def family(name):
    """'LSTM-drift s1' -> 'LSTM-drift'."""
    return re.sub(r" s\d+$", "", name)


def decoder_rows(exp, pid, data, relation):
    ref = data.get(REF, {}).get("errors")
    rows = []
    for name, d in data.items():
        rec = d["errors"]
        trials, rounds = rec.shape
        per = flips_per_trial(rec) / rounds
        row = dict(exp=exp, point=pid, decoder=name, family=family(name),
                   lstm_drift_relation=relation, trials=trials, rounds=rounds,
                   flip_rate=per.mean(), flip_rate_se=per.std(ddof=1) / math.sqrt(trials),
                   ler_final=rec[:, -1].mean(), seconds=d["seconds"], resets=d["resets"])
        if ref is not None and ref.shape == rec.shape:
            ref_per = flips_per_trial(ref) / rounds
            diff = per - ref_per
            row["diff_vs_oracle"] = diff.mean()
            row["diff_vs_oracle_se"] = diff.std(ddof=1) / math.sqrt(trials)
            row["excess_vs_oracle"] = per.mean() / ref_per.mean() if ref_per.mean() > 0 else None
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
        for key in ("flip_rate", "diff_vs_oracle", "excess_vs_oracle", "ler_final"):
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                agg[key] = float(np.mean(vals))
                agg[key + "_seed_min"] = float(np.min(vals))
                agg[key + "_seed_max"] = float(np.max(vals))
        agg["flip_rate_se"] = float(np.mean([r["flip_rate_se"] for r in rs]))
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
    tau = Schedule.concatenate(parts).tau
    return tau[:trials]


def plot_point_curves(ax, data, rounds, title):
    window = max(10, rounds // 20)
    fams = defaultdict(list)
    for name, d in data.items():
        fams[family(name)].append(d["errors"])
    for fam, recs in fams.items():
        curves = [windowed_flip_rate(r, window) for r in recs]
        starts = curves[0][0]
        rate = np.mean([c[1] for c in curves], axis=0)
        ax.plot(starts + window / 2, rate, label=fam, color=COLORS.get(fam),
                lw=2.4 if fam == REF else 1.4)
    ax.set_title(title, fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.grid(alpha=0.3, which="both")


def plot_aligned(ax, data, tau, title, before=50, after=200, smooth=10):
    rounds = next(iter(data.values()))["errors"].shape[1]
    before, after = min(before, rounds // 4), min(after, rounds // 2)
    fams = defaultdict(list)
    for name, d in data.items():
        fams[family(name)].append(d["errors"])
    kernel = np.ones(smooth) / smooth
    for fam, recs in fams.items():
        rates = [aligned_flip_rate(r, tau, before, after) for r in recs]
        offsets = rates[0][0]
        rate = np.nanmean([r[1] for r in rates], axis=0)
        ax.plot(offsets, np.convolve(rate, kernel, mode="same"), label=fam,
                color=COLORS.get(fam), lw=2.4 if fam == REF else 1.4)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_title(title, fontsize=10)
    ax.set_yscale("log")
    ax.set_xlabel("rounds after the unannounced event")
    ax.grid(alpha=0.3, which="both")


def summary_dotplot(exp, rows, fname):
    fam_rows = [r for r in rows if r["family"] == r["decoder"]]
    points = [p for p, _, _ in POINTS[exp]]
    decs = sorted({r["decoder"] for r in fam_rows}, key=lambda n: list(COLORS).index(n)
                  if n in COLORS else 99)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(points) + 1.6))
    for k, dec in enumerate(decs):
        xs, ys = [], []
        for i, p in enumerate(points):
            r = next((r for r in fam_rows if r["point"] == p and r["decoder"] == dec), None)
            if r is not None:
                xs.append(r["flip_rate"])
                ys.append(i + (k - len(decs) / 2) * 0.06)
        ax.scatter(xs, ys, label=dec, color=COLORS.get(dec), s=36 if dec == REF else 22,
                   zorder=3 if dec == REF else 2)
    ax.set_yticks(range(len(points)), points)
    ax.set_xscale("log")
    ax.set_xlabel("logical flips per round (lower is better)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, which="both", axis="x")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"{exp}: flip rate by decoder (oracle = optimal with the true schedule)")
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
    summary_dotplot(exp, all_rows, f"{exp}_flip_rates.png")

    n = len(loaded)
    cols = min(3, n)
    fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(5 * cols, 3.6 * math.ceil(n / cols)),
                             squeeze=False)
    for ax, (pid, point_spec, data) in zip(axes.flat, loaded):
        rounds = next(iter(data.values()))["errors"].shape[1]
        plot_point_curves(ax, data, rounds, pid)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    axes.flat[0].set_ylabel("logical flips per round (windowed)")
    fig.tight_layout()
    fig.savefig(FIG / f"{exp}_over_time.png", dpi=150)
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
        axes.flat[0].set_ylabel("logical flips per round (smoothed)")
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
            summary[exp] = [{k: v for k, v in r.items()} for r in rows
                            if r["family"] == r["decoder"]]
        except Exception:
            import traceback
            summary[exp] = None
            log(f"analyze_nonstationary: {exp} FAILED\n{traceback.format_exc()}")
    (RES / "nonstationary_summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: (
        o.item() if isinstance(o, np.generic) else str(o))))
    log(f"analyze_nonstationary: wrote {RES / 'nonstationary_summary.json'} and figures in {FIG}")


if __name__ == "__main__":
    main()
