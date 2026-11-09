"""Turn cached raw results into tables (CSV/JSON) and figures.

Run from the repo root after experiments/run_all.py:
    MPLBACKEND=Agg python -m experiments.analyze
"""

import json
import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    BROAD_BOX, NOMINAL, effective_eps, flips_per_trial, get_mode, log, out_dir, wilson,
)
from experiments.run_all import (
    E2_LEVELS, E2_RHOS, E3_SQ, E4_SM, E5_RHOS, E5_SM_SCALE, E5_SQ_SCALE, SWITCHING,
)
from simulator.noise import L
from simulator.sim import odd_jump_probability

RES = out_dir()
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

STYLE = {
    "No correction": dict(color="black", marker="x"),
    "NL": dict(color="0.45", marker="s"),
    "Bayes (matched)": dict(color="C0", marker="o"),
    "Bayes (nominal)": dict(color="C9", marker="v", ls="--"),
    "Bayes (per-round)": dict(color="C0", marker="o", ls=":"),
    "Bayes-broad": dict(color="C2", marker="D"),
    "LSTM-nominal": dict(color="C3", marker="^"),
    "LSTM-broad": dict(color="C1", marker="P"),
    "LSTM-matched": dict(color="C4", marker="*", ms=12),
    "Bayes (regime-aware)": dict(color="C0", marker="o"),
    "Bayes (rho=0.9)": dict(color="C9", marker="v"),
    "Bayes (rho=0.97)": dict(color="C5", marker="<"),
    "LSTM-switching": dict(color="C4", marker="*"),
}


def family(name):
    """Group seeds: 'LSTM-nominal s1' -> 'LSTM-nominal'."""
    for fam in ("LSTM-nominal", "LSTM-broad"):
        if name.startswith(fam + " s") and "head" not in name:
            return fam
    return name


# loading
def load_point(exp, point_id):
    pdir = RES / "cache" / exp / point_id
    out = {}
    if not pdir.exists():
        return out
    n_file = pdir / "n_chunks.json"
    n_chunks = json.loads(n_file.read_text())["n_chunks"] if n_file.exists() else None
    for p in sorted(pdir.glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        meta_path = p.with_name(p.name + ".meta.json")
        if not meta_path.exists():
            log(f"warning: {p} has no sidecar, ignored")
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("mode") != get_mode() or (
                n_chunks is not None and meta["config"].get("n_chunks") != n_chunks):
            log(f"warning: {p} is stale (mode or test-set size differs), ignored")
            continue
        with np.load(p) as z:
            trials = int(z["trials"])
            bits = z["bits"]
        rounds = meta["config"]["rounds"]
        rec = np.unpackbits(bits, axis=1, count=rounds).astype(bool)[:trials]
        out[meta["config"]["decoder"]] = dict(
            errors=rec, seconds=meta.get("seconds"), resets=meta.get("resets", 0)
        )
    return out


def stats(rec, ref=None, nl=None):
    """Summary of one decoder's (trials, rounds) record, paired with ref/NL."""
    trials, rounds = rec.shape
    k50 = int(rec[:, -1].sum())
    p50, lo50, hi50 = wilson(k50, trials)
    k10 = int(rec[:, 9].sum())
    p10, lo10, hi10 = wilson(k10, trials)
    per_trial = rec.mean(axis=1)
    flips = flips_per_trial(rec) / rounds
    row = dict(
        trials=trials, ler50=p50, ler50_lo=lo50, ler50_hi=hi50, ler50_events=k50,
        ler50_upper=3.0 / trials if k50 == 0 else None,
        ler10=p10, ler10_lo=lo10, ler10_hi=hi10,
        mean_ler=per_trial.mean(), mean_ler_se=per_trial.std(ddof=1) / math.sqrt(trials),
        flip_rate=flips.mean(), flip_rate_se=flips.std(ddof=1) / math.sqrt(trials),
        eps10=float(effective_eps(p10, 10)), eps50=float(effective_eps(p50, 50)),
        saturated50=bool(p50 > 0.5 - 2 * math.sqrt(0.25 / trials)),
        resolved50=k50 >= 10,
    )
    for tag, other in (("vs_bayes", ref), ("vs_nl", nl)):
        if other is not None and other.shape == rec.shape:
            d = per_trial - other.mean(axis=1)
            row[f"diff_mean_ler_{tag}"] = d.mean()
            row[f"diff_mean_ler_{tag}_se"] = d.std(ddof=1) / math.sqrt(trials)
            # McNemar-style paired difference at round 50
            d50 = rec[:, -1].astype(float) - other[:, -1].astype(float)
            row[f"diff_ler50_{tag}"] = d50.mean()
            row[f"diff_ler50_{tag}_se"] = d50.std(ddof=1) / math.sqrt(trials)
    return row


def in_box(truth):
    return all(BROAD_BOX[k][0] - 1e-12 <= truth[k] <= BROAD_BOX[k][1] + 1e-12
               for k in ("rho", "sigma_q", "sigma_m"))


def is_nominal(truth):
    return all(abs(truth[k] - NOMINAL[k]) < 1e-9 for k in NOMINAL)


def point_rows(exp, point_id, truth, extra=None):
    data = load_point(exp, point_id)
    if not data:
        return []
    ref = data.get("Bayes (matched)", data.get("Bayes (regime-aware)", {})).get("errors")
    nl = data.get("NL", {}).get("errors")
    rows = []
    for name, d in data.items():
        row = dict(exp=exp, point=point_id, decoder=name, family=family(name),
                   seconds=d["seconds"], resets=d["resets"])
        if truth is not None:
            row.update({k: truth[k] / (L if k != "rho" else 1) for k in truth})
            row["in_dist_nominal"] = is_nominal(truth)
            row["in_dist_broad"] = in_box(truth)
        if extra:
            row.update(extra)
        row.update(stats(d["errors"], ref, nl))
        rows.append(row)
    return rows


def aggregate_families(rows):
    """Add one row per LSTM family: mean over seeds, with seed spread."""
    by = defaultdict(list)
    for r in rows:
        if r["family"] != r["decoder"]:
            by[(r["exp"], r["point"], r["family"])].append(r)
    out = []
    for (exp, point, fam), rs in by.items():
        agg = dict(rs[0])
        agg["decoder"] = fam
        agg["seeds"] = len(rs)
        for key in ("ler50", "ler10", "mean_ler", "flip_rate", "eps10", "eps50",
                    "diff_mean_ler_vs_bayes", "diff_mean_ler_vs_nl"):
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals:
                agg[key] = float(np.mean(vals))
                agg[key + "_seed_min"] = float(np.min(vals))
                agg[key + "_seed_max"] = float(np.max(vals))
        # events: total over seeds; zero only if every seed saw zero
        agg["ler50_events"] = int(sum(r["ler50_events"] for r in rs))
        agg["ler50_events_min_seed"] = int(min(r["ler50_events"] for r in rs))
        agg["ler50_lo"] = min(r["ler50_lo"] for r in rs)
        agg["ler50_hi"] = max(r["ler50_hi"] for r in rs)
        agg["ler10_lo"] = min(r["ler10_lo"] for r in rs)
        agg["ler10_hi"] = max(r["ler10_hi"] for r in rs)
        agg["resolved50"] = all(r["resolved50"] for r in rs)
        agg["trials"] = rs[0]["trials"]
        for key in ("ler50_upper", "saturated50", "seconds", "resets"):
            agg[key] = None
        for key in list(agg):
            if key.startswith("diff_ler50"):
                vals = [r[key] for r in rs if r.get(key) is not None]
                agg[key] = float(np.mean(vals)) if vals else None
        out.append(agg)
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")


# plotting helpers
def plot_metric(ax, xs, rows_by_x, decoders, metric="ler50", ood_key=None, log_y=True):
    for dec in decoders:
        style = dict(STYLE.get(dec, {}))
        ls = style.pop("ls", "-")
        X, Y, LO, HI, UP, hollow = [], [], [], [], [], []
        for x in xs:
            r = rows_by_x.get(x, {}).get(dec)
            if r is None:
                continue
            if metric == "ler50" and r["ler50_events"] == 0:
                UP.append((x, 3.0 / r["trials"]))
                continue
            X.append(x)
            Y.append(r[metric])
            if metric == "ler50":
                LO.append(r[metric] - r.get("ler50_lo", r[metric]))
                HI.append(r.get("ler50_hi", r[metric]) - r[metric])
            else:
                se = r.get(metric + "_se", 0) or 0
                lo = r.get(metric + "_seed_min")
                hi = r.get(metric + "_seed_max")
                LO.append(max(r[metric] - lo, 2 * se) if lo is not None else 2 * se)
                HI.append(max(hi - r[metric], 2 * se) if hi is not None else 2 * se)
            hollow.append(bool(ood_key and not r.get(ood_key, True)))
        if X:
            ax.plot(X, Y, ls=ls, color=style.get("color"), lw=1.3, label=dec, zorder=2)
            for x, y, lo, hi, h in zip(X, Y, LO, HI, hollow):
                ax.errorbar([x], [y], yerr=[[lo], [hi]], fmt=style.get("marker", "o"),
                            color=style.get("color"), ms=style.get("ms", 6),
                            mfc="white" if h else style.get("color"), capsize=2, zorder=3)
        for x, u in UP:
            ax.plot([x], [u], marker="v", color=style.get("color"), mfc="white", ms=7,
                    ls="none", zorder=3)
    if log_y:
        ax.set_yscale("log")


def shade_box(ax, lo, hi, scale=1.0):
    ax.axvspan(lo / scale, hi / scale, color="0.9", zorder=0, label="broad training range")


# experiments
SWEEP_DECODERS = ["NL", "Bayes (matched)", "Bayes (nominal)", "Bayes-broad",
                  "LSTM-nominal", "LSTM-broad", "LSTM-matched"]


def analyze_sweep(exp, points, xname, x_of, box_key, title, fname):
    rows = []
    for pid, truth in points:
        rows += point_rows(exp, pid, truth)
    rows += aggregate_families(rows)
    write_csv(RES / f"{exp}_summary.csv", rows)

    groups = defaultdict(list)
    for pid, truth in points:
        groups[pid.split("_")[0] if exp == "E2" else "all"].append((pid, truth))
    n_groups = len(groups)
    fig, axes = plt.subplots(2, n_groups, figsize=(6.5 * n_groups, 8.5), squeeze=False)
    for col, (g, pts) in enumerate(groups.items()):
        rows_by_x = {}
        for pid, truth in pts:
            x = x_of(truth)
            rows_by_x[x] = {r["decoder"]: r for r in rows if r["point"] == pid}
        xs = sorted(rows_by_x)
        lo, hi = BROAD_BOX[box_key]
        scale = 1.0 if box_key == "rho" else L
        any_in_box = any(in_box(t) for _, t in pts)
        for row_i, metric in enumerate(("ler50", "mean_ler")):
            ax = axes[row_i, col]
            if any_in_box:
                shade_box(ax, lo, hi, scale)
            else:
                ax.text(0.02, 0.02, "all points outside the broad training box",
                        transform=ax.transAxes, fontsize=8, color="0.3")
            plot_metric(ax, xs, rows_by_x, SWEEP_DECODERS, metric, "in_dist_broad")
            ax.set_xlabel(xname)
            ax.set_ylabel({"ler50": "logical error rate after 50 rounds",
                           "mean_ler": "time-averaged logical error (rounds 1-50)"}[metric])
            sub = f" (level {g}: " + (
                f"$\\sigma_q={E2_LEVELS[g][0]}L$, $\\sigma_m={E2_LEVELS[g][1]}L$)"
                if exp == "E2" else "")
            ax.set_title(title + (sub if exp == "E2" else ""))
            ax.grid(alpha=0.3, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, -0.05, "Hollow markers: outside the broad training range. "
             "Open triangles: no errors observed (95% upper bound 3/N). "
             "LSTM lines: mean over seeds; bars: seed spread or 2 SE.",
             ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return rows


def analyze_e1():
    truth = NOMINAL
    rows = point_rows("E1", "nominal", truth)
    rows += aggregate_families(rows)
    write_csv(RES / "E1_summary.csv", rows)
    data = load_point("E1", "nominal")
    if not data:
        return rows
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    n = np.arange(1, 51)
    for name, d in data.items():
        fam = family(name)
        style = dict(STYLE.get(fam, STYLE.get(name, {})))
        if "k0 head" in name:
            style = dict(color="C3", ls=":")
        ls = style.pop("ls", "-")
        curve = d["errors"].mean(axis=0)
        lab = name if fam == name else (fam if name.endswith("s0") else None)
        alpha = 1.0 if fam == name or name.endswith("s0") else 0.45
        for ax in axes:
            ax.plot(n, np.where(curve > 0, curve, np.nan), ls=ls, color=style.get("color"),
                    lw=1.6, alpha=alpha, label=lab)
    axes[0].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("round $n$")
        ax.set_ylabel("logical error rate $P_L(n)$")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_title("E1: nominal noise ($\\sigma_q=0.2L$, $\\sigma_m=0.05L$, $\\rho=0.9$), log scale")
    axes[1].set_title("same, linear scale")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "E1_ler_vs_round.png", dpi=160)
    plt.close(fig)
    return rows


def analyze_e5():
    truth = NOMINAL
    rows = point_rows("E5", "nominal", truth)
    rows += aggregate_families(rows)
    write_csv(RES / "E5_summary.csv", rows)
    by = {r["decoder"]: r for r in rows}
    if "Bayes (matched)" not in by:
        return rows
    grid = np.full((len(E5_RHOS), len(E5_SQ_SCALE)), np.nan)
    se = np.full_like(grid, np.nan)
    for i, rho in enumerate(E5_RHOS):
        for j, k in enumerate(E5_SQ_SCALE):
            name = ("Bayes (matched)" if (rho == 0.9 and k == 1.0)
                    else f"Bayes rho'={rho} sq'x{k}")
            r = by.get(name)
            if r is not None:
                grid[i, j] = r["mean_ler"]
                se[i, j] = r["mean_ler_se"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    im = ax.imshow(np.log10(grid), origin="lower", cmap="viridis", aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                sig_inn = E5_SQ_SCALE[j] * 0.2 * math.sqrt(1 - E5_RHOS[i] ** 2)
                ax.text(j, i, f"{grid[i, j]:.4f}\n($\\sigma_{{inn}}'$={sig_inn:.3f}L)",
                        ha="center", va="center", fontsize=7,
                        color="white" if np.log10(grid[i, j]) < np.nanmean(np.log10(grid)) else "black")
    ax.set_xticks(range(len(E5_SQ_SCALE)), [f"{k}x" for k in E5_SQ_SCALE])
    ax.set_yticks(range(len(E5_RHOS)), [str(r) for r in E5_RHOS])
    ax.set_xlabel("assumed $\\sigma_q'$ / true $\\sigma_q$")
    ax.set_ylabel("assumed $\\rho'$ (true 0.9)")
    ax.set_title("E5: Bayes with a mis-specified model\ntime-averaged logical error (log colour)")
    fig.colorbar(im, ax=ax, label="log10 time-averaged logical error")
    ax.add_patch(plt.Rectangle((E5_SQ_SCALE.index(1.0) - 0.5, E5_RHOS.index(0.9) - 0.5), 1, 1,
                               fill=False, ec="red", lw=2))
    ax = axes[1]
    names = ["Bayes sm'x0.5", "Bayes (matched)", "Bayes sm'x2.0", "NL",
             "LSTM-nominal", "LSTM-broad"]
    labels = ["Bayes $\\sigma_m'$=0.5x", "Bayes matched", "Bayes $\\sigma_m'$=2x", "NL",
              "LSTM-nominal (mean of seeds)", "LSTM-broad (mean of seeds)"]
    vals = [by[n]["mean_ler"] if n in by else np.nan for n in names]
    errs = [2 * by[n]["mean_ler_se"] if n in by else 0 for n in names]
    ax.barh(range(len(names)), vals, xerr=errs, color=["C0", "C0", "C0", "0.5", "C3", "C1"])
    ax.set_yticks(range(len(names)), labels)
    ax.set_xscale("log")
    ax.set_xlabel("time-averaged logical error")
    ax.set_title("assumed $\\sigma_m'$ mismatch, with references")
    fig.tight_layout()
    fig.savefig(FIG / "E5_mismatch.png", dpi=160)
    plt.close(fig)
    return rows


def analyze_e6():
    rows = point_rows("E6", "switching", None, extra=dict(noise="switching"))
    rows += aggregate_families(rows)
    write_csv(RES / "E6_summary.csv", rows)
    show = [r for r in rows if r["family"] == r["decoder"] or r["decoder"] in
            ("LSTM-nominal", "LSTM-broad")]
    show = [r for r in show if not (r["decoder"].startswith("LSTM-nominal s")
                                    or r["decoder"].startswith("LSTM-broad s"))]
    if not show:
        return rows
    show.sort(key=lambda r: r["mean_ler"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    names = [r["decoder"] for r in show]
    vals = [r["mean_ler"] for r in show]
    lo = [r["mean_ler"] - r.get("mean_ler_seed_min", r["mean_ler"]) + 2 * r["mean_ler_se"] for r in show]
    hi = [r.get("mean_ler_seed_max", r["mean_ler"]) - r["mean_ler"] + 2 * r["mean_ler_se"] for r in show]
    colors = [STYLE.get(n, {}).get("color", "0.5") for n in names]
    ax.barh(range(len(names)), vals, xerr=[lo, hi], color=colors)
    ax.set_yticks(range(len(names)), names)
    ax.set_xscale("log")
    ax.set_xlabel("time-averaged logical error (rounds 1-50)")
    ax.set_title(f"E6: regime-switching noise ($\\rho$ switches {SWITCHING['rho_lo']}"
                 f"$\\leftrightarrow${SWITCHING['rho_hi']}, p={SWITCHING['p_switch']})")
    ax.grid(alpha=0.3, axis="x", which="both")
    fig.tight_layout()
    fig.savefig(FIG / "E6_switching.png", dpi=160)
    plt.close(fig)
    return rows


def analyze_e7():
    path = RES / "E7_timing.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    rows = data["rows"]
    write_csv(RES / "E7_timing.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, single in zip(axes, (True, False)):
        sel = [r for r in rows if (r["batch"] == 1) == single]
        labels = [f"{r['decoder']} ({r['device']})" for r in sel]
        vals = [r["latency_ms_per_round"] if single else r["throughput_trial_rounds_per_s"]
                for r in sel]
        ax.barh(range(len(sel)), vals, color=["C2" if r["device"] == "cuda" else "C7" for r in sel])
        ax.set_yticks(range(len(sel)), labels)
        ax.set_xscale("log")
        ax.set_xlabel("latency per round, one trial [ms]" if single
                      else "throughput, large batch [trial-rounds / s]")
        ax.grid(alpha=0.3, axis="x", which="both")
    axes[0].set_title("E7: single-trial decoding latency")
    axes[1].set_title("E7: batched throughput")
    fig.tight_layout()
    fig.savefig(FIG / "E7_timing.png", dpi=160)
    plt.close(fig)
    return data


def floors(sigma_q, sigma_m, rho):
    """Per-round error floors: genie (knows eps_{t-1}, nu_{t-1}) and
    Kalman (knows the unwrapped past z_1..z_{t-1})."""
    sig_inn2 = sigma_q**2 * (1 - rho**2)
    genie = odd_jump_probability(math.sqrt(sig_inn2 + sigma_m**2))
    F = np.array([[rho, 0, 0], [0, 0, 0], [0, 1, 0]])
    Q = np.diag([sig_inn2, sigma_m**2, 0.0])
    H = np.array([[1.0, 1.0, -1.0]])
    P = np.diag([sigma_q**2, sigma_m**2, sigma_m**2])
    S = None
    for _ in range(500):
        Pm = F @ P @ F.T + Q
        S = (H @ Pm @ H.T).item() + 1e-12
        K = Pm @ H.T / S
        P = (np.eye(3) - K @ H) @ Pm
    kalman = odd_jump_probability(math.sqrt(S))
    return genie, kalman


def analyze_e8(e2_rows):
    out = {}
    # (a) predictability: effective per-round error vs rho (level A) and floors
    sq, sm = E2_LEVELS["A"]
    rhos = list(E2_RHOS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    g = [floors(sq * L, sm * L, r) for r in rhos]
    ax.plot(rhos, [x[0] for x in g], "k--",
            label="genie floor on $P_L(t)$ (knows $\\varepsilon_{t-1}$, $\\nu_{t-1}$)")
    ax.plot(rhos, [x[1] for x in g], "k:",
            label="Kalman floor on $P_L(t)$ (knows unwrapped past)")
    for dec in ("NL", "Bayes (matched)", "LSTM-nominal", "LSTM-broad"):
        xs, ys, up = [], [], []
        for r in rhos:
            row = next((x for x in e2_rows if x["point"] == f"A_rho{r}" and x["decoder"] == dec), None)
            if row is None:
                continue
            k10 = round(row["ler10"] * row["trials"])
            if k10 == 0:
                up.append((r, 3.0 / row["trials"]))
            else:
                xs.append(r)
                ys.append(row["ler10"])
        st = dict(STYLE.get(dec, {}))
        st.pop("ls", None)
        ax.plot(xs, ys, marker=st.get("marker", "o"), color=st.get("color"),
                label=f"{dec}: $P_L(10)$")
        for r, u in up:
            ax.plot([r], [u], marker="v", mfc="white", color=st.get("color"), ls="none")
    ax.set_yscale("log")
    ax.set_xlabel("$\\rho$")
    ax.set_ylabel("P(logical error at round n)")
    ax.set_ylim(bottom=1e-6)
    ax.set_title("(a) predictability sets the floor ($\\sigma_q=0.2L$, $\\sigma_m=0.05L$)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    out["floors_levelA"] = dict(zip([str(r) for r in rhos], g))

    # (b) why NL improves with rho: odd jumps come in cancelling pairs
    ax = axes[1]
    pair_frac = {}
    for pid in ("rho0", "rho0.5", "nominal", "rho0.99"):
        path = RES / "cache" / "E8" / f"{pid}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            y = z["y"].astype(bool)
        prev = np.zeros_like(y)
        prev[:, 1:] = y[:, :-1]
        nxt = np.zeros_like(y)
        nxt[:, :-1] = y[:, 1:]
        odd = y.sum()
        # fraction of odd jumps with an odd neighbour, and fraction lying in
        # runs of even length (whose parities cancel)
        adjacent = float((y & (prev | nxt)).sum() / max(odd, 1))
        padded = np.pad(y.astype(np.int8), ((0, 0), (1, 1)))
        dif = np.diff(padded, axis=1)
        starts = np.argwhere(dif == 1)
        ends = np.argwhere(dif == -1)
        lengths = ends[:, 1] - starts[:, 1]
        cancel = float(lengths[lengths % 2 == 0].sum() / max(odd, 1))
        pair_frac[pid] = dict(adjacent=adjacent, in_even_runs=cancel)
    if pair_frac:
        labels = {"rho0": "0", "rho0.5": "0.5", "nominal": "0.9", "rho0.99": "0.99"}
        xs = np.arange(len(pair_frac))
        ax.bar(xs - 0.2, [v["adjacent"] for v in pair_frac.values()], 0.4, color="0.6",
               label="has an odd neighbour")
        ax.bar(xs + 0.2, [v["in_even_runs"] for v in pair_frac.values()], 0.4, color="0.3",
               label="in an even-length run (cancels)")
        ax.set_xticks(xs, [labels[k] for k in pair_frac])
        ax.legend(fontsize=8)
        ax.set_xlabel("$\\rho$")
        ax.set_ylabel("fraction of odd NL jumps")
        ax.set_title("(b) NL: correlated jumps cancel in pairs")
    out["nl_pair_fraction"] = pair_frac

    # (c) Bayes vs LSTM: disagreement, who is right, calibration
    ax = axes[2]
    diag = {}
    for pid in ("rho0", "rho0.5", "nominal", "rho0.99", "sq0.10", "sq0.30"):
        path = RES / "cache" / "E8" / f"{pid}.npz"
        meta = path.with_name(path.name + ".meta.json")
        if not path.exists():
            continue
        names = json.loads(meta.read_text())["decoders"]
        with np.load(path) as z:
            K = z["K"].astype(np.int64)
            arr = {k: z[k] for k in z.files}
        idx = {n: i for i, n in enumerate(names)}
        b, l = idx["Bayes (matched)"], idx["LSTM-nominal s0"]
        eb, el = arr[f"d{b}_err"], arr[f"d{l}_err"]
        dis = eb != el
        entry = dict(
            disagree=float(dis.mean()),
            bayes_right_when_disagree=float((~eb[dis]).mean()) if dis.any() else None,
            mean_ler={n: float(arr[f"d{i}_err"].mean()) for n, i in idx.items()},
        )
        # calibration of p_one: correct j_t is K_t xor J_{t-1}
        for n in ("Bayes (matched)", "LSTM-nominal s0"):
            i = idx[n]
            j = arr[f"d{i}_j"].astype(np.int64)
            p = arr[f"d{i}_p"].astype(float)
            J = np.cumsum(j, axis=1) % 2
            Jprev = np.concatenate([np.zeros((J.shape[0], 1), dtype=np.int64), J[:, :-1]], axis=1)
            target = K ^ Jprev
            bins = np.linspace(0, 1, 11)
            which = np.clip(np.digitize(p, bins) - 1, 0, 9)
            curve = []
            for k in range(10):
                m = which == k
                if m.sum() >= 20:
                    curve.append((float(p[m].mean()), float(target[m].mean()), int(m.sum())))
            entry[f"calibration_{n}"] = curve
            if pid == "nominal":
                xs = [c[0] for c in curve]
                ys = [c[1] for c in curve]
                ax.plot(xs, ys, marker="o", label=n,
                        color=STYLE.get(family(n), STYLE.get(n, {})).get("color"))
        diag[pid] = entry
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("predicted P(j=1 is correct)")
    ax.set_ylabel("observed frequency")
    ax.set_title("(c) calibration at nominal noise")
    ax.legend(fontsize=8)
    out["diagnostics"] = diag
    fig.tight_layout()
    fig.savefig(FIG / "E8_diagnostics.png", dpi=160)
    plt.close(fig)

    # (e) long horizon
    path = RES / "cache" / "E8" / "horizon200.npz"
    if path.exists():
        meta = json.loads(path.with_name(path.name + ".meta.json").read_text())
        names = meta["decoders"]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        hz = {}
        with np.load(path) as z:
            for i, n in enumerate(names):
                rec = np.unpackbits(z[f"d{i}_err"], axis=1, count=200).astype(bool)
                rec = rec[: meta["trials"]]
                curve = rec.mean(axis=0)
                hz[n] = dict(ler50=float(curve[49]), ler100=float(curve[99]),
                             ler200=float(curve[199]), mean_ler=float(rec.mean()))
                st = dict(STYLE.get(family(n.replace(" s0", " s0")), STYLE.get(n, {})))
                fam = "LSTM-nominal" if n.startswith("LSTM-nominal") else (
                    "LSTM-broad" if n.startswith("LSTM-broad") else n)
                st = dict(STYLE.get(fam, {}))
                st.pop("ls", None)
                ax.plot(np.arange(1, 201), np.where(curve > 0, curve, np.nan),
                        color=st.get("color"), label=n)
        ax.axvline(50, color="k", ls=":", lw=1)
        ax.text(51, ax.get_ylim()[0] if ax.get_yscale() == "log" else 0.01,
                "training length", fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("round $n$")
        ax.set_ylabel("$P_L(n)$")
        ax.set_title("E8(e): 200 rounds with models trained on 50")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(FIG / "E8_horizon.png", dpi=160)
        plt.close(fig)
        out["horizon200"] = hz

    # (d) value of revising past decisions: from E1
    data = load_point("E1", "nominal")
    if data:
        out["revision"] = {n: float(data[n]["errors"].mean()) for n in
                           ("Bayes (matched)", "Bayes (per-round)", "LSTM-nominal s0",
                            "LSTM-nominal s0 (k0 head)") if n in data}
    return out


def training_summary():
    rows = []
    import torch
    for p in sorted(out_dir("checkpoints").glob("*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        hist = ck["history"]
        rows.append(dict(model=p.stem, head=ck["chosen_head"], stop=ck["stop_reason"],
                         epochs=len(hist), seconds=ck["train_seconds"],
                         val_heads={h: v["mean_ler"] for h, v in ck["val_heads"].items()}))
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for p in sorted(out_dir("checkpoints").glob("*.pt")):
            ck = torch.load(p, map_location="cpu", weights_only=False)
            h = ck["history"]
            axes[0].plot([e["val_loss"] for e in h], label=p.stem)
            axes[1].plot([e["heads"][ck["chosen_head"]]["mean_ler"] for e in h], label=p.stem)
        axes[0].set_ylabel("validation BCE (all heads)")
        axes[1].set_ylabel("validation time-averaged logical error (chosen head)")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.set_yscale("log")
            ax.grid(alpha=0.3, which="both")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "training_curves.png", dpi=160)
        plt.close(fig)
    return rows


def main():
    import traceback
    log(f"analyze: mode={get_mode()}")
    summary = {}

    def section(key, fn, *args):
        try:
            summary[key] = fn(*args)
        except Exception:
            summary[key] = None
            log(f"analyze: section {key} FAILED\n{traceback.format_exc()}")

    summary["E0"] = json.loads((RES / "E0_grid_convergence.json").read_text()) \
        if (RES / "E0_grid_convergence.json").exists() else None
    section("E1", analyze_e1)
    e2_points = [(f"{lv}_rho{r}", dict(sigma_q=sq * L, sigma_m=sm * L, rho=r))
                 for lv, (sq, sm) in E2_LEVELS.items() for r in E2_RHOS]
    section("E2", analyze_sweep, "E2", e2_points, "$\\rho$", lambda t: t["rho"], "rho",
            "E2: correlation sweep", "E2_rho_sweep.png")
    e3_points = [(f"sq{s}", dict(sigma_q=s * L, sigma_m=0.05 * L, rho=0.9)) for s in E3_SQ]
    section("E3", analyze_sweep, "E3", e3_points, "$\\sigma_q / L$",
            lambda t: t["sigma_q"] / L, "sigma_q",
            "E3: displacement-noise sweep ($\\rho=0.9$, $\\sigma_m=0.05L$)",
            "E3_sigma_q_sweep.png")
    e4_points = [(f"sm{s}", dict(sigma_q=0.2 * L, sigma_m=s * L, rho=0.9)) for s in E4_SM]
    section("E4", analyze_sweep, "E4", e4_points, "$\\sigma_m / L$",
            lambda t: t["sigma_m"] / L, "sigma_m",
            "E4: measurement-noise sweep ($\\rho=0.9$, $\\sigma_q=0.2L$)",
            "E4_sigma_m_sweep.png")
    section("E5", analyze_e5)
    section("E6", analyze_e6)
    section("E7", analyze_e7)
    section("E8", analyze_e8, summary.get("E2") or [])
    section("training", training_summary)
    (RES / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: (
        o.item() if isinstance(o, np.generic) else o.tolist() if isinstance(o, np.ndarray)
        else str(o))))
    log(f"analyze: wrote {RES / 'summary.json'} and figures in {FIG}")


if __name__ == "__main__":
    main()
