"""Build an offline metrics viewer and Matplotlib figures from saved results.

Run from the repository root: .venv/bin/python -m writeup.artifact.build_metrics
No simulations or model training are performed. Outputs live in artifact/metrics/.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("metrics")
SOURCES = {}
SOURCE_BYTES = {}
CHARTS = []
SIZE_SOURCE = 'writeup/artifact/gkp-walkthrough/data/decoder-heatmaps.json'
INK, MUTED, GRID = "#172b42", "#5d6d7e", "#e7edf2"
TEAL, PURPLE, CORAL, BLUE = "#087e82", "#7960bd", "#ca634b", "#467fc3"
COLORS = {
    "NL": "#8996a5", "No correction": "#b7c0ca", "Bayes (matched)": TEAL,
    "Bayes-broad": "#58aa9e", "Bayes (nominal)": BLUE, "LSTM-nominal": PURPLE,
    "LSTM-broad": "#b879b2", "LSTM-switching": "#d09246",
    "Bayes (regime-aware)": TEAL, "Bayes (rho=0.9)": BLUE,
    "Bayes (rho=0.97)": "#a6bdcf", "Bayes (per-round)": "#58aa9e",
}
SHORT = {"NL": "Nearest lattice", "Bayes (matched)": "Bayes · matched",
         "Bayes (nominal)": "Bayes · nominal", "Bayes-broad": "Bayes · broad",
         "LSTM-nominal": "LSTM · nominal", "LSTM-broad": "LSTM · broad",
         "Bayes (regime-aware)": "Bayes · regime-aware",
         "Bayes (rho=0.9)": "Bayes · fixed ρ = 0.9",
         "Bayes (rho=0.97)": "Bayes · fixed ρ = 0.97",
         "LSTM-switching": "LSTM · switching", "No correction": "No correction"}
CORE = ["NL", "Bayes (matched)", "Bayes (nominal)", "Bayes-broad",
        "LSTM-nominal", "LSTM-broad"]


def read(path):
    p = ROOT / path
    raw = p.read_bytes()
    SOURCE_BYTES[path] = raw
    SOURCES[path] = {"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    if p.suffix == ".csv":
        data = list(csv.DictReader(io.StringIO(raw.decode())))
    elif p.suffix == ".json":
        data = json.loads(raw)
    else:
        return raw.decode()
    SOURCES[path]["records"] = len(data)
    return data


def pick(rows, **criteria):
    found = [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one result for {criteria}, found {len(found)}")
    return found[0]


def val(row, key):
    x = float(row[key])
    if not math.isfinite(x):
        raise ValueError(f"Nonfinite {key}: {row}")
    return x


def color(name):
    return COLORS.get(name, PURPLE if "LSTM" in name else TEAL)


def axes_style(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(length=0, colors=MUTED, pad=7)


def percentage_axis(ax, which="y", log=False):
    if log:
        getattr(ax, f"set_{which}scale")("log")
    getattr(ax, f"{which}axis").set_major_formatter(PercentFormatter(1, decimals=None))


def mean_scale(ax):
    # A linear region below 1e-5 keeps observed zeros visible without inventing a floor.
    ax.set_yscale("symlog", linthresh=1e-5, linscale=0.6)
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if x == 0 else f"{100*x:g}%"))
    axes_style(ax)


def interval(row, metric):
    # Between-seed spread is not a confidence interval for the aggregate mean.
    if row.get("seeds"):
        return val(row, metric + "_seed_min"), val(row, metric + "_seed_max"), "seed min–max"
    if metric == "ler50":
        return val(row, "ler50_lo"), val(row, "ler50_hi"), "95% Wilson interval"
    y, se = val(row, metric), val(row, metric + "_se")
    return max(0, y - 1.96 * se), y + 1.96 * se, "95% normal MC interval"


def csv_text(rows, columns):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=[c[0] for c in columns], extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def publish(fig, slug, title, subtitle, category, tag, caption, sources, rows, columns, wide=True):
    folder = OUT / "figures"
    svg, png = folder / f"{slug}.svg", folder / f"{slug}.png"
    fig.savefig(svg, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    fig.savefig(png, bbox_inches="tight", facecolor="white", dpi=190)
    plt.close(fig)
    raw_csv = csv_text(rows, columns)
    (OUT / "data" / f"{slug}.csv").write_text(raw_csv)
    CHARTS.append(dict(id=slug, title=title, subtitle=subtitle, category=category, tag=tag,
                       caption=caption, sources=sources, rows=rows, columns=columns,
                       wide=wide, csv=raw_csv,
                       svg="data:image/svg+xml;base64," + base64.b64encode(svg.read_bytes()).decode(),
                       png="data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()))


def nominal(e1):
    order = ["Bayes (matched)", "Bayes-broad", "LSTM-nominal", "LSTM-broad", "NL", "No correction"]
    fig, axs = plt.subplots(1, 2, figsize=(11.7, 4.5), sharey=True)
    rows = []
    for ax, metric, label in zip(axs, ["ler50", "mean_ler"],
                                 ["Logical error after 50 rounds", "Time-averaged logical error"]):
        for i, name in enumerate(order):
            r = pick(e1, decoder=name)
            y = val(r, metric)
            lo, hi, kind = interval(r, metric)
            ax.errorbar(y, i, xerr=[[y-lo], [hi-y]], fmt="D" if r.get("seeds") else "o",
                        color=color(name), capsize=4, markersize=7, elinewidth=1.8)
            ax.annotate(f"{100*y:.3g}%", (y, i), xytext=(8, -15), textcoords="offset points",
                        fontsize=9, color=INK)
            rows.append(dict(decoder=name, metric=metric, estimate=y, lower=lo, upper=hi,
                             interval=kind, trials=int(val(r, "trials")), seeds=int(r.get("seeds") or 1)))
        ax.set_xscale("log")
        ax.set_xlim(3e-4, 0.95)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(label, loc="left", pad=20)
        ax.set_xlabel("Error probability · logarithmic scale")
        axes_style(ax, "x")
    axs[0].set_yticks(range(len(order)), [SHORT[n] for n in order])
    axs[0].invert_yaxis()
    fig.tight_layout(w_pad=2)
    publish(fig, "01-nominal", "Memory changes the baseline", "E1 · 60,000 trials per decoder / seed · 50 rounds", "core", "Full evaluation",
            "Nominal noise: σq = 0.20L, σm = 0.05L, ρ = 0.9. Circles show fixed decoders with 95% Monte Carlo intervals; diamonds show three-seed LSTM means with seed min–max ranges. The two panels use different metrics. Lower is better.",
            ["results/E1_summary.csv"], rows,
            [("decoder", "Decoder"), ("metric", "Metric"), ("estimate", "Estimate"), ("lower", "Lower"), ("upper", "Upper"), ("interval", "Interval type"), ("trials", "Trials / seed"), ("seeds", "Seeds")])


def draw_sweep(ax, rows, xkey):
    plotted = []
    for name in CORE:
        rr = sorted([r for r in rows if r["decoder"] == name], key=lambda r: val(r, xkey))
        if not rr:
            continue
        x, y = [val(r, xkey) for r in rr], [val(r, "mean_ler") for r in rr]
        ax.plot(x, y, marker="D" if "LSTM" in name else "o", ms=4.2, lw=1.8,
                color=color(name), label=SHORT.get(name, name),
                linestyle="--" if name in ("Bayes (nominal)", "LSTM-nominal") else "-")
        if rr[0].get("seeds"):
            ax.fill_between(x, [val(r, "mean_ler_seed_min") for r in rr],
                            [val(r, "mean_ler_seed_max") for r in rr], color=color(name), alpha=.10)
        for r in rr:
            plotted.append(dict(point=r["point"], decoder=name, sigma_q=val(r, "sigma_q"),
                                sigma_m=val(r, "sigma_m"), rho=val(r, "rho"), mean_ler=val(r, "mean_ler"),
                                trials=int(val(r, "trials")), seeds=int(r.get("seeds") or 1)))
    mean_scale(ax)
    ax.set_ylabel("Time-averaged logical error")
    return plotted


SWEEP_COLS = [("point", "Point"), ("decoder", "Decoder"), ("sigma_q", "σq / L"),
              ("sigma_m", "σm / L"), ("rho", "ρ"), ("mean_ler", "Mean error"),
              ("trials", "Trials / seed"), ("seeds", "Seeds")]


def sweeps(e2, e3, e4):
    fig, axs = plt.subplots(1, 2, figsize=(11.7, 4.9))
    rows = []
    for ax, level, title in zip(axs, ("A", "B"), ("σq = 0.20L · σm = 0.05L", "σq = 0.30L · σm = 0.08L")):
        rows += draw_sweep(ax, [r for r in e2 if r["point"].startswith(level + "_")], "rho")
        ax.set_title(title, loc="left", pad=16)
        ax.set_xlabel("Displacement correlation ρ")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.015), fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, 1))
    publish(fig, "02-correlation", "Correlation makes history useful", "E2 · Two noise scales · Nine correlation settings", "core", "Full evaluation",
            "Shading is the min–max spread of three LSTM seeds, not a confidence band. Matched Bayes uses the true noise parameters at each point. The vertical scale is logarithmic above 0.001% and linear down to zero below it. Point estimates are connected as a visual guide.",
            ["results/E2_summary.csv"], rows, SWEEP_COLS)
    fig, axs = plt.subplots(1, 2, figsize=(11.7, 4.9))
    rows = draw_sweep(axs[0], e3, "sigma_q") + draw_sweep(axs[1], e4, "sigma_m")
    for ax, title, label in zip(axs, ("Vary displacement · σm = 0.05L", "Vary readout · σq = 0.20L"), ("Displacement noise σq / L", "Readout noise σm / L")):
        ax.set_title(title, loc="left", pad=16)
        ax.set_xlabel(label)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, -.015), fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, 1))
    publish(fig, "03-noise", "Performance depends on the noise regime", "E3–E4 · Displacement and measurement-noise sweeps · ρ = 0.9", "core", "Full evaluation",
            "Same decoder definitions and seed bands as E2. Observed zeros are shown at zero on a scale that is linear below 0.001% and logarithmic above; zero observed errors do not establish zero risk. Sparse, separately trained matched-LSTM points are not included in these six-series sweeps.",
            ["results/E3_summary.csv", "results/E4_summary.csv"], rows, SWEEP_COLS)


def ablation(e1):
    fig, axs = plt.subplots(1, 2, figsize=(11.7, 3.7))
    rows = []
    cases = [("Bayesian decoder", "Bayes (matched)", "Bayes (per-round)", TEAL),
             ("LSTM · same seed 0", "LSTM-nominal s0", "LSTM-nominal s0 (k0 head)", PURPLE)]
    for ax, (title, full, per, col) in zip(axs, cases):
        vs = [val(pick(e1, decoder=n), "mean_ler") for n in (full, per)]
        ax.barh([0, 1], vs, height=.44, color=[col, "#c8d3dc"])
        for i, (name, v) in enumerate(zip((full, per), vs)):
            ax.text(v + .00025, i, f"{100*v:.3f}%", va="center", fontsize=10, color=INK)
            rows.append(dict(decoder=name, mean_ler=v, reference=full, ratio_to_revision=v/vs[0]))
        ax.set_yticks([0, 1], ["Cumulative revision", "Per-round decisions"])
        ax.invert_yaxis()
        ax.set_xlim(0, .022)
        ax.set_title(f"{title} · {vs[1]/vs[0]:.2f}× lower with revision", loc="left", pad=17)
        ax.set_xlabel("Time-averaged logical error")
        percentage_axis(ax, "x")
        axes_style(ax, "x")
    fig.tight_layout(w_pad=2)
    publish(fig, "04-revision", "Revising past decisions matters", "E1 ablations · Matched Bayes and the same LSTM checkpoint", "core", "Full evaluation",
            "This compares cumulative-parity decoding with per-round decisions. The LSTM comparison uses seed 0 for both heads; it does not compare the three-seed aggregate to a single-seed ablation. Bars show point estimates.",
            ["results/E1_summary.csv"], rows,
            [("decoder", "Decoder / head"), ("mean_ler", "Mean error"), ("reference", "Reference"), ("ratio_to_revision", "Error ÷ revision error")])


def mismatch(e5):
    parsed = []
    for r in e5:
        m = re.fullmatch(r"Bayes rho'=(\S+) sq'x(\S+)", r["decoder"])
        if m:
            parsed.append((float(m[1]), float(m[2]), r))
    parsed.append((.9, 1., pick(e5, decoder="Bayes (matched)")))
    rhos, scales = sorted({a for a, _, _ in parsed}), sorted({b for _, b, _ in parsed})
    base = val(pick(e5, decoder="NL"), "mean_ler")
    matrix = np.full((len(rhos), len(scales)), np.nan)
    rows = []
    for rho, scale, r in parsed:
        ratio = val(r, "mean_ler") / base
        matrix[rhos.index(rho), scales.index(scale)] = ratio
        rows.append(dict(assumed_rho=rho, sigma_q_multiplier=scale, mean_ler=val(r, "mean_ler"),
                         nearest_lattice_mean=base, ratio_vs_nl=ratio))
    assert np.isfinite(matrix).all()
    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    cmap = LinearSegmentedColormap.from_list("mismatch", ["#087e82", "#f3f6f7", "#d36e55"])
    im = ax.imshow(np.log2(matrix), cmap=cmap, norm=TwoSlopeNorm(vmin=-7, vcenter=0, vmax=2), aspect="auto")
    for y in range(len(rhos)):
        for x in range(len(scales)):
            ratio = matrix[y, x]
            ax.text(x, y, f"{ratio:.2f}×", ha="center", va="center", fontsize=11,
                    color="white" if ratio < .08 else INK)
    ax.set_xticks(range(len(scales)), [f"{v:g}×" for v in scales])
    ax.set_yticks(range(len(rhos)), [f"{v:g}" for v in rhos])
    ax.set_xlabel("Assumed displacement noise / true displacement noise")
    ax.set_ylabel("Assumed correlation ρ′")
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=.035, ticks=[-6, -3, 0, 1, 2])
    cb.ax.set_yticklabels(["1/64×", "1/8×", "1×", "2×", "4×"])
    cb.set_label("Mean error relative to nearest lattice")
    fig.tight_layout()
    publish(fig, "05-mismatch", "Wrong assumptions can erase the gain", "E5 · Model mismatch at the nominal point", "robustness", "Full evaluation",
            "Each cell is Bayes time-averaged error divided by nearest-lattice error on the same true noise model. Below 1× is better; above 1× is worse. True σq = 0.20L, σm = 0.05L, ρ = 0.9. The matched reference fills the ρ′ = 0.9, noise ×1 cell. This chart covers E5; E5b exists as a completed log, not a summary table.",
            ["results/E5_summary.csv"], rows,
            [("assumed_rho", "Assumed ρ"), ("sigma_q_multiplier", "σq multiplier"), ("mean_ler", "Mean error"), ("nearest_lattice_mean", "NL mean error"), ("ratio_vs_nl", "Error ÷ NL")])


def switching(e6):
    names = ["Bayes (regime-aware)", "LSTM-broad", "LSTM-switching", "Bayes-broad",
             "Bayes (rho=0.9)", "LSTM-nominal", "Bayes (rho=0.97)", "NL"]
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    rows = []
    for i, name in enumerate(names):
        r = pick(e6, decoder=name)
        y, (lo, hi, kind) = val(r, "mean_ler"), interval(r, "mean_ler")
        ax.errorbar(y, i, xerr=[[y-lo], [hi-y]], color=color(name), fmt="D" if r.get("seeds") else "o",
                    capsize=4, ms=7)
        ax.text(.218, i, f"{100*y:.2f}%", va="center", fontsize=10)
        rows.append(dict(decoder=name, mean_ler=y, lower=lo, upper=hi, interval=kind, trials=int(val(r,"trials"))))
    ax.set_yticks(range(len(names)), [SHORT[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(.11, .236)
    ax.set_xlabel("Time-averaged logical error · truncated horizontal axis")
    percentage_axis(ax, "x")
    axes_style(ax, "x")
    fig.tight_layout()
    publish(fig, "06-switching", "Learned decoding helps under structural mismatch", "E6 · Two-regime switching noise · 20,000 trials per decoder / seed", "robustness", "Full evaluation",
            "Broad LSTM point estimates beat the fixed-model Bayesian baselines here, while regime-aware Bayes remains best. Circles use 95% normal Monte Carlo intervals; diamonds show three-seed means and seed min–max spread. This is a comparison within the switching model, not a claim about all nonstationary noise.",
            ["results/E6_summary.csv"], rows,
            [("decoder", "Decoder"), ("mean_ler", "Mean error"), ("lower", "Lower"), ("upper", "Upper"), ("interval", "Interval type"), ("trials", "Trials / seed")])


def drift(stress):
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    rows = []
    for i, (point, r) in enumerate(stress.items()):
        for off, key, label, col in [(-.12, "excess_fixed", "Fixed nominal Bayes", BLUE),
                                     (.12, "excess_imm", "Adaptive Bayes (IMM)", TEAL)]:
            y = r.get(key)
            if y is not None: ax.plot(y, i+off, "o", color=col, ms=6, label=label if i==0 else None)
        rows.append(dict(point=point, oracle_eps=r["eps"].get("Bayes (oracle)"),
                         fixed_ratio=r.get("excess_fixed"), imm_ratio=r.get("excess_imm"),
                         horizon=r["horizon"], fixed_pl_gap=r["fixed_minus_oracle"][0],
                         fixed_pl_gap_se=r["fixed_minus_oracle"][1]))
    ax.axvline(1, color=MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(stress)), [x.replace("_", " ") for x in stress])
    ax.invert_yaxis()
    ax.set_xlabel("Fitted error per round ε / oracle ε · point estimates")
    ax.set_xlim(.8, 1.58)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=GRID, fontsize=9)
    axes_style(ax, "x")
    fig.tight_layout()
    publish(fig, "07-drift", "Drift stress tests keep Bayes near the oracle", "E9–E11 scenarios · Development random streams", "robustness", "Development benchmark",
            "These are development stress-test results, not the unfinished full E9–E11 evaluation. Runs use 2,000 trials × 1,000 rounds (IMM: 1,000 trials). ε is fitted before saturation. Ratios below 1 can arise from estimation noise and are not evidence of beating an exact oracle. Confidence intervals for ε ratios are not supplied.",
            ["results/benchmarks/stress_bayes.json"], rows,
            [("point", "Scenario"), ("oracle_eps", "Oracle ε / round"), ("fixed_ratio", "Fixed ÷ oracle"), ("imm_ratio", "IMM ÷ oracle"), ("horizon", "Paired horizon"), ("fixed_pl_gap", "Fixed − oracle P_L"), ("fixed_pl_gap_se", "Paired SE")])


def binary(e14):
    fig, axs = plt.subplots(2, 2, figsize=(11.7, 6.8), sharex=True, sharey=True)
    rows = []
    for ax, (scale, rho) in zip(axs.flat, [("hardware", 0.), ("hardware", .9), ("nominal", 0.), ("nominal", .9)]):
        for group, col, marker in [("Bayes (analog, matched)", "#8996a5", "s"),
                                   ("Bayes (binary, matched)", TEAL, "o"), ("Best sBs of 3 δ", CORAL, "D")]:
            ys = []
            for m in (1, 4, 8):
                subset = [r for r in e14 if r["scale"] == scale and r["rho"] == rho
                          and r["m"] == m and r["visibility"] == 1 and r["p_flip"] == 0]
                r = min([r for r in subset if r["decoder"].startswith("sBs")], key=lambda r:r["eps"]) if group.startswith("Best") else pick(subset, decoder=group)
                ys.append(r["eps"])
                rows.append(dict(scale=scale, rho=rho, bits=m, series=group, decoder=r["decoder"],
                                 eps=r["eps"], pl_h=r["pl_h"], horizon=r["horizon"]))
            assert all(y is not None and y > 0 for y in ys)
            ax.plot([1,4,8], ys, marker=marker, color=col, lw=2, ms=5,
                    ls="--" if "analog" in group else "-", label={"Bayes (analog, matched)":"Analog Bayes", "Bayes (binary, matched)":"Binary Bayes"}.get(group,group))
        ax.set_yscale("log")
        ax.set_ylim(1e-6, .11)
        ax.set_title(f"{scale.capitalize()} scale · ρ = {rho:g}", loc="left", pad=12)
        ax.set_xticks([1,4,8])
        ax.set_xlabel("Ancilla bits per round")
        ax.set_ylabel("Fitted error per round ε")
        axes_style(ax)
    fig.legend(*axs[0,0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0,.055,1,1), h_pad=1.8)
    publish(fig, "08-binary", "Bits per round determine the memory advantage", "E14 · Ideal visibility and no readout flips · 4,000 trials × 200 rounds", "readout", "Baselines complete",
            "The best sBs curve selects the lowest ε among the three tested δ values at each point; it is not an independently tuned policy. Analog and binary measurements carry different information. All bits in a round see the same displacement, so these are per-round comparisons, not equal-time hardware comparisons. The trained bit-LSTM is absent. All 120 E14 rows, including imperfect readout, are available in the data explorer.",
            ["results/E14_summary.json"], rows,
            [("scale", "Noise scale"), ("rho", "ρ"), ("bits", "Bits / round"), ("series", "Series"), ("decoder", "Selected decoder"), ("eps", "ε / round"), ("pl_h", "P_L at paired horizon"), ("horizon", "Horizon")])


def continuous(e15):
    order = ["nominal", "hw_sm0.05", "hw_sm0.10", "hw", "hw_sm0.15"]
    labels = ["Nominal", "HW\nσm=.05L", "HW\nσm=.10L", "HW\nσm=.105L", "HW\nσm=.15L"]
    fig, axs = plt.subplots(1,2,figsize=(11.7,4.6))
    for off, name, col, mark in [(-.12,"NL","#8996a5","s"), (0,"Bayes (parity)",TEAL,"o"), (.12,"Bayes (fine)",PURPLE,"D")]:
        ys = [pick(e15, point=p, decoder=name)["pl_h"] for p in order]
        axs[0].scatter(np.arange(5)+off, ys, color=col, marker=mark, s=42, label=name, zorder=3)
    axs[0].set_yscale("symlog", linthresh=1e-4, linscale=.7)
    axs[0].set_ylim(0,.85)
    axs[0].set_yticks([0,1e-4,1e-3,1e-2,1e-1])
    axs[0].yaxis.set_major_formatter(FuncFormatter(lambda y,_:"0" if y==0 else f"{100*y:g}%"))
    axs[0].set_ylabel("Logical-error probability P_L(200)")
    axs[0].set_title("Final-round error",loc="left",pad=16)
    axs[0].legend(frameon=False,fontsize=9,loc="upper left")
    fine = [pick(e15, point=p, decoder="Bayes (fine)") for p in order]
    axs[1].axhline(0,color=MUTED,lw=1)
    axs[1].errorbar(range(5),[100*r["diff_vs_bayes_parity"] for r in fine],
                    yerr=[196*r["diff_se"] for r in fine],fmt="o",color=PURPLE,capsize=5,ms=6)
    axs[1].set_ylabel("Fine − parity P_L · percentage points")
    axs[1].set_title("Paired difference · 95% normal intervals",loc="left",pad=16)
    for ax in axs:
        ax.set_xticks(range(5),labels)
        axes_style(ax)
    fig.tight_layout(w_pad=2)
    publish(fig, "09-continuous", "Fine corrections barely change logical error here", "E15 · Closed-loop baselines · 20,000 trials × 200 rounds", "readout", "DQN pending",
            "Left: point estimates, with a linear region below 0.01% to show zero observations. Right: paired fine-minus-parity differences with ±1.96 SE; negative values favor fine correction. Zero errors were observed for both Bayesian policies at HW σm = 0.05L; this does not establish zero risk. DQN checkpoints and evaluated DQN rows are absent. The shift model omits finite-energy dissipation and ancilla back-action.",
            ["results/E15_summary.json"], e15,
            [("point","Point"),("decoder","Decoder"),("pl_h","P_L(200)"),("eps","ε / round"),("diff_vs_bayes_parity","ΔP_L vs parity"),("diff_se","Paired SE"),("trials","Trials"),("horizon","Horizon")])


def compute(cost):
    rs = cost["matched_accuracy"]
    fig, axs = plt.subplots(1,2,figsize=(11.7,5),gridspec_kw={"width_ratios":[1.5,1]})
    offsets = [(5,-16),(8,10),(0,-22),(0,12),(0,-24),(5,10),(0,14),(5,7),(-5,14),(4,-22),(7,2)]
    rows = []
    for i, (r, offset) in enumerate(zip(rs, offsets)):
        name=r["decoder"]
        col=PURPLE if "LSTM" in name else ("#8996a5" if name=="NL" else TEAL)
        x,y=r["us_per_trial_round"],r["mean_ler"]
        key=chr(65+i)
        axs[0].scatter(x,y,c=col,s=50,zorder=3)
        axs[0].annotate(key,(x,y),xytext=offset,textcoords="offset points",fontsize=10,fontweight="bold",color=col)
        axs[1].text(0, .96-i*.082, key, color=col, weight="bold", va="center")
        axs[1].text(.09, .96-i*.082, name.replace("Bayes (matched) n_x=","Bayes · grid "),fontsize=9,va="center")
        rows.append(dict(key=key,decoder=name,mean_ler=y,us_per_trial_round=x,trials=r["trials"],
                         rounds=r["rounds"],device="CPU" if name=="NL" else "CUDA",peak_gpu_gb=r["peak_gpu_gb"]))
    axs[0].set_xscale("log");axs[0].set_yscale("log")
    axs[0].set_xlim(.02,160);axs[0].set_ylim(.0009,.3)
    axs[0].set_xlabel("Microseconds per trial-round · lower is faster")
    axs[0].set_ylabel("Time-averaged logical error · lower is better")
    axs[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes_style(axs[0],"both")
    axs[1].axis("off")
    fig.tight_layout(w_pad=2)
    publish(fig,"10-compute","Accuracy and throughput have different tradeoffs", "Recorded cost benchmark · Accuracy and timing from the same runs", "compute", "Development benchmark",
            f"GPU: {cost['env']['gpu']}; PyTorch {cost['env']['torch']}. Nearest lattice runs on CPU; other shown decoders use CUDA. Batches are 8,000 trials except IMM and window filters (2,000), all over 50 rounds. This is batched throughput, not single-stream feedback latency, and excludes training cost. The y-axis uses mean error rather than the potentially misleading long-run flip-rate metric. Timings were not remeasured on this machine.",
            ["results/benchmarks/cost.json"],rows,
            [("key","Plot key"),("decoder","Decoder"),("mean_ler","Mean error"),("us_per_trial_round","µs / trial-round"),("trials","Batch trials"),("rounds","Rounds"),("device","Device"),("peak_gpu_gb","Peak GPU GB")])


def validation(v):
    checks = [dict(check=f"Single round · σ = {r['sigma']:g}",z=r["z"]) for r in v["A"]]
    checks += [dict(check="First-round flip",z=v["B"][0]["z1"])]
    checks += [dict(check=f"Later-round flip · ρ = {r['rho']:g}",z=r["z_later"]) for r in v["B"]]
    checks += [dict(check="ρ = 0 decay · max |z| across 50 rounds",z=v["C1"]["zmax"])]
    fig,ax=plt.subplots(figsize=(10.6,5.9))
    ax.axvspan(-2,2,color="#e5f3f0")
    ax.axvline(0,color="#99c7bd",lw=1)
    ax.axvline(-2,color=TEAL,ls="--",lw=1);ax.axvline(2,color=TEAL,ls="--",lw=1)
    ax.scatter([r["z"] for r in checks],range(len(checks)),color=TEAL,s=40,zorder=3)
    ax.set_yticks(range(len(checks)),[r["check"] for r in checks])
    ax.invert_yaxis();ax.set_xlim(-2.6,2.6)
    ax.set_xlabel("Deviation from the analytic reference / Monte Carlo standard error")
    axes_style(ax,"x");fig.tight_layout()
    publish(fig,"11-validation","Exact-check summaries stay within two standard errors", "Simulator validation rerun · Analytic references", "validation", "Development benchmark",
            "The 15 plotted summaries cover exact single-round and flip formulas plus the maximum absolute z-score of the 50-round independent-noise decay check. The shaded interval is −2 to +2 standard errors. Approximate limiting checks are excluded from this exact-check plot. Agreement validates these simplified-model identities; it does not validate full hardware dynamics.",
            ["results/benchmarks/validation_rerun.json"],checks,[("check","Check"),("z","z-score / max |z|")])


def coverage(cov):
    fig,ax=plt.subplots(figsize=(10.6,5))
    rows=[]
    for i,(name,xy) in enumerate(cov["feature_medians"].items()):
        dev=name.startswith(("E9/","E10/","E11/"))
        group="E9–E11 schedules" if dev else "Core scenarios"
        col=PURPLE if dev else BLUE
        ax.scatter(*xy,color=col,s=42,alpha=.78,marker="D" if dev else "o")
        rows.append(dict(scenario=name,group=group,syndrome_sd_L=xy[0],syndrome_lag1=xy[1]))
        if name in ("E1 nominal","E2 rho=0.0"):
            ax.annotate(name,xy,xytext=(8,10),textcoords="offset points",fontsize=9,color=INK)
    for name,r in cov["hardware_anchor_coverage"].items():
        x,y=r["median_syndrome_sd"],r["median_lag1"]
        ax.scatter(x,y,color=CORAL,s=100,marker="*",zorder=4)
        ax.annotate(name.replace("hardware scale, ","Assumed "),(x,y),xytext=(10,-4),textcoords="offset points",fontsize=9,color=CORAL)
        rows.append(dict(scenario=name,group="Hardware-scale simulation",syndrome_sd_L=x,syndrome_lag1=y))
    for col,marker,label in [(BLUE,"o","Core scenarios"),(PURPLE,"D","E9–E11 schedules"),(CORAL,"*","Hardware-scale simulations")]:
        ax.scatter([],[],color=col,marker=marker,s=45,label=label)
    ax.axhline(0,color=GRID,lw=1)
    ax.set_xlabel("Median syndrome standard deviation / L")
    ax.set_ylabel("Median lag-1 syndrome correlation")
    ax.set_ylim(-.48,.75)
    ax.legend(loc="upper left",frameon=False,fontsize=9)
    axes_style(ax,"both");fig.tight_layout()
    publish(fig,"12-coverage","Hardware-scale simulations occupy a different regime", "Coverage benchmark · Snapshot before the E12 expansion", "validation", "Development benchmark",
            "Each point is a simulated feature median, including the stars. Hardware-scale anchors use σq ≈ 0.072L and larger readout noise; ρ is assumed, not measured on a GKP device. This pre-E12 benchmark explains the coverage gap that motivated E12. It does not include the subsequent partial E12 evaluation or show experimental syndrome records.",
            ["results/benchmarks/coverage_map.json"],rows,
            [("scenario","Scenario"),("group","Group"),("syndrome_sd_L","Syndrome SD / L"),("syndrome_lag1","Lag-1 correlation")])


def lstm_sizes(data):
    """One published snapshot, with missing models kept explicitly pending."""
    sweep, cfg = data['size_sweep'], data['config']
    sizes = [m for m in cfg['decoders'] if m.get('group') == 'size_sweep']
    n = cfg['trials']
    nominal_id = 'sigma_q0.200_sigma_m0.050_rho0.900'
    rows = []
    for model in sizes:
        records = [cell['models'].get(model['id']) for cell in data['cells'].values()]
        complete = len(records) == 61 and all(r and r['status'] == 'complete' for r in records)
        row = dict(model=model['id'], width=model['hidden'], layers=model['layers'], seed=model['seed'],
                   status='complete' if complete else 'pending', trials_per_setting=n,
                   evaluated_settings=61 if complete else 0, round=cfg['rounds'])
        if complete:
            k = data['cells'][nominal_id]['models'][model['id']]['counts'][-1]
            z = 1.959963984540054
            p, den = k/n, 1 + z*z/n
            center = (p + z*z/(2*n))/den
            half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
            row.update(nominal_errors=k, nominal_probability=p,
                       nominal_wilson95_lo=max(0, center-half), nominal_wilson95_hi=min(1, center+half),
                       mean_probability_over_grid=float(np.mean([r['counts'][-1]/n for r in records])),
                       cells_estimate_above_10pct=sum(r['counts'][-1]/n > .1 for r in records),
                       parameters=model['n_params'], epochs=model['epochs'], best_epoch=model['best_epoch'],
                       chosen_head=model['head'])
        rows.append(row)
    completed = sum(r['status'] == 'complete' for r in rows)
    assert len(rows) == 36 and completed * 61 == sweep['evaluated']
    cols = [(k, label) for k, label in [
        ('model','Model'),('width','Width'),('layers','Layers'),('seed','Seed'),('status','Status'),
        ('round','Readout round'),('trials_per_setting','Trials / setting'),('evaluated_settings','Evaluated settings'),
        ('nominal_errors','Nominal errors'),('nominal_probability','Nominal P_L(50)'),
        ('nominal_wilson95_lo','Nominal 95% lower'),('nominal_wilson95_hi','Nominal 95% upper'),
        ('mean_probability_over_grid','Equal-weight mean over 61 settings'),
        ('cells_estimate_above_10pct','Settings estimated above 10%'),('parameters','Parameters'),
        ('epochs','Epochs'),('best_epoch','Best epoch'),('chosen_head','Chosen head')]]
    cmap = LinearSegmentedColormap.from_list('size_error', ['#edf6f2','#b1dbcc','#f3d790','#db8657','#9d2c27'])
    shared = (f"{completed}/36 evaluated models · {sweep['evaluated']:,}/2,196 model/settings · "
              f"{n:,} trials per setting · snapshot {data['updated']}")
    for key, slug, title, caption in [
        ('nominal_probability','13-lstm-size-nominal','LSTM size at nominal noise',
         'P_L(50) at σq/L = 0.20, σm/L = 0.05, ρ = 0.90. Each cell is one independently trained model. '
         'Pointwise 95% Wilson intervals are in the data table and CSV; they describe trial uncertainty for a fixed model.'),
        ('mean_probability_over_grid','14-lstm-size-noise-grid','LSTM size across the noise grid',
         'Each cell averages P_L(50) equally across the 61 stationary noise settings. This descriptive grid mean is '
         'not expected error under a physical operating distribution, and is different from averaging error over rounds. '
         'Settings share random draws, so they are not 61 independent experiments.')]:
        fig, axs = plt.subplots(1, 3, figsize=(11.8, 3.7), layout='constrained')
        for ax, seed in zip(axs, sweep['config']['seeds']):
            rr = [r for r in rows if r['seed'] == seed]
            matrix = np.full((3,4), np.nan)
            for i, depth in enumerate(sweep['config']['depths']):
                for j, width in enumerate(sweep['config']['widths']):
                    r = next(r for r in rr if r['layers'] == depth and r['width'] == width)
                    if r['status'] == 'complete':
                        matrix[i,j] = r[key]
                    else:
                        ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='#f8fafb',edgecolor='#dfe7ec',hatch='///',lw=.5,zorder=2))
                        ax.text(j,i,'pending',ha='center',va='center',fontsize=8,color=MUTED,zorder=3)
            im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=.5, aspect='auto')
            for i,j in zip(*np.where(np.isfinite(matrix))):
                v = matrix[i,j]
                ax.text(j,i,f'{100*v:.1f}%',ha='center',va='center',fontsize=10,color='white' if v>.38 else INK)
            ax.set_xticks(range(4),sweep['config']['widths']); ax.set_yticks(range(3),sweep['config']['depths'])
            ax.set_xlabel('Hidden units'); ax.set_ylabel('Layers')
            ax.set_title(f'Seed {seed} · {sum(r["status"]=="complete" for r in rr)}/12 models',loc='left',pad=14)
            ax.tick_params(length=0)
            for spine in ax.spines.values(): spine.set_visible(False)
        cb = fig.colorbar(im,ax=axs,shrink=.86,pad=.025,format=PercentFormatter(1))
        cb.set_label('Error probability · lower is better')
        cb.set_ticks([0,.1,.2,.3,.4,.5],labels=['0%','10%','20%','30%','40%','≥50%'])
        publish(fig,slug,title,shared,'sizes','Development sweep',
                caption+' Seeds remain separate. Stripes mean pending, not zero error. Colors saturate at 50%; labels retain the measured values. '
                'All models train at nominal noise with the same 200,000-sequence, up-to-40-epoch protocol; off-nominal error reflects noise mismatch as well as capacity.',
                [SIZE_SOURCE],rows,cols)
    return dict(run_id=data['run_id'],trained=sweep['trained'],evaluated=sweep['evaluated'],
                evaluated_models=completed,models=36,expected_evaluations=2196,updated=data['updated'])


def build():
    SOURCES.clear(); SOURCE_BYTES.clear(); CHARTS.clear()
    OUT.mkdir(exist_ok=True)
    for folder in ("figures","data","sources"):
        (OUT/folder).mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "axes.titlesize":11, "axes.titleweight":"bold", "axes.labelsize":10,
        "text.color":INK, "axes.labelcolor":MUTED, "xtick.labelsize":9, "ytick.labelsize":10,
        "legend.fontsize":9, "savefig.pad_inches":.18, "svg.fonttype":"path"})
    es = {i: read(f"results/E{i}_summary.csv") for i in range(1,7)}
    e14,e15=read("results/E14_summary.json"),read("results/E15_summary.json")
    stress=read("results/benchmarks/stress_bayes.json")
    cost=read("results/benchmarks/cost.json")
    v=read("results/benchmarks/validation_rerun.json")
    cov=read("results/benchmarks/coverage_map.json")
    size_info = None
    if (ROOT / SIZE_SOURCE).exists():
        size_data = read(SIZE_SOURCE)
        if size_data.get('size_sweep'):
            size_info = lstm_sizes(size_data)
    nominal(es[1]);sweeps(es[2],es[3],es[4]);ablation(es[1]);mismatch(es[5]);switching(es[6])
    drift(stress);binary(e14);continuous(e15);compute(cost);validation(v);coverage(cov)
    read("docs/findings.md")
    logs={name:read(f"results/logs/{name}.log") for name in ("full_E5b","full_E9-E11","full_E12","quick_E13","full_E14","full_E15")}
    assert "failed experiments: none" in logs["full_E5b"]
    assert len({r["point"] for r in e14})==24 and len({r["point"] for r in e15})==5
    assert all(not r["decoder"].startswith("DQN") for r in e15)
    def git(*args):
        return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
    b,n,l=[pick(es[1],decoder=d) for d in ("Bayes (matched)","NL","LSTM-nominal")]
    payload=dict(title="GKP decoder metrics",commit=git("rev-parse","HEAD"),
        commit_date=git("show","-s","--format=%cI","HEAD"),
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        matplotlib=matplotlib.__version__,charts=CHARTS,sources=list(SOURCES.values()),
        headline=dict(gain=val(n,"mean_ler")/val(b,"mean_ler"),bayes_p50=val(b,"ler50"),
                      lstm_gap=val(l,"mean_ler")/val(b,"mean_ler"),binary_points=24),
        all_binary_rows=e14,
        size_sweep=size_info,
        status=[
            dict(id="E0–E8",name="Core decoder study",state="Complete",kind="complete",detail="Committed summaries, figures, checkpoints and report.",source="RESULTS.md"),
            dict(id="E5b",name="Expanded model mismatch",state="Complete",kind="complete",detail="Full run log finishes without failed experiments or decoders; no summary CSV to chart.",source="results/logs/full_E5b.log"),
            dict(id="E9–E11",name="Drift and generalization",state="Partial",kind="partial",detail="Development stress benchmark complete. The committed full-run log stops during E9.",source="results/logs/full_E9-E11.log"),
            dict(id="E12",name="Hardware-scale evaluation",state="Partial",kind="partial",detail="Latest findings report 1 of 16 points complete. Full run log is incomplete.",source="docs/findings.md"),
            dict(id="E13",name="Learned architectures",state="Smoke-tested",kind="partial",detail="Quick-run log completes; no full architecture-comparison summary is committed.",source="results/logs/quick_E13.log"),
            dict(id="E14",name="Binary syndromes",state="Baselines ready",kind="baseline",detail="24 points / 120 baseline rows; no evaluated binary-LSTM rows.",source="results/E14_summary.json"),
            dict(id="E15",name="Continuous corrections",state="DQN pending",kind="baseline",detail="5 points / 15 baseline rows; DQN checkpoints were deferred by the runner.",source="results/E15_summary.json")])
    if size_info:
        payload['status'].insert(0,dict(id='LSTM sizes',name='Width × depth × seed sweep',
            state=f'{size_info["evaluated_models"]}/36 evaluated',kind='complete' if size_info['evaluated_models']==36 else 'partial',
            detail=f'{size_info["evaluated"]:,}/2,196 measured model/settings. Widths 32/64/128/256 × depths 1/2/3 × three seeds. '
                   'Nominal training; 61 stationary noise settings; 1,024 development trials per setting. '
                   'This is separate from the E13 architecture-family smoke test.',source=SIZE_SOURCE))
    # Also preserve every E14 baseline row, beyond the ideal-readout plot subset.
    binary_cols=[("point","Point"),("decoder","Decoder"),("scale","Scale"),("rho","ρ"),("m","Bits / round"),
                 ("visibility","Visibility"),("p_flip","Bit-flip probability"),("eps","ε / round"),
                 ("pl_h","P_L at paired horizon"),("horizon","Horizon"),("diff_se","Paired SE")]
    payload["binary_columns"]=binary_cols
    payload["binary_csv"]=csv_text(e14,binary_cols)
    (OUT/"data"/"E14-all-baselines.csv").write_text(payload["binary_csv"])
    # Register the status table's additional source before freezing the manifest.
    read("RESULTS.md")
    payload["sources"]=list(SOURCES.values())
    template=Path(__file__).with_name("metrics_template.html").read_text()
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(",",":")).replace("<","\\u003c")
    (OUT/"viewer.html").write_text(template.replace("__METRICS_PAYLOAD__",encoded))
    compact={**payload,"charts":[{k:v for k,v in c.items() if k not in ("png","svg","csv")} for c in CHARTS]}
    (OUT/"metrics_data.json").write_text(json.dumps(compact,indent=2,ensure_ascii=False,allow_nan=False))
    for src in SOURCES:
        dest=OUT/"sources"/src
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(SOURCE_BYTES[src])
    (OUT/"README.txt").write_text(
        "QKP metrics overview\n\nOpen viewer.html in any modern browser; no server or internet is required.\n"
        "The HTML embeds all charts and chart data. PNG, SVG and CSV downloads also work offline.\n"
        "figures/ contains standalone Matplotlib plots. data/ contains plotted tables.\n"
        "sources/ contains the source snapshots; metrics_data.json records SHA-256 hashes.\n\n"
        "Regenerate inside the qkp repository:\n  .venv/bin/python -m writeup.artifact.build_metrics\n"
        "This only renders saved data, including completed models in the active size sweep. It does not train models.\n"
        "Use --watch-size-sweep to rebuild when another model finishes; reload the offline page to see updates.\n"
        f"Source commit: {payload['commit']}\nBuilt: {payload['generated']}\n")
    archive=OUT.parent/"qkp-metrics.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file(): z.write(p,p.relative_to(OUT))
    print(json.dumps(dict(viewer=str(OUT/"viewer.html"),charts=len(CHARTS),sources=len(SOURCES),
                         viewer_bytes=(OUT/"viewer.html").stat().st_size,bundle=str(archive)),indent=2))
    return size_info


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch-size-sweep', action='store_true')
    args = parser.parse_args()
    info = build()
    while args.watch_size_sweep and info and info['evaluated'] < info['expected_evaluations']:
        time.sleep(30)
        latest = json.loads((ROOT / SIZE_SOURCE).read_text())
        if latest['run_id'] != info['run_id']:
            raise RuntimeError('Active size-sweep run changed; restart the metrics watcher explicitly.')
        if latest['size_sweep']['evaluated'] != info['evaluated']:
            info = build()
