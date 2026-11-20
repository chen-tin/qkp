"""Plot saved QuTiP reference diagnostics without changing or rerunning physics.

Run from the repository root:
    .venv/bin/python -m experiments.plot_qutip_root_cause
"""
import argparse
import base64
import csv
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

from experiments.qutip_reference import estimate
from quantum.qutip_reference import Ancilla, CARDINAL, L


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/qutip_reference"
LONG = BASE / "convergence/e042e0a1b30d34d0"
LARGER = BASE / "convergence/dd4e949e01da259d"
COARSE = BASE / "reference/27ef7fdefecc9020"
FINE = BASE / "convergence/86ab35fc13213ba0"
COLORS = {3072: "#b94c38", 4096: "#177d83", 5120: "#644eb3"}
STYLES = {3072: "--", 4096: "-.", 5120: "-"}
LABELS = {"nearest": "Nearest", "bayes_gaussian": "Bayes", "lstm_256x3_s0": "LSTM"}
MEAN_GATE = 1e-4
MAX_GATE = .01


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Inputs:
    def __init__(self):
        self.manifest = {}

    def record(self, path):
        digest = sha(path)
        self.manifest[str(path.relative_to(ROOT))] = digest
        return digest

    def result(self, run):
        path = run / "result.json"
        self.record(path)
        result = json.loads(path.read_text())
        if not result["complete"]:
            raise ValueError(f"Incomplete run: {run}")
        snapshot = run / "source_snapshot/quantum/qutip_reference.py"
        if self.record(snapshot) != result["config"]["source_hashes"]["quantum/qutip_reference.py"]:
            raise ValueError(f"Physics source snapshot hash mismatch: {run}")
        return result

    def raw(self, run, config, row, arm):
        path = run / f'{row["point"]}_N{row["cutoff"]}_dx{config["dx"]:g}' / f'{row["decoder"]}_{arm}.npz'
        if self.record(path) != row["raw_hashes"][path.name]:
            raise ValueError(f"Raw data hash mismatch: {path}")
        # Final state snapshots are unnecessary for this read-only analysis.
        with np.load(path, allow_pickle=False) as z:
            raw = {key: z[key] for key in z.files if key not in ("score_snapshot", "score_indices")}
        shape = (6 * config["per_state"], config["rounds"] + 1)
        for key in ("fidelity", "energy", "edge"):
            if raw[key].shape != shape or not np.isfinite(raw[key]).all():
                raise ValueError(f"Invalid {key}: {path}")
        if raw["fidelity"].min() < -1e-9 or raw["fidelity"].max() > 1 + 1e-9:
            raise ValueError(f"Fidelity outside probability range: {path}")
        if raw["edge"].min() < 0 or raw["edge"].max() > 1 + 1e-9:
            raise ValueError(f"Boundary mass outside probability range: {path}")
        if raw["energy"].min() < 0 or raw["energy"].max() > row["cutoff"]:
            raise ValueError(f"Energy outside truncated basis: {path}")
        saved = row["finite_ancilla" if arm == "finite_ancilla" else "finite_energy_replayed"]
        for key, values in estimate(raw["fidelity"], config["per_state"]).items():
            np.testing.assert_allclose(values, saved[key], rtol=1e-10, atol=1e-12)
        return raw


def assert_paired(ca, cb, *, same_dx=True):
    keys = ["seed", "per_state", "rounds", "data_delta", "logical_bins", "points", "decoders"]
    if same_dx:
        keys.append("dx")
    for key in keys:
        if ca[key] != cb[key]:
            raise ValueError(f"Not a paired comparison: {key}")
    if ca["source_hashes"]["quantum/qutip_reference.py"] != cb["source_hashes"]["quantum/qutip_reference.py"]:
        raise ValueError("Physics implementation differs")


def check_noise(a, b):
    for key in ("dq", "dp", "nq", "np"):
        np.testing.assert_array_equal(a[key], b[key])


def savefig(fig, out, stem):
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=175, facecolor="white")
    plt.close(fig)


def axes_style(ax, ylabel, title):
    ax.set_title(title, loc="left", weight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Completed correction cycles")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def load_long(inputs):
    result = inputs.result(LONG)
    larger = inputs.result(LARGER)
    config = result["config"]
    assert_paired(config, larger["config"])
    # The diagnostic below instantiates this implementation's ancilla. Stop
    # rather than silently mixing a future smooth evaluator with old results.
    if inputs.record(ROOT / "quantum/qutip_reference.py") != config["source_hashes"]["quantum/qutip_reference.py"]:
        raise ValueError("Current ancilla implementation differs from the saved experiment")
    inputs.record(ROOT / "experiments/qutip_reference.py")
    cases = {}
    for run, data in ((LONG, result), (LARGER, larger)):
        for row in data["rows"]:
            n = row["cutoff"]
            cases[n] = {"row": row}
            for arm, name in (("finite_ancilla", "physical"), ("replayed_corrections", "replay")):
                cases[n][name] = inputs.raw(run, data["config"], row, arm)
    assert set(cases) == {3072, 4096, 5120}
    for case in cases.values():
        check_noise(cases[3072]["physical"], case["physical"])
        check_noise(case["physical"], case["replay"])
        np.testing.assert_array_equal(cases[3072]["replay"]["classical_fidelity"], case["replay"]["classical_fidelity"])
        np.testing.assert_allclose(cases[3072]["replay"]["fidelity"], case["replay"]["fidelity"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(case["physical"]["corrections"], case["physical"]["syndromes"], atol=1e-14, rtol=0)
    return config, cases


def summarize(config, cases):
    metrics = []
    comparisons = {}
    comparison_rows = []
    per_state = config["per_state"]
    for n, case in cases.items():
        raw, replay = case["physical"], case["replay"]
        case["estimate"] = estimate(raw["fidelity"], per_state)
        mean_edge = np.maximum(raw["edge"].mean(0), replay["edge"].mean(0))
        max_edge = np.maximum(raw["edge"].max(0), replay["edge"].max(0))
        case["mean_edge"] = mean_edge
        case["max_edge"] = max_edge
        case["gate"] = np.logical_and.accumulate((mean_edge < MEAN_GATE) & (max_edge < MAX_GATE))
        case["energy_quantiles"] = np.quantile(raw["energy"], [.5, .9, .99], axis=0)
        for t in range(config["rounds"] + 1):
            metrics.append({
                "cutoff": n, "cycle": t, "trajectories": 6 * per_state,
                "logical_fidelity_mean": case["estimate"]["mean"][t],
                "logical_fidelity_ci95_low": case["estimate"]["ci95_low"][t],
                "logical_fidelity_ci95_high": case["estimate"]["ci95_high"][t],
                "energy_mean": float(raw["energy"][:, t].mean()),
                "energy_median": float(case["energy_quantiles"][0, t]),
                "energy_q90": float(case["energy_quantiles"][1, t]),
                "energy_q99": float(case["energy_quantiles"][2, t]),
                "energy_max": float(raw["energy"][:, t].max()),
                "replay_energy_mean": float(replay["energy"][:, t].mean()),
                "reference_mean_boundary_mass": float(raw["edge"][:, t].mean()),
                "reference_max_boundary_mass": float(raw["edge"][:, t].max()),
                "max_over_arms_mean_boundary_mass": float(mean_edge[t]),
                "max_over_arms_max_boundary_mass": float(max_edge[t]),
                "reference_trajectories_above_1pct_boundary_mass": int((raw["edge"][:, t] >= MAX_GATE).sum()),
                "population_gate_pass_through_cycle": bool(case["gate"][t]),
                "max_norm_error_all_cycles": case["row"]["max_norm_error"],
                "logical_extraction_trace_error": case["row"]["extraction_trace_error"],
            })
    for a, b in ((3072, 4096), (4096, 5120)):
        e = estimate(cases[b]["physical"]["fidelity"] - cases[a]["physical"]["fidelity"], per_state)
        e["within_tolerance"] = ((np.array(e["ci95_low"]) >= -.01) & (np.array(e["ci95_high"]) <= .01)).tolist()
        comparisons[f"{b}_minus_{a}"] = e
        for t in range(config["rounds"] + 1):
            comparison_rows.append({"smaller_cutoff": a, "larger_cutoff": b, "cycle": t,
                                    "mean_fidelity_difference": e["mean"][t],
                                    "paired_ci95_low": e["ci95_low"][t], "paired_ci95_high": e["ci95_high"][t],
                                    "interval_inside_plus_minus_0_01": e["within_tolerance"][t]})
    return metrics, comparisons, comparison_rows


def overview(out, config, cases, comparisons):
    t = np.arange(config["rounds"] + 1)
    fig, axs = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    ax = axs[0, 0]
    for n, case in cases.items():
        e = case["estimate"]
        ax.plot(t, 100 * np.array(e["mean"]), color=COLORS[n], ls=STYLES[n], label=f"N = {n:,}")
        if n == 5120:
            ax.fill_between(t, 100 * np.array(e["ci95_low"]), 100 * np.array(e["ci95_high"]), color=COLORS[n], alpha=.13, label="N = 5,120: 95% CI")
    ref = cases[5120]["replay"]
    ax.plot(t, 100 * ref["classical_fidelity"].mean(0), color="#475569", ls=":", label="Classical control")
    ax.plot(t, 100 * ref["fidelity"].mean(0), color="#555", ls="--", label="Finite-energy replay")
    axes_style(ax, "Six-state average logical fidelity (%)", "A  Logical performance")
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    case = cases[5120]
    ax.plot(t, case["physical"]["energy"].mean(0), color="#171f32", lw=2, label="Mean (N = 5,120)")
    ax.plot(t, case["energy_quantiles"][0], color="#64748b", label="Median (N = 5,120)")
    ax.plot(t, case["energy_quantiles"][2], color="#64748b", ls=":", label="99th percentile (N = 5,120)")
    for n, case in cases.items():
        ax.plot(t, case["physical"]["energy"].max(0), color=COLORS[n], ls=STYLES[n], label=f"Maximum (N = {n:,})")
    axes_style(ax, "Mean photon number of a trajectory", "B  Rare trajectories dominate the extremes")
    ax.legend(fontsize=8)

    for ax, key, gate, title in ((axs[0, 2], "mean_edge", MEAN_GATE, "C  Mean boundary population"),
                                  (axs[1, 0], "max_edge", MAX_GATE, "D  Worst boundary population")):
        for n, case in cases.items():
            # Plot cycles 1..8; initial-state values are zero or below 1e-216.
            ax.semilogy(t[1:], 100 * case[key][1:], marker="o", ms=3, color=COLORS[n], ls=STYLES[n], label=f"N = {n:,}")
        ax.axhline(100 * gate, color="#111827", ls=":", label=f"Gate: {100 * gate:g}%")
        axes_style(ax, "Probability in highest 10% of Fock levels (%)", title)
        ax.legend(fontsize=8)

    ax = axs[1, 1]
    for j, (n, case) in enumerate(cases.items()):
        count = (case["physical"]["edge"] >= MAX_GATE).sum(0)
        ax.plot(t, count, color=COLORS[n], ls=STYLES[n], marker=["o", "s", "x"][j],
                ms=5, label=f"N = {n:,}")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes_style(ax, "Trajectories with ≥1% boundary population", "E  Failure is rare in the 3,072-level run")
    ax.set_ylim(-.2, 3.4)
    ax.legend(fontsize=8)

    ax = axs[1, 2]
    ax.axhspan(-1, 1, color="#d8eadf", alpha=.7, label="Registered ±1 percentage point band")
    ax.axhline(0, color="#555", lw=.7)
    for (a, b), offset, color in (((3072, 4096), -.065, COLORS[3072]), ((4096, 5120), .065, COLORS[5120])):
        e = comparisons[f"{b}_minus_{a}"]
        m, lo, hi = [100 * np.array(e[k]) for k in ("mean", "ci95_low", "ci95_high")]
        ax.errorbar(t + offset, m, yerr=[np.maximum(m-lo, 0), np.maximum(hi-m, 0)], color=color,
                    fmt="o-", ms=3, capsize=2, label=f"{b:,} − {a:,}")
    axes_style(ax, "Paired fidelity difference (percentage points)", "F  Cutoff equivalence: the full CI must fit")
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Numerical health and logical performance are different questions\n"
                 "Nominal noise · nearest-lattice decoder · 3,072 trajectories per cutoff · 8 cycles", fontsize=15)
    savefig(fig, out, "01_overview")


def trajectory_plots(out, config, cases, selected):
    t = np.arange(config["rounds"] + 1)
    cycle = t[1:]
    rows = []
    for rank, index in enumerate(selected):
        fig, axs = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
        for n, case in cases.items():
            raw = case["physical"]
            style = {"color": COLORS[n], "ls": STYLES[n], "label": f"N = {n:,}"}
            axs[0, 0].plot(t, raw["energy"][index], **style)
            axs[0, 1].semilogy(cycle, 100 * raw["edge"][index, 1:], **style)
            axs[0, 2].plot(t, 100 * raw["fidelity"][index], **style)
            for a, axis in enumerate(("q", "p")):
                axs[1, 1+a].plot(cycle, raw["syndromes"][a, index] / L, marker="o", ms=3, **style)
            for k in t:
                rows.append({"rank_by_peak_boundary_mass_at_N3072": rank+1, "trajectory": int(index),
                             "initial_state": CARDINAL[index // config["per_state"]][0], "cutoff": n, "cycle": int(k),
                             "logical_fidelity": float(raw["fidelity"][index, k]),
                             "energy": float(raw["energy"][index, k]), "boundary_mass": float(raw["edge"][index, k]),
                             "dq_over_L": float(raw["dq"][index, k-1] / L) if k else "",
                             "dp_over_L": float(raw["dp"][index, k-1] / L) if k else "",
                             "q_syndrome_over_L": float(raw["syndromes"][0, index, k-1] / L) if k else "",
                             "p_syndrome_over_L": float(raw["syndromes"][1, index, k-1] / L) if k else "",
                             "q_correction_over_L": float(raw["corrections"][0, index, k-1] / L) if k else "",
                             "p_correction_over_L": float(raw["corrections"][1, index, k-1] / L) if k else ""})
        axs[0, 0].plot(t, cases[5120]["replay"]["energy"][index], color="#4b5563", ls=":", label="No-backaction replay (5,120)")
        axs[0, 1].axhline(100 * MAX_GATE, color="#444", ls=":", label="1% boundary gate")
        axes_style(axs[0, 0], "Mean photon number", "A  Conditional energy")
        axes_style(axs[0, 1], "Boundary population (%)", "B  Weight near the Fock boundary")
        axes_style(axs[0, 2], "Logical fidelity (%)", "C  Conditional logical fidelity")
        axs[0, 2].set_ylim(-2, 102)
        raw = cases[5120]["physical"]
        axs[1, 0].plot(cycle, raw["dq"][index]/L, marker="o", color="#176d86", label="Applied q displacement")
        axs[1, 0].plot(cycle, raw["dp"][index]/L, marker="s", color="#b56929", label="Applied p displacement")
        axs[1, 0].axhline(0, color="#555", lw=.7)
        axes_style(axs[1, 0], "Displacement / √π", "D  External noise is identical across cutoffs")
        for a, axis in enumerate(("q", "p")):
            ax = axs[1, a+1]
            axes_style(ax, "Wrapped syndrome / √π", f"{'E' if a == 0 else 'F'}  {axis} measurement history")
            ax.set_ylim(-.54, .54)
        for ax in axs.flat:
            ax.legend(fontsize=8)
        state = CARDINAL[index // config["per_state"]][0]
        fig.suptitle(f"Trajectory {index} · initial state {state} · boundary-risk rank {rank+1} at N = 3,072\n"
                     "Common random numbers do not force identical measurement histories; nearest-lattice correction equals wrapped syndrome",
                     fontsize=13)
        savefig(fig, out, f"02_trajectory_{index}")
    return rows


def tail_heatmap(out, cases, top):
    fig, axs = plt.subplots(1, 3, figsize=(15, 7.2), layout="constrained", sharey=True)
    for ax, (n, case) in zip(axs, cases.items()):
        values = 100 * case["physical"]["edge"][top, 1:]
        im = ax.imshow(values, aspect="auto", norm=LogNorm(vmin=1e-5, vmax=100), cmap="magma", interpolation="nearest")
        yy, xx = np.where(values >= 1)
        ax.scatter(xx, yy, marker="s", s=80, facecolors="none", edgecolors="#65f4e4", lw=1.2, label="≥1% boundary mass")
        ax.set_title(f"N = {n:,}")
        ax.set_xticks(np.arange(8), np.arange(1, 9))
        ax.set_xlabel("Completed correction cycles")
        ax.set_yticks(np.arange(len(top)), [str(i) for i in top])
    axs[0].set_ylabel("Trajectory index (ranked by peak N = 3,072 boundary mass)")
    axs[0].legend(fontsize=8, loc="lower left", labelcolor="white")
    fig.colorbar(im, ax=axs, label="Probability in top 10% of Fock levels (%)", shrink=.85)
    fig.suptitle("The 20 highest-risk trajectories in the 3,072-level run\n"
                 "Same selected indices at all cutoffs; this is an enriched diagnostic sample", fontsize=14)
    savefig(fig, out, "03_tail_heatmap")


def refinement_data(inputs):
    coarse, fine = inputs.result(COARSE), inputs.result(FINE)
    assert_paired(coarse["config"], fine["config"], same_dx=False)
    rows = []
    lookup = {(r["point"], r["decoder"]): r for r in coarse["rows"] if r["cutoff"] == 3072}
    for row in fine["rows"]:
        a = inputs.raw(COARSE, coarse["config"], lookup[row["point"], row["decoder"]], "finite_ancilla")
        b = inputs.raw(FINE, fine["config"], row, "finite_ancilla")
        check_noise(a, b)
        diff = estimate(b["fidelity"] - a["fidelity"], fine["config"]["per_state"])
        rows.append({"point": row["point"], "decoder": row["decoder"], "cutoff": 3072, "cycles": 2,
                     "trajectories": 6 * fine["config"]["per_state"],
                     "coarse_dx": coarse["config"]["dx"], "fine_dx": fine["config"]["dx"],
                     "coarse_final_mean_boundary_mass": float(a["edge"][:, -1].mean()),
                     "fine_final_mean_boundary_mass": float(b["edge"][:, -1].mean()),
                     "fine_minus_coarse_final_fidelity": diff["mean"][-1],
                     "paired_ci95_low": diff["ci95_low"][-1], "paired_ci95_high": diff["ci95_high"][-1],
                     "all_cycle_CIs_inside_plus_minus_0_005": bool(np.all(np.array(diff["ci95_low"]) >= -.005) and np.all(np.array(diff["ci95_high"]) <= .005))})
    return rows


def discretization_plot(out, refinement):
    ancillas = [Ancilla(.05 * L, dx) for dx in (.003, .0015)]
    diagnostics = []
    for anc in ancillas:
        jump_sum = float(np.sum(np.diff(np.r_[0., anc.amp, 0.]) ** 2))
        derivative = float(np.sum(np.gradient(anc.amp, anc.dx) ** 2) * anc.dx)
        diagnostics.append({"dx": anc.dx, "sum_squared_jumps": jump_sum, "smooth_grid_derivative_p2": derivative})
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    anc = ancillas[0]
    # Evaluate the same smooth Mehler sum on a dense local grid, with the
    # original normalization. This is a visual diagnostic, not a new sampler.
    x = np.linspace(.045, .065, 2001)
    variance = np.tanh(anc.t)
    centres = np.arange(-20, 21) * L

    def smooth(xx):
        return np.exp(-.5 * variance * centres[:, None] ** 2
                      -(xx[None, :] - centres[:, None] / np.cosh(anc.t)) ** 2 / (2 * variance)).sum(0)

    normalizer = anc.amp[np.argmin(np.abs(anc.x))] / smooth(np.array([anc.x[np.argmin(np.abs(anc.x))]]))[0]
    axs[0, 0].plot(x, smooth(x) * normalizer, color="#162c46", lw=2, label="Smooth Mehler wavefunction")
    axs[0, 0].plot(x, anc.amplitude(x), color="#b94c38", drawstyle="steps-mid", label="Current bin lookup (dx = .003)")
    axs[0, 0].set(title="A  Wavefunction lookup introduces jumps", xlabel="Ancilla position q", ylabel="Wavefunction amplitude")
    axs[0, 0].legend(fontsize=9)

    ax = axs[0, 1]
    for d, color in zip(diagnostics, ("#b94c38", "#177d83")):
        h = d["dx"] / (2 * 2. ** np.arange(0, 10))
        p2 = d["sum_squared_jumps"] / (2 * h)
        ax.loglog(h, p2, marker="o", ms=3, color=color, label=f'Step function, dx = {d["dx"]:g}')
    ax.axhline(diagnostics[1]["smooth_grid_derivative_p2"], color="#162c46", ls="--", label="Smooth-grid derivative ≈ 31.83")
    ax.invert_xaxis()
    ax.set(title="B  Step-wavefunction derivative energy diverges", xlabel="Derivative half-step h (smaller →)",
           ylabel="∫ |[φ(q+h)−φ(q−h)] / (2h)|² dq")
    ax.legend(fontsize=8)

    xx = np.arange(len(refinement))
    labels = [f'{r["point"].replace("readout_high", "high readout")}\n{LABELS[r["decoder"]]}' for r in refinement]
    ax = axs[1, 0]
    ax.bar(xx-.18, [100*r["coarse_final_mean_boundary_mass"] for r in refinement], width=.36, color="#b94c38", label="dx = .003")
    ax.bar(xx+.18, [100*r["fine_final_mean_boundary_mass"] for r in refinement], width=.36, color="#177d83", label="dx = .0015")
    ax.set_yscale("log")
    ax.set_xticks(xx, labels, rotation=40, ha="right", fontsize=8)
    ax.set(title="C  Finer bins reduce two-cycle boundary mass", ylabel="Mean boundary population at cycle 2 (%)")
    ax.legend(fontsize=9)

    ax = axs[1, 1]
    m = np.array([r["fine_minus_coarse_final_fidelity"] for r in refinement]) * 100
    lo = np.array([r["paired_ci95_low"] for r in refinement]) * 100
    hi = np.array([r["paired_ci95_high"] for r in refinement]) * 100
    ax.axhspan(-.5, .5, color="#d8eadf", alpha=.65, label="Registered ±0.5 percentage point band")
    ax.axhline(0, color="#555", lw=.7)
    ax.errorbar(xx, m, yerr=[np.maximum(0, m-lo), np.maximum(0, hi-m)], fmt="o", color="#644eb3", capsize=3)
    ax.set_xticks(xx, labels, rotation=40, ha="right", fontsize=8)
    ax.set(title="D  Two-cycle fidelity is less sensitive", ylabel="Fine − coarse fidelity (percentage points)")
    ax.legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Measurement discretization is a credible mechanism to test\n"
                 "A–B: mathematical diagnostic · C–D: separate two-cycle experiment, 1,536 trajectories per case", fontsize=14)
    savefig(fig, out, "04_measurement_discretization")
    return diagnostics


def image_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def report(out, config, cases, comparisons, selected):
    def figure(stem, alt):
        return (f'<a class="figure" href="{stem}.svg" title="Open scalable figure"><img src="{image_uri(out/(stem+".png"))}" alt="{html.escape(alt)}"></a>'
                f'<p class="download"><a href="{stem}.png">PNG</a> · <a href="{stem}.svg">SVG</a></p>')

    table = []
    for n, case in cases.items():
        horizon = int(np.flatnonzero(case["gate"])[-1])
        e = case["estimate"]
        table.append(f'<tr><td>{n:,}</td><td>{horizon}</td><td>{100*case["max_edge"].max():.4g}%</td>'
                     f'<td>{100*case["mean_edge"].max():.4g}%</td><td>{100*e["mean"][-1]:.2f}% '
                     f'[{100*e["ci95_low"][-1]:.2f}, {100*e["ci95_high"][-1]:.2f}]</td>'
                     f'<td>{case["physical"]["energy"][:,-1].mean():.2f}</td></tr>')
    selectors = ''.join(f'<option value="{int(i)}">Trajectory {int(i)} · {CARDINAL[i//config["per_state"]][0]} · risk rank {j+1}</option>' for j, i in enumerate(selected))
    trajectory_images = {str(int(i)): image_uri(out/f"02_trajectory_{i}.png") for i in selected}
    e = comparisons["5120_minus_4096"]
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuTiP reference: root cause review</title><style>
:root{color-scheme:light;font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172438;background:#f2f5f8}
body{margin:0}main{max-width:1300px;margin:auto;padding:36px 24px 60px}h1{font-size:clamp(26px,4vw,42px);letter-spacing:-.025em;margin:8px 0 12px}h2{font-size:25px;margin:32px 0 12px}p,li{line-height:1.65;max-width:1040px}a{color:#3650a0}nav{display:flex;flex-wrap:wrap;gap:16px;margin:22px 0}article,.card{background:white;border:1px solid #d7e0ea;border-radius:12px;padding:22px;margin:20px 0}.tag{font-size:12px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:#52647b}.warning{border-left:5px solid #b94c38}.evidence{border-left:5px solid #177d83}.figure img,#trajectory{width:100%;height:auto;display:block}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce3ed}th{background:#f3f6fa}.scroll{overflow:auto}.download,.caption{font-size:13px;color:#536278}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}.metrics .card{margin:0}.value{font-size:27px;font-weight:750;margin:8px 0}select{font:inherit;padding:9px 12px;max-width:100%;border-radius:7px;border:1px solid #aab8c9}code{background:#edf1f6;padding:2px 5px;border-radius:4px}dt{font-weight:700;margin-top:14px}dd{margin:5px 0 12px;line-height:1.6}.muted{color:#536278}.equation{padding:12px;background:#f5f7fa;border-radius:7px;font-family:serif;font-size:20px}footer{font-size:13px;color:#536278;margin-top:30px}.nowrap{white-space:nowrap}
@media print{main{padding:0}nav,select,.download{display:none}article,.card{break-inside:avoid}body{background:white}}
</style></head><body><main><div class="tag">Numerical diagnostics · saved-data analysis · 19 November 2026</div>
<h1>What happens after cycle 5?</h1>
<div class="card warning"><p><strong>The boundary spike is bad for the numerical reliability of the affected 3,072-level run.</strong>
It means its finite basis is too small for some conditional states. It does not establish a physical five-cycle limit or identify a decoder defect.
Average energy growth can occur under this measurement protocol, which has no energy stabilization; the abrupt rare jumps still need a controlled explanation.</p></div>
<nav><a href="#overview">Performance &amp; health</a><a href="#trajectories">Rare trajectories</a><a href="#discretization">Measurement approximation</a><a href="#causes">Root cause assessment</a><a href="#metrics">Metric definitions &amp; data</a></nav>
<div class="metrics"><div class="card"><div class="tag">N = 3,072</div><div class="value">Valid through cycle 5</div><p>Boundary gates fail at cycle 6; three trajectories exceed 1% boundary mass at cycle 8.</p></div>
<div class="card"><div class="tag">N = 4,096 and 5,120</div><div class="value">Pass through cycle 8</div><p>Both pass population gates. All pointwise paired fidelity intervals fit inside the registered ±1 percentage point band.</p></div>
<div class="card"><div class="tag">Causality remains open</div><div class="value">Energy jump persists</div><p>The largest-basis run still has a rare high-energy jump. A smooth measurement-resource intervention is needed.</p></div></div>
<article id="overview"><h2>Performance and numerical health</h2><p>Read panels A and F for logical performance and cutoff sensitivity, B for energy tails, and C–E for numerical warning signs. The mean can look smooth while individual trajectories reach the basis boundary.</p>
''' + figure("01_overview", "Six panels showing logical fidelity, energy, boundary populations, failure counts, and paired cutoff differences") + '''
<p class="caption">Nearest-lattice decoder only, nominal σ<sub>q</sub>/√π = 0.20, σ<sub>m</sub>/√π = 0.05, ρ = 0.90; Δ = 0.30. Each cutoff uses 512 trajectories per cardinal input state (3,072 total), common noise/random-number streams and eight q-then-p cycles. Shading/error bars are pointwise 95% trajectory intervals, stratified equally over six input states; they do not include model error. Boundary gates use the worse value of the physical and replay arms. Initial boundary values are below the logarithmic panels' range.</p>
<div class="scroll"><table><thead><tr><th>Fock levels N</th><th>Population gates pass through cycle</th><th>Worst boundary mass, any cycle</th><th>Largest ensemble-mean boundary mass</th><th>Cycle-8 fidelity [95% CI]</th><th>Cycle-8 mean photons</th></tr></thead><tbody>''' + ''.join(table) + '''</tbody></table></div>
<p><strong>New evidence:</strong> the 5,120-level follow-up completed. At cycle 8, its paired fidelity change relative to 4,096 is ''' + f'{100*e["mean"][-1]:+.3f} percentage points, with 95% CI [{100*e["ci95_low"][-1]:+.3f}, {100*e["ci95_high"][-1]:+.3f}].' + ''' This supports cutoff stability of the <em>average logical fidelity</em> at the registered tolerance. It does not establish convergence of every trajectory, energy tail, measurement approximation, noise setting, or decoder.</p>
<p>The classical control has ideal encoding and classical readout. The replay arm applies its corrections to the finite-energy state without measurement backaction. These controls help expose protocol differences; their fidelity gaps do not isolate a single physical mechanism.</p></article>
<article id="trajectories"><h2>Inspect the rare trajectories</h2><p>The selector contains the three trajectories with the largest peak boundary population in the 3,072-level run, selected by that rule before comparing cutoffs. They are diagnostic outliers, not representative samples.</p>
<label for="pick">Choose trajectory: </label><select id="pick">''' + selectors + '''</select><img id="trajectory" alt="Selected trajectory: energy, boundary population, fidelity, applied displacements and measurement histories">
<p class="download"><a id="trajectory-png">PNG</a> · <a id="trajectory-svg">SVG</a> · <a href="selected_trajectories.csv">Trajectory data CSV</a></p>
<p>For trajectory 1659 (initial X−), mean photon number jumps from 46.44 at cycle 5 to 2,602.07 at cycle 6 at N = 3,072. Its boundary mass reaches 94.99% at cycle 7. At N = 5,120 the same random-number index still jumps to 3,080.67 photons at cycle 6, while boundary mass remains below 0.061%. Thus a boundary failure and a high-energy event are separate observations.</p>
<p>Wrapped syndromes determine the nearest-lattice corrections here, so those curves also show the feedback commands. Common random numbers reduce comparison variance but do not enforce identical conditional measurement outcomes across cutoffs. Even at N = 5,120 this outlier's energy path is not shown to be converged.</p>
''' + figure("03_tail_heatmap", "Boundary population heatmaps for the same 20 highest-risk trajectories at three cutoffs") + '''</article>
<article id="discretization"><h2>A concrete measurement-approximation concern</h2><p>The saved simulator evaluates the ancilla wavefunction by a piecewise-constant position-bin lookup. Sampling uses the matching bin density, so the two operations agree for that approximation. The jumps in the wavefunction nevertheless create artificial high-momentum components. This supplies a credible mechanism for rare high-energy events, not yet a demonstrated cause of the observed trajectory.</p>
''' + figure("04_measurement_discretization", "Wavefunction discontinuities, divergent derivative diagnostic and two-cycle grid-refinement results") + '''
<p class="equation">For derivative half-step h &lt; dx/2: &nbsp; ∫ |D<sub>h</sub>φ|² dq = Σ<sub>j</sub>|jump<sub>j</sub>|² / (2h).</p>
<p class="caption">Panel B is a mathematical finite-difference diagnostic of the step-function approximation, not a measured physical momentum energy from these trajectories. For a continuous smooth resource this integral has a finite limit (about 31.83 in the nominal case). Panels C–D reuse a separate two-cycle refinement, 256 trajectories per input state, at N = 3,072. All nine pointwise fidelity comparisons pass its ±0.5 percentage point band; that short test does not resolve eight-cycle energy tails.</p></article>
<article id="causes"><h2>Root cause assessment</h2><div class="scroll"><table><thead><tr><th>Candidate / observation</th><th>Evidence</th><th>Current conclusion</th></tr></thead><tbody>
<tr><td>Fock basis too small at N = 3,072</td><td>Up to 94.99% probability reaches its top 10% of levels. Increasing N removes the boundary-gate failure through cycle 8.</td><td>Confirmed numerical limitation for that cutoff. This explains why its affected results are unreliable, but not the origin of the high-energy event.</td></tr>
<tr><td>Measurement backaction and no energy stabilization</td><td>The ensemble mean rises from 5.07 to 261.93 photons at N = 5,120 (average increase 32.11 per cycle); the replay arm has no comparable growth. The nominal smooth ancilla momentum variance is about 31.83.</td><td>Consistent with expected heating under this protocol. The scale agreement does not prove the abrupt rare jumps are physical.</td></tr>
<tr><td>Piecewise-constant resource wavefunction</td><td>Its derivative-energy diagnostic diverges. Finer bins reduce boundary mass in the separate two-cycle study.</td><td>Confirmed approximation defect for continuum energy; a plausible contributor to the spikes. Its causal contribution to this event is unresolved.</td></tr>
<tr><td>External displacement outlier</td><td>For trajectory 1659 at cycle 6: dq/√π ≈ 0.292 and dp/√π ≈ −0.121, versus stationary standard deviation 0.20.</td><td>No large new applied displacement at the event. This does not rule out a conditional-state / measurement interaction.</td></tr>
<tr><td>Norm loss or logical extraction failure</td><td>Largest norm error across the three runs is below 8×10⁻¹²; extraction trace error is below 4×10⁻¹¹.</td><td>These diagnostics show no failure. Small norm/trace errors alone cannot certify physical accuracy.</td></tr>
<tr><td>LSTM or Bayes defect</td><td>This eight-cycle cutoff study uses the nearest-lattice controller only.</td><td>No architecture-specific conclusion is supported by this study.</td></tr>
</tbody></table></div>
<h3>What would resolve the remaining uncertainty?</h3><ol>
<li>Replace the bin lookup with a smooth analytic Mehler wavefunction and matching continuous outcome sampler; hold the physical protocol and decoder fixed. Validate it against the explicit two-mode reference before rerunning.</li>
<li>Repeat the nominal eight-cycle comparison with the same noise and sampling budget. Compare fidelity, energy quantiles, rare jumps and boundary populations. Include a long-horizon bin-refinement control if retaining the binned implementation.</li>
<li>For the outliers, record energy before/after displacement, q measurement, q feedback, p measurement and p feedback, plus unwrapped homodyne outcomes and their log probabilities. Saved round-end data cannot identify the first failing substep or the rarity of the full outcome.</li>
</ol><p>Adding damping or energy feedback changes the physical protocol. It is a separate modeling experiment, not a numerical convergence fix. No new simulation or decoder training was launched for this review.</p></article>
<article id="metrics"><h2>Metric definitions and reusable data</h2><dl>
<dt>Logical fidelity</dt><dd>Overlap of the extracted logical state with its intended input state, averaged equally across ±X, ±Y and ±Z. Higher is better; 100% is perfect. Finite-energy encoding starts at about 97.72% before any cycle. This is not a classification accuracy or a per-round logical error rate.</dd>
<dt>Mean photon number ⟨n⟩</dt><dd>Energy of a conditional oscillator state, in photon-number units. The mean tracks the ensemble; median and 99th percentile describe its spread; the maximum exposes rare events. Higher energy needs a larger basis and can have a physical cost, but does not by itself imply a logical decoding error.</dd>
<dt>Boundary population</dt><dd>Probability weight in the highest 10% of Fock levels (a cutoff-dependent region). It is a warning diagnostic, not a decoder error probability. Registered gates require ensemble mean &lt;0.01% and every trajectory &lt;1%, in both arms, at every preceding cycle.</dd>
<dt>Paired cutoff difference and 95% interval</dt><dd>The average trajectory-level fidelity difference when the numerical cutoff increases. The entire pointwise interval must fit inside ±1 percentage point. This gate addresses mean fidelity at the tested cutoffs; it is not a universal convergence proof or a simultaneous confidence band.</dd>
<dt>Norm and extraction trace errors</dt><dd>Checks of state normalization and logical-observable quadrature completeness. They help catch arithmetic or integration problems; they can remain tiny while a finite basis or measurement approximation is inadequate.</dd>
</dl><p><a href="cycle_metrics.csv">Cycle metrics CSV</a> · <a href="cutoff_comparisons.csv">Paired cutoff CSV</a> · <a href="selected_trajectories.csv">Selected trajectories CSV</a> · <a href="bin_refinement.csv">Two-cycle refinement CSV</a> · <a href="review_data.json">Summary JSON</a> · <a href="provenance.json">Input hashes and checks</a></p>
<p class="muted">Source experiments: e042e0a1b30d34d0 (N = 3,072/4,096), dd4e949e01da259d (N = 5,120), 27ef7fdefecc9020 / 86ab35fc13213ba0 (two-cycle bin refinement). Physical operators and raw observations are unchanged. Every loaded raw file is checked against its recorded SHA-256; saved means and intervals are recomputed. Only common noise arrays, not measurement outcomes, are required to coincide.</p></article>
<footer>Figures are embedded for offline viewing. PNG, SVG and CSV downloads are neighboring files. Scope: finite-energy GKP model with ideal SUM/homodyne and classical displacement noise; no photon loss, dephasing, gate faults or hardware validation.</footer>
<script>const images = ''' + json.dumps(trajectory_images) + ''';
function choose(){const id=document.getElementById('pick').value;document.getElementById('trajectory').src=images[id];document.getElementById('trajectory-png').href='02_trajectory_'+id+'.png';document.getElementById('trajectory-svg').href='02_trajectory_'+id+'.svg';}document.getElementById('pick').addEventListener('change',choose);choose();</script></main></body></html>'''
    (out / "index.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "root_cause_review_20261119")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.frameon": False})
    inputs = Inputs()
    config, cases = load_long(inputs)
    metrics, comparisons, comparison_rows = summarize(config, cases)
    # Reproduce the completed follow-up's registered cutoff comparison.
    comparison_path = LARGER / "comparison_with_4096.json"
    inputs.record(comparison_path)
    registered = json.loads(comparison_path.read_text())
    for key in ("mean", "se", "ci95_low", "ci95_high", "within_tolerance"):
        np.testing.assert_allclose(comparisons["5120_minus_4096"][key], registered[key], atol=1e-12, rtol=1e-10)
    top = np.argsort(cases[3072]["physical"]["edge"].max(1), kind="stable")[::-1][:20]
    selected = top[:3]
    assert selected.tolist() == [1659, 1973, 2776]
    assert [int(np.flatnonzero(cases[n]["gate"])[-1]) for n in cases] == [5, 8, 8]
    assert (cases[3072]["physical"]["edge"] >= MAX_GATE).sum(0).tolist() == [0, 0, 0, 0, 0, 0, 1, 2, 3]
    overview(out, config, cases, comparisons)
    trajectory_rows = trajectory_plots(out, config, cases, selected)
    tail_heatmap(out, cases, top)
    refinement = refinement_data(inputs)
    diagnostics = discretization_plot(out, refinement)
    findings = {
        "scope": "Saved-data numerical root cause review; no new physical simulation",
        "nominal_config": {k: config[k] for k in ("seed", "data_delta", "per_state", "rounds", "dx", "logical_bins", "points", "decoders")},
        "population_valid_through_cycle": {str(n): int(np.flatnonzero(c["gate"])[-1]) for n, c in cases.items()},
        "cutoff_comparisons": comparisons,
        "selected_trajectory_rule": "Descending maximum boundary mass over all cycles at N3072, stable argsort reversed",
        "selected_trajectory_indices": [int(i) for i in selected],
        "measurement_step_diagnostic": diagnostics,
        "bin_refinement_two_cycles": refinement,
        "interpretation": "N3072 has a demonstrated boundary limitation. N4096/N5120 pass population gates and mean-fidelity cutoff equivalence through cycle8. Rare energy jumps persist. A binned-wavefunction defect is confirmed mathematically; its causal contribution to observed spikes is not isolated.",
        "limitations": ["Eight-cycle study covers nominal nearest-lattice only", "Pointwise trajectory CIs exclude model/systematic uncertainty", "No substep energies, unwrapped homodyne outcomes or outcome log likelihoods were saved", "No long-horizon smooth-resource intervention has been run", "Cutoff equivalence does not establish convergence of energy tails or individual trajectories"],
    }
    write_csv(out / "cycle_metrics.csv", metrics)
    write_csv(out / "cutoff_comparisons.csv", comparison_rows)
    write_csv(out / "selected_trajectories.csv", trajectory_rows)
    write_csv(out / "bin_refinement.csv", refinement)
    write_json(out / "review_data.json", findings)
    provenance = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                  "analysis_script": str(Path(__file__).relative_to(ROOT)), "analysis_script_sha256": sha(__file__),
                  "raw_hashes_verified": True, "physics_source_snapshots_verified": True,
                  "saved_fidelity_means_and_intervals_recomputed": True,
                  "paired_config_and_noise_arrays_verified": True,
                  "registered_4096_5120_comparison_reproduced": True,
                  "input_sha256": inputs.manifest,
                  "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    write_json(out / "provenance.json", provenance)
    report(out, config, cases, comparisons, selected)
    print(json.dumps({"report": str(out / "index.html"), "figures": 6,
                      "verified_inputs": len(inputs.manifest), "population_horizons": findings["population_valid_through_cycle"],
                      "all_4096_5120_cutoff_intervals_pass": all(comparisons["5120_minus_4096"]["within_tolerance"])}, indent=2))


if __name__ == "__main__":
    main()
