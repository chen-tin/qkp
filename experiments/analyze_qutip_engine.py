"""Audit and render an explicit QuTiP/local measurement comparison as HTML."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
L = np.sqrt(np.pi)
ARMS = ["qutip_explicit", "local_binned_0.003", "local_binned_0.0015", "local_smooth"]
LABELS = {"qutip_explicit": "Explicit QuTiP", "local_binned_0.003": "Current bins: .003",
          "local_binned_0.0015": "Finer bins: .0015", "local_smooth": "Smooth Mehler"}
COLORS = {"qutip_explicit": "#26354a", "local_binned_0.003": "#ba513b",
          "local_binned_0.0015": "#b08320", "local_smooth": "#168780"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def savefig(fig, out, name):
    for extension in ("png", "svg"):
        fig.savefig(out/f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", weight="bold", pad=10)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)


def analyze_gates(result):
    config = result["config"]
    last_na = max(config["ancilla_cutoffs"])
    refinement = {c["to_case"]+"/"+c["state"]: c for c in result["convergence"]}
    final = [row for case in result["cases"] for row in case["rows"] if row["ancilla_cutoff"] == last_na]
    reference_valid = {}
    for row in final:
        if row["arm"] != "qutip_explicit":
            continue
        key = row["case"]+"/"+row["state"]
        ref = refinement[key]
        components = {"mass": abs(row["integrated_mass"]-1) < 1e-5,
                      "norm": row["joint_norm_error"] < 1e-7,
                      "ancilla_refinement": ref["ancilla_refinement_pass"],
                      "grid_energy": abs(row["grid_energy_change"]) < .01,
                      "grid_fidelity": abs(row["grid_logical_fidelity_change"]) < 1e-4}
        reference_valid[key] = {"pass": all(components.values()), "checks": components}
    for row in final:
        key = row["case"]+"/"+row["state"]
        row["reference_valid"] = reference_valid[key]["pass"]
        row["ancilla_tv_sensitivity"] = refinement[key]["outcome_tv"]
        row["ancilla_state_infidelity_sensitivity"] = refinement[key]["weighted_conditional_infidelity"]
        row["smooth_threshold_pass"] = (row["outcome_tv"] < 1e-4 and row["weighted_conditional_infidelity"] < 1e-5)
    smooth = [r for r in final if r["arm"] == "local_smooth"]
    # If measured integration sensitivity reaches a decision boundary, avoid
    # claiming support based on the finest grid alone.
    robust = all(r["outcome_tv"]+abs(r["grid_tv_change"])+r["ancilla_tv_sensitivity"] < 1e-4 and
                 r["weighted_conditional_infidelity"]+abs(r["grid_conditional_infidelity_change"])+r["ancilla_state_infidelity_sensitivity"] < 1e-5 for r in smooth)
    if not all(v["pass"] for v in reference_valid.values()):
        verdict = "inconclusive_reference_gates"
    elif all(r["smooth_threshold_pass"] for r in smooth) and robust:
        verdict = "smooth_agreement_supported_in_tested_measurements"
    elif any(r["outcome_tv"]-abs(r["grid_tv_change"])-r["ancilla_tv_sensitivity"] > 1e-4 or
             r["weighted_conditional_infidelity"]-abs(r["grid_conditional_infidelity_change"])-r["ancilla_state_infidelity_sensitivity"] > 1e-5 for r in smooth):
        verdict = "smooth_agreement_contradicted_in_tested_measurements"
    else:
        verdict = "inconclusive_threshold_sensitivity"
    return final, reference_valid, verdict


def verify_saved_metrics(run, case):
    """Recompute scalar metrics from saved outcome curves, independently of plots."""
    states = {name: i for i, name in enumerate(("Z+", "Z-", "X+", "X-", "Y+", "Y-"))}
    with np.load(run/case["raw_file"], allow_pickle=False) as z:
        w, coarse = z["weights"], z["coarse_weights"]
        for row in case["rows"]:
            arm, index = row["arm"], states[row["state"]]
            p = z[f"{arm}_probability_density"][index]
            pr = z["qutip_explicit_probability_density"][index]
            mass, mr = float(p@w), float(pr@w)
            np.testing.assert_allclose(mass, row["integrated_mass"], rtol=1e-11, atol=1e-12)
            for name, key in (("energy_density", "expected_energy"), ("logical_fidelity_density", "expected_logical_fidelity"), ("edge_density", "expected_boundary_mass")):
                values = z[f"{arm}_{name}"][index]
                np.testing.assert_allclose(values@w/mass, row[key], rtol=1e-11, atol=1e-12)
            tv = .5*float(np.abs(pr/mr-p/mass)@w)
            infid = z[f'{arm}_{row["state"]}_conditional_infidelity']
            np.testing.assert_allclose(tv, row["outcome_tv"], rtol=1e-10, atol=1e-13)
            np.testing.assert_allclose((pr*infid)@w/mr, row["weighted_conditional_infidelity"], rtol=1e-10, atol=1e-13)
            coarse_mass = float(p@coarse)
            for name, key in (("energy_density", "grid_energy_change"), ("logical_fidelity_density", "grid_logical_fidelity_change")):
                values = z[f"{arm}_{name}"][index]
                change = values@coarse/coarse_mass-values@w/mass
                np.testing.assert_allclose(change, row[key], rtol=1e-8, atol=1e-12)


def agreement_plot(out, result):
    nd = max(result["config"]["data_cutoffs"])
    resources = list(result["config"]["resources"])
    fig, axes = plt.subplots(2, len(resources), figsize=(max(12, 7*len(resources)), 9), layout="constrained", squeeze=False)
    for column, resource in enumerate(resources):
        rows = [r for c in result["cases"] for r in c["rows"] if r["data_cutoff"] == nd and r["resource"] == resource]
        for line, (metric, threshold, ylabel) in enumerate([
                ("outcome_tv", 1e-4, "Outcome distribution total variation"),
                ("weighted_conditional_infidelity", 1e-5, "Probability-weighted conditional state infidelity")]):
            ax = axes[line, column]
            for arm in ARMS[1:]:
                xx = sorted({r["ancilla_cutoff"] for r in rows})
                groups = [[r[metric] for r in rows if r["arm"] == arm and r["ancilla_cutoff"] == n] for n in xx]
                lo, hi = np.array([min(v) for v in groups]), np.array([max(v) for v in groups])
                ax.semilogy(xx, np.maximum(hi, 1e-16), color=COLORS[arm], marker="o", label=LABELS[arm])
                ax.fill_between(xx, np.maximum(lo, 1e-16), np.maximum(hi, 1e-16), color=COLORS[arm], alpha=.10)
            ax.axhline(threshold, color="#59677c", ls=":", label="Pre-registered smooth-arm threshold")
            ax.set_xlabel("Ancilla Fock cutoff")
            ax.set_xticks(xx)
            style(ax, resource.replace("_", " ").title(), ylabel)
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Explicit QuTiP versus local measurement updates · data cutoff {nd}\n"
                 "Line: worst of six inputs × two extraction axes · band: tested range, not a confidence interval", fontsize=14)
    savefig(fig, out, "01_agreement")


def observable_plot(out, result):
    nd = max(result["config"]["data_cutoffs"])
    resources = list(result["config"]["resources"])
    fig, axes = plt.subplots(2, len(resources), figsize=(max(12, 7*len(resources)), 9), layout="constrained", squeeze=False)
    for column, resource in enumerate(resources):
        rows = [r for c in result["cases"] for r in c["rows"] if r["data_cutoff"] == nd and r["resource"] == resource]
        lookup = {(r["case"], r["state"]): r for r in rows if r["arm"] == "qutip_explicit"}
        for arm in ARMS:
            xx = sorted({r["ancilla_cutoff"] for r in rows})
            fidelities, energies, grid_fidelity_sensitivity = [], [], []
            for n in xx:
                rr = [r for r in rows if r["arm"] == arm and r["ancilla_cutoff"] == n]
                fidelities.append(100*np.mean([r["expected_logical_fidelity"]-lookup[r["case"],r["state"]]["expected_logical_fidelity"] for r in rr]))
                grid_fidelity_sensitivity.append(100*abs(np.mean([r["grid_logical_fidelity_change"]-lookup[r["case"],r["state"]]["grid_logical_fidelity_change"] for r in rr])))
                energies.append(np.mean([r["expected_energy"]-lookup[r["case"], r["state"]]["expected_energy"] for r in rr]))
            axes[0, column].plot(xx, fidelities, color=COLORS[arm], marker="o", label=LABELS[arm],
                                 ls="--" if arm == "qutip_explicit" else "-")
            if arm != "qutip_explicit":
                axes[0, column].fill_between(xx, np.array(fidelities)-grid_fidelity_sensitivity,
                                             np.array(fidelities)+grid_fidelity_sensitivity,
                                             color=COLORS[arm], alpha=.10)
            if arm != "qutip_explicit":
                axes[1, column].plot(xx, energies, color=COLORS[arm], marker="o", label=LABELS[arm])
        axes[1, column].axhline(0, color="#59677c", ls=":")
        ref_final = [r for r in rows if r["arm"] == "qutip_explicit" and r["ancilla_cutoff"] == max(xx)]
        absolute = 100*np.mean([r["expected_logical_fidelity"] for r in ref_final])
        axes[0, column].text(.03, .97, f"QuTiP mean fidelity at Na={max(xx)}: {absolute:.3f}%",
                             transform=axes[0, column].transAxes, va="top", fontsize=9,
                             bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
        style(axes[0, column], resource.replace("_", " ").title(), "Local − QuTiP logical fidelity (percentage points)")
        style(axes[1, column], "Energy sensitivity", "Local − QuTiP expected energy (photons)")
        for ax in axes[:, column]:
            ax.set_xlabel("Ancilla Fock cutoff");ax.set_xticks(xx);ax.legend(fontsize=8)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.suptitle(f"Average observable differences · data cutoff {nd}\n"
                 "Six fixed inputs × q/p extraction; fidelity bands show grid sensitivity, not confidence intervals", fontsize=13)
    savefig(fig, out, "02_observables")


def timing_plot(out, result):
    na = max(result["config"]["ancilla_cutoffs"])
    cases = [c for c in result["cases"] if c["rows"][0]["ancilla_cutoff"] == na]
    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    xx = np.arange(len(cases));width = .19
    for j, arm in enumerate(ARMS):
        med = np.array([c["timing"][arm]["median_seconds"] for c in cases])
        lo = np.array([c["timing"][arm]["min_seconds"] for c in cases])
        hi = np.array([c["timing"][arm]["max_seconds"] for c in cases])
        ax.bar(xx+(j-1.5)*width, med, width=width, color=COLORS[arm], label=LABELS[arm])
        ax.errorbar(xx+(j-1.5)*width, med, yerr=[np.maximum(0, med-lo), np.maximum(0, hi-med)],
                    fmt="none", color="#26354a", capsize=2, lw=.8)
    ax.set_yscale("log")
    ax.set_xticks(xx, [f'Nd={c["rows"][0]["data_cutoff"]}\n{c["rows"][0]["resource"].replace("_", " ")} · {c["rows"][0]["axis"]}' for c in cases], fontsize=9)
    style(ax, f"Measurement batch runtime · ancilla cutoff {na}", "Seconds for all six inputs and the full outcome grid")
    ax.legend(ncols=2, fontsize=9)
    fig.suptitle("Median with min/max whiskers; setup, common feedback and scoring excluded\n"
                 "Explicit two-mode evolution and reduced one-mode evaluation perform different amounts of work", fontsize=13)
    savefig(fig, out, "03_runtime")


def example_plot(out, run, result):
    nd, na = max(result["config"]["data_cutoffs"]), max(result["config"]["ancilla_cutoffs"])
    label = next(iter(result["config"]["resources"]))
    case = next(c for c in result["cases"] if c["rows"][0]["data_cutoff"] == nd and
                c["rows"][0]["ancilla_cutoff"] == na and c["rows"][0]["resource"] == label and c["rows"][0]["axis"] == "q")
    with np.load(run/case["raw_file"]) as z:
        y = z["y"]
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
        reference_p = z["qutip_explicit_probability_density"][0]
        relevant = reference_p > 1e-5
        for arm in ARMS:
            p = z[f"{arm}_probability_density"][0]
            axes[0, 0].plot(y/L, p, color=COLORS[arm], label=LABELS[arm], ls="--" if arm == "qutip_explicit" else "-")
            if arm != "qutip_explicit":
                axes[0, 1].plot(y/L, p-reference_p, color=COLORS[arm], label=LABELS[arm])
                infid = z[f"{arm}_Z+_conditional_infidelity"]
                axes[1, 0].semilogy(y[relevant]/L, np.maximum(infid[relevant], 1e-16), color=COLORS[arm], label=LABELS[arm])
            f = z[f"{arm}_logical_fidelity_density"][0]
            axes[1, 1].plot(y[relevant]/L, 100*f[relevant]/p[relevant], color=COLORS[arm], label=LABELS[arm], ls="--" if arm == "qutip_explicit" else "-")
        titles = ["A  Outcome density", "B  Outcome density difference from QuTiP",
                  "C  Conditional state discrepancy", "D  Conditional logical fidelity after feedback"]
        ys = ["Probability density p(y)", "Local p(y) − QuTiP p(y)", "Pure-state infidelity", "Logical fidelity (%)"]
        for ax, title, ylabel in zip(axes.flat, titles, ys):
            style(ax, title, ylabel);ax.set_xlabel("Homodyne outcome y / √π");ax.set_xlim(-2.5, 2.5);ax.legend(fontsize=8)
        fig.suptitle(f"Matched-outcome example · {label.replace('_', ' ')} · initial Z+ · q extraction\n"
                     f"Data cutoff {nd}; ancilla cutoff {na}; input displaced identically in every arm", fontsize=14)
        savefig(fig, out, "04_outcome_example")


def html_report(out, run, result, final, gates, verdict):
    def figure(stem, caption):
        uri = "data:image/png;base64,"+base64.b64encode((out/f"{stem}.png").read_bytes()).decode()
        return f'<a href="{stem}.svg"><img src="{uri}" alt="{html.escape(caption)}"></a><p class="caption">{caption} <a href="{stem}.png">PNG</a> · <a href="{stem}.svg">SVG</a></p>'
    rows = []
    for nd in result["config"]["data_cutoffs"]:
        for resource in result["config"]["resources"]:
            selected = [r for r in final if r["data_cutoff"] == nd and r["resource"] == resource]
            refs = {(r["case"], r["state"]): r for r in selected if r["arm"] == "qutip_explicit"}
            for arm in ARMS[1:]:
                group = [r for r in selected if r["arm"] == arm]
                rows.append(f'<tr><td>{nd}</td><td>{resource.replace("_", " ")}</td><td>{LABELS[arm]}</td>'
                            f'<td>{max(r["outcome_tv"] for r in group):.3g}</td>'
                            f'<td>{max(r["weighted_conditional_infidelity"] for r in group):.3g}</td>'
                            f'<td>{100*np.mean([r["expected_logical_fidelity"]-refs[r["case"],r["state"]]["expected_logical_fidelity"] for r in group]):+.5f}</td>'
                            f'<td>{np.mean([r["expected_energy"]-refs[r["case"],r["state"]]["expected_energy"] for r in group]):+.5f}</td>'
                            f'<td>{sum(r["reference_valid"] for r in group)}/{len(group)}</td></tr>')
    valid_count = sum(g["pass"] for g in gates.values())
    conclusion = {"smooth_agreement_supported_in_tested_measurements": "Smooth evaluation agrees within the registered thresholds",
                  "smooth_agreement_contradicted_in_tested_measurements": "Smooth agreement fails a registered threshold",
                  "inconclusive_reference_gates": "Some reference gates remain unresolved",
                  "inconclusive_threshold_sensitivity": "Agreement is inconclusive at the registered thresholds"}[verdict]
    failing = [key for key, value in gates.items() if not value["pass"]]
    failed_text = html.escape(", ".join(failing)) if failing else "None."
    setup = sum(result["setup_seconds"].values())
    solver_text = html.escape(json.dumps(result.get("solver_checks", []), indent=2))
    health = []
    for nd in result["config"]["data_cutoffs"]:
        for resource in result["config"]["resources"]:
            group = [r for r in final if r["data_cutoff"] == nd and r["resource"] == resource and r["arm"] == "qutip_explicit"]
            health.append(f'<tr><td>{nd}</td><td>{resource.replace("_", " ")}</td>'
                          f'<td>{max(abs(r["integrated_mass"]-1) for r in group):.3g}</td>'
                          f'<td>{max(r["joint_norm_error"] for r in group):.3g}</td>'
                          f'<td>{100*max(r["ancilla_final_boundary_mass"] for r in group):.3g}%</td>'
                          f'<td>{100*max(r["expected_boundary_mass"] for r in group):.3g}%</td>'
                          f'<td>{max(abs(r["grid_logical_fidelity_change"]) for r in group):.3g}</td></tr>')
    text = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Explicit QuTiP versus local engine</title><style>
body{margin:0;background:#f1f5f8;color:#1c2c40;font:16px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:36px 24px 60px}h1{font-size:clamp(28px,4vw,42px);line-height:1.18;letter-spacing:-.025em}h2{line-height:1.25;margin-top:12px}article,.notice{background:white;border:1px solid #d7e0eb;border-radius:12px;padding:24px;margin:22px 0}.notice{border-left:5px solid #168780}.tag{font-size:12px;letter-spacing:.09em;text-transform:uppercase;font-weight:700;color:#53657d}p,li{max-width:1080px}img{width:100%;height:auto}a{color:#3856a0}nav{display:flex;flex-wrap:wrap;gap:20px}.caption{font-size:13px;color:#53657d}.scroll{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #dce4ed}th{background:#f1f5f9}code,pre{background:#edf2f7;border-radius:5px;padding:3px 6px}pre{padding:14px;overflow:auto;font-size:12px}.numbers{display:flex;flex-wrap:wrap;gap:14px}.number{background:#eaf1f6;padding:14px 20px;border-radius:9px;flex:1;min-width:220px}.number b{display:block;font-size:26px}.muted{color:#53657d}@media print{nav,.caption a{display:none}body{background:white}main{padding:0}article{break-inside:avoid}}
</style></head><body><main><div class="tag">Local benchmark · explicit two-oscillator control</div><h1>QuTiP versus our measurement engine</h1><div class="notice"><h2>''' + conclusion + '''</h2><p>This benchmark compares the <strong>same finite data operator and the same homodyne outcomes</strong>. QuTiP propagates an explicit data-plus-ancilla ket. Local arms apply the reduced measurement operator using the current lookup, finer lookup bins, or a smooth analytic resource. Input preparation, feedback and logical scoring are shared to isolate measurement extraction.</p></div>
<div class="numbers"><div class="number"><b>''' + f'{valid_count}/{len(gates)}' + '''</b>final reference input/axis/resource checks pass</div><div class="number"><b>''' + str(len(result["cases"])) + '''</b>explicit circuit configurations</div><div class="number"><b>''' + f'{result["total_wall_seconds"]:.1f} s' + '''</b>recorded run time, including setup and analysis</div></div>
<nav><a href="#agreement">Agreement</a><a href="#observables">Logical fidelity and energy</a><a href="#cost">Runtime</a><a href="#outcomes">Individual outcomes</a><a href="#scope">Interpretation and limits</a><a href="#evidence">Data and numerical checks</a></nav>
<article id="agreement"><h2>How closely do the measurement updates agree?</h2><p>Smaller is better. Outcome total variation measures differences in the probability distribution. Conditional infidelity measures whether both engines produce the same quantum state when given the same outcome. Both are needed: nearly identical average fidelities can conceal different conditional states.</p>''' + figure("01_agreement", "Worst-case agreement over the fixed six states and q/p axes; shaded ranges are descriptive, not confidence intervals.") + '''<div class="scroll"><table><thead><tr><th>Data cutoff</th><th>Resource</th><th>Local arm</th><th>Worst outcome TV</th><th>Worst weighted state infidelity</th><th>Mean logical fidelity difference (pp)</th><th>Mean energy difference (photons)</th><th>Reference checks passed</th></tr></thead><tbody>''' + ''.join(rows) + '''</tbody></table></div><p class="caption">Table uses the largest ancillary cutoff. Maxima cover all tested input states and extraction axes; means weight those fixed cases equally. Values are not averages over a stochastic noise ensemble.</p></article>
<article id="observables"><h2>Logical fidelity and energy after common feedback</h2>''' + figure("02_observables", "Local minus explicit QuTiP after a single q or p extraction and common feedback. Equal averages over fixed matched inputs; shaded fidelity ranges show the observed integration-grid sensitivity of the difference.") + '''<p>These are observables of a finite-data-operator benchmark. The small data cutoffs were chosen to make the explicit two-mode comparison tractable. Agreement at these cutoffs does not certify convergence of the physical oscillator's energy or long-time decoder performance.</p></article>
<article id="cost"><h2>Runtime under matched output requirements</h2>''' + figure("03_runtime", "All six inputs and the same full outcome grid. Three repetitions at the largest ancilla cutoff in the main benchmark; whiskers show min/max.") + f'''<p>The shared input/scoring and quadrature-basis setup totalled {setup:.2f} seconds. Per-case setup is recorded separately in the result JSON. Timed QuTiP work includes full joint-state propagation and projection; local work includes resource evaluation and reduced conditional-ket formation. Common feedback, logical scoring, disk output and shared precomputation are excluded from these bars.</p>''' + '''<p>These timings compare two implementations with different state-space costs. They are not a comparison of QuTiP and NumPy doing identical linear algebra, and are not per-shot decoder latency. They were measured on a shared machine after memory-related setup interruptions. Treat them as descriptive throughput measurements; the min/max spread does not capture all machine-load variability.</p></article>
<article id="outcomes"><h2>Inspect an individual outcome distribution</h2>''' + figure("04_outcome_example", "A displaced Z+ example. Conditional panels omit nodes with QuTiP outcome density below 1e-5; integrated summary metrics use the full window.") + '''</article>
<article id="scope"><h2>What this establishes</h2><p>This tests whether reduced measurement updates reproduce an explicit circuit, with ancillary cutoff, integration-grid and solver-precision checks. The smooth arm evaluates the continuous Mehler wavefunction with analytic normalization; it has no new continuous outcome sampler and has not replaced the production implementation.</p><p>A smaller discrepancy after smoothing supports a discretization contribution to the tested measurement. It does not, on its own, establish the origin of the earlier cycle-6 rare energy jump. That requires an end-to-end smooth sampler and a matched long-horizon replay with substep logging.</p><p>The comparison covers one q <em>or</em> p extraction, finite data cutoffs, two resource widths in the main run, and six seeded displaced input states. It is not a q/p cycle, an LSTM/Bayes comparison, a temporal-noise robustness test or hardware validation. State preparation and scoring are shared, so their correctness is not independently established here.</p><p>QuTiP provides the Schrödinger evolution and harmonic-oscillator wavefunction kernel. Benchmark code constructs the GKP ancilla and contracts continuous homodyne bras; there is no packaged QuTiP GKP decoder involved. The kernel is the same one used by its public <a href="https://qutip.readthedocs.io/en/stable/apidoc/visualization.html#qutip.distributions.HarmonicOscillatorWaveFunction">wavefunction evaluator</a>; the evolution uses <a href="https://qutip.readthedocs.io/en/stable/apidoc/solver.html#qutip.solver.sesolve.sesolve">sesolve</a>.</p></article>
<article id="evidence"><h2>Data, provenance and numerical checks</h2><p><a href="metrics.csv">All per-state metrics CSV</a> · <a href="result.json">Full results and timings</a> · <a href="audit.json">Numerical gates and source hashes</a> · <a href="plan.md">Pre-run methods and thresholds</a></p><p>Reference gates require outcome mass within 1e-5 of one, joint norm error below 1e-7, final ancilla refinement below the registered TV/state/energy/fidelity limits, and stable integrated energy/fidelity on the finer grid. There are no Monte Carlo intervals: these are deterministic quadrature comparisons. Refinement differences are numerical sensitivities, not rigorous error bounds.</p><div class="scroll"><table><thead><tr><th>Data cutoff</th><th>Resource</th><th>Worst outcome-mass error</th><th>Worst joint norm error</th><th>Worst final ancilla boundary mass</th><th>Worst expected data boundary mass after feedback</th><th>Worst grid fidelity change</th></tr></thead><tbody>''' + ''.join(health) + '''</tbody></table></div><p class="caption">Boundary mass uses the highest 10% of levels. Ancilla mass is evaluated in the joint state after coupling. Data mass is the expected value after feedback. These small-cutoff data operators are held fixed across engines: a passing measurement comparison does not imply that their physical data populations are converged.</p><p>Failed reference checks: ''' + failed_text + '''</p><details><summary>ODE tolerance checks</summary><pre>''' + solver_text + '''</pre></details><p class="muted">All raw-case hashes and frozen source snapshots were verified before plotting. Old reference data and the current local engine were left unchanged. The report's figures are embedded for offline viewing; PNG/SVG downloads are neighboring files.</p></article></main></body></html>'''
    (out/"index.html").write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    result = json.loads((run/"result.json").read_text())
    if not result["complete"]:
        raise ValueError("Benchmark is incomplete")
    for source, expected in result["config"]["source_hashes"].items():
        if sha(run/"source_snapshot"/source) != expected:
            raise ValueError(f"Source snapshot hash mismatch: {source}")
    for case in result["cases"]:
        if sha(run/case["raw_file"]) != case["raw_sha256"]:
            raise ValueError(f"Raw hash mismatch: {case['raw_file']}")
        verify_saved_metrics(run, case)
    final, gates, verdict = analyze_gates(result)
    out = run/"review";out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.frameon": False})
    agreement_plot(out, result)
    observable_plot(out, result)
    timing_plot(out, result)
    example_plot(out, run, result)
    for name in ("metrics.csv", "result.json"):
        shutil.copyfile(run/name, out/name)
    shutil.copyfile(run/"source_snapshot/docs/qutip_engine_benchmark.md", out/"plan.md")
    audit = {"verdict": verdict, "reference_checks": gates, "final_rows": final,
             "all_raw_hashes_verified": True, "all_source_snapshots_verified": True,
             "saved_scalar_metrics_recomputed_from_raw_curves": True,
             "result_sha256": sha(run/"result.json"), "analysis_script_sha256": sha(__file__),
             "input_run": str(run), "scope": result["scope"]}
    save(out/"audit.json", audit)
    html_report(out, run, result, final, gates, verdict)
    print(json.dumps({"report": str(out/"index.html"), "verdict": verdict,
                      "reference_valid": sum(g["pass"] for g in gates.values()), "reference_checks": len(gates)}, indent=2))


if __name__ == "__main__":
    main()
