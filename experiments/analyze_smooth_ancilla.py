"""Audit and render the smooth-ancilla accuracy decision and eight-cycle run."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {3072: "#ba513b", 4096: "#1b7b8f", 5120: "#694ca6"}
ARM_COLORS = {"binned": "#ba513b", "smooth": "#168780"}
L = np.sqrt(np.pi)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimate(values, per_state):
    v = np.asarray(values).reshape(6, per_state, -1)
    mean = v.mean(axis=(0, 1));se = np.sqrt(v.var(axis=1, ddof=1).sum(axis=0)/(36*per_state))
    return {"mean": mean, "se": se, "ci95_low": mean-1.95996398454*se, "ci95_high": mean+1.95996398454*se}


def audit_validation(run, validation):
    by_case = {}
    for row in validation["rows"]:by_case.setdefault(row["case"], []).append(row)
    names = {name: i for i, name in enumerate(("Z+", "Z-", "X+", "X-", "Y+", "Y-"))}
    for case, rows in by_case.items():
        with np.load(run/"validation"/(case+".npz")) as z:
            for row in rows:
                i, arm = names[row["state"]], row["arm"]
                p = z[f"{arm}_probability_{i}"];pr = z[f"qutip_probability_{i}"];w = z["weights"]
                mr, m = float(pr@w), float(p@w)
                infid = z[f"{arm}_infidelity_{i}"]
                observed = float((pr*infid)@w/mr)
                tv = .5*float(np.abs(pr/mr-p/m)@w)
                np.testing.assert_allclose(observed, row["weighted_conditional_infidelity"], rtol=1e-10, atol=1e-14)
                np.testing.assert_allclose(tv, row["outcome_tv"], rtol=1e-10, atol=1e-13)
                np.testing.assert_allclose(1-observed, row["accuracy"], rtol=0, atol=1e-14)


def savefig(fig, out, name):
    for suffix in ("png", "svg"):fig.savefig(out/f"{name}.{suffix}", dpi=175, facecolor="white")
    plt.close(fig)


def style(ax, title, ylabel, xlabel="Completed correction cycles"):
    ax.set_title(title, loc="left", weight="bold", pad=10);ax.set_ylabel(ylabel);ax.set_xlabel(xlabel)
    ax.grid(alpha=.17);ax.spines[["top", "right"]].set_visible(False)


def accuracy_plot(out, validation):
    rows = validation["rows"]
    resources = list(dict.fromkeys(r["resource"] for r in rows))
    fig, axes = plt.subplots(1, len(resources), figsize=(max(12, 7*len(resources)), 5.8), layout="constrained", squeeze=False)
    for ax, resource in zip(axes[0], resources):
        selected = [r for r in rows if r["resource"] == resource]
        nn = sorted({r["ancilla_cutoff"] for r in selected})
        for arm in ("binned", "smooth"):
            maxima = [max(r["weighted_conditional_infidelity"] for r in selected if r["arm"] == arm and r["ancilla_cutoff"] == n) for n in nn]
            ax.semilogy(nn, np.maximum(maxima, 1e-16), marker="o", color=ARM_COLORS[arm], label="Current binned lookup" if arm == "binned" else "Continuous resource + sampler")
        ax.axhline(1e-5, color="#4c596d", ls=":", label="Accuracy ≥99.999% threshold")
        ax.set_xticks(nn)
        style(ax, resource.replace("_", " ").title(), "1 − accuracy A (lower is better)", "QuTiP ancilla cutoff")
        ax.legend(fontsize=9)
    fig.suptitle("Accuracy against independent two-mode QuTiP on fresh inputs\n"
                 "A = reference-weighted squared state fidelity; worst over six displaced states × q/p extraction", fontsize=14)
    savefig(fig, out, "01_accuracy")


def memory_plot(out, result, raw, controls):
    rows = result["long"]["rows"];t = np.arange(result["config"]["rounds"]+1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    for row in rows:
        n = row["cutoff"];color = COLORS[n]
        for key, ls in (("smooth_fidelity", "-"), ("binned_fidelity", "--")):
            axes[0, 0].plot(t, 100*np.array(row[key]["mean"]), color=color, ls=ls, label=f"{'Smooth' if ls == '-' else 'Binned'} N={n:,}")
        if n == 5120:
            e = row["smooth_fidelity"]
            axes[0, 0].fill_between(t, 100*np.array(e["ci95_low"]), 100*np.array(e["ci95_high"]), color=color, alpha=.12)
        d = row["smooth_minus_binned"]
        axes[0, 1].plot(t, 100*np.array(d["mean"]), color=color, label=f"N={n:,}")
        axes[0, 1].fill_between(t, 100*np.array(d["ci95_low"]), 100*np.array(d["ci95_high"]), color=color, alpha=.12)
        for ax, key, threshold in ((axes[1, 0], "max_edge", .01), (axes[1, 1], "mean_edge", 1e-4)):
            ax.semilogy(t[1:], 100*np.array(row[key][1:]), color=color, marker="o", ms=3, label=f"Smooth N={n:,}")
            binned = controls[n]["edge"]
            values = binned.max(0) if key == "max_edge" else binned.mean(0)
            ax.semilogy(t[1:], 100*values[1:], color=color, ls="--", alpha=.6, label=f"Binned N={n:,}")
            ax.axhline(100*threshold, color="#555", ls=":", lw=.8)
    n = max(raw)
    for arm, values, ls in (("smooth", raw[n]["energy"], "-"), ("binned", controls[n]["energy"], "--")):
        for metric, arr, color in (("Mean", values.mean(0), "#24384d"), ("99th percentile", np.quantile(values, .99, axis=0), "#77899b"), ("Maximum", values.max(0), "#9b4872")):
            axes[0, 2].plot(t, arr, color=color, ls=ls, label=f"{metric}, {arm}")
    axes[0, 1].axhline(0, color="#555", lw=.8)
    final = next((c for c in result["long"]["cutoff_comparisons"] if c["from_cutoff"] == 4096 and c["to_cutoff"] == 5120), None)
    if final:
        axes[1, 2].axhspan(-1, 1, color="#d6e9dc", alpha=.65, label="Registered ±1 pp band")
        mean, low, high = [100*np.array(final[k]) for k in ("mean", "ci95_low", "ci95_high")]
        axes[1, 2].errorbar(t, mean, yerr=[np.maximum(0, mean-low), np.maximum(0, high-mean)], color=COLORS[5120], fmt="o-", capsize=3, ms=4, label="Smooth 5,120 − 4,096")
        axes[1, 2].axhline(0, color="#555", lw=.8)
    labels = [("A  Logical memory fidelity", "Six-state average logical fidelity (%)"),
              ("B  Paired change from the binned control", "Smooth − binned fidelity (percentage points)"),
              ("C  Energy tails at the largest cutoff", "Trajectory mean photon number"),
              ("D  Worst boundary population", "Probability in highest 10% of Fock levels (%)"),
              ("E  Ensemble-mean boundary population", "Mean probability in highest 10% of levels (%)"),
              ("F  Smooth cutoff stability", "Fidelity difference (percentage points)")]
    for ax, (title, label) in zip(axes.flat, labels):
        style(ax, title, label);ax.set_xticks(t);ax.legend(fontsize=7.5)
    axes[1, 0].set_ylim(top=110)
    fig.suptitle("Matched eight-cycle experiment · nominal noise · nearest-lattice controller\n"
                 "512 trajectories per cardinal state (3,072 total); pointwise 95% intervals are stratified trajectory intervals", fontsize=14)
    savefig(fig, out, "02_memory")


def forensic_plot(out, run, result):
    n = max(r["cutoff"] for r in result["long"]["rows"])
    row = next(r for r in result["long"]["rows"] if r["cutoff"] == n)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    events = []
    for arm in ("binned", "smooth"):
        with np.load(run/row["forensic"][arm]["path"]) as z:
            energy, edge, outcomes, logdensity = [z[k] for k in ("stage_energy", "stage_edge", "outcomes", "log_outcome_density")]
            indices = z["original_indices"]
            for i, index in enumerate(indices):
                flat = energy[i].ravel()
                steps = np.repeat(np.arange(energy.shape[1]), 6)+np.tile(np.linspace(0, 1, 6), energy.shape[1])
                axes[0, 0].plot(steps, flat, color=ARM_COLORS[arm], alpha=1 if i == 0 else .35, lw=2 if i == 0 else .8,
                                label=f"{arm.title()} {index}" if i == 0 else None)
                axes[0, 1].semilogy(steps, np.maximum(edge[i].ravel()*100, 1e-8), color=ARM_COLORS[arm], alpha=1 if i == 0 else .35, lw=2 if i == 0 else .8,
                                   label=f"{arm.title()} {index}" if i == 0 else None)
                delta = np.diff(flat);at = int(np.argmax(delta))+1;cycle, stage = divmod(at, 6)
                event = {"arm": arm, "cutoff": n, "trajectory": int(index), "cycle": cycle+1,
                               "stage": result["long"]["stages"][stage], "energy_increase": float(delta[at-1]),
                               "energy_after": float(flat[at]), "boundary_after": float(edge[i].ravel()[at])}
                if stage in (2, 4):
                    axis = (stage-2)//2
                    event["unwrapped_homodyne_over_L"] = float(outcomes[axis, i, cycle]/L)
                    event["log_outcome_density"] = float(logdensity[axis, i, cycle])
                events.append(event)
                if i == 0:
                    for a, axis in enumerate(("q", "p")):
                        axes[1, a].plot(np.arange(1, outcomes.shape[2]+1), logdensity[a, i],
                                       marker="o", color=ARM_COLORS[arm], label=arm.title())
    axes[0, 1].axhline(1., color="#555", ls=":", label="1% boundary gate")
    style(axes[0, 0], "A  Energy at every substep", "Mean photon number", "Cycle progress (6 saved stages per cycle)")
    style(axes[0, 1], "B  Boundary weight at every substep", "Boundary population (%)", "Cycle progress (6 saved stages per cycle)")
    style(axes[1, 0], "C  q-measurement outcome density", "Natural log p(y) [density, not probability]", "Correction cycle")
    style(axes[1, 1], "D  p-measurement outcome density", "Natural log p(y) [density, not probability]", "Correction cycle")
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.suptitle(f"Selected forensic replays · cutoff {n:,}\n"
                 "Dark lines: trajectory 1659; faint lines: 1973 and 2776. Selection is based on the historical binned run.", fontsize=13)
    savefig(fig, out, "03_forensics")
    return events


def write_report(out, result, summary):
    def figure(stem, alt):
        uri = "data:image/png;base64,"+base64.b64encode((out/f"{stem}.png").read_bytes()).decode()
        return f'<a href="{stem}.svg"><img src="{uri}" alt="{html.escape(alt)}"></a><p class="caption"><a href="{stem}.png">PNG</a> · <a href="{stem}.svg">SVG</a></p>'
    final = summary["accuracy_rows"]
    table = []
    for arm in ("binned", "smooth"):
        rows = [r for r in final if r["arm"] == arm]
        table.append(f'<tr><td>{arm.title()}</td><td>{100*min(r["accuracy"] for r in rows):.10f}%</td>'
                     f'<td>{max(r["weighted_conditional_infidelity"] for r in rows):.5g}</td>'
                     f'<td>{max(r["outcome_tv"] for r in rows):.5g}</td></tr>')
    longtable = []
    for row in result["long"]["rows"]:
        n = row["cutoff"];d = row["smooth_minus_binned"]
        longtable.append(f'<tr><td>{n:,}</td><td>{row["valid_through_cycle"]}</td>'
                         f'<td>{"Pass" if all(row["stage_population_pass"]) else "Fail"}</td>'
                         f'<td>{100*row["smooth_fidelity"]["mean"][-1]:.3f}%</td>'
                         f'<td>{100*d["mean"][-1]:+.3f} [{100*d["ci95_low"][-1]:+.3f}, {100*d["ci95_high"][-1]:+.3f}]</td>'
                         f'<td>{100*max(row["max_stage_edge"]):.5g}%</td></tr>')
    adopted = result["decision"] == "adopt_smooth_for_reference"
    title = "Decision: adopt the smooth measurement implementation" if adopted else "Decision: retain a limited validation scope"
    decision_text = ("The smooth resource and matching continuous sampler are now the default for new reference CLI runs. Historical binned mode remains available for reproduction."
                     if adopted else "The smooth implementation has a limited validated scope; long-horizon certification is not established.")
    events = ''.join(f'<tr><td>{e["arm"]}</td><td>{e["trajectory"]}</td><td>{e["cycle"]}</td><td>{e["stage"].replace("_", " ")}</td><td>{e["energy_increase"]:.3f}</td></tr>' for e in summary["forensic_events"])
    validation = result["validation"]
    main = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Smooth measurement: accuracy decision</title><style>
body{margin:0;color:#1d2d42;background:#f2f5f8;font:16px/1.6 system-ui,sans-serif}main{max-width:1300px;margin:auto;padding:36px 24px 60px}h1{font-size:clamp(27px,4vw,42px);line-height:1.2;letter-spacing:-.025em}h2{line-height:1.3}article,.notice{background:white;padding:24px;border:1px solid #d6e0eb;border-radius:12px;margin:24px 0}.notice{border-left:5px solid #168780}img{width:100%;height:auto}a{color:#36559b}nav{display:flex;gap:18px;flex-wrap:wrap}.tag{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#55677f}.caption{color:#52647c;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #dce5ed;padding:12px}th{background:#f2f6f9}.scroll{overflow:auto}.formula{font:22px/1.6 Georgia,serif;background:#edf3f7;padding:16px;border-radius:8px}p,li{max-width:1100px}pre{background:#edf3f7;padding:16px;overflow:auto;font-size:12px}code{background:#edf3f7;padding:2px 5px;border-radius:4px}@media print{nav{display:none}body{background:white}article{break-inside:avoid}}
</style></head><body><main><div class="tag">Preregistered local experiment · accuracy-driven implementation decision</div><h1>''' + title + '''</h1><div class="notice"><p><strong>The selection metric is agreement with an independent QuTiP conditional state.</strong> A higher logical memory fidelity alone cannot win this comparison. The smooth implementation must also pass outcome-distribution accuracy, continuous-sampler validation, and the registered long-horizon numerical gates.</p><p>''' + decision_text + ''' Raw observations, engineering tolerances and the pre-run protocol remain available below.</p></div><nav><a href="#accuracy">Accuracy and decision rule</a><a href="#memory">Eight-cycle results</a><a href="#forensics">Rare-event analysis</a><a href="#sources">Method and metric citations</a><a href="#data">Data and limits</a></nav>
<article id="accuracy"><h2>The accuracy metric</h2><div class="formula">A = ∫ p<sub>QuTiP</sub>(y) |⟨ψ<sub>QuTiP</sub>(y) | ψ<sub>model</sub>(y)⟩|² dy.</div><p>The inner term is squared quantum-state fidelity: 1 means identical pure states up to global phase. We average it over the reference outcome distribution and report the <strong>worst accuracy over the fixed held-out inputs</strong>. The squared-fidelity convention follows <a href="https://doi.org/10.1080/09500349414552171">Jozsa (1994)</a>. This reference weighting and the numerical acceptance budget are our engineering choices, not thresholds prescribed by that paper.</p><p>Before these runs, acceptance was set at <strong>A ≥ 99.999%</strong>, with outcome total variation ≤10⁻⁴. The new test uses six freshly displaced cardinal states, seed26111923, both extraction axes, two resource widths, and ancillary cutoff refinement from512 to768. It is a finite-data-operator comparison at96 data levels; it does not establish a worst-case channel bound.</p>''' + figure("01_accuracy", "Held-out conditional-state accuracy against full two-mode QuTiP") + '''<div class="scroll"><table><thead><tr><th>Implementation</th><th>Minimum reference accuracy A</th><th>Maximum weighted infidelity (1−A)</th><th>Maximum outcome TV</th></tr></thead><tbody>''' + ''.join(table) + '''</tbody></table></div><p>The inverse-CDF sampler uses the positive Gaussian-mixture representation of the same smooth wavefunction density. Its quantile residual and independently integrated CDF are checked. The empirical outcome check uses65,536 samples per input/axis/resource combination, with a strict familywise95% <a href="https://doi.org/10.1214/aop/1176990746">DKW–Massart bound</a>. No Monte Carlo intervals are attached to the deterministic fidelity integrals.</p><p>Accuracy gate: ''' + str(validation["primary_accuracy_pass"]) + ''' · Sampler gate: ''' + str(validation["sampler_pass"]) + ''' · Ancilla reference-refinement gate: ''' + str(validation["reference_refinement_pass"]) + '''.</p></article>
<article id="memory"><h2>Matched eight-cycle experiment</h2><p>Both implementations use nominal displacement/readout settings(.20,.05,.90), nearest-lattice feedback, Δ=.30,512 trajectories for each of six input states, and the historical seed26111901. The binned observations are frozen controls whose hashes and input noise arrays are checked. Smooth outcomes use a matching continuous sampler. Common random numbers do not force the two models to have identical measurement histories.</p>''' + figure("02_memory", "Eight-cycle fidelity, energy and numerical health") + '''<div class="scroll"><table><thead><tr><th>Cutoff</th><th>Cycle-end gates valid through</th><th>All substep population gates</th><th>Smooth cycle8 fidelity</th><th>Smooth−binned cycle8 (pp) [95% interval]</th><th>Largest individual substep boundary mass</th></tr></thead><tbody>''' + ''.join(longtable) + '''</tbody></table></div><p>Population limits are ensemble mean top-decile mass &lt;.01% and every trajectory &lt;1%, checked at every recorded substep. The final4096-versus5120 paired fidelity interval must fit entirely within±1 percentage point at every cycle. Trajectory intervals are pointwise, stratified over the six states; they exclude model uncertainty. Logical memory fidelity is a physical performance measure for this protocol, while A measures implementation accuracy against QuTiP.</p></article>
<article id="forensics"><h2>What happened to the rare energy event?</h2>''' + figure("03_forensics", "Substep energy, boundary mass and unwrapped-outcome density diagnostics") + '''<div class="scroll"><table><thead><tr><th>Replay arm</th><th>Selected trajectory</th><th>Cycle of largest substep increase</th><th>Substep</th><th>Photon-number increase</th></tr></thead><tbody>''' + events + '''</tbody></table></div><p>At cutoff5,120, the selected binned trajectory also exceeds the1% boundary limit at an intermediate substep, despite passing its historical cycle-end checks. The continuous run passes at every recorded substep. These indices were selected from the old binned run before the new experiment. Replay errors against full-batch observations are saved in the audit. The plots identify where energy changes, not a universal physical mechanism. Differences across stochastic histories alone cannot prove that every rare high-energy event is artificial.</p><p>Decision interpretation and maximum observed energy jumps across all trajectories are recorded in <a href="summary.json">summary.json</a>. This is a combined numerical implementation change: the resource representation and its matching sampler change together. The experiment does not separately assign their contributions. It does not add damping, loss, stabilization or decoder retraining.</p></article>
<article id="sources"><h2>Method and metric sources</h2><p>The physical method is finite-energy GKP syndrome extraction using a fresh ancilla, SUM interaction and homodyne measurement, followed by a displacement correction. The local update M<sub>y</sub>=φ(y−q) is the reduced form of the explicit circuit; momentum extraction uses the Fourier conjugate. See <a href="https://doi.org/10.1103/PhysRevA.64.012310">Gottesman, Kitaev &amp; Preskill (2001)</a> and <a href="https://doi.org/10.1103/PRXQuantum.2.020101">Grimsmo &amp; Puri (2021)</a>. Approximate finite-energy state representations are discussed by <a href="https://doi.org/10.1103/PhysRevA.102.032408">Matsuura, Yamasaki &amp; Koashi (2020)</a>.</p><p>Metric: squared state fidelity, <a href="https://doi.org/10.1080/09500349414552171">Jozsa, Fidelity for Mixed Quantum States (1994)</a>, reducing to|⟨ψ|χ⟩|² for pure states. Sampler-distribution check: <a href="https://doi.org/10.1214/aop/1176990746">Massart (1990)</a>. Bibliographic correction (19 November 2026): the frozen pre-run protocol misattributes PRX Quantum 2, 020101; its authors are Grimsmo and Puri, as cited here. The Gaussian-mixture identity, optimized evaluation, inverse-CDF algorithm and reference weighting are our implementation, independently checked here. This experiment is not a reproduction of a published numerical performance value.</p></article>
<article id="data"><h2>Evidence and limits</h2><p><a href="summary.json">Decision summary and numerical checks</a> · <a href="validation.csv">Held-out accuracy CSV</a> · <a href="memory.csv">Memory metrics CSV</a> · <a href="result.json">Complete run record</a> · <a href="protocol.md">Frozen pre-run protocol</a></p><p>The accuracy comparison covers fixed states and finite operators, not all states or hardware. The eight-cycle study covers one noise setting and the nearest-lattice controller; LSTM and Bayes conclusions do not follow. The thresholds are empirical numerical budgets and refinement checks, not rigorous infinite-cutoff bounds. All source snapshots and raw hashes are verified before rendering. Figures are embedded for offline viewing.</p></article></main></body></html>'''
    (out/"index.html").write_text(main)


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument("run", type=Path);args = ap.parse_args()
    run = args.run.resolve();result = json.loads((run/"result.json").read_text())
    if not result.get("complete") or "long" not in result:
        raise ValueError("Complete validation and long-horizon records required")
    for source, digest in result["config"]["source_hashes"].items():
        if sha(run/"source_snapshot"/source) != digest:raise ValueError(f"Source hash mismatch: {source}")
    for name, digest in result["validation"]["raw_hashes"].items():
        if sha(run/"validation"/name) != digest:raise ValueError(f"Validation hash mismatch: {name}")
    audit_validation(run, result["validation"])
    raw, controls, jumps = {}, {}, []
    for row in result["long"]["rows"]:
        n = row["cutoff"]
        if sha(run/row["raw_file"]) != row["raw_sha256"]:raise ValueError("Long raw hash mismatch")
        if sha(ROOT/row["control_file"]) != row["control_sha256"]:raise ValueError("Historical control hash mismatch")
        with np.load(run/row["raw_file"]) as z:raw[n] = {k: z[k] for k in ("fidelity", "energy", "edge", "stage_energy", "stage_edge", "dq", "dp")}
        with np.load(ROOT/row["control_file"]) as z:controls[n] = {k: z[k] for k in ("fidelity", "energy", "edge", "dq", "dp")}
        for key in ("dq", "dp"):np.testing.assert_array_equal(raw[n][key], controls[n][key])
        per_state = result["config"]["per_state"]
        for key, actual in (("smooth_fidelity", raw[n]["fidelity"]), ("binned_fidelity", controls[n]["fidelity"]),
                            ("smooth_minus_binned", raw[n]["fidelity"]-controls[n]["fidelity"])):
            for metric, values in estimate(actual, per_state).items():
                np.testing.assert_allclose(values, row[key][metric], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(raw[n]["stage_energy"][:, :, -1], raw[n]["energy"][:, 1:], atol=1e-8, rtol=1e-10)
        for arm in ("binned", "smooth"):
            if sha(run/row["forensic"][arm]["path"]) != row["forensic"][arm]["sha256"]:raise ValueError("Forensic raw hash mismatch")
            e = controls[n]["energy"] if arm == "binned" else raw[n]["energy"]
            jumps.append({"cutoff": n, "arm": arm, "maximum_cycle_energy_increase": float(np.diff(e, axis=1).max()),
                          "trajectories_with_cycle_energy_increase_above_1000": int((np.diff(e, axis=1).max(1) > 1000).sum())})
    out = run/"review";out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.frameon": False})
    accuracy_plot(out, result["validation"]);memory_plot(out, result, raw, controls)
    events = forensic_plot(out, run, result)
    maximum = max(r["ancilla_cutoff"] for r in result["validation"]["rows"])
    final = [r for r in result["validation"]["rows"] if r["ancilla_cutoff"] == maximum]
    reference = {r["case"]+r["state"]: r for r in result["validation"]["refinements"]}
    sensitivity = []
    for row in final:
        if row["arm"] == "smooth":
            ref = reference[row["case"]+row["state"]]
            empirical_error = (np.sqrt(row["weighted_conditional_infidelity"])+np.sqrt(ref["weighted_conditional_infidelity"]))**2+abs(row["grid_infidelity_change"])
            sensitivity.append({"case": row["case"], "state": row["state"], "accuracy_minus_observed_sensitivity": 1-float(empirical_error)})
    summary = {"decision": result["decision"], "accuracy_rows": final, "energy_jumps": jumps,
               "forensic_events": events, "all_raw_hashes_verified": True, "all_source_snapshots_verified": True,
               "analysis_source_sha256": sha(__file__), "forensic_replay_checks": {str(r["cutoff"]): r["forensic"] for r in result["long"]["rows"]},
               "primary_metric": "Reference-probability-weighted squared conditional-state fidelity; worst case over held-out states/axes/resources",
               "primary_accuracy_threshold": .99999, "threshold_provenance": "Preregistered engineering choice; not prescribed by literature",
               "accuracy_observed_sensitivity_checks": sensitivity,
               "minimum_accuracy_minus_observed_sensitivity": min(r["accuracy_minus_observed_sensitivity"] for r in sensitivity),
               "sensitivity_note": "Observed ancilla/grid refinements summarized via a pure-state distance triangle; not a rigorous infinite-cutoff error bound",
               "saved_scalar_metrics_recomputed": True}
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    for name, rows in (("validation.csv", result["validation"]["rows"]),):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (out/name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys);writer.writeheader();writer.writerows(rows)
    memory_rows = []
    for row in result["long"]["rows"]:
        for t in range(result["config"]["rounds"]+1):
            memory_rows.append({"cutoff": row["cutoff"], "cycle": t,
                                "smooth_fidelity": row["smooth_fidelity"]["mean"][t],
                                "smooth_ci95_low": row["smooth_fidelity"]["ci95_low"][t],
                                "smooth_ci95_high": row["smooth_fidelity"]["ci95_high"][t],
                                "binned_fidelity": row["binned_fidelity"]["mean"][t],
                                "smooth_minus_binned": row["smooth_minus_binned"]["mean"][t],
                                "difference_ci95_low": row["smooth_minus_binned"]["ci95_low"][t],
                                "difference_ci95_high": row["smooth_minus_binned"]["ci95_high"][t],
                                "mean_energy": row["mean_energy"][t], "max_energy": row["max_energy"][t],
                                "mean_edge": row["mean_edge"][t], "max_edge": row["max_edge"][t],
                                "cycle_population_gate_pass": row["population_pass"][t]})
    with (out/"memory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(memory_rows[0]));writer.writeheader();writer.writerows(memory_rows)
    (out/"result.json").write_bytes((run/"result.json").read_bytes())
    (out/"protocol.md").write_bytes((run/"source_snapshot/docs/smooth_ancilla_experiment.md").read_bytes())
    write_report(out, result, summary)
    print(json.dumps({"report": str(out/"index.html"), "decision": result["decision"], "energy_jumps": jumps}, indent=2))


if __name__ == "__main__":main()
