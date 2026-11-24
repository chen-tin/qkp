"""Build a self-contained review of fetched production summaries; no simulation."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DECODERS = ("nearest", "bayes_gaussian", "lstm_256x3_s0")
NAMES = ("Nearest lattice", "Bayesian", "LSTM 256×3, seed 0")
COLORS = ("#64748b", "#047c91", "#9356c7")
LABELS = {"baseline": "Nominal", "cross_correlated": "Cross-quadrature correlation",
          "negative_memory": "Negative memory (ρ = −0.5)", "student_innovations": "Student-t innovations",
          "gradual_drift": "Gradual drift", "unexpected_step": "Unexpected correlation step",
          "burst": "Displacement burst", "loss": "Loss", "dephasing": "Dephasing",
          "loss_dephasing": "Loss + dephasing", "inefficient_detector": "95% detector efficiency",
          "electronic_readout": "Electronic readout noise", "coupling_error": "1% coupling error",
          "control_rotation": "Control rotation", "finite_duration_lumped": "Finite-duration approximation",
          "passive_cooling": "Passive cooling"}


def build(metrics, output):
    metrics, output = Path(metrics), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    hashes = {}
    for source in sorted(metrics.rglob("*.json")):
        if source.parent.name in ("live", "logs"):
            continue
        relative = source.relative_to(metrics)
        if len(relative.parts) > 2:
            continue
        target = inputs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    read = lambda path: json.loads((inputs / path).read_text())
    status = read("status.json")
    suites = {suite: read(suite + "/summary.json") for suite in ("numerical", "physics", "noise")}
    plans = {suite: read(suite + "/plan.json") for suite in suites}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    figures = []

    def group(suite, variant, decoder):
        return suites[suite]["groups"].get(f"nominal/{variant}/{decoder}/N5120")

    def savefig(fig, name, caption):
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(output / f"{name}.{ext}", dpi=165, bbox_inches="tight")
        plt.close(fig)
        figures.append((name, caption))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, variant in zip(axes, ("baseline", "negative_memory", "burst")):
        for decoder, name, color in zip(DECODERS, NAMES, COLORS):
            g = group("noise", variant, decoder)
            y = np.array(g["mean_logical_fidelity"])
            t = np.arange(len(y))
            ax.plot(t, y, "o-", color=color, label=name, ms=3)
            ax.fill_between(t, *g["pointwise_fidelity_ci95"], color=color, alpha=.13)
        ax.set(title=LABELS[variant], xlabel="Correction cycle", ylim=(.5, 1.01), xticks=range(9))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.16)
    axes[0].set_ylabel("Six-cardinal mean logical fidelity")
    axes[2].legend(loc="lower left", fontsize=8)
    savefig(fig, "01_fidelity_cycles", "Eight correction cycles plus initialization; shaded bands are pointwise 95% Monte Carlo intervals. N = 5,120; 3,072 trajectories per decoder/configuration. All three displayed noise cases pass the registered numerical sample checks.")

    def heatmap(suite, name):
        variants = list(dict.fromkeys(job["variant"] for job in plans[suite]["jobs"]))
        values = np.full((len(variants), 3), np.nan)
        labels = np.full(values.shape, "pending", dtype=object)
        for i, variant in enumerate(variants):
            for j, decoder in enumerate(DECODERS):
                g = group(suite, variant, decoder)
                if not g or g["mean_logical_fidelity"] is None:
                    continue
                values[i, j] = g["mean_logical_fidelity"][-1]
                eq = suites[suite]["cutoff_equivalence"].get(f"nominal/{variant}/{decoder}", {})
                marker = "†" if not eq.get("pass", False) else ""
                labels[i, j] = f"{100*values[i,j]:.2f}%{marker}"
        fig, ax = plt.subplots(figsize=(9.5, .47 * len(variants) + 2))
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#e5e7eb")
        im = ax.imshow(values, vmin=.5, vmax=1, cmap=cmap, aspect="auto")
        for (i, j), text in np.ndenumerate(labels):
            ax.text(j, i, text, ha="center", va="center", color="white" if values[i,j] > .86 else "#172335", fontsize=10)
        ax.set(xticks=range(3), xticklabels=NAMES, yticks=range(len(variants)),
               yticklabels=[LABELS.get(v, v) for v in variants], title=f"{suite.title()}: fidelity after cycle 8, N = 5,120")
        bar = fig.colorbar(im, ax=ax, pad=.025)
        bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
        bar.set_label("Higher is better")
        savefig(fig, name, "Each cell is a separate configuration, not an interpolation. Gray cells have no completed high-cutoff observation. † marks cutoff equivalence not established; these cells remain provisional. Values use the original saved-operator scoring; refinement tests its stability.")
    heatmap("noise", "02_noise_heatmap")
    heatmap("physics", "03_physics_heatmap")

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.axhspan(-1, 1, color="#d8eee5", alpha=.7, label="Registered equivalence margin ±1 percentage point")
    ax.axhline(0, color="#64748b", lw=.8)
    for j, (decoder, name, color) in enumerate(zip(DECODERS, NAMES, COLORS)):
        eq = suites["numerical"]["cutoff_equivalence"][f"nominal/baseline/{decoder}"]
        d = eq["difference"]
        mean, low, high = [100*np.array(d[k]) for k in ("mean", "ci95_low", "ci95_high")]
        ax.errorbar(np.arange(len(mean)) + (j-1)*.12, mean, yerr=[mean-low, high-mean],
                    fmt="o-", color=color, capsize=2, ms=3, label=name)
    ax.set(title="Numerical baseline: paired cutoff agreement", xlabel="Cycle",
           ylabel="Fidelity at N5,120 minus N4,096 (percentage points)", xticks=range(9))
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, loc="upper left")
    savefig(fig, "04_cutoff_agreement", "All baseline paired 95% intervals are wholly inside ±1 percentage point, at every cycle. This establishes the registered sample-equivalence criterion; it is not an all-state truncation bound.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1))
    for decoder, name, color in zip(DECODERS, NAMES, COLORS):
        g = group("numerical", "baseline", decoder)
        for ax, field in zip(axes, ("mean_energy", "maximum_energy")):
            ax.plot(g[field], "o-", label=name, color=color, ms=3)
    axes[1].axhline(1000, color="#bd641b", ls="--", label="Predefined energy-event threshold")
    for ax, title in zip(axes, ("Mean photon number", "Maximum observed photon number")):
        ax.set(title=title, xlabel="Cycle", ylabel="Photons", xticks=range(9))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    savefig(fig, "05_energy", "Energy is a state-growth diagnostic, not a logical-error probability. The maximum is sample-dependent. Crossing the 1,000-photon event threshold is recorded separately from Fock-boundary population and norm failures.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, event, title in zip(axes, ("energy_limit", "energy_jump", "boundary_failure"),
                                ("Energy > 1,000", "Energy jump > 1,000", "Boundary population ≥1%")):
        for j, (decoder, name, color) in enumerate(zip(DECODERS, NAMES, COLORS)):
            e = group("numerical", "baseline", decoder)["events"][event]
            ax.plot(j, 100*e["rate"], "o", color=color)
            ax.vlines(j, 100*e["interval"][0], 100*e["interval"][1], color=color, alpha=.7)
            ax.plot(j, 100*e["one_sided_upper"], "_", color=color, ms=13)
            ax.annotate(f'{e["errors"]}/{e["trials"]}', (j, 100*e["rate"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=title, xticks=range(3), xticklabels=["Nearest", "Bayes", "LSTM"],
               ylabel="Per-trajectory event probability (%)", xlim=(-.5, 2.5))
        ax.set_ylim(bottom=-.15)
        ax.grid(alpha=.15)
    savefig(fig, "06_event_bounds", "Dots are empirical event rates; bars are conservative fixed-budget, state-stratified Clopper–Pearson/Bonferroni intervals; underscores mark one-sided upper bounds. Zero observed events do not establish zero probability or a 10⁻⁴ bound. The dedicated rare-event suite is still unrun.")

    def table(suite):
        rows = []
        for key, g in suites[suite]["groups"].items():
            if not key.endswith("/N5120") or g["mean_logical_fidelity"] is None:
                continue
            _, variant, decoder, _ = key.split("/")
            ci = g["pointwise_fidelity_ci95"]
            eq = suites[suite]["cutoff_equivalence"].get(f"nominal/{variant}/{decoder}", {})
            rows.append(f'<tr><td>{html.escape(LABELS.get(variant, variant))}</td><td>{NAMES[DECODERS.index(decoder)]}</td>'
                        f'<td>{100*g["mean_logical_fidelity"][-1]:.2f}%</td><td>{100*ci[0][-1]:.2f}–{100*ci[1][-1]:.2f}%</td>'
                        f'<td>{"pass" if eq.get("pass") else "unresolved"}</td><td>{g["trials"]:,}</td></tr>')
        return '<table><tr><th>Scenario</th><th>Decoder</th><th>Cycle-8 fidelity</th><th>Pointwise 95% interval</th><th>Cutoff check</th><th>Trajectories</th></tr>' + ''.join(rows) + '</table>'

    definitions = [
        ("Logical fidelity", "Overlap of the extracted logical qubit with its intended input, averaged equally over six cardinal inputs. Higher is better. Initial fidelity can be below 100% for finite-energy states. One minus fidelity is not generally the discrete logical-flip probability."),
        ("Pointwise 95% interval", "Monte Carlo uncertainty for that decoder, configuration and cycle. It excludes many model/numerical uncertainties and is not simultaneous coverage for the entire plotted grid. A paired comparison is needed for precise decoder-ranking claims."),
        ("Cutoff equivalence", "Paired N4,096 versus N5,120 fidelity intervals must lie entirely inside ±0.01 at every cycle. A failed equivalence test can be inconclusive because its interval is too wide; it does not automatically prove large numerical bias."),
        ("Scoring refinement", "Recalculate saved representative and hot states at 192,384,768 integration bins. Each maximum fidelity change must be ≤0.001, logical trace error ≤10⁻⁷ and minimum eigenvalue ≥−10⁻⁷. This checks selected states, not every possible state."),
        ("Primitive numerical gates", "At every recorded evolution stage: mean top-10%-Fock population <10⁻⁴, individual population <0.01, and norm error <10⁻⁷. Failure limits the valid horizon and makes affected curves diagnostic."),
        ("Photon number", "Oscillator energy in photon units. Growth can indicate demanding state support or instability. It is separate from logical fidelity, boundary population and hardware accuracy."),
        ("Event probability and upper bound", "Fraction of trajectories with a registered event at any recorded stage. Exact-binomial bounds are combined across cardinal states with multiplicity adjustment. Rare tails require the separate fixed-budget rare-event study.")]
    glossary = '<table><tr><th>Metric</th><th>Meaning and limit</th></tr>' + ''.join(
        f'<tr><td>{html.escape(name)}</td><td>{html.escape(text)}</td></tr>' for name, text in definitions) + '</table>'
    progress = ''.join(f'<tr><td>{html.escape(s)}</td><td>{v.get("completed_jobs",0)}/{v["jobs"]}</td><td>{html.escape(v["status"])}</td></tr>'
                       for s, v in status["stages"].items())
    params = {}
    for suite, plan in plans.items():
        params[suite] = {}
        for job in plan["jobs"]:
            params[suite].setdefault(job["variant"], {k: job["config"][k] for k in
                ("points", "noise_extension", "physics", "per_state", "rounds", "data_delta")})
    def percent(suite, variant, decoder):
        g = group(suite, variant, decoder)
        return f'{100*g["mean_logical_fidelity"][-1]:.2f}%' if g and g["mean_logical_fidelity"] else "pending"
    nominal = [percent("noise", "baseline", decoder) for decoder in DECODERS]
    negative = [percent("noise", "negative_memory", decoder) for decoder in DECODERS]
    burst = [percent("noise", "burst", decoder) for decoder in DECODERS]
    detector = [percent("physics", "inefficient_detector", decoder) for decoder in DECODERS]
    energy_group = group("numerical", "baseline", "bayes_gaussian")
    energy_note = (f'Mean energy grows from about {energy_group["mean_energy"][0]:.0f} to '
                   f'{energy_group["mean_energy"][-1]:.0f} photons; the largest sampled cycle-8 value is '
                   f'about {energy_group["maximum_energy"][-1]:.0f}. Passing eight-cycle numerical checks '
                   'does not establish stable energy or reliability over longer horizons.')
    content = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GKP production review</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#203047;background:#f7f8fc;max-width:1250px;margin:32px auto;padding:0 22px}}h1,h2{{line-height:1.2}}section,.callout{{background:white;border:1px solid #dce2ed;border-radius:12px;padding:24px;margin:24px 0}}.callout{{border-left:5px solid #047c91}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #e0e5ed;vertical-align:top}}th{{background:#eef2f8}}a{{color:#006e88}}small,figcaption{{color:#536277}}pre{{white-space:pre-wrap;font-size:12px}}.scroll{{overflow-x:auto}}</style></head><body>
<h1>GKP production results so far</h1><p>Saved EC2 observations through <strong>{html.escape(status['updated'])}</strong>. Eight cycles, six cardinal inputs,512 trajectories per input (3,072 per decoder/configuration); plots use N5,120.</p>
<div class="callout"><strong>What the data currently show</strong><ul><li>At nominal noise, cycle-8 fidelity is {nominal[1]} Bayesian,{nominal[2]} LSTM and{nominal[0]} nearest-lattice. Bayesian/LSTM intervals overlap; a paired ranking analysis has not been run.</li><li>When true temporal correlation becomes −0.5, the nearest-lattice point estimate is{negative[0]}, Bayesian{negative[1]}, and LSTM{negative[2]}. The controller inputs remain nominal; this is a mismatch stress test with fixed decoders.</li><li>Burst point estimates are {', '.join(burst)} for nearest/Bayesian/LSTM. The Bayesian and LSTM estimates are close.</li><li>The 95%-efficiency detector intervention gives {', '.join(detector)} for nearest/Bayesian/LSTM. Its physical parameters are project-selected and uncalibrated.</li><li>{energy_note}</li></ul></div>
<section><h2>Coverage and validity</h2><div class="scroll"><table><tr><th>Suite</th><th>Saved / planned jobs</th><th>Worker stage</th></tr>{progress}</table></div><p>The numerical baseline and complete noise sweep pass their registered numerical sample checks. Physics remains partial; three nearest-lattice physics comparisons have unresolved cutoff equivalence. Small-basis finite-gate controls failed population gates and are excluded from valid performance comparisons. Independent instrument coverage for these production states is incomplete. Hardware accuracy and general robustness remain unestablished.</p></section>'''
    for i, (name, caption) in enumerate(figures, 1):
        content += f'<section><h2>{i}. {html.escape(name[3:].replace("_", " ").capitalize())}</h2><a href="{name}.png"><img src="{name}.png" alt="{html.escape(name)}"></a><figcaption>{html.escape(caption)}</figcaption><p><a href="{name}.svg">Download SVG</a> · <a href="{name}.png">PNG</a></p></section>'
    content += f'<section><h2>What each metric measures</h2>{glossary}</section>'
    for suite in ("noise", "physics"):
        content += f'<section><h2>{suite.title()} values and uncertainty</h2><div class="scroll">{table(suite)}</div></section>'
    content += '<section><h2>Exact settings and provenance</h2><p>Displacement/readout resource scales are given relative to L=√π; baseline scales are0.20L and0.05L withρ=0.90. The frozen learned controller uses one training seed. Physical extraction and logical scoring are described in the repository methods; these are simulations.</p><details><summary>Scenario settings</summary><pre>' + html.escape(json.dumps(params, indent=2)) + '</pre></details><p><a href="review_manifest.json">Review manifest</a> · <a href="inputs/numerical/summary.json">Numerical input</a> · <a href="inputs/noise/summary.json">Noise input</a> · <a href="inputs/physics/summary.json">Physics input</a></p><p>Methods: modular subsystem extraction (Pantaleoni,Baragiola,Menicucci, <a href="https://doi.org/10.1103/PhysRevLett.125.040501">PRL125,040501</a>); GKP review (Grimsmo,Puri, <a href="https://doi.org/10.1103/PRXQuantum.2.020101">PRX Quantum2,020101</a>). Project-specific thresholds and interval contracts are retained in the copied plans. This report generates no trajectories and does not alter source results.</p></section></body></html>'
    (output / "index.html").write_text(content)
    manifest = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_status_updated": status["updated"], "classification": "partial_production_review",
                "input_sha256": hashes, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "note": "Fetched summary snapshots may reflect adjacent exporter checkpoints. No new trajectories or significance tests."}
    (output / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.metrics, args.output))
