# GKP decoding and simulator validation

Research code for Gottesman–Kitaev–Preskill (GKP) error correction: classical displacement-noise models, nearest-lattice and Bayesian decoders, learned sequence decoders, and finite-energy oscillator references checked against QuTiP.

Start with [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for the current evidence, corrections, methods, and unresolved questions. Historical baseline results are in [RESULTS.md](RESULTS.md). The project directory is named `qkp`; the quantum code is GKP.

## Current evidence

| Study | Status |
| --- | --- |
| Decoder heatmaps | 20 decoders across 61 stationary noise settings; development evaluation |
| LSTM size sweep | 36 width × depth × seed models, 2,196 evaluations; [coverage audit](docs/lstm_size_sweep.md) |
| Published-result reproductions | Partial and mismatched targets remain explicit; [reproduction status](docs/published_reproductions.md) |
| Other architectures | Eleven paper-specific workspaces with reproduction infrastructure; [catalog and readiness](misc%20methods/README.md) |
| Local instrument versus explicit QuTiP | Completed within the stated finite-operator, single-extraction scope; [method and results](docs/qutip_engine_benchmark_results.md) |
| Simulation-quality production suite | Running on EC2; numerical baseline passed, remaining suites incomplete; [dated metrics snapshot](results/ec2_runs/production_20261119/snapshots/20261120T194213Z/README.md) |

The EC2 snapshot was captured on **20 November 2026 at 19:42 UTC**. It is a saved checkpoint, not a live GitHub status feed. The small-basis finite-gate controls fail numerical validity checks, and two partial physics comparisons do not yet establish cutoff equivalence. Passing the numerical baseline does not establish hardware accuracy or complete independent instrument validation.

## Setup and checks

Use Python 3.13 for the recorded reference environment. The exact production package versions and platform are saved in the [environment record](results/ec2_runs/production_20261119/snapshots/20261120T194213Z/environment.json); `requirements.txt` lists the top-level dependencies without pinning them.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q tests/test_simulation_quality.py tests/test_continuous_ancilla.py tests/test_qutip_reference.py tests/test_qutip_engine_benchmark.py tests/test_quantum_channels.py
```

These are bounded software/reference checks. They do not launch the production benchmark, train the full LSTM sweep, or validate hardware. Paper-specific workspaces have their own dependencies and protocols.

## Browse the HTML reports

From the repository root:

```sh
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
```

Then open:

- [Illustrated walkthrough and decoder heatmaps](http://127.0.0.1:8765/writeup/artifact/gkp-walkthrough/viewer.html).
- [Metrics dashboard](http://127.0.0.1:8765/writeup/artifact/metrics/viewer.html).
- [Captured EC2 production plots](http://127.0.0.1:8765/results/ec2_runs/production_20261119/snapshots/20261120T194213Z/live/).

The HTML and its assets are included in Git; GitHub's file viewer does not execute the interactive pages. The walkthrough contains illustrative trajectories as well as measured results; their labels distinguish them.

## Reproduce or extend

- [Simulation-quality methods and acceptance criteria](docs/simulation_quality.md).
- [Infrastructure validation and its limits](docs/simulation_quality_validation.md).
- [EC2 execution and metrics retrieval](docs/ec2_production.md).
- [LSTM sweep configuration and regeneration](docs/lstm_size_sweep.md).
- [Smooth-resource experiment and decision](docs/smooth_ancilla_results.md).
- [Research recordkeeping requirements](AGENTS.md).

Compact summaries, protocols, source snapshots, figures, and the frozen production controller are versioned. Large raw trajectory arrays, training datasets, resume checkpoints, logs, live EC2 downloads, and duplicate ZIP bundles are excluded from Git. Exact raw/config/checkpoint hashes remain in the saved manifests. Production raw data is retained separately in the private S3 run archive; it is not publicly downloadable from this repository. Reproducing analyses that consume omitted raw arrays requires regenerating them or obtaining the corresponding archive.
