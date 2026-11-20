# Resgru - Syndrome decoder for heavy-hexagon quantum error correction code

[2026 publication](https://doi.org/10.1088/1402-4896/ae87e2) · transferable; heavy-hexagon, not GKP

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| d5_reduction | 0.28 | fraction | Abstract |
| d7_reduction | 0.14 | fraction | Abstract |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
Generate seeded heavy-hexagon Stim detector events with the author circuit. Train the released residual GRU on short sequences. Evaluate all decoders on shared held-out shots across durations through 300 rounds; fit the same parity-decay model and compute 1 - epsilon_GRU/epsilon_MWPM. Short-sequence validation accuracy alone is not the published metric.

Baseline: MWPM using exactly the same detector error model and held-out shots
## Commands
Run from this folder (quote its path because it contains spaces):
```sh
python3 reproduce.py check
python3 reproduce.py plan
python3 reproduce.py fetch --kind paper
python3 reproduce.py fetch --kind code
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 reproduce.py run --mode smoke --python .venv/bin/python
```
Production commands above are explicit opt-ins. `train` runs the released training example; unresolved gaps below prevent calling it a full paper reproduction. `smoke` is infrastructure validation only. `replay` recomputes author data, without retraining.

## Known gaps
- Original experimental datasets are available on request.
- Released example configuration must be reconciled against final paper training settings before a metric reproduction claim.
- Full multi-duration evaluation and matched MWPM comparison adapter not yet implemented.

## Author source
[Ashu-20/Resgru](https://github.com/Ashu-20/Resgru) pinned at `5765954cf34f0b776d44ef94fa65a8b77140ab6b`. This is a reviewed current snapshot, not a claim that it is the paper-era commit. `fetch` leaves existing checkouts intact.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.

Additional audit: the author train script assumes distributed CUDA and fixed 500k validation. Its generator filenames also include a basis tag that the loader does not accept. Production requires an explicit adapter; smoke uses the circuit and GRU directly.
