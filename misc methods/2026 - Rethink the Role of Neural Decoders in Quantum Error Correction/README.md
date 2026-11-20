# Rethink the Role of Neural Decoders in Quantum Error Correction

[2026 publication](https://proceedings.mlr.press/v306/yan26g.html) · transferable; surface-code benchmark

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| d3_calibrated_finetuned | 0.067 | fraction | Table 1 |
| d5_calibrated_finetuned | 0.1147 | fraction | Table 1 |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
Use the author TCN-small architecture, and preserve the hardware split and calibration used in Table 1. Compare uniform pretraining, calibrated pretraining, and fine-tuning separately. Table 1 values are shot-level percentages, not AlphaQubit-style fitted errors per round. Record INT4 resource estimates separately from measured inference times.

Baseline: Standard and correlated MWPM, plus uniform-prior TCN on identical hardware test split
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
- Released code contains model, data, compression and training utilities, but no complete Table 1 hardware-calibration/fine-tuning entry point.
- The local smoke test validates the released TCN on generated Stim data; it cannot establish Table 1 accuracy.

## Author source
[GrahamYan/QEC-neural-decoder-benchmark](https://github.com/GrahamYan/QEC-neural-decoder-benchmark) pinned at `aeca633b4c30022193dfe1b18a41e860d3496b87`. This is a reviewed current snapshot, not a claim that it is the paper-era commit. `fetch` leaves existing checkouts intact.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.
