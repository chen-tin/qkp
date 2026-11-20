# Learning high-accuracy error decoding for quantum processors

[2024 publication](https://www.nature.com/articles/s41586-024-08148-8) · transferable; surface-code decoding

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| hardware_d3 | 0.02901 | fraction_per_round | Figure 3 and Results |
| hardware_d5 | 0.02748 | fraction_per_round | Figure 3 and Results |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
Acquire the Sycamore data, preserve the even/odd partition, implement each supplementary pseudocode component, pretrain on the calibrated detector-error model, fine-tune and ensemble 20 independently trained models. Score fitted logical errors per round. The existing qkp generic Transformer is not an AlphaQubit reproduction.

Baseline: Tensor-network decoder: d3 .03028 +/- .00023; d5 .02915 +/- .00016 per round
## Commands
Run from this folder (quote its path because it contains spaces):
```sh
python3 reproduce.py check
python3 reproduce.py plan
python3 reproduce.py fetch --kind paper
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
Production commands above are explicit opt-ins. `train` runs the released training example; unresolved gaps below prevent calling it a full paper reproduction. `smoke` is infrastructure validation only. `replay` recomputes author data, without retraining.

## Known gaps
- Paper releases detailed pseudocode, not an executable author training implementation or model checkpoints.
- Hardware calibration, soft-input preprocessing, ensemble and fit must be implemented faithfully.
- Full pretraining is a large accelerator workload.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.
