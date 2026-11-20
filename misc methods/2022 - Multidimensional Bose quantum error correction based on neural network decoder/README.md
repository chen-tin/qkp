# Multidimensional Bose quantum error correction based on neural network decoder

[2022 publication](https://www.nature.com/articles/s41534-022-00650-z) · direct surface-GKP decoding

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| perfect_measurement_threshold | 0.78 | paper_sigma | Abstract; threshold figures |
| noisy_measurement_threshold | 0.34 | paper_sigma | Abstract; threshold figures |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
Rebuild the surface-GKP simulator and CNN specified in the paper and supplement, or supply the author implementation. First reproduce the MWPM thresholds, then evaluate the NN on the same noise grid. Retain multiple-distance raw counts and threshold-fit covariance. The sigma values are paper claims, not verified qkp results.

Baseline: MWPM under the identical perfect/noisy measurement regimes
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
- Author neural decoder implementation and simulation data are available on request.
- The linked ConcatenatedCodedRepeaters repository is a cited reference implementation, not this CNN.
- Training sample-count wording needs reconciliation before a full run.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.
