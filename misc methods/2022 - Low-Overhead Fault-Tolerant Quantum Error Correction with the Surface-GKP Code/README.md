# Low-Overhead Fault-Tolerant Quantum Error Correction with the Surface-GKP Code

[2022 publication](https://doi.org/10.1103/PRXQuantum.3.010315) · direct GKP concatenated fault tolerance

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| cnot_ML_failure | 0.0036 | fraction | Abstract and gate-decoding results |
| cnot_rounding_failure | 0.0087 | fraction | Abstract and gate-decoding results |
| threshold_squeezing | 9.9 | dB | Abstract and threshold figure |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
First reproduce the joint two-mode CNOT displacement decoder against closest-integer rounding at 12 dB. Then implement the Appendix C memory circuit with dynamically weighted, correlated matching graphs and perform the squeezing sweep. Preserve the paper’s variance-to-dB convention and count both relevant gate failure components.

Baseline: Closest-integer gate decoding and conventional rotated surface-code resource comparison
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
- No verified public author simulation repository located.
- Full correlated gate model and dynamically weighted matching implementation remain to be implemented from the paper appendices.
- A 1e-7 failure claim requires adequate rare-event statistics, not a smoke run.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.
