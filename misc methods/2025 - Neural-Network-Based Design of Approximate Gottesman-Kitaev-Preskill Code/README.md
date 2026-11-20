# Neural-Network-Based Design of Approximate Gottesman-Kitaev-Preskill Code

[2025 publication](https://doi.org/10.1103/PhysRevLett.134.060601) · direct GKP encoding optimization; not a decoder

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| state_count_ratio | 0.3333333333333333 | ratio | Abstract and Figure 2 |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
Construct normalized finite Fock-basis codewords from squeezed coherent states, optimize coefficients against the paper’s error operators, and compare KL residuals on identical loss/dephasing grids. Save coefficients and residual matrices. Report the component-count reduction only if the optimized code also beats the conventional code on the specified correctability objective.

Baseline: Best conventional finite-energy GKP encoding at the same squeezing and channel
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
- No verified public author training implementation/checkpoints located.
- Optimized coefficient tables and complete supplementary constraints must be obtained and implemented.
- Component-count ratio alone is not evidence of improved correctability.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.
