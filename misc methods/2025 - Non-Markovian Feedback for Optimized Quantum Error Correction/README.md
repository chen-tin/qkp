# Non-Markovian Feedback for Optimized Quantum Error Correction

[2025 publication](https://doi.org/10.1103/PhysRevLett.134.020601) · direct GKP control; not a post-processing decoder

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| lifetime_gain | 2.0 | ratio | Main text and Figure 3 |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
Run the released feedback-GRAPE example to establish the environment, then reproduce the paper’s matched protocols and six logical-state channel-fidelity trajectories. Fit lifetimes with intervals across trajectories and training seeds. This controller changes physical gates; compare it in a separate experiment from classical post-processing decoders.

Baseline: Standard sBs, Markovian feedback, autonomous QEC under matched physical durations and noise
## Commands
Run from this folder (quote its path because it contains spaces):
```sh
python3 reproduce.py check
python3 reproduce.py plan
python3 reproduce.py fetch --kind paper
python3 reproduce.py fetch --kind code
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
python3 reproduce.py run --mode train --python .venv/bin/python
```
Production commands above are explicit opt-ins. `train` runs the released training example; unresolved gaps below prevent calling it a full paper reproduction. `smoke` is infrastructure validation only. `replay` recomputes author data, without retraining.

## Known gaps
- Default author runner is a short example, not the full Figure 3 recipe.
- All noise regimes, trained checkpoints, six-state evaluation and lifetime fitting need configuration before a Figure 3 claim.
- TensorFlow and old Keras APIs need an isolated compatible environment.

## Author source
[Matteo-Puviani/GQF](https://github.com/Matteo-Puviani/GQF) pinned at `c9ab1ef2b3ff6fa6d6d24cd95fbd06e2872e016d`. This is a reviewed current snapshot, not a claim that it is the paper-era commit. `fetch` leaves existing checkouts intact.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.
