# Analog Information Decoding of Bosonic Quantum Low-Density Parity-Check Codes

[2024 publication](https://doi.org/10.1103/PRXQuantum.5.020349) · bosonic-code outer decoder; reported model is not our single-mode GKP simulator

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| single_shot_threshold | 0.099 | physical_error_probability | Figure 7b |
| non_single_shot_analog_threshold | 0.0166 | physical_error_probability | Appendix F, Figure 15 |
| non_single_shot_hard_threshold | 0.0126 | physical_error_probability | Appendix F, Figure 15 |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
Use the author ATD and quasi-single-shot simulators with the supplied parity-check matrices. Preserve X/Z component distinctions and repeated-round conventions. Fit threshold crossings for multiple sizes and bootstrap independent simulation batches. Never compare the single-shot 9.9% threshold with the non-single-shot 1.66% as though they were the same task.

Baseline: Hard-syndrome decoder under the same phenomenological model; historical 7.1% single-shot threshold is not a newly paired run
## Commands
Run from this folder (quote its path because it contains spaces):
```sh
python3 reproduce.py check
python3 reproduce.py plan
python3 reproduce.py fetch --kind paper
python3 reproduce.py fetch --kind code
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
Production commands above are explicit opt-ins. `train` runs the released training example; unresolved gaps below prevent calling it a full paper reproduction. `smoke` is infrastructure validation only. `replay` recomputes author data, without retraining.

## Known gaps
- Paper-era data/configuration locations differ from current MQT layout.
- Simulation driver must be reconciled with pinned ATD/QSS APIs before full threshold sweeps.
- Analog measurement channel must match the paper; no direct GKP-noise transfer claim.

## Author source
[cda-tum/mqt-qecc](https://github.com/cda-tum/mqt-qecc) pinned at `8d8544ddf182997126cd007b23a8f63ab728ad2b`. This is a reviewed current snapshot, not a claim that it is the paper-era commit. `fetch` leaves existing checkouts intact.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.
