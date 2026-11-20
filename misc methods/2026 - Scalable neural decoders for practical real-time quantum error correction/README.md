# Scalable neural decoders for practical real-time quantum error correction

[2026 publication](https://www.nature.com/articles/s41534-026-01403-y) · transferable; surface-code decoding

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| hardware_d3 | 0.02981 | fraction_per_round | Accepted article, Sycamore results |
| hardware_d5 | 0.02916 | fraction_per_round | Accepted article, Sycamore results |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
Replay the released actual/predicted observable files for both architectures and noise-injection conditions. Export errors, shots and Wilson intervals. To regenerate observations or reproduce hardware per-round targets, obtain the author model and training implementation. Do not use the preprint threshold to label the accepted article benchmark.

Baseline: Transformer on identical simulation inputs; retain both with- and without-latency-noise conditions
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
python3 reproduce.py run --mode replay --python .venv/bin/python
```
Production commands above are explicit opt-ins. `train` runs the released training example; unresolved gaps below prevent calling it a full paper reproduction. `smoke` is infrastructure validation only. `replay` recomputes author data, without retraining.

## Known gaps
- Author training/decoder code is withheld for intellectual-property reasons; available on request.
- Public repository releases numerical real-time data, not the hardware model or training code.
- Replay results are shot failure rates and must not be compared directly to hardware per-round targets.

## Author source
[qDNA-yonsei/NeuralDecoder_v1](https://github.com/qDNA-yonsei/NeuralDecoder_v1) pinned at `f8db26927c061c25fa6828c70ea29bb477430e3b`. This is a reviewed current snapshot, not a claim that it is the paper-era commit. `fetch` leaves existing checkouts intact.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.

The default sparse checkout selects the d=3, 28-round, p=0.002 real-time anchor for Mamba and Transformer. Replay covers those four with/without-noise conditions. Expand the registered checkout_paths to acquire other points; the paper hardware LER targets cannot be scored from these files.
