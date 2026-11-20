# Quaternary Neural Belief Propagation Decoding of Quantum LDPC Codes With Overcomplete Check Matrices

[2025 publication](https://doi.org/10.1109/ACCESS.2025.3539475) · transferable; toric and qLDPC codes

published_metrics_verified; no new paper-metric reproduction claimed

## Published targets
| Target | Value | Units | Source |
|---|---:|---|---|
| toric_d8_nobp18 | 0.001 | fraction | Figure 9, d=8 green curve; journal PDF page 10 |

Exact conditions and uncertainties are in `targets.json`. Visual-reading ranges are explicitly marked; they are not statistical confidence intervals.
## Reproduction procedure
Build the author C++ decoder and load released toric weights. The adapter copies sources into the run folder and changes only code n/k/m, decoder iterations, decoder initialization ep0, and trained/control choice; all changes are recorded. Score frame errors, counting nontrivial logical errors even when the syndrome matches. Foundational reference: Liu and Poulin, PRL 122, 200501 (2019), https://doi.org/10.1103/PhysRevLett.122.200501; this newer paper adds public code and weights.

Baseline: Untrained BP4 and overcomplete BP4; MWPM reference from paper
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
python3 reproduce.py run --mode evaluate --python .venv/bin/python
```
Production commands above are explicit opt-ins. `train` runs the released training example; unresolved gaps below prevent calling it a full paper reproduction. `smoke` is infrastructure validation only. `replay` recomputes author data, without retraining.

## Known gaps
- The released C++ executable defaults to GB [[46,2]] with six iterations; the adapter makes an explicit copy with toric/18-iteration parameters.
- Training weights are public; a fresh-training replication needs the notebook settings and independent seeds.
- The adapter compiles a serial C++17 copy; OpenMP includes/pragmas are removed and every edit is saved in source.patch. No libomp is required.

## Author source
[kit-cel/Quantum-Neural-BP4-demo](https://github.com/kit-cel/Quantum-Neural-BP4-demo) pinned at `801ccc34a92f013422a90a421e6ad4e2149db86e`. This is a reviewed current snapshot, not a claim that it is the paper-era commit. `fetch` leaves existing checkouts intact.

## Recording results
Use the evidence format in [RESULTS.md](../RESULTS.md), then `python3 reproduce.py score --input /absolute/path/to/evidence.json`. The checker reports numerical differences; it cannot certify that an independently supplied implementation matches the paper.

No full benchmark has been executed by creating this folder.

Decoder initialization is ep0=0.37, from the released toric training script for m=3n. Physical test error remains p=0.045. The generic C++ executable’s GB default ep0=0.1 is not used. The source patch records this change.

Stopping limits: the serial author loop uses <=, so the adapter passes -e299 and -w44999999 for effective caps of 300 frame errors or 45,000,000 decoded words. Smoke uses -w127 for 128 frames.
