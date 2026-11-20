# Research-backed decoder and encoding methods

Eleven paper-specific reproduction workspaces. All have published quantitative targets, explicit code/noise/metric conventions, a local protocol, source retrieval, dependency instructions, and evidence checking. **These are reproduction preparations, not eleven completed reproductions.**

The first goal is to reproduce each paper in its original setting. Transferring it to the existing correlated-displacement GKP simulator is a separate experiment. None of these papers establishes an advantage on our simulator merely because it uses the same architecture.

| ID | Method / paper | Published anchor | Current executable coverage |
|---|---|---|---|
| `resgru` | [Resgru (2026)](2026%20-%20Resgru%20-%20Syndrome%20decoder%20for%20heavy-hexagon%20quantum%20error%20correction%20code/README.md) | 28% / 14% lower per-round LER than MWPM at d=5 / 7, p=0.001 | Author circuit + GRU gradient/inference smoke; production launcher gaps documented |
| `gkp_cnn` | [Multidimensional Bose QEC (2022)](2022%20-%20Multidimensional%20Bose%20quantum%20error%20correction%20based%20on%20neural%20network%20decoder/README.md) | Paper-sigma thresholds ≈0.78 / 0.34 for perfect / noisy measurements | Source/supplement, protocol and scorer; author CNN code unavailable publicly |
| `tcn` | [Rethink the Role of Neural Decoders (2026)](2026%20-%20Rethink%20the%20Role%20of%20Neural%20Decoders%20in%20Quantum%20Error%20Correction/README.md) | Table 1 calibrated + fine-tuned TCN-small: 6.70% / 11.47% shot failures, d=3 / 5 | Author TCN + Stim gradient/inference smoke; hardware pipeline incomplete upstream |
| `alphaqubit` | [Learning high-accuracy error decoding (2024)](2024%20-%20Learning%20high-accuracy%20error%20decoding%20for%20quantum%20processors/README.md) | 2.901% / 2.748% LER per round, d=3 / 5, 20-model ensemble | Supplementary pseudocode, hardware-data link, protocol and scorer; implementation required |
| `mamba` | [Scalable neural decoders (accepted 2026)](2026%20-%20Scalable%20neural%20decoders%20for%20practical%20real-time%20quantum%20error%20correction/README.md) | Hardware LER ≈2.981% / 2.916%, d=3 / 5 | Reanalysis of released simulated actual/predicted bits; model training code withheld |
| `gnn` | [Data-driven decoding using GNNs (2025)](2025%20-%20Data-driven%20decoding%20of%20quantum%20error%20correcting%20codes%20using%20graph%20neural%20networks/README.md) | Figure 3 d=3,r=3 ≈6e-4 shot failures at p=0.001; marked visual reading range | Pinned author source/checkpoints, protocol and scorer; graph/evaluation integration pending |
| `nbp4` | [Quaternary Neural BP (2025)](2025%20-%20Quaternary%20Neural%20Belief%20Propagation%20Decoding%20of%20Quantum%20LDPC%20Codes%20With%20Overcomplete%20Check%20Matrices/README.md) | Figure 9 toric d=8, p=.045, NOBP4/18 iterations: FER ≈1e-3; marked visual reading range | Compiled author decoder + trained weights, bounded smoke and full anchor evaluator |
| `feedback` | [Non-Markovian Feedback (2025)](2025%20-%20Non-Markovian%20Feedback%20for%20Optimized%20Quantum%20Error%20Correction/README.md) | About 2× logical lifetime versus standard sBs | Released feedback-GRAPE training-example launcher; full lifetime protocol still required |
| `learned_gkp` | [Neural-Network-Based GKP Design (2025)](2025%20-%20Neural-Network-Based%20Design%20of%20Approximate%20Gottesman-Kitaev-Preskill%20Code/README.md) | 7 versus 21 squeezed coherent components at 9.55 dB, with better correctability objective | Protocol and scorer; optimized coefficients and full implementation needed |
| `analog_ldpc` | [Analog Information Decoding (2024)](2024%20-%20Analog%20Information%20Decoding%20of%20Bosonic%20Quantum%20Low-Density%20Parity-Check%20Codes/README.md) | 9.9% single-shot-component threshold; separate non-single-shot ≈1.66% analog / 1.26% hard | Pinned author MQT source, protocol and scorer; paper-era configuration reconciliation pending |
| `surface_gkp` | [Low-Overhead Surface-GKP (2022)](2022%20-%20Low-Overhead%20Fault-Tolerant%20Quantum%20Error%20Correction%20with%20the%20Surface-GKP%20Code/README.md) | At 12 dB, CNOT failure .36% joint ML versus .87% rounding; threshold 9.9 dB | Algorithms/physical protocol and scorer; simulator/decoder implementation required |

The proposed **Bayesian–neural residual decoder for our GKP simulator** is excluded from the evidence-backed benchmark list: we have not identified a paper benchmarking that exact design. Neural BP is a published model-based neural alternative, but does not validate that proposed GKP hybrid. Liu and Poulin (2019) is included as the foundational reference in the newer neural-BP workspace; the 2025 paper supplies executable code and weights.

## Common workflow

From `qkp/`:

```sh
python3 'misc methods/bench.py' check
python3 'misc methods/bench.py' plan --paper nbp4
python3 'misc methods/bench.py' fetch --paper nbp4 --kind code
python3 'misc methods/bench.py' run --paper nbp4 --mode smoke
# Explicit production anchor evaluation; may take substantial time:
python3 'misc methods/bench.py' run --paper nbp4 --mode evaluate
python3 'misc methods/bench.py' score --paper nbp4 --input /absolute/path/to/evidence.json
```

Each folder also has `reproduce.py`, so the same commands work from that folder without `--paper`. `fetch --kind paper` retrieves the primary paper/supplement where available; `fetch --kind code` retrieves a pinned author repository. Pins are current snapshots, not asserted publication-era revisions. Sources already inspected locally are cached under `artifacts/` with SHA-256 hashes. Git checkouts, PDFs, environments and generated runs are ignored by Git, and remain fetchable.

For neural smoke checks, create an **isolated** Python 3.11 environment; do not change the running benchmark environment:

```sh
uv venv --python 3.11 'misc methods/.smoke-venv'
uv pip install --python 'misc methods/.smoke-venv/bin/python' -r 'misc methods/smoke-requirements.txt'
python3 'misc methods/bench.py' fetch --paper resgru --kind code
python3 'misc methods/bench.py' run --paper resgru --mode smoke --python "$PWD/misc methods/.smoke-venv/bin/python"
python3 'misc methods/bench.py' fetch --paper tcn --kind code
python3 'misc methods/bench.py' run --paper tcn --mode smoke --python "$PWD/misc methods/.smoke-venv/bin/python"
```

The TCN repository's NumPy 2.2.6 and Brevitas 0.12.1 pins conflict. Our smoke environment explicitly uses NumPy 1.26.4. This compatibility adjustment is not evidence of identical production numerics.

Mamba's sparse checkout selects the published simulation anchor d=3, 28 rounds, p=.002 for both Mamba and Transformer. `run --mode replay` recomputes four sets of outcome counts and intervals. These are **author-data reanalysis**, not newly trained models or the hardware per-round targets.

## What each folder contains

- `paper.json`: citation, evidence scope, pinned repository, source hashes and known blockers.
- `targets.json`: quantitative published targets with exact conditions, metric units and figure/table locators.
- `protocol.json`: planned control/intervention, settings, uncertainty and decision rule.
- `requirements.txt`, `environment.md`: isolated setup and environment limitations.
- `reproduce.py`, `analyze.py`: fetch/check/run where supported, raw-count analysis, and comparison.
- `adapter.py`, where author code/data can currently be invoked meaningfully.

## Evidence and verification

See [RESULTS.md](RESULTS.md) for raw-data contracts and [VALIDATION.md](VALIDATION.md) for checks actually executed. Smoke observations cannot be submitted to the paper-metric scorer. The scorer checks units, conditions and raw-artifact hashes, but it cannot prove physical-model equivalence or certify a reproduction from a scalar match.

Run infrastructure checks with `python3 -m unittest discover -s 'misc methods/tests' -v`. Existing LSTM, Convy, Baireuther and Wan results are separate and remain unchanged by this directory.
