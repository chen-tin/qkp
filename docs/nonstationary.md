# Nonstationary noise: models, decoders and experiments E9–E11

Status (14 Nov 2026): **built and tested, not run.** The test suite covers every piece listed
here, including one end-to-end pipeline run on a toy budget (train → evaluate → analyse). No
quick-mode or full experiment runs have been made, and no LSTM has been trained on drift.

This implements the three conditions in the updated proposal ("Scope and Evaluation"):
gradual drift, unexpected changes and unseen conditions. The model shapes follow prior work.
The sources are listed in the review page and below.

## Noise models: `simulator/nonstationary.py`

An AR(1) whose parameters can change every round and differ between trials. A `Schedule`
holds `sigma_q`, `rho`, `sigma_m` and `kick`, each with shape (trials, rounds), plus the
event round `tau` for each trial:

    eps_t = rho_t (sigma_t / sigma_{t-1}) eps_{t-1} + sigma_t sqrt(1 - rho_t^2) z_t

This keeps `Var eps_t = sigma_t^2` and `corr(eps_{t-1}, eps_t) = rho_t` exactly. A burst
adds a one-round displacement `kick_t z'_t` that is not part of the AR state.

| Family | What changes | Prior work it follows |
|---|---|---|
| `stationary` | nothing (the control) | — |
| `sinusoid` | sigma_q (relative amplitude) or rho (absolute), random phase per trial | Bhardwaj et al. arXiv:2511.09491; Spitz et al. arXiv:1712.02360 |
| `ramp` | linear change of the target | — |
| `one_over_f` | sigma_q from a sum of log-spaced OU processes, clipped at 3 sd | arXiv:2502.12296; Zhu et al. T1 data |
| `step` | one unannounced jump at round tau | — |
| `telegraph` | two-level switching with p_switch per round | Klimov et al. arXiv:1809.01043 |
| `burst` | one-round extra displacement at an unannounced round | Tan, Pattison, McEwen, Preskill arXiv:2406.18897 |

Any scalar can be given as `[lo, hi]` to draw it uniformly per trial. `sample_mixture` draws
a different family for each trial; the drift-LSTM training data uses it.

## Decoders: `decoders/bayesian/adaptive.py`

| Decoder | Knows | Notes |
|---|---|---|
| `Bayes (oracle)` | the true schedule of every trial | exact HMM, so optimal for parity corrections; the **reference** for E9–E11. Gets the schedule through `run_blocked(..., schedule=)`. |
| `Bayes-IMM` | a parameter grid and a jump hazard | joint filter over (regime, eps, X); regimes jump with probability `hazard` per round (default 0.005) |
| `Bayes-window W` | a parameter grid | one fixed filter per grid point, weighted by its syndrome likelihood over the last W rounds (W = 50 and 200). This is the GKP analogue of Bhardwaj et al.'s sliding window. `forgetting=λ` gives exponential forgetting instead. |

Existing decoders also run on every point: NL, `Bayes (nominal)` (fixed, possibly wrong
model), `Bayes-broad` (full-history model average), and LSTM-nominal and LSTM-broad (trained
on 50 rounds). The new **LSTM-drift** (`lstm_drift_s*`) is trained on `DRIFT_TRAIN_MIX` in
`training/datasets.py`: stationary, sinusoidal σ_q and ρ, and steps in σ_q and ρ, with
per-sequence random parameters. It is trained on `budget.ns_rounds`-round sequences.

## Experiments: `experiments/run_nonstationary.py`

Base point σ_q = 0.20L, σ_m = 0.05L, ρ = 0.9. Each point records how the drift LSTM relates to
it: `in` (inside its training mixture), `shape` (held-out shape) or `values` (seen shape,
values outside the training ranges).

- **E9 gradual drift:** control, sinusoidal σ_q (periods 1000 and 200), sinusoidal ρ, ramp, 1/f.
- **E10 unexpected changes:** step σ_q (0.20 → 0.26L), step ρ (0.9 → 0.3), telegraph σ_q,
  telegraph ρ, one-round burst (0.4L).
- **E11 unseen conditions:** amplitude 0.6, period 50, step to 0.34L, step to ρ = 0, and
  stationary σ_q = 0.32L.

Each point uses a fixed test-set size (`ns_max_test_trials`) of `ns_rounds`-round sequences.
The main metric is logical flips per round, paired against the oracle. P_L at the last round
saturates on long runs and is reported only for completeness. `analyze_nonstationary.py`
writes `E9/E10/E11_summary.csv`, a dot plot per experiment, flip rate over time per point,
and flip rate aligned on the event round for steps and bursts.

## Running (not done yet)

```bash
QKP_MODE=quick .venv/bin/python -m pytest -q                 # 30 tests
QKP_MODE=quick .venv/bin/python -m training.train_lstm        # smoke sizes first
QKP_MODE=quick .venv/bin/python -m experiments.run_nonstationary
QKP_MODE=quick .venv/bin/python -m experiments.analyze_nonstationary
# full run: drop QKP_MODE=quick; E9-E11 are also available as run_all --only E9 E10 E11
```

Cost has not been measured. The sizes are in `Budget.ns_*` in `experiments/common.py`: full
mode uses 1000 rounds and 5000 test trials per point; quick mode uses 120 rounds and 600
trials. In a short sanity check the oracle filter took about 30 µs per trial-round on the
laptop GPU, which suggests about 2.5 min per point for the oracle alone in full mode. The
IMM and window decoders carry a 12-model grid and will cost more.

## Compatibility with E1–E8

Old cache keys and checkpoint configs are unchanged:
- no file in `run_all.CODE_HASH` was edited;
- the new configs carry their own `NS_CODE_HASH` inside the noise spec;
- long-sequence keys are added only for the `drift` dataset;
- E7 keys on the original model list.

A full-mode dry run of E0–E8 and of training, with every compute call patched to fail,
completed without recomputing anything.
