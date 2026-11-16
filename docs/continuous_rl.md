# E15: continuous corrections and a DQN decoder (closed loop)

Status: **built and tested, not run** (16 Nov 2026). Tests: `tests/test_closed_loop.py`,
`tests/test_continuous.py`, `tests/test_rl.py`.

## Why a closed loop

The main experiments use parity-only corrections c_t = s_t + j_t L. With those, the
syndromes do not depend on the decoder, which makes comparisons exactly paired and the
matched Bayes filter optimal. Continuous corrections and RL agents break that: a correction
changes where the oscillator sits, and so changes later syndromes. `simulator/closed_loop.py`
runs any decoder with `reset(trials)` / `decide(obs) -> real correction` round by round. For
parity decoders it reproduces `simulator/sim.py` exactly. Because the analog syndrome is
deterministic given the noise, decoders run on the same `(d, nu)` arrays remain paired.

## Bayes filter with continuous corrections (`decoders/bayesian/continuous.py`)

The grid filter's hidden state X_t, the cumulative displacement, does not depend on the
corrections. They enter only through the known cumulative correction C in
s_t = wrap(X_t − C_{t−1} + ν_t). So the existing `GridFilter` is an exact HMM for any real
C: there is no interpolation, and C is simply real-valued.

| Policy | Correction | Notes |
|---|---|---|
| `parity` | s + jL | identical to `BayesGridDecoder` (tested) |
| `fine` | s + jL + δ, with δ the posterior mean of the within-cell offset of X from the chosen centre | removes the estimated readout imprint −ν_t (Lemma 2), so the oscillator sits closer to a lattice point (tested) |

**Early finding (smoke-test sizes, not a result).** For the *Bayes* decoder, `fine` is almost
information-neutral in the shift model. Its belief over X is unchanged by corrections, so the
fine offset only changes where the oscillator sits within its cell. Logical errors were
essentially unchanged: P_L(100) 0.2787 → 0.2733 at σ_q = 0.072L, σ_m = 0.15L, ρ = 0.9, with
3 000 trials. Fine corrections should matter more for memoryless decoders, and in physics the
shift model omits: finite-energy envelopes, where displacement size matters, and noise that
depends on the state.

## RL (`rl/`)

- **`rl/env.py`**: a vectorised environment with `reset()` and `step(actions)` over a batch
  of trials.
  - **Observation:** the last `window` syndromes (sin, cos and s/L), the agent's cumulative
    parity J as ±1, and its last `window` fine offsets.
  - **Actions:** `2 × n_fine`, giving j ∈ {0, 1} times fine bins in [−fine_max, fine_max];
    `n_fine = 1` is parity-only.
  - **Rewards:** `dense` (−1 per round in a wrong logical state, the default), `terminal`
    (−1 only at the last round), or `flip` (−1 per change of logical state; penalises
    dithering, but it is not a logical error rate once P_L saturates).
  - **Noise:** any `NoiseModel` or nonstationary spec.
- **`rl/dqn.py`**: an MLP Q-network on the observation window, with a replay buffer, a target
  network, linear ε decay, Huber loss and Adam. `train()` returns the agent and its history;
  `DQNDecoder` runs a trained agent greedily in the closed loop. Tests check that it takes the
  same actions as in the environment.
- **`rl/train.py`**: one agent per (E15 point, action set), saved as
  `checkpoints/dqn_<point>_<actions>.pt` with a sidecar. Full size is 60 000 steps × 512
  trials, 200 rounds and a 30 min cap. Quick mode is 400 steps.

**Sanity check.** With parity-only actions the task is a contextual bandit, and its optimal
policy is the matched Bayes decision rule. A trained DQN should approach matched Bayes at easy
points; E15 measures that gap directly (`DQN (parity)` against `Bayes (parity)`).

## Experiment E15 (`experiments/run_continuous.py`)

- **Points:**
  - `nominal`: σ_q = 0.20L, σ_m = 0.05L, ρ = 0.9
  - `hw`: σ_q = 0.072L, σ_m = 0.105L, ρ = 0.9 (hardware scale)
  - `hw_sm0.05`, `hw_sm0.10`, `hw_sm0.15`: σ_q = 0.072L, ρ = 0.9, σ_m swept
- **Decoders:** NL, Bayes (parity), Bayes (fine), DQN (parity), DQN (fine).
- **Sizes:** 20 000 trials × 200 rounds (full); 400 × 60 (quick).
- **Caching:** results go to `results/cache/E15/` with sidecars. DQN decoders without a
  checkpoint are deferred.
- **Scoring:** the decay-fitted ε and paired P_L at the common horizon against Bayes (parity),
  written to `results/E15_summary.json`.

```bash
python -m rl.train                      # train DQN agents (2 per point)
python -m experiments.run_continuous    # evaluate + summary
QKP_MODE=quick ...                      # smoke sizes
```

## Open choices

1. Q-network architecture: the MLP on a window of 8 is Markov for parity actions only; a GRU
   on the full history is the principled choice for fine actions.
2. Reward: dense per round, terminal only, or flip. Dense matches P_L(t) best.
3. Discount γ = 0.9, which plans about 10 rounds ahead. Longer horizons may matter for fine
   actions.
4. Fine-action grid: 9 bins up to ±0.25L. Finer bins or a continuous-action method (DDPG/SAC)
   are alternatives.
5. Whether E15 should also run the nonstationary schedules. The environment supports `spec=`.
6. The DQN training budget, which is untested at full size.
