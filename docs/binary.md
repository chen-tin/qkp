# E14: one-bit (binary) syndromes — built, not run

Cavity GKP hardware returns ancilla **bits**, not an analog syndrome. E14 asks whether the
memory advantage survives that. Infrastructure is built and tested; nothing has been run at
full size.

## Model (closed loop, `simulator/closed_loop.py`)

Each round: x ← x + ε_t (AR(1)); m ancilla bits are measured; the decoder applies any real
correction c_t; x ← x − c_t. Logical error after round t: round(x_t/L) odd (as everywhere).

Bit j of a round (`PhasedBits`, `decoders/bayesian/binary_filter.py`):

    P(b_j = 1 | x) = (1 − v cos(2π (x + ν_j)/L − φ_j)) / 2,  then a bit flip with prob. p_flip

- φ = −π/2 (sin type): the bit says on which side of the nearest lattice point the state is
  (what sBs-style feedback uses). φ = 0 (cos type): magnitude only.
- `make_measurement(v, p_flip, m, σ_m)` alternates sin and cos phases, m bits per round,
  each with independent readout noise ν_j ~ N(0, σ_m²).
- Bits are packed into one integer per round.

**Why m > 1.** With one weak bit per round (P(b|x) ≈ 0.7 at x = 0.07 L), no decoder follows
per-round displacements of σ_q ≳ 0.07 L: at the nominal and hardware scales every decoder
saturates at P_L ≈ ½. Development runs (2 000 trials × 200 rounds, not saved):

| σ_q, σ_m (L), ρ | m | best sBs P_L(200) | binary Bayes P_L(200) |
|---|---|---|---|
| 0.072, 0.105, 0.9 | 1 | 0.50 | 0.49 |
| 0.072, 0.105, 0.9 | 4 | 0.47 | 0.24 |
| 0.072, 0.105, 0.9 | 8 | 0.47 | 0.027 |
| 0.072, 0.105, 0.0 | 8 | 0.020 | 0.008 |
| 0.20, 0.05, 0.9 | 8 | 0.51 | 0.17 |

Real devices take several ancilla interactions per correction cycle, so m is a modelling
choice that needs the user's judgement (see below).

## Decoders

- **sBs baseline** (`SBSDecoder`): c = δ × mean over this round's sin-type bits of (2b − 1).
  Memoryless, like the feedback of Campagne-Ibarcq 2020 / Sivak 2023 (without their
  finite-energy dissipation). δ ∈ {0.08, 0.15, 0.25} L in E14.
- **Binary Bayes** (`BinaryBayesDecoder`): an exact HMM on a grid over (ε, x mod 2L). The
  likelihood integrates ν analytically (v_eff = v·exp(−2π²σ_m²/L²); the grid adds h²/12). Each
  round it chooses a correction on the grid (multiples of h = 2L/96) that minimises
  P(odd cell) + 0.01·(centring), evaluated on the **predicted next-round state**
  x − c + ε_{t+1} (lookahead), then shifts its belief exactly by −c. Lookahead matters: the
  greedy variant (centre the current state) lost to sBs under correlated drift
  (P_L(200) 0.41 vs 0.27 at σ_q 0.05 L, ρ 0.9).
- **Binary LSTM** (`decoders/binary_lstm.py`, `training/train_binary.py`):
  - Inputs: this round's bits and the previous correction.
  - Outputs: distributions over 48 bins of x_pre and of x_pre + ε_{t+1}. The second does not
    depend on the correction, so the same lookahead cost picks c.
  - Training data come from a behaviour policy: sBs with δ ~ U(0.05, 0.2) L plus 0.03 L
    jitter, with per-sequence noise drawn from a box around each scale.
  - **Caveat:** at test time the LSTM's own corrections replace the behaviour policy, so
    there is distribution shift. A DAgger-style second round is the fix, and is not built.
  - Models: `binary_lstm_{nominal,hardware}_m{1,4,8}_s0`.
- **Analog reference**: matched analog-syndrome Bayes on the same displacement noise.

## Points (`experiments/run_binary.py`)

- **16 points with m = 4:** scale {nominal (0.20, 0.05), hardware (0.072, 0.105)} × ρ {0, 0.9}
  × v {1, 0.95} × p_flip {0, 0.01}.
- **8 points for the bits-per-round axis:** m ∈ {1, 8} at v = 1, p_flip = 0.

Full size is 4 000 trials × 200 rounds per point (quick mode: 300 × 40). Every decoder gets
the same noise and the same bit-draw seed. Results: `results/cache/E14/` and
`results/E14_summary.json`, with ε from the decay of 1 − 2P_L and paired P_L against
binary Bayes.

## What is and isn't realistic

- Realistic: bit-valued syndromes, readout flips, reduced visibility, feedback in the loop,
  and a memoryless sBs-like baseline.
- Not modelled:
  - finite-energy dissipation (the "small" steps of sBs);
  - ancilla errors that propagate into the oscillator;
  - the p quadrature;
  - per-bit timing (all m bits see the same x).
- Corrections are quantised to the grid spacing (0.037 L).

## Running (not done yet)

```bash
QKP_MODE=quick python -m pytest -q tests/test_binary.py tests/test_closed_loop.py
python -m training.train_binary          # 6 LSTMs
python -m experiments.run_binary         # E14; --analyze-only to re-score
```
