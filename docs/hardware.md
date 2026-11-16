# E12: decoders at hardware-scale noise (built, not run)

**Why.** The coverage map (`docs/benchmarks.md` §H) shows that no simulated scenario covers
the regime real cavities run in:
- σ_q ≈ 0.072 L (the loss of a T1 = 610 µs cavity over a 9.85 µs cycle, Sivak et al. 2023);
- readout noise σ_m ≈ 0.10 L, the value that reproduces their p_X = 1.81 × 10⁻³ (§B).

There, consecutive syndromes are anti-correlated for any ρ (lag-1 correlation −0.24 to
−0.38), so a decoder's memory would mostly average readout noise rather than track
correlated drift. E12 tests every decoder in that regime.

**Points** (`experiments/run_hardware.py`, `HW_POINTS`; 16 in total):

| Group | Points |
|---|---|
| Stationary, at σ_q = 0.072 L | σ_m ∈ {0.08, 0.105, 0.136} L × ρ ∈ {0, 0.5, 0.9, 0.97} |
| Drifting, at the anchor | sinusoidal σ_q (amplitude 0.3, period 1 000); telegraph σ_m (0.08 ↔ 0.136 L, the first σ_m-drift point); step ρ 0.9 → 0.3; 1/f σ_q (rel 0.12, the amplitude calibrated on measured T1 drift) |

**Decoders**
- the oracle (reference) and nearest lattice;
- Bayes fixed at the anchor (ρ = 0.9 assumed);
- Bayes at the nominal model (0.2 L, 0.05 L, 0.9), which is what our filters assume today;
- IMM and window 50 / 200 over a hardware-scale grid (σ_q ∈ {0.05, 0.072, 0.10} L ×
  ρ ∈ {0, 0.5, 0.9, 0.97}, σ_m = 0.105 L);
- LSTM-hw, trained on `HW_DATASETS["hw"]`: a hardware-scale mixture with σ_q 0.05–0.10 L,
  σ_m 0.08–0.14 L, ρ 0–0.97, plus sinusoidal, σ_m-telegraph and ρ-step variants;
- the existing LSTM-nominal and LSTM-broad.

The GPU fast paths are used by default, as in E9–E11.

**Sizes and metric.** E12 uses the `ns_*` budget (1 000 rounds × 5 000 trials in full mode)
and the P_L-based metric: `eps_fit`, plus paired P_L at the common horizon. At these noise
levels errors per round are small, so check the C1c sample-size table before trusting
small differences.

**Running** (after the wiring below):
```bash
python -m training.train_lstm        # trains lstm_hw_s* (with the other models)
python -m experiments.run_hardware   # or: python -m experiments.run_all --only E12
```

**Tests:** `tests/test_hardware.py`. They check the point specs (including the σ_m drift),
the training mixture and decoder construction, plus one sealed tiny end-to-end run (train
LSTM-hw → evaluate one E12 point).
