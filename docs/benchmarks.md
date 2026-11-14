# Benchmarks: real data and simulation variance (14 Nov 2026)

Scripts: `experiments/fetch_real_data.py`, `experiments/benchmark_real.py` and
`experiments/benchmark_variance.py`. Outputs are in `results/benchmarks/` (JSON, CSV, PNG).
All runs use the dev random streams.

## A. Drift generators vs measured T1 drift

Data: Zhu et al. (arXiv:2409.09926), zenodo 16754689, CC-BY-4.0. That is 33 transmon T1 traces
(3 qubits × 11 temperatures, about 72 h each); the headline uses the 27 traces at ≤ 119 mK.
Photon loss maps to displacement variance in proportion to the decay rate, so σ_q(t) ∝ √Γ(t).
Features are compared per sample, because the timescales differ: T1 samples are 4.5–13 min
apart, QEC rounds about 10 µs. Each entry below is the median, with the 10th–90th percentile
range in brackets.

| Series | relative σ spread | spectral α | lag-1 autocorr. | roughness |
|---|---|---|---|---|
| Real T1, ≤ 119 mK | 0.088 (0.056–0.131) | 0.65 (0.44–1.02) | 0.38 (0.15–0.56) | 0.79 (0.67–0.92) |
| 1/f, rel = 0.25 (E9) | 0.18 | 1.23 | 0.89 | 0.34 |
| sinusoid P = 1000 / 200 (E9) | 0.21 | 2.2 / 1.9 | 1.00 | ≤ 0.02 |
| telegraph σ_q (E10) | 0.13 | 1.9 | 0.99 | 0.11 |
| 1/f, rel = 0.12 (calibrated) | 0.089 | 1.28 | 0.90 | 0.32 |

Roughness is 1 for white noise and goes to 0 for smooth drift.

Findings:
- Our drift points vary σ_q about twice as much as the real data does. Setting rel = 0.12
  matches the spread.
- No generator reproduces the texture of the real data. Real T1 drift is close to white at the
  sampling interval (α ≈ 0.65, autocorrelation ≈ 0.4), while every generator is smooth
  (α ≥ 1.2, autocorrelation ≥ 0.85). Matching it would need a 1/f component plus a white
  component.
- Caveat: each T1 point is a fit with its own uncertainty, which also adds white noise. The
  data files carry no fit errors, so the two cannot be separated here.
- Increments of the real data are close to Gaussian (kurtosis about 4). There are no dominant
  telegraph jumps at this sampling rate, while our step and telegraph generators have
  kurtosis of 100 or more by construction.

## B. Calibration against a real GKP experiment

Sivak et al. 2023 (arXiv:2211.09116) report a cavity T1 of 610 µs, a cycle of 9.848 µs,
Δ = 0.34, and p_X = 1.81×10⁻³ per cycle. Using σ² = γ/(1−γ) for loss followed by
amplification (Albert et al., arXiv:1708.05010, Sec. II.2), loss alone gives
**σ_q = 0.072L** per cycle, with ρ = 0. Our nominal σ_q = 0.20L would correspond to a cavity
T1 of about **83 µs**, roughly 7× worse than that experiment.

At σ_q = 0.072L, our model reaches the measured 1.81×10⁻³ per cycle when the effective
measurement noise is **σ_m ≈ 0.102L (nearest lattice) or 0.109L (matched Bayes)**. Taking
σ_m = Δ/√2 = 0.136L instead gives 1.4×10⁻² (NL) and 9.7×10⁻³ (Bayes), 5–8× above the
experiment. The simulated NL rates match the closed form for σ_m ≤ 0.12L. Above that, the
closed form's assumption |ν| < L/2 starts to fail.

Caveats:
- This compares a single number, not data. The experiment measures one ancilla bit per round
  using finite-energy states and real ancilla errors.
- The σ ↔ loss and σ_m ↔ Δ conversions assume the √π-lattice convention used in this repo.
  The quadrature convention of the Albert et al. formula was not stated in the text read.

## C. Variance of the generated data

**C1. Monte Carlo variance at the nominal point:** 30 seeds × 2000 trials × 50 rounds.

| Decoder | Metric | Mean | Seed-to-seed SD ÷ reported SE (95% CI) | Trials for 10% relative SE |
|---|---|---|---|---|
| NL | P_L(50) | 0.245 | 1.09 (0.87–1.47) | 330 |
| NL | flip rate | 0.0185 | 1.12 (0.89–1.50) | 420 |
| Bayes (matched) | P_L(50) | 8.2×10⁻⁴ | 0.92 (0.73–1.24) | 67 000 |
| Bayes (matched) | time-averaged error | 1.4×10⁻³ | 0.88 (0.70–1.19) | 12 000 |
| Bayes (matched) | flip rate | 2.2×10⁻³ | 0.73 (0.58–0.98) | 6 000 |

The reported standard errors match the seed-to-seed scatter; the CI includes 1 everywhere
except the Bayes flip rate, where the reported SE is slightly conservative. The NL−Bayes
difference is resolved at 3σ with about 50 trials.

**C1b. Nonstationary step in σ_q:** 12 seeds × 600 trials × 400 rounds. The fixed-model Bayes
filter makes 3.2×10⁻⁴ more flips per round than the oracle, about 11%. Again the SE matches
the scatter (ratio 0.81, CI includes 1). Resolving this difference at 3σ takes about 1 500
trials at 400 rounds, or about **610 trials at E9–E11's 1 000 rounds**, so the planned 5 000
trials per point is enough for effects of this size.

**C2. Spread of the generated sequences, as the decoders see them:** 4 000 sequences per family.
- Within every family, the per-sequence syndrome statistics vary much more than the family
  medians differ. The median syndrome SD is 0.20–0.22L for all nonstationary families.
- At 1 000 rounds, the burst point and the two sinusoid periods are indistinguishable from the
  control by these statistics: one burst in 1 000 rounds hardly changes them.
- In the LSTM-drift training mixture, the family label explains only **1.6%** of the variance in
  sequence difficulty (NL flip rate). The rest comes from the per-sequence parameter draws and
  Monte Carlo noise.
- E11 and the held-out shapes therefore test robustness to parameter regimes more than to
  recognisably different syndrome statistics.
