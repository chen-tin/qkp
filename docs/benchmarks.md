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

**Caution on the flip-rate metric.** The flip rate is a valid per-round logical error
rate only while P_L stays small, as in the 50-round runs above (nominal P_L(50) ≤ 0.24).
For long or high-noise runs, where P_L approaches 1/2, it is not: revising decoders dither,
and each dither counts as a flip. Use the decay-fitted eps and paired P_L instead (C1c and
`docs/nonstationary.md`). The earlier recommendation to use flip rate for long runs is
withdrawn. In B, the Bayes points at σ_m ≥ 0.15L may be inflated for the same reason;
the crossing with the Sivak value, at σ_m ≈ 0.11L where P_L(100) ≈ 0.1, is not affected.

**C1c. Sample sizes for the P_L-based metric on long runs** (`benchmark_variance.py --pl`;
8 replicates × 1 000 trials × 1 000 rounds; fixed nominal Bayes against the oracle;
`results/benchmarks/benchmark_variance_pl.json`).

| Point | eps (oracle) | Trials for 10% relative SE on eps | Paired P_L gap at the horizon | Reported SE ÷ replicate SD | Trials to resolve the gap at 3σ |
|---|---|---|---|---|---|
| stationary nominal | 1.4×10⁻⁵ | 9 500 | 0 (identical decoders) | — | — |
| sinusoid σ_q, P = 200 | 3.3×10⁻⁵ | 2 000 | +0.003 | 1.00 | 11 300 |
| 1/f σ_q | 6.4×10⁻⁵ | 2 550 | +0.028 | 0.96 | 720 |
| step ρ 0.9 → 0.3 | 9.9×10⁻⁴ | 70 | +0.005 | 1.08 | 124 000 |

The paired P_L standard errors match the replicate scatter, so these error bars are honest.
At error rates around 10⁻⁵ per round, eps needs about 2 000–9 500 trials for 10% precision.
The planned 5 000 trials per E9–E11 point resolve eps to about 1–14%, and P_L gaps of about
0.01 or more. Smaller mis-specification costs, such as sinusoidal drift or ρ steps, cannot
be distinguished from zero at that budget.

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

## D. Numerical precision (float32/TF32 vs float64)

Script: `experiments/benchmark_precision.py`. Outputs: `results/benchmarks/precision.{json,csv}`.
The Bayes filters store the belief in float32 and allow TF32 in the prediction GEMM. The same
noise was rerun with three settings: fp32+TF32 (the default for every experiment), fp32 with
TF32 off, and fp64 (the reference). The filters were switched at runtime; no decoder file was
changed.

| Case | Decoder | Trials × rounds | Decisions differing (fp32+TF32) | Flip rate (fp64) | Flip-rate change | max \|Δp\| |
|---|---|---|---|---|---|---|
| nominal | Bayes (matched) | 4000 × 50 | 0 | 2.09×10⁻³ | 0 | 1.4×10⁻⁴ |
| σ_m = 0.02L, ρ = 0.99 | Bayes (matched) | 4000 × 50 | 0 | 1.01×10⁻² | 0 | 3.9×10⁻⁴ |
| σ_q = 0.30L | Bayes (matched) | 4000 × 50 | 0 | 3.73×10⁻² | 0 | 2.7×10⁻⁴ |
| ρ = 0 | Bayes (matched) | 4000 × 50 | 1.3×10⁻⁴ (25 of 200 000) | 2.03×10⁻² | +8.5×10⁻⁵ ± 5.1×10⁻⁵ | 1.5×10⁻⁴ |
| nominal, 500 rounds | Bayes (matched) | 1000 × 500 | 0 | 5.3×10⁻⁴ | 0 | 2.6×10⁻⁴ |
| step σ_q, 200 rounds | Bayes (oracle) | 1000 × 200 | 0 | 3.56×10⁻³ | 0 | 1.8×10⁻⁷ |
| step σ_q, 200 rounds | Bayes-IMM | 500 × 200 | 0 | 3.80×10⁻³ | 0 | 3.9×10⁻⁴ |

Notes on the table:
- Plain fp32 (TF32 off) matched fp64 exactly in every case (max |Δp| ≤ 2.4×10⁻⁷, which is
  the float32 precision at which p is recorded).
- TF32 moves the posterior probability p by up to 4×10⁻⁴.
- That changes decisions only at ρ = 0, where p sits close to ½ most often. There, 25 of
  200 000 decisions differed, and the flip-rate change of +0.4% is not significant (1.7 SE).
- No filter resets occurred in any setting.
- Error does not build up over 500 rounds.
- The oracle's per-trial batched matrix multiply did not use TF32 in this run.

**Verdict:** float32 with TF32 is safe for every reported metric. Its effect is at least
100× smaller than the statistical errors. For exact reproducibility of individual decisions
near p = ½, turn TF32 off: this costs nothing in accuracy.

## E. Finite-energy GKP states (HyQBench-style checks)

Script: `experiments/benchmark_finite_energy.py` (QuTiP 5.3, CPU, 8 min). Outputs:
`results/benchmarks/finite_energy.json` and `finite_energy_truncation.csv`.

The state is exp(−Δ² n̂)|0⟩, built as in `quantum/encoding.py` (a coherent-state lattice sum
with k_max = 8). The reference uses N = 600 Fock levels and k_max = 12. |+⟩ is the π/2 rotation
of |0⟩, so its truncation metrics are identical to those of |0⟩; this was checked.

| Δ | ⟨n⟩ | cutoff for 1−F < 1e-6 | 1−F at N = 100 | squeezing, dB (from ⟨S_q⟩, nominal) | Wigner negativity | peak SD / L (Δ/√2 / L) | odd-cell mass (shift model) |
|---|---|---|---|---|---|---|---|
| 0.20 | 12.0 | 173 | 4.0e-4 | 13.98 / 13.98 | 0.498 | 0.0798 (0.0798) | 4.2e-10 (3.7e-10) |
| 0.25 | 7.5 | 109 | 4.6e-6 | 12.04 / 12.04 | 0.488 | 0.0997 (0.0997) | 5.28e-7 (5.35e-7) |
| 0.30 | 5.1 | 77 | 1.9e-8 | 10.46 / 10.46 | 0.463 | 0.1195 (0.1197) | 2.94e-5 (2.95e-5) |
| 0.34 | 3.8 | 61 | 1.1e-10 | 9.38 / 9.37 | 0.433 | 0.1353 (0.1356) | 2.26e-4 (2.28e-4) |
| 0.40 | 2.4 | 41 | 1.5e-14 | 7.97 / 7.96 | 0.372 | 0.1589 (0.1596) | 1.69e-3 (1.73e-3) |

Notes:
- Δ_eff is computed as Δ_eff² = −ln(|⟨S⟩|²)/(2π) (Duivenvoorden, Terhal & Weigand 2017). It
  agrees with the nominal Δ from ⟨S_q⟩. ⟨S_p⟩ gives a slightly larger value at Δ = 0.40
  (0.411).
- Wigner negativity is (∫|W| − 1)/2. It is converged to 4 digits across grid spacings 0.08,
  0.05 and 0.035, with ∫W = 1.
- Two QuTiP cross-checks agree with the marginal-based calculation to 1e-12: ⟨S_q⟩ from the
  displacement operator, and the odd-cell mass after an explicit displacement.

**Verdict:**
- **Truncation.** The cutoff of 100 in `encoding.py` is enough for Δ ≥ 0.30 (1−F ≤ 2e-8),
  borderline at Δ = 0.25, and not enough at Δ = 0.20 (1−F = 4e-4; it needs N ≥ 173). k_max = 8
  also loses 1.5e-5 of fidelity at Δ = 0.20.
- **Shift model.** For position readout, the finite-energy state behaves almost exactly like
  an ideal GKP state with Gaussian shift noise of σ = Δ/√2. Peak widths agree to 0.5%. The odd-cell
  (logical) mass agrees with `odd_jump_probability` within 0.1–2.3%, and within 0.3–1.1% after
  an added displacement σ_q ∈ {0.072, 0.10, 0.20} L. The finite envelope also pulls peaks
  0.1–1.3% toward the origin.
- **Implied σ_m.** Δ = 0.34 (Sivak) gives an intrinsic σ = 0.136 L. The 0.10–0.11 L needed to
  reproduce Sivak's p_X (section B) corresponds to Δ ≈ 0.25–0.27. So the state's width alone
  overstates the effective measurement noise in that experiment. Its error per cycle is set
  by more than readout of an ideal-plus-Δ state (binary ancilla, feedback); this check does
  not model those.

## F. Effective-length metrics (from existing E1–E8 results)

`experiments/benchmark_effective_length.py` produces `results/benchmarks/effective_length.{json,csv}`
from cached results only, with no new simulations. It is the analogue of the "effective
fermionic length" benchmark (arXiv:2003.01862): for each decoder, how far it can be pushed
before a quality threshold is crossed. Noise levels are in units of L. Sweep boundaries are
interpolated in log space.

Column definitions:
- **σ_q max:** the largest σ_q with P_L(50) < 0.05, along E3 (ρ = 0.9, σ_m = 0.05L).
- **σ_q max (mean):** the largest σ_q with time-averaged error < 0.01.
- **σ_m max:** the largest σ_m with P_L(50) < 0.05, along E4.
- **ρ min A / B:** the smallest ρ with P_L(50) < 0.05, along E2 levels A and B.
- **Rounds to P_L > 0.01:** at the nominal point, the first round from which P_L stays above
  0.01, with a Wilson-interval range. Two horizons are shown: E1 (50 rounds, LSTM averaged over
  3 seeds) and E8 (200 rounds, seed s0 only).
- **≥ 2× NL:** the range where the time-averaged error is at most half of NL's.
- **σ_q\*:** above this σ_q, the decoder's logical flip rate per round is below the odd-jump
  probability of one bare, unprotected round. Below it, measurement noise makes repeated
  correction worse than not correcting.

| Decoder | σ_q max | σ_q max (mean) | σ_m max | ρ min A / B | Rounds to P_L > 0.01 (E1 / E8) | ≥ 2× NL: ρ (E2-A) | ≥ 2× NL: σ_q (E3) | σ_q\* |
|---|---|---|---|---|---|---|---|---|
| NL | 0.145 | 0.129 | nowhere | nowhere / nowhere | 1 / 1 | – | – | never |
| Bayes (matched) | 0.314 | 0.257 | 0.084 | 0.714 / 0.953 | > 50 / > 200 | 0.7–0.995 (6/6 at 2σ) | 0.08–0.34 (8/9) | ≤ 0.14 |
| Bayes (nominal) | 0.306 | 0.255 | 0.081 | 0.736 / 0.965 | not run | 0.7–0.995 (6/6) | 0.08–0.34 (8/9) | 0.123 |
| Bayes-broad | 0.308 | 0.251 | 0.075 | 0.722 / 0.977 | > 50 / not run | 0.7–0.995 (6/6) | 0.08–0.34 (8/9) | 0.125 |
| LSTM-nominal | 0.256 | 0.217 | 0.070 | 0.779 / nowhere | > 50 / 102 [81–115] | 0.7–0.995 (6/6) | 0.10–0.34 (8/8) | 0.127 |
| LSTM-broad | 0.241 | 0.200 | 0.066 | 0.758 / nowhere | 27 [20–33] / 114 [100–143] | 0.7–0.995 (6/6) | 0.08–0.34 (8/9) | 0.139 |

Notes:
- "nowhere" means the threshold fails even at the easiest point of the sweep.
- Bayes (matched) has no σ_q\* crossing in the sweep. It beats a bare round at every resolved
  point from σ_q = 0.14L, and makes zero flips below that.
- The two LSTM-broad round lengths differ (27 and 114) because E1 averages 3 seeds and E8 uses
  s0 alone. The LSTM-broad seeds differ widely (see E1).
- On level B of E2 (σ_q = 0.30L, σ_m = 0.08L), only the Bayes filters keep P_L(50) below 0.05,
  and they need ρ ≥ 0.95.

## G. Cost: latency, throughput, scaling, matched accuracy, time-to-solution

Script `experiments/benchmark_cost.py`; output in `results/benchmarks/cost.json` and
`cost_*.csv`. Measured on 15 Nov 2026 on an idle laptop RTX 5080 (no other GPU processes,
load average about 1.2), torch 2.11. "µs/tr" means microseconds per trial-round.

| Decoder (GPU unless noted) | Latency, 1 trial (ms/round) | Best batched cost (µs/tr) | Flip rate at nominal (8 000 × 50) | Peak memory at best batch |
|---|---|---|---|---|
| NL (CPU, numpy) | 0.009 | 0.039 | 1.91e-2 | — |
| Bayes matched, n_x = 64 | 2.24 | **1.04** | **2.18e-3** | 0.22 GB |
| Bayes matched, n_x = 96 (default) | 2.27 | 1.75 | 2.19e-3 | 0.47 GB |
| Bayes matched, n_x = 128 | 3.04 | 3.62 | 2.19e-3 | 0.83 GB |
| Bayes matched, n_x = 192 | 2.23 | 8.2 | 2.19e-3 | 0.24 GB (at batch 512) |
| Bayes-broad (8 models) | 18.0 | 34.9 | 2.53e-3 | 1.6 GB |
| Bayes (oracle) | 4.70 | 46.0 | 2.19e-3 | 2.3 GB |
| Bayes-IMM (12 regimes) | 3.72 | 130 | 2.78e-3 ± 0.41e-3 | 1.4 GB |
| Bayes-window 50 / 200 (12 models) | 24.6 / 27.5 | 38.0 / 34.7 | 2.71e-3 ± 0.41e-3 | 0.65 GB |
| LSTM-nominal (hidden 128) | 1.11 | 3.25 | 4.12e-3 | 0.13 GB |

Findings:
- **Matched accuracy.** At the nominal point every matched Bayes grid gives the same flip rate.
  So n_x = 64 is the cheapest accurate decoder: 1.0 µs/tr, 13× cheaper than n_x = 192 and
  1.7× cheaper than the default n_x = 96. The LSTM costs 3× more per trial-round than
  n_x = 64 for twice the error, so it is beaten on both axes. NL is about 30× cheaper again,
  but 9× less accurate. The oracle gives the same answer as matched Bayes on stationary noise,
  at 44× the cost (one small matrix product per trial). IMM and the window averages pay
  about 25% in accuracy (1.4 SE, 2 000 trials) for not knowing the model.
- **Time-to-solution, QUARK-style.** The LSTM needs 339 s of training plus data generation
  (lstm_nominal_s0). Its best throughput (3.1e5 trial-rounds/s) is *below* Bayes n_x = 96
  (5.7e5) and n_x = 64 (9.6e5). Training therefore never pays off in throughput at this
  operating point: decoding 1e9 trial-rounds takes 1 040 s with Bayes n_x = 64 and 3 590 s
  with the LSTM, including training.
- **Single-trial latency.** Latency is set by Python and kernel-launch overhead: 1.1 ms/round
  for the LSTM and 2.2–4.7 ms/round for the filters, against a 9.85 µs hardware cycle. This
  confirms the lab section: per-round real-time use would need an FPGA or compiled
  implementation.
- **Scaling.** Cost per trial-round stays flat from 50 to 1 000 rounds for every decoder
  (within ±15% timing noise), so cost is linear in sequence length. Cost per trial-round
  falls with batch size until the GPU is saturated (batch 4 096 for small grids). At n_x =
  192 or with 12-model banks it rises again at large batches, because memory bandwidth limits.
- **Components.** For matched Bayes at 4 096 × 50, noise generation (6.6 ms) and the
  simulator loop (5 ms) are under 1% of the 673 ms total; decoding is 99%.
- **E9–E11 projection.** With the full budget (5 000 trials × 1 000 rounds per point, 16
  points): about **7.0 h** of decoding in total. Per point: IMM 650 s, oracle 230 s,
  window 50 190 s, Bayes-broad 174 s, window 200 173 s, the nine LSTMs 146 s together, and
  the rest under 10 s. LSTM-drift training is not included. The adaptive decoders
  dominate the total; the oracle has not been timed at smaller grids.

## H. Coverage map: where our scenarios sit, and where hardware sits

Script `experiments/coverage_map.py`; output `results/benchmarks/coverage_map.png` and `.json`.
Panels: (a) σ_q against ρ and (b) σ_q against σ_m, for the E1–E8 sweep points, the 5–95%
range of every E9–E11 schedule, and the LSTM training box; (c) per-sequence syndrome spread
against lag-1 syndrome correlation, which is what a decoder sees, for each scenario family,
plus the hardware-scale point (σ_q = 0.072L, σ_m = 0.105L) simulated at ρ = 0, 0.5 and 0.9.

Findings:
- **ρ has no data anchor.** Every scenario assumes a value of ρ. No GKP measurement constrains
  it, because the T1 data is sampled minutes apart.
- **Hardware sits outside every simulated scenario.**
  - Its σ_q (0.072L) is below the smallest we simulate (0.08L), and no nonstationary point
    goes below 0.081L.
  - Its σ_m (about 0.10L) appears only in the E4 sweep, at σ_q = 0.20L.
  - No E9–E11 point varies σ_m.
- **Measurement noise dominates the syndromes at hardware scale.** Consecutive syndromes share
  ν_t with opposite signs, so their lag-1 correlation is negative: −0.38 at ρ = 0 and
  still −0.24 at ρ = 0.9. All our scenarios sit at +0.23 to +0.58, except ρ = 0 (−0.04). In
  standardised feature units, the nearest scenario is 2.1–2.8 away. In this regime ρ is
  barely visible in the syndromes, so any memory advantage comes mainly from averaging out
  readout noise, not from tracking correlated drift. This is a regime the decoder
  comparisons have not tested.
- **Syndrome spread is similar across our scenarios** (median SD 0.19–0.22L for most). The
  families differ mainly in lag-1 correlation, consistent with C2.

