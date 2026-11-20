# Findings log

New project-wide work is consolidated in [RESEARCH_NOTES.md](../RESEARCH_NOTES.md), with ongoing logging required by [AGENTS.md](../AGENTS.md). This historical findings log is retained as evidence, including its dated corrections.

A running record of results, in the order they came in. Each entry gives the claim, the
evidence (script and result file) and the caveats. Corrections are recorded as new entries
that point back to what they correct. Nothing earlier is silently rewritten.

Conventions:
- Noise levels are in units of L = √π.
- ε is the logical error per round, fitted from the decay of 1 − 2P_L(t) before saturation
  (`experiments.common.eps_fit`).
- "Paired P_L" means P_L at a common horizon, on the same noise for every decoder.
- "Hardware scale" means σ_q = 0.072L, σ_m ≈ 0.105L. That is the cavity loss of Sivak et
  al. 2023, with σ_m calibrated to their measured logical error; see `docs/benchmarks.md` B.

---

## 2026-11-15 · Stationary decoder comparison (E1–E8)

Complete; see `RESULTS.md` and `writeup/report.pdf`.
- At the nominal point (σ_q = 0.20L, σ_m = 0.05L, ρ = 0.9), matched Bayes has 95× lower
  time-averaged error than nearest lattice (NL).
- Revising past decisions matters: per-round variants are 3–7× worse.
- The correctly specified Bayes filter is best at every point. LSTMs come within 2–17% of it
  where errors are common, and are 3–4× worse where they are rare.
- Wrong memory is worse than none (E5). Under regime switching, a broadly trained LSTM beats
  every fixed-model filter (E6).

## 2026-11-15 · Simulator validation, hardware calibration, cost (benchmarks A–H)

See `docs/benchmarks.md`.
- **Validation:** all 15 exact checks fall within 2σ. The shift model matches QuTiP
  finite-energy states to 0.1–2.3% for single-round readout. float32/TF32 does not change
  results.
- **Calibration:** our nominal noise is about 7× harsher than the Sivak et al. cavity. The
  measured T1 drift is rougher and about half as large as our drift generators.
- **Cost:** a 64-point Bayes grid is the cheapest accurate decoder. The LSTM's training never
  pays off in throughput.

## 2026-11-15 · Correction: flip rate is not a logical error rate for long runs

Corrects the stress test's first version and the earlier recommendation in
`docs/benchmarks.md` C.
- Once P_L nears ½, a decoder that revises its estimate of the logical parity dithers, and
  each dither counts as a logical flip.
- At stationary ρ = 0.3 over 600 rounds, flip rates made every Bayes filter look 1.4–16×
  worse than NL. P_L(t) shows them better than NL at every round before saturation (0.391
  against 0.425 at round 50). The fast and reference implementations agree there.
- Long runs are now scored by ε and paired P_L (`experiments/common.py`; E9–E11 analysis
  rewritten).

## 2026-11-15 · Drifting noise does not break the Bayes decoder (stress test)

`experiments/stress_bayes.py`; `results/benchmarks/stress_bayes.json`. All 16 E9–E11
schedules, 2 000 trials × 1 000 rounds (IMM: 1 000), dev streams.
- No belief collapses anywhere.
- The fixed nominal filter beats NL at every point, and its ε is 0.97–1.48× the oracle's.
- Its largest losses come where the noise leaves the assumed range:
  - amplitude 0.6: paired P_L +0.055
  - σ_q 0.32–0.34L: +0.053
  - sinusoidal ρ: +0.061
  - 1/f σ_q: +0.028
- The adaptive IMM recovers most of it (within 0.003 of the oracle at the clearly significant
  points). The exceptions are 1/f drift (+0.025) and a step to ρ = 0, which is outside its
  grid (+0.046, 2.2σ).

## 2026-11-15 · Sample sizes for the long-run metric

`experiments/benchmark_variance.py --pl`; `results/benchmarks/benchmark_variance_pl.json`.
- At ε of about 10⁻⁵ per round over 1 000 rounds, 10% precision on ε needs 2 000–9 500 trials.
- The reported paired-P_L SEs match the replicate scatter (ratio 0.96–1.08).
- With 5 000 trials per point, P_L gaps of about 0.01 or more are resolved; smaller ones are not.

## 2026-11-15 · Coverage: hardware-scale noise is a regime our scenarios miss

`experiments/coverage_map.py`; `results/benchmarks/coverage_map.png`; `docs/benchmarks.md` H.
- ρ has no data anchor for GKP hardware.
- Hardware σ_q (0.072L) is below everything simulated before E12.
- At hardware scale, readout noise dominates and consecutive syndromes are anti-correlated
  (lag-1 correlation −0.24 to −0.38 for any ρ). All earlier scenarios sit at +0.23 to +0.58,
  except ρ = 0 (−0.04). In that regime ρ is barely visible in the syndromes.

## 2026-11-16 · Bug found while launching the full runs (fixed)

When `experiments/run_all.py` runs as a script, the new experiment modules imported a second
copy of it. That copy had no GPU device set, and its failure list was never reported. The
first full E5b run failed every decoder while reporting "failed: none". Fixed in
`run_all._share_state()`; E5b was re-run. E9–E11 were unaffected, because they run through
`run_nonstationary`.

## 2026-11-16 · E14: one-bit syndromes. Memory is what makes one-bit readout survive correlated noise

`experiments/run_binary.py`; `results/E14_summary.json`. 24 points, 4 000 trials × 200 rounds,
closed loop. Paired P_L at a common horizon. The bit-LSTM is still training.

| Point | sBs, memoryless (best δ) | Bayes, binary | Bayes, analog (reference) |
|---|---|---|---|
| hardware, ρ = 0, m = 4 bits/round | 0.088 | 0.074 | 0.015 |
| hardware, ρ = 0, m = 8 | 0.018 | 0.014 | 0.016 |
| hardware, ρ = 0.9, m = 4 | 0.495 | 0.26 | 0.038 |
| hardware, ρ = 0.9, m = 8 | 0.36 | **0.030** | 0.040 |
| hardware, ρ = 0.9, m = 1 | 0.45 | 0.40 | 0.012 |
| nominal, ρ = 0.9, m = 4 | 0.495 | 0.40 | 0.002 |
| nominal, ρ = 0.9, m = 8 | 0.49 | 0.16 | 0.004 |

- **Under white noise (ρ = 0), memoryless sBs nearly matches binary Bayes.** Under correlated
  noise (ρ = 0.9), sBs loses the qubit at every δ tried (P_L ≈ 0.5), while binary Bayes holds
  on. At 8 bits per round it matches the analog-syndrome decoder (0.030 against 0.040).
- **Bits per round is the deciding knob.** One bit per round throws away most of the memory
  advantage: 0.40 against 0.012 analog at hardware scale. At nominal noise even 4 bits leave
  binary Bayes at 0.40, against 0.002 analog.
- Visibility 0.95 and 1% bit flips cost binary Bayes about 0.01–0.10 in P_L at hardware
  scale and m = 4.
- Caveats:
  - The binary filter's correction policy (lookahead centring) is one design choice.
  - Corrections are quantised to 0.037L.
  - All m bits in a round see the same displacement.
  - No finite-energy dissipation or ancilla back-action is modelled.

## 2026-11-16 · E15: fine corrections do not help a Bayes decoder in the shift model

`experiments/run_continuous.py`; `results/E15_summary.json`. 5 points, closed loop. The DQN
agents are still training.

| Point | NL P_L(200) | Bayes, parity only | Bayes, fine corrections |
|---|---|---|---|
| nominal | 0.465 | 0.0032 | 0.0032 |
| hardware | 0.306 | 0.0354 | 0.0352 |
| hardware, σ_m = 0.10L | 0.236 | 0.0205 | 0.0204 |
| hardware, σ_m = 0.15L | 0.504 | 0.382 | 0.381 (−0.001 ± 0.0005) |

- Fine corrections, which also undo the readout error's imprint, change P_L by at most 0.001.
  In the shift model, the Bayes filter's belief does not depend on where the corrections
  leave the oscillator inside a cell. A gain would need physics the model lacks, such as
  finite-energy damping or a memoryless decoder.
- At hardware scale, Bayes is about 9× better than NL (0.035 against 0.306).

## 2026-11-16 · E5b: over-confident models are the dangerous mismatch, everywhere

`experiments/run_mismatch.py`; cache `results/cache/E5b/`. 40 000 trials × 50 rounds per
decoder; reference decoders; test streams. Values are P_L(50) / time-averaged error.

| True point | Matched Bayes | NL | Best mismatched | Worst mismatched |
|---|---|---|---|---|
| ρ = 0.5 (σ_q 0.20L, σ_m 0.05L) | 0.234 / 0.134 | 0.399 / 0.255 | σ_m′ ×0.5: 0.242 / 0.138 | ρ′ = 0.99, σ_q′ ×0.5: 0.500 / 0.404; ρ′ = 0, σ_q′ ×2: 0.502 / 0.319 |
| σ_q = 0.26L (ρ 0.9) | 0.0102 / 0.0117 | 0.418 / 0.286 | σ_m′ ×0.5: 0.0147 / 0.0146 | ρ′ = 0, σ_q′ ×2: 0.500 / 0.405 |
| hardware (0.072L, σ_m 0.105L, ρ 0.9) | 0.0097 / 0.0058 | 0.101 / 0.052 | ρ′ = 0.99, σ_q′ ×2: 0.0138 / 0.0086 | ρ′ = 0, σ_q′ ×0.5: 0.442 / 0.296 |

- **E5's rule holds at every point.** Over-estimating the noise (σ_q′ ×2) is cheap. Under-estimating
  it (σ_q′ ×0.5), or assuming no correlation (ρ′ = 0), is often **worse than NL**. At all three
  points, ρ′ = 0 with σ_q′ ≤ 1 loses to NL.
- **At hardware scale, matched Bayes is 10× better than NL** (P_L(50) 0.0097 against 0.101).
  - It is sensitive to the assumed readout noise: σ_m′ ×2 costs 4.6×, and σ_m′ ×0.5 costs 2.7×.
  - So a hardware decoder needs σ_m calibrated. Over-estimating ρ is cheaper (ρ′ = 0.99,
    σ_q′ ×2 costs 1.4×).
- At weak correlation (ρ = 0.5), mismatch costs least, because memory helps least there to begin with.

## 2026-11-16 · E14 re-scored per round (ε). Same conclusions, sharper numbers

The E14 table above compares paired P_L at each point's own horizon, and the horizons
differ (15–200 rounds). ε, the error per round, is comparable across points.
Source: `results/E14_summary.json`; visibility 1, no bit flips.

| Point | sBs, best δ | Bayes, binary | Bayes, analog |
|---|---|---|---|
| hardware, ρ = 0.9, m = 1 | 2.2×10⁻² | 1.5×10⁻² | 1.9×10⁻⁴ |
| hardware, ρ = 0.9, m = 4 | 1.0×10⁻² | 1.9×10⁻³ | 2.0×10⁻⁴ |
| hardware, ρ = 0.9, m = 8 | 3.3×10⁻³ | **1.4×10⁻⁴** | 1.9×10⁻⁴ |
| hardware, ρ = 0, m = 8 | 8.6×10⁻⁵ | 7.1×10⁻⁵ | 6.9×10⁻⁵ |
| nominal, ρ = 0.9, m = 4 | 4.3×10⁻² | 7.5×10⁻³ | 6.7×10⁻⁶ |
| nominal, ρ = 0.9, m = 8 | 3.3×10⁻² | 9.2×10⁻⁴ | 8.7×10⁻⁶ |

- With correlated noise at hardware scale and 8 bits per round, binary Bayes is 24× better
  than the best sBs and matches the analog decoder.
- With white noise, sBs is within 1.2× of binary Bayes.
- At nominal noise, binary readout costs about 100–1 000× against analog syndromes.

## 2026-11-16 · First E12 point: decoders tuned to our stress regime fail at hardware scale (preliminary)

`experiments/run_hardware.py`. One of 16 points finished (σ_q 0.072L, σ_m 0.08L, ρ = 0;
test streams, 1 000 rounds). ε relative to the oracle:

| Decoder | ε ÷ oracle |
|---|---|
| Bayes-IMM (hardware grid) | 1.19× |
| Bayes-window 200 / 50 | 1.55× / 1.86× |
| Bayes at the hardware anchor (assumes ρ = 0.9) | 5.8× |
| LSTM-broad | 22× |
| NL | 22× |
| Bayes assuming the nominal model (0.2L / 0.05L / 0.9) | **170×** |
| LSTM-nominal | **477×** |

- A filter or LSTM tuned to our nominal stress regime is far worse than memoryless NL on
  hardware-like white noise. The nominal model over-estimates σ_q, which E5 found cheap. Its
  other two assumptions are the costly ones:
  - strong correlation (ρ′ = 0.9) where there is none;
  - less readout noise (σ_m′ = 0.05L) than there is (0.08L).

  Both make it over-confident, which is E5's dangerous direction, here at its most extreme.
- Adaptive filters over a hardware-scale grid stay within 2× of the oracle.
- Preliminary: one point only. The remaining 15 are running.
