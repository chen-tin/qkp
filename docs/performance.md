# Performance: where time goes, and the GPU fast paths (15 Nov 2026)

Hardware: laptop RTX 5080 (Blackwell, sm_120, 16 GB), torch 2.11 + CUDA 12.8, Triton 3.6.

**The laptop's GPU power limit changes during a session.** It was 33 W on battery, 80 W just
after plugging in, then 45 W at 73–77 °C. Timings move by up to 8× with it. Every run now logs
the power state (`experiments/runlog.py`). Speedups below come from interleaved A/B runs
(reference and fast alternating in one process, five repeats, medians), so drift affects
both sides equally. Absolute numbers are for a 45 W limit, on mains power.

## Run logger

`experiments/runlog.py` appends one JSON line per run to `results/runlog/runs.jsonl` (or
`results/quick/runlog/`). Each line records the run id, script and arguments, git hash, mode,
device, GPU power limit and clocks, battery state, and timed phases with metadata. GPU work is
synchronised at phase boundaries.

These write to the log:
- `run_all` and `run_nonstationary`: one phase per experiment, plus a `decode` phase per
  decoder and point with trials, rounds, implementation, noise-generation time and resets.
- `train_lstm`: one phase per model.
- `profile_pipeline`: per phase.

`runlog.summarize()` aggregates time by phase.

## Where the time went (reference implementation, profiled)

`experiments/profile_pipeline.py --impl reference`; GPU breakdown from the PyTorch profiler.

| Component | Finding |
|---|---|
| Simulate | Noise generation and the NL loop take ~5 ms for 4 096 × 50: negligible. |
| Matched Bayes | GPU busy only 25–75% of wall time. The rest is Python and launch overhead, plus a host sync every round (decisions returned to numpy). |
| Multi-model filters (broad, window, IMM) | One small GEMM per model in a Python loop. Time splits evenly between GEMM, shear gather, elementwise passes and reductions. |
| Oracle | 71% of GPU time in float64 elementwise math: per-trial transition kernels built in float64, which the laptop GPU runs at a small fraction of float32 speed. |
| LSTM | One cuDNN call per round, with the hidden state carried in Python. |
| Encode | Building the state takes 30–65 ms (a Python double loop over 289 coherent states). **The Wigner function took 0.9–3.7 s per 201² grid, and 11 s at 401², at N = 200**: the slowest single step in the pipeline. |

## What changed

New modules; the reference code is unchanged, so E0–E8 caches stay valid.

1. **`simulator/fast.py`**: closed-form, vectorised syndromes and NL record (Lemma 2 and
   Prop. 3 of `sim_validation.md`). It is exactly equal to `Simulator.run`: 0 mismatches in
   4 × 10⁶ trial-rounds, syndromes equal to 10⁻¹⁴.
2. **`decoders/fast.py`**: GPU drivers for every decoder.
   - The syndrome stream is computed once and kept on the GPU; decisions are copied back once
     per block, not every round.
   - All elementwise work is in float32; only C and the evidences stay float64.
   - Normalisation is folded into the next round's likelihood multiply (prediction is
     linear), which removes two passes over the belief.
   - Models that share a grid advance with one batched GEMM.
   - **CUDA graphs**: one round is captured and replayed, so there is one launch per round.
   - **Fused Triton kernel** (`torch.compile`): shear gather, likelihood multiply and
     eps-marginal in one pass. Disable with `QKP_FUSED=0`.
   - Oracle: kernels in float32, plus a kernel table when the schedule takes few distinct
     values.
   - LSTM: the whole sequence in one cuDNN call, and the windowed heads vectorised.
   - Resets are flagged without a host sync, and flagged trials are re-run with the reference
     decoder.
3. **`quantum/fast_encoding.py`**: the log-domain vectorised `encode_gkp` (equal to 10⁻¹⁵),
   and a Wigner function computed as a GEMM over the position wavefunction (equal to QuTiP to
   10⁻¹⁴).
4. **Wiring**: `experiments.common.run_blocked` sends `is_fast` decoders to the GPU driver.
   E9–E11 use the fast paths by default (`QKP_FAST=0` for the reference). `impl` is part of
   each cache key.

**Equivalence.** Fast and reference decisions are identical on stationary, step and burst
noise. Under strongly mismatched models (telegraph ρ 0.3↔0.97), float32 summation order flips
near-tie decisions (posterior ≈ ½) in ≤ 6 × 10⁻⁴ of trial-rounds. Flip-rate differences are
within 1.6 SE, the same size as the TF32 effect in benchmark D. CUDA-graph and eager runs are
bit-identical. All of this is enforced by `tests/test_fast.py` (8 tests; 42 in the suite).

## Results (interleaved A/B, `results/benchmarks/profile_ab.json`)

Throughput: batch 4 096 × 50 (1 024 for the heavy decoders). Latency: one trial.

| Decoder | Throughput, µs/trial-round (ref → fast) | Speedup | Latency, ms/round (ref → fast) | Speedup |
|---|---|---|---|---|
| Bayes (matched) | 1.43 → 0.61 | 2.3× | 1.76 → 0.13 | 14× |
| Bayes-broad (8 models) | 20.6 → 19.7 | 1.05× | 11.2 → 0.27 | 41× |
| Bayes (oracle) | 19.2 → 2.25 | 8.5× | 2.38 → 1.21 | 2.0× |
| Bayes-IMM (12 regimes) | 89.9 → 44.5 | 2.0× | 3.39 → 0.15 | 23× |
| Bayes-window 200 (12 models) | 38.8 → 30.5 | 1.27× | 19.0 → 0.34 | 57× |
| LSTM-nominal | 2.32 → 0.23 | 10× | 0.92 → 0.028 | 33× |

| Encode / simulate | Reference | Fast | Speedup |
|---|---|---|---|
| NL simulation, 4 096 × 50 | 6.2 ms | 5.4 ms | 1.2× |
| GKP state, Δ = 0.30, N = 100 | 32 ms | 1.9 ms | 17× |
| GKP state, Δ = 0.20, N = 200 | 66 ms | 2.1 ms | 32× |
| Wigner 201², N = 100 | 923 ms | 23 ms | 40× |
| Wigner 201², N = 200 | 3 677 ms | 31 ms | 119× |

What each piece contributes (same runs):
- **CUDA graphs** give the latency gains: matched Bayes goes from 1.18 to 0.13 ms per round,
  window 200 from 4.37 to 0.34. They barely change large-batch throughput.
- **The fused kernel** gives 1.1–1.4× throughput on the grid filters (matched 0.87 → 0.61,
  IMM 53.5 → 44.5 µs/tr).

## Where time goes now

From the logged quick-mode E9–E11 cycle (run `92b6585446a8`: 16 points × 600 trials × 120
rounds, fast path, 45 W): 174 s in total.

| Decoder | Share of decode time |
|---|---|
| Bayes-IMM | 36% |
| Bayes-window 50 + 200 | 42% |
| Bayes-broad | 13% |
| Oracle | 8% |
| Matched, LSTMs, NL | < 2% together |

Noise generation is about 0.1%. The grid filters now keep the GPU 80–98% busy, and their
remaining cost is the per-model GEMMs, which scale with the number of models. Options for
further gains:
- n_x = 64 instead of 96: same accuracy at the nominal point (benchmark G), about 1.7×.
- A banded kernel, since the AR(1) transition matrix is mostly zeros.
- Fewer components in the adaptive grids.

**Projection for the full E9–E11 run.** Scaling the cost-benchmark projection (7.0 h at
80 W, reference implementation) by the measured throughput speedups gives **about 3.6 h**. This
is an estimate, not a measurement. The oracle's speedup will be smaller on continuously
drifting schedules, where the kernel table does not apply.
