# Results: GKP decoding under correlated displacement noise

**Location:** `/home/babay/programs/QC/qkp-experiment`, a git worktree on the local branch
`experiment/decoder-comparison`. The branch has not been pushed. Your original folder
`/home/babay/programs/QC/qkp` and `chris-branch` were not modified.

**Start here:**
- [`writeup/report.pdf`](writeup/report.pdf): the full LaTeX write-up (20 pages).
  The source is `writeup/report.tex`, with sections in `writeup/sections/` and the methods
  in `writeup/methods.tex`.
- `results/figures/`: every figure. `results/E*_summary.csv`: every number.
- `results/run_log_pass*.txt`, `results/train_log*.txt`: what ran, and when.

## Status of each experiment

| ID | What | Status |
|---|---|---|
| E0 | Simulator validation re-run + Bayes grid convergence | complete (reproduces `docs/sim_validation`; grid converged) |
| E1 | Main comparison at the nominal point (60k trials) | complete |
| E2 | ρ sweep, two noise levels (9 ρ values each) | complete |
| E3 | σ_q sweep (9 points, 20k trials each) | complete |
| E4 | σ_m sweep (7 points, 20k trials each) | complete |
| E5 | Bayes model mismatch (35 + 2 assumed models, 40k trials) | complete |
| E6 | Regime-switching noise | complete, including the stretch items (regime-aware Bayes, LSTM trained on switching noise) |
| E7 | Timing, CPU and GPU (measured on an idle GPU) | complete |
| E8 | Diagnostics (floors, jump runs, calibration, revision ablations, 200-round horizon) | complete |
| — | DQN / reinforcement learning | **not done** (out of scope by your instruction) |
| — | Experimental GKP data | **not done** (out of scope) |

There were no failures in any run (`run_all finished; failed experiments: none`). All 17
tests pass (`QKP_MODE=quick .venv/bin/python -m pytest -q`).

## Key numbers (nominal point: σ_q = 0.2L, σ_m = 0.05L, ρ = 0.9)

| Decoder | P_L(50) | Time-averaged error |
|---|---|---|
| No correction | 0.501 | 0.481 |
| Nearest lattice (NL) | 0.242 | 0.144 |
| **Bayes (matched)** | **8.5e-4** | **1.5e-3** |
| Bayes-broad (8-model average) | 9.7e-4 | 1.9e-3 |
| Bayes, per-round (no revision) | 0.012 | 0.011 |
| LSTM-nominal (3 seeds, mean) | 6.6e-3 | 5.5e-3 |
| LSTM-nominal, per-round head | 0.020 | 0.017 |
| LSTM-broad (3 seeds, mean; large seed spread) | 0.015 | 0.010 |

## Main findings (details and caveats in the report)

1. **Memory helps a lot under correlated noise.** At the nominal point, matched Bayes
   reduces the time-averaged error by about 95× relative to NL. The gain grows with ρ and
   tracks the predictability of the next displacement, σ_q√(1−ρ²).
2. **Being able to revise past decisions matters.** The optimal decoder estimates the
   *cumulative* logical parity. Per-round variants are about 3–7× worse, for both Bayes
   and the LSTM.
3. **A correctly specified Bayes filter is best at every test point.** Matched LSTMs come
   within 2–17% of it where errors are common (ρ = 0.5: 2%; σ_q = 0.26L: 17%). The LSTM is
   about 3–4× worse where errors are rare (the nominal point). A moderately mis-specified
   filter usually still beats the LSTMs. The exception is weak correlation (ρ = 0.3–0.5),
   where the broadly trained LSTM beats the model-averaged Bayes-broad.
4. **Wrong memory is worse than no memory.** Assuming ρ' = 0.9 when the truth is ρ = 0
   (Bayes or LSTM) is worse than NL. In E5, the filter tolerates over-estimated noise but
   not over-confident models (too small an innovation variance) or ρ' = 0.
5. **ML helps when the Bayes model is structurally wrong.** Under regime-switching noise,
   the broadly trained LSTM beats every fixed-model Bayes filter. Only a regime-aware
   filter does better.
6. **Cost.** On the GPU, the LSTM and the grid filter have similar throughput
   (about 3e5 trial-rounds/s). On the CPU, single-trial latency is similar (about 0.9 ms
   per round), but batched LSTM throughput is about 27× the filter's. Training an LSTM took
   6–11 minutes (on a shared GPU).

## Things to know

- **Discrete corrections only.** Decoders apply c_t = s_t + j_t·L. This makes syndromes
  decoder-independent, so all comparisons are exact paired comparisons, but it excludes
  continuous fine corrections. See the report's §2 and limitations.
- **Retrained LSTMs.** Some LSTMs first hit a 10-minute training guard because they shared
  the GPU with the evaluation run. Those 5 models were retrained to their full schedule,
  and every affected result was re-evaluated automatically (the cache keys include
  checkpoint identity). The training table in the report lists the stop reason for each
  model.
- **Compute.** About 2 h 20 min of wall-clock compute (03:28–05:48), slightly over the
  2-hour target, because training and evaluation competed for one GPU. On top of that came
  a quick smoke run, two independent multi-agent reviews (plan and code), and a
  claim-by-claim audit of the report against the result files. Thirty-two
  corrections from that audit were applied to the report.
- **Large files not in git.** `results/cache/` (38 MB of per-trial records) is gitignored
  but kept on disk; `experiments/analyze.py` needs it to rebuild the figures. The LSTM
  checkpoints (`checkpoints/*.pt`, 29 MB) are committed.

## Reproduce

```bash
cd /home/babay/programs/QC/qkp-experiment
QKP_MODE=quick .venv/bin/python -m pytest -q
.venv/bin/python -m training.train_lstm          # skips up-to-date checkpoints
.venv/bin/python -m experiments.run_all          # skips cached results
.venv/bin/python -m experiments.analyze
.venv/bin/python -m writeup.make_tables
cd writeup && xelatex report.tex && xelatex report.tex
```

The environment is `.venv` (uv, Python 3.13, PyTorch 2.11 + CUDA 12.8).
`QKP_MODE=quick` runs everything small, using only the development random streams.
