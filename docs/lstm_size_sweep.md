# LSTM width and depth sweep

The approved scope is widths **32, 64, 128, 256**, depths **1, 2, 3**, and seeds
**0, 1, 2**: 36 independently trained models. Each model is evaluated at all 61
stationary noise settings used by the decoder heatmaps, giving 2,196 new
model/setting evaluations. These are three intersecting 5×5 parameter planes,
not the full 125-point cube and not a nonstationary-noise sweep.

The question is whether capacity changes logical-error boundaries under noise
mismatch. The control is a freshly trained 128-unit, two-layer LSTM under the
same protocol. Width comparisons hold depth fixed; depth comparisons hold
width fixed. The factorial grid also exposes their interaction.

## Shared protocol

- Nominal training: σq = 0.20L, σm = 0.05L, ρ = 0.90, 50 rounds.
- 200,000 training and 20,000 validation sequences per seed. All architectures
  within a seed share the exact sequences and minibatch permutations.
- Existing three syndrome features and five parity heads. Float32 training,
  AdamW, learning rate 0.002, weight decay 0.0001, batch 2,048, gradient norm
  clipping at 1, cosine learning-rate schedule.
- Up to 40 epochs; stop after five epochs without validation-loss improvement
  greater than 0.00001. No wall-time cutoff for larger models. Restore the best
  validation-loss weights, then select the head with lowest validation mean
  logical error. Evaluation heatmaps never select weights or heads.
- Separate deterministic train, validation and development random streams.
  Evaluation uses the same 1,024 noise trajectories per cell as the existing
  heatmaps. Each seed is reported separately with pointwise Wilson intervals;
  three trained seeds do not count as 3,072 independent noise trials.
- Reuse the original measured classical/learned comparisons from run
  `e940f6fb63229b90`; preserve their raw records and provenance.

The expectation is that greater capacity may help some settings, but gains
need not be monotonic. Report per-seed error rates and paired differences.
Treat trends that reverse across seeds as inconclusive. This is a development
study; it does not establish a universal winning architecture or replace the
benchmark test split.

## Training method and literature support

This is supervised sequence learning. The simulator supplies the true parity
labels; the network receives only the syndrome history. At round t its inputs
are `s/L`, `sin(2πs/L)` and `cos(2πs/L)`. A causal LSTM predicts five parity
targets: windows of 1, 2, 3 and 5 rounds, and the entire prefix through t.
Binary cross-entropy is averaged across sequences, rounds and heads, and
gradients are backpropagated through each 50-round training sequence. Hidden
state starts fresh for each independent sequence. The decoder does not receive
future syndromes, hidden displacements or true labels at inference.

The methodological precedent is [Baireuther et al., *Quantum* 2, 48
(2018)](https://quantum-journal.org/papers/q-2018-01-29-48/), especially
[Appendix A](https://ar5iv.labs.arxiv.org/html/1705.07855). They train an LSTM
surface-code decoder on syndrome histories and parity labels with
cross-entropy, validation-based selection, and separate evaluation data.
Their architecture, code, data volumes and training settings differ from
ours; this sweep is not a reproduction of their experiment.

For GKP codes, [Wan, Neville and Kolthammer, *Physical Review Research* 2,
043280 (2020)](https://arxiv.org/abs/1912.00829) demonstrate benefits from
combining syndrome information across rounds using Bayesian estimation.
This supports the motivation for using history, not the efficacy of our
particular LSTM. Our optimizer is the standard decoupled-weight-decay method
of [Loshchilov and Hutter, *Decoupled Weight Decay Regularization*
(2019)](https://arxiv.org/abs/1711.05101).

The periodic feature encoding, five parity heads, and numerical training
settings are project choices being evaluated, not literature-proven optima.
All 36 models train at the same nominal noise setting. Consequently the
off-nominal heatmaps measure distribution-shift robustness as well as model
capacity: failure there does not establish that the architecture could not
learn that regime with appropriate training data. Likewise, a common
40-epoch budget compares performance under that budget; it does not establish
each architecture's best attainable performance or convergence. The simulator
is a displacement/parity model, not a full finite-energy oscillator or
hardware validation.

## Run and resume

From the repository root:

```sh
.venv/bin/python -m experiments.lstm_size_sweep
```

CUDA is selected when available; otherwise Apple MPS, then CPU. An explicit
`--device cpu|mps|cuda` is supported. Results are separated by a configuration
digest including source hashes and device. Repeating an unchanged command
resumes at the last saved epoch and reuses completed evaluations. Every epoch
preserves weights, optimizer, scheduler, shuffle state, best weights and
history. Finished checkpoints are distinct from partial resume files.

Small validation, isolated from full results and the viewer:

```sh
.venv/bin/python -m experiments.lstm_size_sweep --smoke --only h128_l2_s0 h32_l1_s0
```

`--only` can select individual models without changing the shared protocol.
Smoke mode uses 256 training and 64 validation sequences for two epochs and
never counts toward full coverage.

## Artifacts and completion

Each `results/lstm_size_sweep/full/<digest>/` contains the plan, status,
per-epoch resume checkpoints, final checkpoints with training histories,
reference validation, per-cell packed logical-error records and summaries.
The original `checkpoints/lstm_*.pt` files are preserved.

After each model finishes, its 61 evaluations are embedded in the offline
walkthrough. Reload `viewer.html` and choose a **LSTM sizes** seed selector.
The width/depth coverage table distinguishes pending entries from evaluated
ones. The final ZIP is rebuilt at the end of the selected run.

Full coverage requires all 36 valid final checkpoints and all 2,196 measured
evaluations. A launch, smoke test, time-limited checkpoint, or missing cell
does not meet that condition. `completion.json` records the completion check.

Audit all checkpoints and raw records after completion:

```sh
.venv/bin/python -m experiments.audit_lstm_sizes --run results/lstm_size_sweep/full/<digest>
```

This verifies the source/checkpoint hashes, unchanged original comparisons,
training completion rules, chosen heads, reference checks and all recorded
counts. It writes `audit.json` and `size_comparison.csv`. `--partial` audits
fully evaluated models during the run. CSV grid means are descriptive averages
over these 61 equally weighted settings, not expected error under a physical
distribution of operating conditions.

## Completed run: 18 November 2026

**Belief change — suggestive within this protocol.** More width improved
nominal-noise performance, but neither extra depth nor the largest parameter
count guaranteed the lowest error across the noise grid. Increasing width from
32 to 256 reduced nominal P_L(50) in all nine comparisons holding depth and
seed fixed. This is evidence about the specified training budget and simulator,
not each architecture's best attainable performance or hardware performance.

Run `9aab4963bc37dcea` completed all **36 models and 2,196 model/setting
evaluations**. Thirty-two models reached 40 epochs and four used the registered
validation early-stopping rule. All 72 fast/reference checks agreed exactly.
The audit reproduced every recorded count from the raw trial records, preserved
the original 20-decoder comparisons, and verified the walkthrough and ZIP.
The metrics dashboard, its 21 source hashes, plotted tables and ZIP were also
verified against the completed snapshot. No training or evaluation failures
were reported. Training took 259.4 minutes on Apple MPS, excluding evaluation
and packaging.

For one-layer models, the nominal error averaged across three seeds fell from
**7.715% at width 32** (seed range 6.055–8.887%) to **0.488% at width 256**
(0.293–0.586%). The lowest nominal architecture mean was **0.391% for width
256, depth 3**, with 3, 3 and 6 errors in 1,024 trials for the three seeds.
Seed ranges are descriptive spread, not confidence intervals; each model's
pointwise 95% Wilson interval remains available in the viewer and exports.

The lowest equal-weight grid mean was **27.289% for width 256, depth 1**
(seed range 26.479–27.746%), versus 27.515% for width 256, depth 3
(27.392–27.750%). These overlapping ranges and exploratory comparisons do not
establish a decisive winner among the larger models. The grid mean weights
all 61 settings equally and is not an operational noise distribution.

The 256-by-3 model illustrates the failure boundary: holding σq/L = 0.20 and
σm/L = 0.05 fixed, changing ρ from 0.90 to 0 raised its three-seed mean
P_L(50) from 0.391% to **47.884%**. At ρ = 0.90, increasing σm/L to 0.16
gave **51.432%**. Matched Bayes also had high error in these difficult settings
(423/1,024 and 490/1,024 respectively), so these examples do not isolate
training mismatch from intrinsic decoding difficulty.

Evidence: [audit](../results/lstm_size_sweep/full/9aab4963bc37dcea/audit.json),
[per-seed results](../results/lstm_size_sweep/full/9aab4963bc37dcea/size_comparison.csv),
and [architecture means and seed ranges](../results/lstm_size_sweep/full/9aab4963bc37dcea/architecture_comparison.csv).
Per-cell records and checkpoints are in the same run directory.

**Next decision:** retain this as the completed capacity baseline. A subsequent
robustness experiment should vary training-noise coverage while holding the
architecture fixed, with a separate held-out evaluation before claiming a
general improvement.
