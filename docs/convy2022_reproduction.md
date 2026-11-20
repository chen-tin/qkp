# Convy 2022: first reproduction checkpoint

Registered before training on 18 November 2026. Status: planned.

## Question and scope

Can an independently implemented supervised LSTM reproduce the final-fidelity
result in Convy et al., *New Journal of Physics* 24, 063019 (2022), Figure 2,
Scheme A at gamma = 0.04 per microsecond? The control is their empirical
eight-state Bayesian filter; the intervention is their 32-unit, two-layer LSTM.
This validates our ability to reproduce a published sequence-decoding result
before extending the separate GKP study. It does not validate GKP noise values.

Prior work searched: local experiments, docs, results, and the authors' public
source links. Our completed GKP width/depth sweep uses different physics,
labels, features and metrics and cannot answer this reproduction question.
No earlier local reproduction of this paper was found.

Primary sources: [paper](https://arxiv.org/html/2110.10378v2),
[journal DOI](https://doi.org/10.1088/1367-2630/ac66f9), and
[author's Bayesian code](https://github.com/IanConvy/bayesian-error-correction/tree/9bb00616add5fb8cd004f7ca64d14f7803dfa33b).
Source snapshots, hashes, extracted Figure 2 coordinates and retrieval notes
are in `reproductions/convy2022/reference/`. The original LSTM Colab link
returns 404 through both the Drive connector and public file export.
Consequently this is an **independent methods reproduction**, not a rerun of
the authors' exact training code or original random samples.

## Matched methods

- Three-qubit bit-flip code, all eight initial basis states equally represented.
  Known initial state is supplied to both decoders. Measurement happens after
  each interval's flips. The LSTM receives two measurements and the scalar
  initial state at every step; no future measurements or current true state.
- Scheme A: independent Poisson flips, gamma = 0.04/us; dt = 0.032 us;
  625 steps = 20 us; syndrome means +/-1; independent Gaussian measurement
  noise variance 1/(4.7 * 0.032) = 6.64893617. Odd-Poisson probability sampling
  is mathematically identical to sampling integer counts and taking parity.
- Training on 100,000 trajectories; testing on 30,000 new trajectories.
  Bayesian transition probabilities and state-conditioned Gaussian means and
  covariances are fitted on the same training set. Full 2x2 covariances follow
  the available author implementation at history depth zero.
- Two stacked LSTM layers with 32 hidden units, linear eight-state readout,
  softmax classification; 13,448 trainable parameters, matching Appendix D.
  Cross entropy averages across trajectories and all time steps; Adam optimizer.
- Primary outcome: fraction of exactly correct final basis-state predictions,
  equivalent to final recovery fidelity for this basis-state tracking task.
  All-step accuracy is secondary. We do not use a majority-vote relaxation.

## Explicit reconstruction choices and limits

The unavailable notebook prevents recovering selected batch size, learning
rate, epoch count, initialization, validation split and random seeds. Choose
Adam lr=0.001, batch=512, gradient norm cap=1, standard PyTorch initialization,
no dropout or weight decay. Use an additional independent 10,000-trajectory
validation set. Reduce learning rate by half after five plateau epochs,
minimum 1e-5. Stop after at least 20 epochs and 12 epochs without validation-CE
improvement >1e-5, or at 100 epochs; restore lowest validation CE. These are
reproduction settings, not settings claimed from the paper. Full-sequence
backpropagation through all 625 steps; no truncated history or feature changes.

Training seeds 0, 1, 2 share data; data streams are fixed at train=210378,
validation=210379, test=210380. Three seeds quantify initialization/training
variability, not three independent test datasets. Keep each seed separately;
do not pick the best seed using the test results. No test-driven retraining.
Epoch checkpoints save optimizer, scheduler and shuffle RNG. There are no
stochastic model layers. Device and library versions are recorded per run;
MPS kernels need not be bitwise deterministic.

The paper caption names initial state 000 while the methods specify all eight
states equally represented; follow the methods and retain per-state labels.
Use the paper's positive stabilizer sign convention (the public empirical
simulator uses a global negative sign). Both are equivalent after sign
reversal; our independently implemented generator follows Scheme A directly.
The author's tracking helper initializes from the first label; ours uses the
actual pre-measurement initial state so first-interval flips are not leaked.

This first checkpoint excludes other gamma points, Schemes B/C/D, double
threshold tuning, active correction, T1 extension and quantum annealing.
They remain unrun. Do not describe this checkpoint as the full paper reproduced.

## Predeclared interpretation

Extracted Figure 2 Scheme A targets at gamma=.04/us: Bayesian 0.8868 and
LSTM approximately 0.8835667. These are vector-digitized published points,
not raw author data. Extraction tolerance is 0.0002 absolute. Paper error bars
are SEM from 30,000 trajectories. Use Bernoulli SEM and 95% Wilson intervals
for our final fidelity; retain paired per-trajectory differences vs Bayesian.

Numerical compatibility for a model means distance from its published point
is at most 1.96 * sqrt(our_SEM^2 + paper_SEM^2) + 0.0002. For the LSTM, report
each of the three seeds' compatibility separately. The practical hypothesis
is that LSTM final fidelity is within one percentage point of matched Bayes.
Support requires each seed's paired 95% interval to lie within +/-0.01 and
at least two of three seeds to be numerically compatible with the paper.
Contradiction requires Bayesian compatibility but all three seeds to be more
than one percentage point worse than Bayes with paired intervals wholly below
-0.01. All other cases are inconclusive; investigate protocol/convergence
before any further experiment, and do not tune against this held-out test.
These are checkpoint criteria, not a universal statistical replication standard.

If the Bayesian control is incompatible, investigate the simulator, convention
and metric before interpreting LSTM discrepancies. Infrastructure errors,
nonfinite losses and failed source/data checks are invalid runs, not model
failures. An epoch-budget stop is reported as such, not called convergence.

## Commands and artifacts

From the `qkp` repository root, dependencies already in `.venv`:

```sh
.venv/bin/python -m pytest tests/test_convy2022_reproduction.py -q
.venv/bin/python -m experiments.reproduce_convy2022 --smoke --seeds 0 --device cpu
.venv/bin/python -m experiments.reproduce_convy2022
```

The smoke run exercises one Bayesian control and one LSTM intervention with
512/128/256 trajectories and two epochs. Its statistics do not count as a
reproduction. Before scaling, an independent agent should execute these
documented smoke instructions and check the outputs.

Full runs live in `results/reproduce_convy2022/anchor/<digest>/`, separately
from smoke results and all existing GKP runs. Repeating an unchanged command
resumes. `--seeds 0` runs one planned seed; `--baselines-only` evaluates just
the control. `--device cpu|mps|cuda` selects the backend. `--output` is useful
for isolated checks; it refuses to mix protocols in an existing directory.

Every run saves the protocol and source hashes before data generation, dataset
hashes and arrays, fitted baseline, training histories, best and resume
checkpoints, predictions for every test trajectory/time step, and intervals.
Completion requires all three final checkpoints and evaluations, plus a
separate audit against the protocol and recorded numerical predictions.
