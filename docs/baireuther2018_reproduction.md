# Baireuther 2018 Figure 3 reproduction

Registered 18 November 2026 before production training. Status: planned.
Question: can the published two-network LSTM recover epsilon=0.00209 logical
errors/cycle on the paper's Surface-17 Pauli circuit? This is separate from
both the current GKP sweep and the Convy continuous-measurement experiment.
Prior local search found no surface-code reproduction. Published source is
pinned in `reproductions/baireuther2018/reference/provenance.json`.

## Published protocol

Use the seven-step circuit and ancilla reuse in the author's `1705p07855`
circuit branch, with px=py=pz=0.00048 independently at each circuit location
and pm=0.0014. Preserve CNOT order, data readout before measurement-step idle
errors, syndrome second differences, final increments and parity of all nine
data-qubit bit flips. Our batched frame simulator is checked against the
original code using identical random fault draws, including at high rates.

The neural decoder has two separate networks, each with two LSTM layers of
width 64 and a 64-unit ReLU dense layer. The first uses the full syndrome
history; the second uses the last three cycles and final four syndrome
increments. Their probabilities combine by probabilistic XOR. Dropout=.2
after both LSTM outputs and the hidden dense layer; L2 penalty=1e-5 times
half the squared norm of all dense/output weights. Adam lr=.001, batch=64.
Preserve TF BasicLSTMCell's constant +1 forget bias and single trainable bias.

Train on 4,000,000 sequences, lengths 11–20 equally represented. Each epoch
uses 10,000 batches (640,000 examples). Validate on 10,000 distinct sequences,
lengths 81–100. Group validation fidelity by length and fit the per-cycle error
rate. Stop only after 100 epochs without a strictly lower fitted validation
error rate; preserve the best checkpoint. Train seeds 0,1,2 and select the
seed using that fitted validation rate. Test on 50,000
new 300-cycle histories at t=2+n(n+1)/2 <=300, fitting
F(t)=.5+.5(1-2 epsilon)^(t-t0). Bootstrap whole trajectories 1,000 times.

Match the executable author's unweighted point-fit bounds:
epsilon in [.0001,.1], t0 in [.0001,10], default SciPy initialization.
Their validation helper intends to pass t0_max=.01 but passes it into the
n_sampling argument position; actual t0_max remains 10. Preserve that actual
behavior explicitly. Bootstrap fit bounds are epsilon in [0,.99], t0 in [0,10].
Prediction ties at probability .5 are classified as odd parity.

The source rebuilds the TensorFlow graph for longer validation histories and
restores only neural weights, resetting Adam state after every epoch. Preserve
this executable behavior by recreating Adam each epoch. Match TF1's epsilon
placement using PyTorch eps=1e-8/sqrt(1-.999^update_number) within each epoch.
These are source-code details absent from the paper's prose, not new tuning.

## Reconstruction choices and boundaries

This is a PyTorch port, because the original TensorFlow 1 contrib runtime is
unavailable in the current Python environment. The neural and circuit source
snapshots remain unmodified. Seeds and random streams are newly chosen; the
paper did not publish trained weights or datasets in the inspected branch.
Each seed shares the same frozen data and differs in initialization, dropout
and batch sampling. CPU uses one thread for these small batches; MPS is
optional. Data generation is batched and float32 uniform draws replace the
original float64 RandomState draws; the stochastic model is preserved, but
original trajectories are not recreated. The exact-fault reference test uses
float64 supplied draws and demands equality of all output bits.

No standard Stim-generated circuit or generic depolarizing model substitutes
for the seven-step source circuit. The published blossom result .00274 is
a reference line, not a measurement of a new MWPM implementation. A fresh
MWPM baseline is not claimed until one is implemented and checked separately.
This first target excludes Figure 4's py sweep and the density-matrix hardware
model in Figure 5.

## Decision and execution

Paper Figure 3 reports LSTM epsilon=.00209. Numerical reproduction is supported
if this lies in the 95% bootstrap interval enlarged by .000005 for printed
rounding, and the source-equivalence checks pass. It is contradicted at this
checkpoint if the interval excludes the target; incomplete training, failing
controls or invalid fits are inconclusive. Do not shorten the published
stopping rule and call it complete. Record hardware/software failures separately
from valid model errors. Do not inspect test metrics during training.

Before scaling, independent validation executes both the no-fault control and
the noisy learned decoder smoke run from these commands:

```sh
.venv/bin/python -m pytest tests/test_baireuther_reproduction.py -q
.venv/bin/python -m experiments.reproduce_baireuther2018 --smoke --seeds 0
.venv/bin/python -m experiments.reproduce_baireuther2018
```

Smoke uses 1,280/128/256 train/validation/test trajectories, 10 batches/epoch,
and two epochs. It exercises the pipeline only. Full training can take hours:
100 unsuccessful validation epochs alone require one million updates per seed,
and improvements restart the patience counter. `--data-only` generates and
hashes the full data without training. `--seeds 0` runs one planned seed.

Results are isolated under `results/reproduce_baireuther2018/<profile>/<digest>`.
Run the same command to resume from the last epoch, including CPU/MPS dropout
RNG. Optimizer snapshots are retained for audit; the published source's epoch
boundary reset is applied on continuation. Files include immutable plan, data manifests/arrays,
per-epoch checkpoints, best completed checkpoints and histories. Only after
all three seeds finish is the validation-selected decoder tested and its
fidelity curve, fitted error rate and bootstrap interval written to result.json.
