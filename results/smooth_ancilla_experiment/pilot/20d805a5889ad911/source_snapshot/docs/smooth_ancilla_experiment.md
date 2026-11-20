# Smooth finite-energy measurement: accuracy decision experiment

Registered locally before the new runs, 19 November 2026. This continues the
measurement benchmark and root-cause audit; no Solar writes or external
uploads are involved. Historical observations and their source remain frozen.

## Method and metric

The physical method is finite-energy GKP syndrome extraction with a fresh
ancilla, ideal SUM interaction, homodyne measurement and nearest-lattice
feedback. Its reduced q instrument is M_y = phi(y-q); p extraction is the
Fourier conjugate. The smooth phi is the normalized Mehler representation
of exp(-t n)|+ideal>, t=atanh(2 sigma_m^2). This follows the GKP framework
and finite-energy constructions; our analytic Gaussian-mixture inverse-CDF
sampler is a numerical implementation derived below, not a published
performance reproduction.

Primary accuracy metric, fixed BEFORE these new observations:

    A = integral p_Q(y) |<psi_Q(y)|psi_M(y)>|^2 dy.

The normalized QuTiP outcome distribution weights the pure-state squared
fidelity. This is our reference-weighted aggregation of the standard squared
state fidelity F(psi,chi)=|<psi|chi>|^2. It is simulator agreement, not decoder
classification accuracy. Report the minimum A over all held-out fixed inputs,
axes and resource widths, not only their average. Reference numerical
sensitivities are reported separately; no deterministic-quadrature confidence
intervals are invented.

Literature: [Gottesman, Kitaev & Preskill (2001), PRA 64, 012310](https://doi.org/10.1103/PhysRevA.64.012310);
[Terhal, Conrad & Vuillot (2021), PRX Quantum 2, 020101](https://doi.org/10.1103/PRXQuantum.2.020101);
[Matsuura, Yamasaki & Koashi (2020), PRA 102, 032408](https://doi.org/10.1103/PhysRevA.102.032408);
[Jozsa (1994), J. Mod. Opt. 41, 2315–2323](https://doi.org/10.1080/09500349414552171).
The sources justify the method and state-fidelity convention. The particular
reference weighting and tolerances below are our engineering choices.

## Prior work and decisions

The completed two-mode benchmark `main/5f1d5766c1aa2659` found worst weighted
infidelity 2.16787e-5 for dx=.003, 5.38942e-6 for dx=.0015, and 3.85760e-10
for smooth evaluation. It did not include a smooth sampler or multi-cycle
experiment. These known results motivate the thresholds; they are not blind
predictions. The binned eight-cycle control at N=3072 fails population gates
from cycle 6, while N=4096/5120 pass mean-fidelity cutoff equivalence. These
controls will be reused with hash verification, not relabeled as new runs.

Adopt the smooth implementation for new reference runs only if:

1. Every held-out QuTiP comparison has A >= .99999 and outcome total variation
   <=1e-4, after the reference's normalization and ancilla-refinement gates.
2. The optimized amplitude agrees with the previously validated smooth sum
   to absolute error <1e-12 over a broad deterministic position grid.
3. Inverse-CDF residual max |CDF(sample(u))-u| <=1e-10 for fixed random and
   extreme quantiles at both benchmark widths; independent quadrature agrees
   with the mixture CDF. Empirical sampler checks use a familywise 95% DKW
   bound over all tested input/axis/resource combinations, with 65,536 draws
   per combination (Massart, 1990, doi:10.1214/aop/1176990746).
4. The eight-cycle smooth run at N=4096 and 5120 passes the existing population
   gates (ensemble mean top-decile mass <1e-4 and every trajectory <.01), norm
   error <1e-7, and every pointwise paired fidelity-difference 95% interval
   lies inside +/- .01. The intervals use independent trajectories, stratified
   equally over the six cardinal inputs, as in the original reference.

Support: all four requirements pass. Contradict the proposed implementation
if reference-valid fidelity or sampler-accuracy checks fail beyond numerical
sensitivities. Otherwise mark the decision inconclusive and retain the
validated scope; do not switch models on memory fidelity alone. If only the
long-horizon cutoff gate fails, smooth evaluation may remain preferred for
diagnostics, but eight-cycle physical claims stay unvalidated. No automatic
unbounded cutoff or sampling escalation is scheduled.

## Continuous sampling derivation

Let v=tanh(t), mu_j=j L/cosh(t), w_j=exp[-v(j L)^2/2], and
phi(x)=C sum_j w_j exp[-(x-mu_j)^2/(2v)]. Then |phi|^2 is a positive Gaussian
mixture with variance v/2 in every component, means (mu_j+mu_k)/2, and weights
proportional to w_j w_k exp[-(mu_j-mu_k)^2/(4v)]. Terms sharing j+k are
combined. The normalized mixture CDF is inverted with safeguarded Newton
iterations. Thus sampling and amplitude evaluation use the same continuous
density. Fast amplitude evaluation omits only Gaussian tails below its fixed
1e-16 target and is checked against the full smooth sum.

Use the same first ancilla uniform as the historical sampler for inverse
CDF sampling; the old second jitter uniform is unused. Keeping all original
draw arrays preserves common-random-number coupling. It does not guarantee
identical measurement histories or identical trajectories across cutoffs.

## Fixed settings and sequence

Validation: six fresh displaced cardinal states, seed 26111923 (different
from the old measurement benchmark), data cutoff 96, ancillary cutoffs 512
and 768, sigma_m/L=.05 and .16, both axes. Full two-mode QuTiP sesolve and
QuTiP wavefunctions are independent of the local updater. Use the prior
cell-split outcome rule, radius36, maximum dy=.005, with its nested coarse
rule for numerical sensitivity. QuTiP vern9 atol1e-11/rtol1e-10;
normalize_output=False. A broad-width, small-cutoff pilot is independently
run from this document before scaling.

Long-horizon: nominal noise (.20,.05,.90), nearest lattice, Delta=.30,
seed26111901, 512 trajectories for each of six inputs, eight cycles,
cutoffs3072/4096/5120, logical quadrature192. Run the smooth arm with
stage-level energy and boundary mass after displacement, each measurement,
and each feedback; save unwrapped homodyne outcomes and log probability
densities. Compare with frozen binned controls e042e0a1b30d34d0 (3072/4096)
and dd4e949e01da259d (5120). Re-run the three predefined highest-boundary-risk
indices [1659,1973,2776] with binned and smooth instruments and the original
full-batch random streams for forensic logging; check reproduction against
the saved binned histories. This is a selected diagnostic sample.

Secondary outcomes: six-state logical memory fidelity, energy mean/quantiles,
boundary mass, maximum per-cycle energy jump, and stage/outcome likelihoods.
The tested causal statement is limited to whether replacing binning removes
the observed defect under this fixed protocol; no claim of universal GKP
robustness follows. Do not add damping, loss, energy feedback or decoder
retraining. Any future physics intervention remains Planned.

Commands (from repo root, one BLAS thread):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_continuous_ancilla.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.smooth_ancilla_experiment --pilot
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.smooth_ancilla_experiment
```

Sources, versions, run identities, raw hashes, intermediate results and a
self-contained HTML decision report are saved with each run. No observations
are excluded merely for failing a numerical gate.
