# Accuracy decision: adopt the smooth measurement implementation

Belief change: the continuous Mehler resource with a matching inverse-CDF
sampler meets the registered accuracy and numerical-stability requirements.
Adopt it for new quantum-reference runs. The binned approximation remains an
explicit legacy mode. This decision concerns implementation accuracy for the
chosen idealized GKP protocol, not hardware validation or decoder superiority.

The preregistered experiment completed in569.3 seconds:
`results/smooth_ancilla_experiment/main/06c34206dd1a1b00`.
[HTML review](../results/smooth_ancilla_experiment/main/06c34206dd1a1b00/review/index.html)
contains figures, raw-metric exports, numerical gates, and method/metric citations.
The frozen [protocol](smooth_ancilla_experiment.md) was written before the new
observations; source snapshots and raw hashes are preserved.

## Primary accuracy result

The primary metric is reference-probability-weighted squared conditional-state
fidelity, A=integral p_Q(y)|<psi_Q(y)|psi_model(y)>|^2dy. We report its minimum
over24 fresh input/axis/resource combinations, using96 data levels and
ancillary refinement512 to768. The squared pure-state fidelity convention is
standard; the reference weighting and99.999% acceptance threshold are our
engineering choices. [Jozsa (1994)](https://doi.org/10.1080/09500349414552171).

| Implementation | Minimum observed accuracy A | Maximum outcome TV |
| --- | ---: | ---: |
| Historical bins, dx=.003 | 99.9978529259% | .002075096 |
| Continuous resource and sampler | 99.9999999615% | 7.64244e-7 |

The continuous arm passes A>=99.999% and TV<=1e-4 in every held-out case.
Its worst infidelity is3.85059e-10. Allowing for the observed ancilla/grid
refinement sensitivity gives a minimum accuracy summary of99.9999171%, still
above the registered threshold. This sensitivity summary is not a rigorous
bound on the infinite-cutoff limit. Deterministic quadrature results have no
Monte Carlo confidence intervals.

Sampler validation: maximum inverse-CDF residual1.99840e-14, below1e-10.
All24 empirical checks pass the strict familywise95% DKW–Massart criterion:
maximum CDF distance.00637173, bound.00723813, with65,536 independent draws
per combination. The mixture density is also checked by independent
quadrature. [Massart (1990)](https://doi.org/10.1214/aop/1176990746).

## Eight-cycle results

Nearest-lattice controller, nominal noise(.20,.05,.90), Delta=.30,
seed26111901,512 trajectories per cardinal input state (3,072 total), eight
q/p cycles. Historical binned controls are reused after hash and exact
displacement-array checks. Stage energies, boundary mass, full homodyne
outcomes and log densities are recorded in the new run.

| Cutoff | Smooth cycle-end horizon | Every substep passes through8 | Smooth cycle8 logical fidelity |
| --- | ---: | --- | ---: |
| 3072 | 6 | No | 87.6160% |
| 4096 | 8 | Yes | 88.1578% |
| 5120 | 8 | Yes | 88.1566% |

Every pointwise paired4096-to5120 fidelity interval fits within+/-1 percentage
point. At cycle8 the difference is-0.001219 percentage points,95% interval
[-0.465922,+0.463485]. All registered high-cutoff population and norm gates
pass at every logged substep. Do not use3072 levels for an eight-cycle physical
claim merely because the smooth resource is more accurate.

At5120, smooth minus binned logical fidelity is+0.188692 percentage points,
95% interval[-0.115726,+0.493111]. This does not establish a logical-memory
performance improvement. The adoption decision is based on accuracy against
the independent circuit and numerical stability, not on a larger memory score.

## Root-cause evidence

The historical selected trajectory1659 is reproduced to floating-point
precision by the instrumented binned replay. Its catastrophic increase occurs
at the p measurement of cycle6. At5120 levels, cycle-end energy changes from
46.53 to3080.67 photons in the binned replay; the continuous replay changes
from46.50 to216.82 and ends at82.77 photons by cycle8. Other selected hot
trajectories continue growing, consistent with the unstabilized protocol's
energy accumulation.

At5120, the selected binned trajectory reaches1.5687% boundary population
after q feedback in cycle7, which its historical cycle-end-only gate missed.
The continuous ensemble passes the stronger substep gate; its worst substep
boundary population is.00151684%. This illustrates why norm preservation and
cycle-end mean fidelity alone do not certify energy tails.

Across all3,072 trajectories, each tested cutoff has one binned trajectory
with a cycle energy increase above1000 photons and zero in the continuous
run. These are descriptive counts using a diagnostic threshold, not an
additional registered adoption metric. The intervention changes the resource
representation and matching sampler together; it does not independently
assign their contributions or prove that every rare event is numerical.

## Method, implementation and validation

The method is finite-energy GKP ancilla-assisted SUM/homodyne syndrome
extraction with conditional Kraus updates and nearest-lattice feedback.
See [Gottesman, Kitaev & Preskill (2001)](https://doi.org/10.1103/PhysRevA.64.012310),
[Grimsmo & Puri (2021)](https://doi.org/10.1103/PRXQuantum.2.020101),
and [Matsuura, Yamasaki & Koashi (2020)](https://doi.org/10.1103/PhysRevA.102.032408).
The Gaussian-mixture inverse-CDF sampler is our derived implementation, not
a claim to reproduce a published performance number. Full two-mode QuTiP
evolution and its separate wavefunction kernel provide the independent control.

New CLI runs use `--measurement-resource smooth` by default. Historical runs
can be reproduced with `--measurement-resource binned`; `--dx` is restricted
to that mode. New directories/configuration fields prevent merging the
implementations, and convergence analysis rejects cross-resource comparisons
as cutoff refinements. The legacy bin-refinement finalizer explicitly requires
binned data. Historical orchestration commands now request binned mode.

Validation:23 tests pass, including independent q/p circuit checks, continuous
density/quadrature/inverse checks, extreme quantiles, negative controls and
instrumented replay regression. Both CLI modes execute successfully; their
deliberately undersized96-level smoke runs correctly fail population gates.
Scope remains nominal nearest-lattice behavior through eight cycles at the
validated high cutoffs, plus the fixed-input accuracy tests. Loss, dephasing,
energy stabilization, hardware matching and decoder retraining remain future
experiments.

## Bibliographic correction — 19 November 2026

The original response and report misattributed PRX Quantum2,020101 (2021).
Its authors are Arne L. Grimsmo and Shruti Puri. The active results text and
report generator now use the corrected attribution; the frozen pre-run
protocol/source snapshots retain their original wording. This correction
changes no methods, metrics, thresholds or measured results. See the
[research notebook](../RESEARCH_NOTES.md) for the correction record.
