# Explicit QuTiP versus local measurement engine

Registered locally before running, 19 November 2026. This is a bounded
measurement-implementation benchmark, following the planning, defect
separation and independent pilot checks in the experiment-plan skill. No
Solar record or external upload is requested or created.

## Question and prior work

Does the current position-bin lookup reproduce an explicit two-oscillator
QuTiP SUM/homodyne circuit, and does smooth Mehler evaluation improve the
agreement? This comparison can help other oscillator simulators distinguish
measurement approximation errors from finite ancillary Hilbert spaces.

Prior work: `tests/test_qutip_reference.py` has a fixed-outcome two-mode check
at a broad ancilla width. The eight-cycle reference's N=3072 population gate
fails at cycle 6; N=4096/5120 pass average-fidelity cutoff equivalence. The
measurement-discretization audit identifies discontinuities but has not
established the cause of rare long-horizon events. Existing data and the
baseline `quantum/qutip_reference.py` remain unchanged.

## Shared methods

Units: [q,p]=i, L=sqrt(pi), data Delta=.30, six cardinal input kets. Each input
gets a fixed, seeded displacement (seed 26111907, independent q,p Gaussian
draws of standard deviation .20 L), generated once and shared by all arms.
The same finite data operator, input vector, homodyne outcomes, nearest-lattice
feedback and logical scoring are held fixed. There is one q OR p extraction
per test, not a complete q/p correction cycle or a multi-cycle decoder trial.
Temporal correlation is not tested by a single extraction.

Control: QuTiP `sesolve` evolves the full data-plus-ancilla ket with H=q_d p_a
for q extraction or H=-p_d q_a for p extraction, for unit time. The p ancilla
and measurement bra receive the appropriate Fourier phases. The ancilla
coefficients and continuous homodyne bras are computed with QuTiP's own
harmonic-oscillator wavefunction recurrence, the same kernel used by its
`HarmonicOscillatorWaveFunction` class. This avoids the local bin lookup and
local Hermite recurrence. Projection and integration over recorded outcomes
are benchmark code; this is not a packaged QuTiP GKP decoder. Common input
preparation, feedback and score operators deliberately isolate the extraction.

Interventions, isolated at identical data cutoff and outcomes:

| Arm | Change |
| --- | --- |
| local_binned_0.003 | Existing `Ancilla.amplitude`, unchanged |
| local_binned_0.0015 | Halve its lookup-bin width only |
| local_smooth | Analytic continuous Mehler amplitude, with analytic normalization; evaluation only, no replacement sampler |

The smooth arm is an experimental diagnostic; no production sampler or prior
benchmark is silently replaced. For p extraction use the same scalar
wavefunction in the momentum eigenbasis.

## Bounded settings and convergence

Pilot: data cutoff 16, ancilla cutoffs 64 and 128, sigma_m=.5 in absolute
quadrature units, both axes and all six inputs, y in [-18,18], dy=.02.

Main: data cutoffs 64 and 96; ancilla cutoffs 256, 512 and 768;
sigma_m/L=.05 and .16 (nominal and high-readout resource widths); both axes,
all six inputs; y in [-36,36], maximum dy=.005. Split the trapezoid rule at
every nearest-lattice feedback boundary, using separate one-sided endpoint
values. The coarse rule uses every other node within each cell (maximum
dy=.01) for an integration-refinement check. Retain all cases, including
nonconverged ones, and label them. Data cutoffs are small on purpose: this
tests agreement for the same finite data operator, not data-basis convergence
of an eight-cycle physical memory.

QuTiP uses `vern9`, atol=1e-11, rtol=1e-10, normalize_output=False,
nsteps=100000. Check the tight-solver setting (atol=1e-13, rtol=1e-12) for
one nominal input at the largest data/ancilla cutoff on each axis.

Before interpreting a reference case require integrated outcome mass within
1e-5 of one and joint-state norm error below 1e-7. Require the final ancilla
refinement (512 to 768) to change total-variation distance by less than 1e-4,
reference-weighted conditional infidelity by less than 1e-5, expected energy
by less than .01 photons and expected logical fidelity by less than 1e-4,
for every input/axis/resource at fixed data cutoff. Integration refinement
must change expected energy by less than .01 photons and logical fidelity by
less than 1e-4; separately report its effect on comparisons to the binned
arms. If these gates fail, report numerical agreement as inconclusive for
that configuration. Do not automatically increase budgets.

## Hypothesis and decision rule

Pre-run hypothesis: once the reference gates pass, smooth evaluation will
agree with the explicit circuit to outcome TV <1e-4 and probability-weighted
conditional infidelity <1e-5. Bin lookup may disagree more strongly even
when average logical fidelity barely changes. This would support a local
discretization contribution in the tested measurement, not establish the
cause of the previously observed cycle-6 event.

Support: the smooth arm meets both thresholds in every reference-valid
configuration. Contradict: any reference-valid configuration robustly exceeds
them beyond the measured numerical sensitivities. Inconclusive: a reference
or integration gate fails, or sensitivities straddle a threshold. A reduction
in binned-arm discrepancies is descriptive unless it exceeds those same
sensitivities. A small/absent difference is a valid result.

## Metrics, uncertainty, costs and evidence

Primary metrics: integrated outcome-distribution total variation and
reference-probability-weighted conditional pure-state infidelity at common y.
Use all grid nodes with nonzero reference probability; probability weighting
prevents near-zero-density nodes from dominating. Also report the worst
conditional infidelity among nodes carrying at least 1e-8 probability mass.
Secondary: expected logical fidelity and photon number after common feedback,
boundary population, normalization, reference ancilla edge population, and
runtime. Outcome densities are normalized within the integration window for
TV; record the raw integrated masses alongside them.

These are deterministic quadrature results for six fixed inputs: there are no
Monte Carlo confidence intervals or population-generalization claims. Report
the ancilla, grid and ODE refinements as separate numerical sensitivities.
Average across the six states only after checking each state's gates.

Time circuit propagation plus outcome projection for QuTiP and amplitude
evaluation plus conditional-ket formation for local arms. Exclude common
feedback/scoring, input construction and precomputed quadrature bases from
that comparison; report all setup times separately. At the largest ancilla
cutoff repeat each arm three times, report median and min/max. Lower
ancilla cutoffs and the pilot use one repeat. These are batch-throughput
measurements on this computer, not single-shot decoder latency.

Save configs, source snapshots/hashes, package versions, all input kets,
displacements, joint output coefficients, outcome densities and conditional
metrics. Source/unit tests must detect a wrong SUM sign and test p Fourier
conventions before the main run. An independent agent runs the documented
pilot control and interventions before scaling. Planned until launched;
conclusions are appended only after the saved result is complete.

Pilot-driven numerical amendment, before the main run: the original uniform
outcome rule completed and validated the measurement reduction, but failed
the post-feedback integration gate (energy change up to .01290 photons and
logical fidelity change .000647 at dy=.02 versus .04). Its frozen run is
`pilot/5f7b6276e80e413b`. Feedback has jumps at half-integer lattice cells;
the amended quadrature integrates each cell separately and evaluates its
one-sided endpoint limits. This changes numerical integration only, retaining
the original thresholds. Rerun the independent pilot after the amendment.

The revised pilot `264018cc5fc6057e` passes the norm, probability, ancillary
cutoff and energy-grid checks; its q-axis Z-minus logical-grid change
(.000128) remains above .0001. The independent validator supports the
registered main diagnostic at its four-times-finer spacing, retaining the
same gates. The first main attempt `4ab0843f59a999a7` was killed with exit
137 during basis setup under severe machine memory pressure; it produced no
measurement observations. A memory-allocation amendment preallocates arrays,
shares the largest quadrature basis across cutoffs and processes local arms
sequentially. Physics, outcome nodes and thresholds are unchanged. Preserve
the incomplete attempt and identify shared-machine limitations in timings.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_qutip_engine_benchmark.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.benchmark_qutip_engine --pilot
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.benchmark_qutip_engine
```

Official references: [QuTiP Schrödinger solver](https://qutip.readthedocs.io/en/stable/apidoc/solver.html#qutip.solver.sesolve.sesolve),
[oscillator wavefunction evaluator](https://qutip.readthedocs.io/en/stable/apidoc/visualization.html#qutip.distributions.HarmonicOscillatorWaveFunction).
