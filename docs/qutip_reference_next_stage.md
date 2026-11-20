# Next quantum-reference stage

Registered before the larger-sample eight-cycle continuation, 19 November 2026.

The immediate question is whether the original eight-cycle nearest-lattice
reference is stable between3072 and4096 Fock levels after increasing the
sampling budget from64 to512 trajectories per cardinal input state. Physical
operators, data/ancilla states, noise, decoder policy, seed and scoring remain
the protocol in `docs/qutip_reference.md`. This changes sampling precision
only; it is not a new physical model or decoder training run.

The independent control/intervention check and explicit two-mode QuTiP tests
already cover this evolution. Boundary-population, norm and extraction gates
remain unchanged. Compare paired fidelity differences at every cycle, using
the registered +/- .01 numerical-equivalence band. If an interval fits wholly
inside the band and all population gates pass, that cycle is provisionally
stable; if it excludes the band, investigate numerical bias; otherwise it is
inconclusive. Do not infer failure of the decoder from a numerical failure.
This fixed continuation adds one budget of512 trajectories per input state,
with no automatic unbounded growth. Subsequent stages remain Planned.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.qutip_reference \
  --convergence --cutoffs 3072 4096 --rounds 8 --per-state 512 \
  --points nominal --decoders nearest
```

The follow-up physics stage will introduce photon loss and oscillator
dephasing separately, then together, under the same measurement protocol.
`quantum/channels.py` implements exact pure-loss Kraus trajectories and
Lindblad number-dephasing random rotations. Their independent tests compare
with QuTiP `mesolve`. These channels are not yet used in the reference
benchmark, so no loss/dephasing performance result is claimed.

For loss, eta=exp(-cycle_time/T1); the environment's loss count is hidden from
the decoder. For dephasing, strength=gamma_phi*cycle_time with collapse
operator sqrt(gamma_phi)*n. The convention is explicit because reported Tphi
definitions can differ by factors of two. Validate identity/noise-free limits
and Fock convergence before production runs. Avoid adding physical loss on
top of a displacement variance already derived from that same loss mechanism.

Hardware comparison will use primary-source public data where available,
document protocol and parameter mappings, fit on a calibration subset and
evaluate separate observables or held-out records. The existing Sivak
parameter mapping is a rough anchor, not validated hardware agreement. New
laboratory measurement collection requires laboratory access. No external
messages, paid compute or Solar writes are part of this local continuation.

## Fixed follow-up after the first continuation

The512-per-input-state continuation completed at15:04 UTC on19 November2026.
Its3072-level boundary-population gate is valid only through cycle5, whereas
4096 levels pass through cycle8. The cycle8 paired fidelity difference between
these resolutions is0.005418 with95% interval[0.000061,0.010774], which does
not fit wholly inside the registered +/- .01 band. The result remains
numerically inconclusive rather than a physical decoder failure.

One further fixed diagnostic uses5120 levels with exactly the same512
trajectories per cardinal state, seed, eight cycles, nominal noise and
nearest-lattice controller. It reuses the completed4096-level observations
for the paired comparison. This changes Fock resolution, not sample size or
physics. Apply the same population/norm gates and +/- .01 equivalence band;
record an inconclusive result if it does not pass. No automatic further
cutoff increases are scheduled. Loss/dephasing integration and hardware
calibration remain Planned pending a suitable baseline.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.qutip_reference \
  --convergence --cutoffs 5120 --rounds 8 --per-state 512 \
  --points nominal --decoders nearest
```


## Measurement-discretization audit

A subsequent diagnostic found that the original ancilla amplitude evaluator
uses piecewise-constant position bins. A discontinuous wavefunction has
artificial high-momentum tails and does not retain finite continuum kinetic
energy. This is a credible source of the rare high-Fock spikes and requires
an explicit smooth-resource comparison before identifying their physical
origin. Typical ensemble heating is consistent with SUM backaction, but that
does not establish that the abrupt outliers are physical.

The5120-level check still measures basis sensitivity. Passing it alone cannot
resolve the measurement-approximation concern. The next numerical task is a
smooth analytic ancilla amplitude with consistent continuous sampling,
independent explicit-circuit validation, and new fidelity/energy convergence
checks. Keep the physical model otherwise fixed; do not add damping merely to
make a numerical problem disappear. Loss/dephasing and hardware calibration
remain subsequent stages. Evidence: `results/qutip_reference/measurement_discretization_audit.json`.

## Completed cutoff follow-up and review plots

The fixed 5,120-level follow-up completed on 19 November 2026 at 20:52 UTC
(`convergence/dd4e949e01da259d`). Both 4,096 and 5,120 levels pass the
population gates through cycle 8, and every pointwise paired fidelity
interval fits within the registered +/- .01 band. At cycle 8 the difference
(5,120 minus 4,096) is -0.001281, with 95% interval
[-0.006468, 0.003905]. This supports cutoff stability of the average logical
fidelity at that tolerance for nominal noise and the nearest-lattice
controller. It does not establish convergence of energy tails or individual
trajectories.

The rare energy event persists at 5,120 levels: trajectory 1659 rises from
46.53 photons at cycle 5 to 3,080.67 at cycle 6, while its boundary population
remains below 0.061%. The event's physical origin remains unresolved; passing
the cutoff test does not remove the measurement-discretization concern above.
The smooth-resource intervention and substep logging remain Planned.

Saved-data plots, a trajectory selector, metric definitions and an evidence
assessment are in
[`root_cause_review_20261119/index.html`](../results/qutip_reference/root_cause_review_20261119/index.html).
They include the separate two-cycle bin-refinement results without treating
them as an eight-cycle causal test. Generate the review without resimulation:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.plot_qutip_root_cause
```

## Completed smooth-resource decision experiment

The smooth-resource and continuous-sampler experiment completed on19 November
2026 (`results/smooth_ancilla_experiment/main/06c34206dd1a1b00`). The decision
is to adopt the continuous implementation for new reference runs: its minimum
held-out reference-weighted squared-state fidelity is99.9999999615%, exceeding
the registered99.999% threshold, and all sampler/reference checks pass.
Both4096 and5120 levels pass the eight-cycle population/norm gates at every
logged substep, and their paired mean-fidelity intervals fit the registered
equivalence band. Cutoff3072 remains valid only through cycle6.

The original catastrophic selected event is traced to the p measurement in
cycle6 and greatly reduced by the continuous implementation. Other physical
heating remains. The binned5120 replay also exposes an intermediate boundary
violation that was invisible at cycle ends. Full results, scope, citations and
the implemented CLI decision are in [smooth_ancilla_results.md](smooth_ancilla_results.md).
The default CLI now uses smooth mode; the historical commands earlier in this
document require `--measurement-resource binned`. The historical continuation
and cutoff wrappers specify that mode explicitly. Loss/dephasing integration,
energy stabilization and hardware calibration remain future stages.
