# Explicit QuTiP/local measurement benchmark result

Belief change: smooth Mehler evaluation reproduces the independently evolved
two-mode QuTiP circuit within the registered tolerances for the tested finite
data operators. The current piecewise-constant lookup introduces measurable
outcome and conditional-state discrepancies. Confidence: supported within
this measurement-level scope. The cause of the earlier cycle-6 energy event
remains unresolved.

Scope: one q OR p extraction, six fixed seeded displaced cardinal inputs,
data cutoffs 64 and 96, ancilla cutoffs 256/512/768, and resource widths
sigma_m/L=.05 and .16. Input preparation, feedback and scoring are shared.
This is not a multi-cycle memory, sampler, LSTM/Bayes, or hardware benchmark.

The completed run is `results/qutip_engine_benchmark/main/5f1d5766c1aa2659`.
All 24 circuit configurations completed; all 48 final per-input reference
checks pass. The frozen pre-run plan, two pilot runs and two interrupted
attempts are retained. An independent agent ran both pilots from the
documentation; the pilot grid limitations prompted the documented
cell-split integration amendment before the main run. The interrupted
attempts contribute no timings to the completed comparison.

| Local arm | Worst outcome TV | Worst reference-weighted conditional infidelity |
| --- | ---: | ---: |
| Current bins, dx=.003 | .002139734 | 2.16787e-5 |
| Finer bins, dx=.0015 | .001024482 | 5.38942e-6 |
| Smooth Mehler evaluation | 1.20899e-6 | 3.85760e-10 |

These maxima cover all 48 input/axis/resource/data-cutoff tests at ancilla
cutoff 768. They are deterministic quadrature results; no Monte Carlo
confidence intervals apply. The observed ancilla, grid and ODE sensitivities
are recorded separately, not presented as rigorous infinite-cutoff bounds.
All final 512-to-768 ancilla refinements pass the pre-registered gates.
The largest final reference grid-induced fidelity change is 1.39644e-5;
the largest joint norm error is 1.09055e-11. Tightening the ODE tolerances in
the two registered checks changes outcome TV by at most 1.08497e-10.

At the current bin width, the largest absolute logical-fidelity difference
after common feedback is only .0010323 percentage points; the largest
expected-energy difference is .0005223 photons. Thus good agreement of these
averages does not certify agreement of conditional states. Halving the bin
width reduces both primary discrepancies, while the smooth arm agrees much
more closely with the explicit circuit. This supports a bin-discretization
effect in this measurement test, without proving the long-horizon mechanism.

The current binned batch implementation is 39–106 times faster than the
explicit circuit across tested final-cutoff configurations (median ratio
54.3). The smooth batch is 12–19 times faster (median 16.9). Each largest-
ancilla batch has three timing repetitions; figures show median and min/max.
Timings exclude common input/scoring and quadrature-basis setup, and compare
different state-space costs on a shared machine. They are not production
per-shot decoder latencies. The completed run's total wall time is 197.0 s.

Data-basis convergence is not established here. Expected data-boundary mass
reaches 5.79% in these small finite-operator tests, so their absolute logical
fidelities and energies should not be used as physical memory predictions.
Their role is to test two implementations of the same truncated data
operator while converging the ancillary and outcome representations.

Verification: 16 tests pass, including explicit q and p circuit comparisons,
QuTiP wavefunction-kernel checks, resource normalization, wrong-sign negative
control, existing displacement/measurement reference tests, and feedback
boundary integration. All 24 raw hashes and frozen source snapshots were
verified. Scalar metrics were recomputed from saved outcome-density curves.

Review: [HTML report](../results/qutip_engine_benchmark/main/5f1d5766c1aa2659/review/index.html),
[all metrics CSV](../results/qutip_engine_benchmark/main/5f1d5766c1aa2659/metrics.csv),
[registered methods](qutip_engine_benchmark.md).

Next decision: retain historical results; implement a matching continuous
sampler for the independently validated smooth resource, then run the
registered long-horizon comparison with measurement-substep energy and
unwrapped-outcome logging. That follow-up remains Planned.
