# Simulation-quality infrastructure verification — 19 November 2026

This record covers the earlier infrastructure validation. Production was
subsequently launched on EC2; see [the current execution record](ec2_production.md).

Classification: bounded software/infrastructure validation. No production
decoder-performance, hardware or rare-event benchmark was launched.

Implementation and commands: [simulation_quality.md](simulation_quality.md).
Research record: [RESEARCH_NOTES.md](../RESEARCH_NOTES.md).

## Checks executed

The final targeted suite:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_simulation_quality.py tests/test_continuous_ancilla.py tests/test_qutip_engine_benchmark.py tests/test_qutip_reference.py tests/test_quantum_channels.py -q

Result: **50 passed in8.13s**, including22 dedicated quality-infrastructure
test cases. This includes:

- Ideal-limit agreement with historical evolution in physical and replay arms.
- Detection of a transient boundary violation and a later norm failure.
- Separation of energy events from numerical gates.
- Dense/streamed logical scoring and independent QuTiP wavefunction agreement.
- Finer scoring and detection of a deliberately corrupted original score.
- Per-cycle snapshots and explicit final-only legacy limitations.
- Physical parameter/time validation and prevention of double-counted loss.
- Detector record versus latent outcome, and unchanged exogenous noise draws.
- Correlated/negative-memory/nonstationary noise and dynamic resource widths.
- Fixed-budget rare-event bounds and sample-size calculations.
- Nonunit SUM coupling versus independent two-mode evolution.
- Explicit finite-duration gate identity/dissipative paths and runner integration.
- Six-state default input selection without silently truncating saved states.
- Calibration model recovery and rejection of overlapping split IDs.
-125-point cube, custom matrices, immutable plan hashes and unsafe-path rejection.
- Interrupted/resumed jobs, honest partial status and raw-corruption rejection.

A broader regression adding tests/test_nonstationary.py was interrupted
after439.15s (7min19s), with58 checks passed and no failures reported before
interruption. It was executing the unchanged adaptive Bayesian stress path
in decoders/bayesian/adaptive.py:162. That broader suite is **incomplete**;
its partial passes are not added to the50 as independent extra checks.

## CLI and saved-artifact checks

All of these commands completed successfully:

1. Plan and run a two-job infrastructure smoke, first with --max-jobs1 and
   then resumed to2/2. The intermediate report retained1/2 completion.
2. Rescore both jobs at192/384/768 logical bins; every saved cycle checked,
   all representative/hot sample refinements pass.
3. Generate HTML, PNG/SVG figures, CSV event counts, fixed-budget intervals,
   paired cutoff analysis and JSON status. The figure was visually inspected.
4. Run independent ideal-instrument controls at data cutoff6, Fock inputs0/2,
   ancilla64/96, sigma=.5, radius12 and dy=.04. Registered finite-operator
   fidelity, distribution, grid, norm and ancillary-refinement checks pass.
5. Audit saved instrument raw hashes/source snapshots and recompute scalar
   density metrics; attach the report to the benchmark plan. Its6-level
   scope does not cover the32/48-level smoke and is not promoted to do so.
6. Execute the normal reference CLI at24 levels, two trajectories/cardinal
   state, one cycle, nearest-lattice, broad ancillary width. Analyze its
   stage gates and rescore all saved cycles at192/384/768; scoring passes.
7. Fit the calibration CLI on an explicitly synthetic fixture with disjoint
   train/test IDs and independent draws. No hardware interpretation is made.
8. Generate six production plans without executing them. Verify each frozen
   source snapshot and current working source against its plan hashes.

The tiny trajectory smokes deliberately fail population gates. This is the
expected negative control; their final fidelities are diagnostic only.

Evidence:

- [Smoke HTML](../results/simulation_quality_infrastructure/v1/smoke/index.html)
- [Smoke summary](../results/simulation_quality_infrastructure/v1/smoke/summary.json)
- [Independent instrument result](../results/simulation_quality_infrastructure/v1/instrument/result.json)
- [Synthetic calibration fixture specification](../results/simulation_quality_infrastructure/v1/calibration_fixture.json)
- [Calibration smoke](../results/simulation_quality_infrastructure/v1/calibration_smoke.json)
- [Normal CLI result](../results/qutip_reference/smoke/9dfe2e71a2bac772/result.json)
- [Normal CLI stage audit](../results/qutip_reference/smoke/9dfe2e71a2bac772/audit.json)

## Unrun benchmark plans

| Preset | Jobs | Estimated uncompressed result size |
| --- | ---: | ---: |
| Numerical convergence |2|.951GB|
| Physical interventions |20|9.508GB|
| Noise robustness |14|6.656GB|
|125-point coverage |250|118.850GB|
| Rare events, target1e-4 |300|142.619GB|
| Explicit finite-duration comparison |8|.006GB|

Plans: [numerical](../results/simulation_quality_infrastructure/benchmark_plans_v2/numerical/plan.json),
[physics](../results/simulation_quality_infrastructure/benchmark_plans_v2/physics/plan.json),
[noise](../results/simulation_quality_infrastructure/benchmark_plans_v2/noise/plan.json),
[coverage](../results/simulation_quality_infrastructure/benchmark_plans_v2/coverage/plan.json),
[rare](../results/simulation_quality_infrastructure/benchmark_plans_v2/rare/plan.json),
and [finite gates](../results/simulation_quality_infrastructure/benchmark_plans_v2/finite_gate/plan.json).

The v2 drafts include the final plan-size guard and match the final source
hashes. Earlier unrun drafts are retained without silently editing them.

The explicit-gate draft uses16 trajectories/state and2cycles with nearest
lattice to bound reference cost. Other defaults are specified in each plan.
Storage estimates are conservative array budgets, not measured compression
ratios or timing forecasts. Execution checks available output space.

## Limits retained

The scalable duration model is a labeled lumping approximation; the explicit
backend enables its comparison, not automatic approval of that approximation.
Independent controls and rescoring are supplied-state/sample checks, not
rigorous channel bounds. Cooling is passive loss, not a newly implemented
GKP stabilizer. Synthetic calibration does not equate the classical and
quantum measurement channels. Physics settings remain uncalibrated
interventions. Benchmark jobs and confirmatory numerical results remain
Planned until actually executed.
