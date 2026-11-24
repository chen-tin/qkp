# Local simulator-accuracy work, 24 November 2026

This records completed local repairs and targeted studies, including negative and inconclusive outcomes. The canonical research notebook is [RESEARCH_NOTES.md](../RESEARCH_NOTES.md). All new implementation, tests and numerical execution in this pass are local. Original EC2 plans, source snapshots and raw observations remain unchanged. No remote scientific jobs or deployment are part of this pass.

## Evidence and scope

| Issue | Current evidence | Status |
| --- | --- | --- |
| Scheduler drain, raw archive ordering, stale running status | [Prior source-directed repairs](../results/simulator_accuracy_actions/20261123/summary.json) | Fixed locally; undeployed |
| Missing physics raw job | Original six sealed files recovered during the earlier recovery-only session | Recovered; original bytes retained |
| Missing saved-state scoring refinement | [Recovered-job completion](../results/simulator_accuracy_completion/20261124/scoring_635d50e4441cc5d1/completion_summary.json): 162 deterministic checks pass | Closed for all 16 saved physics jobs |
| High-order independent wavefunction evaluator underflow | Generalized-Laguerre evaluator checked against arbitrary-precision Hermite values, including high-order tails | Fixed locally; explicit new reference identity |
| Independent production-state instrument validation | [Selected full4096 result](../results/simulator_accuracy_completion/20261124/measurement/full4096_cycle8_spectral_v1/result.json), eight checks and four refinements; sealed raw audit passes | Selected-state check complete |
| Uniform ideal finite-operator comparison | All 9,216 eigenvalues across 4096/5120; finite-grid comparison with ancilla12288/16384 | Both gains1 and1.01 pass fresh computations and sealed audits; [summary](../results/simulator_accuracy_completion/20261124/measurement/completion_summary.json) |
| Four original unresolved physics cutoff intervals | [Fixed matched-sampler result](../results/simulator_accuracy_completion/20261124/coupled_cutoff_checks/main_128_static_planned/completion_report.json),16/16 jobs complete | Direct-CDF comparisons pass every registered cycle; matched legacy comparisons remain inconclusive; original evidence preserved |
| Invalid small-basis finite-duration controls | Original failures preserved; fixed two-cycle GKP study and raw audit pass numerical checks; lumped-versus-explicit fidelity intervals cross the±.01 margin | GKP numerical gap closed, logical approximation inconclusive; low-energy density controls and raw audit pass |
| Four unrun physics configurations | [Frozen local plan](../results/simulator_accuracy_completion/20261124/remaining_physics/plan.json), exact original scientific parameters with fresh local source/environment identity | All four observed/refined; one4096passive-nearest stage failure, nearest cutoff CIs inconclusive; [report](../results/simulator_accuracy_completion/20261124/remaining_physics/completion_report.json) |
| Coverage250 and rare300 grids | Original plans retained | Unrun; substantial separate compute/storage requirement |

A completed audit can have a negative or inconclusive scientific result. No failed control is relabeled as a pass, and no sample budget is increased until a desired outcome appears.

## Completed matched-sampler result

The fixed prospective study used128trajectories per cardinal input ×6 =768total per job/arm, eight cycles, nearest feedback, nominal sigma_q/L=.20, sigma_m/L=.05, rho=.90, and data cutoffs4096/5120. All16jobs pass stage and192/384/768saved-state scoring checks. Cutoff equivalence requires every pointwise paired95%logical-fidelity interval to lie inside±.01.

| Condition | Largest absolute legacy CI endpoint | Largest absolute direct-CDF CI endpoint |
| --- | ---: | ---: |
| Loss | .01424 | 1.40e-11 |
| Loss + dephasing | .02116 | 6.30e-11 |
| Coupling error | .03173 | 3.26e-8 |
| Control rotation | .01407 | 6.29e-12 |

All four direct-CDF comparisons pass; all four legacy comparisons remain inconclusive. The displayed endpoints describe only Monte Carlo variation of the paired logical-fidelity difference. Deterministic scoring, finite-grid and CDF approximation sensitivities are checked separately; these tiny intervals are not a claim of total numerical or physical accuracy to those decimal places. This supports a common-random-number variance explanation within the registered model and horizon. It does not imply improved physical decoder fidelity. The controlled samplers have the same mathematical marginal instrument, with a declared numerical CDF budget and an explicitly assumed floating-error allowance; their joint trajectory coupling differs.

The mechanism audit verified440replay arrays and40finite-arm exogenous arrays bitwise unchanged across eight pairs, plus reconstructed uniform streams. All eight CDF diagnostic files satisfy the registered contract. No extra shots were added. [Sealed analysis packet](../results/simulator_accuracy_completion/20261124/analysis_completion_packet.json).

## Remaining scientific and resource limits

- Passive-cooling4096nearest has one of3,072paths crossing the individual boundary threshold in cycle8, reaching.04950against.01. Preserve this numerical-domain failure; the separate5120sample passes recorded stages. Legacy cross-cutoff indices share noise/cardinal identity but can follow different conditional physical paths.
- The newly completed eight-cycle finite-duration/passive nearest cutoff intervals remain inconclusive. Bayes and frozen LSTM satisfy their registered cutoff-interval checks. These results are separate from the four-condition sampler study above.
- The two-cycle GKP lumped-versus-explicit comparison remains inconclusive at±1percentage point. Numerically valid low-energy density controls support their narrow one-extraction margin; they do not establish high-energy or hardware accuracy. [Finite-gate report](local_finite_gate_accuracy_2026-11-24.md).
- Original250coverage and300rare jobs remain unrun. Their declared uncompressed output is243.5GiB; existing local storage is insufficient. A suitable local archive path and an explicit multi-day runtime budget are still needed. [Resource assessment](../results/simulator_accuracy_completion/20261124/unrun_550_resource_assessment.json).

All targeted study, scoring, audit, manager and observer processes have exited. The original640/640/2048/640confirmation proposal remains Planned/unrun; its formerly reserved26112401stream was used in the completed matched-sampler study, so a later independent confirmation needs a newly registered stream.

## Source-directed issue ledger

These are local fixes and diagnostics. EC2 deployment remains outside this pass.

| Issue | Source / directed verification | Disposition |
| --- | --- | --- |
| Logging `tee` prevented supervisor drain | `ops/ec2/parallel_worker.py`, `tests/test_ec2_parallel.py` | Recognize only the exact persistent direct-child logger; scientific descendants still block drain |
| Raw archive depended on completing scoring | EC2 supervisors and checkpoint-order regressions | Checkpoint sealed observations before expensive refinement |
| Archived status could appear currently running after stop | `ops/ec2/monitoring.py`, `tests/test_ec2_monitoring.py` | Separate observed status from immutable archived status |
| Explicit homodyne accepted invalid quantiles | `quantum/finite_gate.py`, finite-gate directed tests | Reject endpoints, nonfinite and out-of-range quantiles before solver work |
| Raw audit trusted saved numerical pass flags | `experiments/validate_reference_states.py`, directed measurement tests | Recompute gates, metrics and refinements from sealed raw inputs |
| Empty, duplicate or incomplete decoder coverage could pass | `experiments/quality_benchmark.py`, directed analysis tests | Require exact declared row and raw-arm coverage |
| Empty or partial scoring could pass | Scoring/refinement acceptance tests | Require complete configured files, cycles and resolutions |
| Small parameter-only controls could imply production-state coverage | Provided-state validator and directed coverage tests | Require declared actual input-state identities and cutoffs |
| High-order QuTiP scalar evaluator underflow | `quantum/measurement_benchmark.py`, `tests/test_reference_wave_basis.py` | Stabilized independent generalized-Laguerre reference with arbitrary-precision gold checks |
| Dense high-cutoff validation exceeded local memory | Streaming/spectral/operator modules and small independent circuit controls | Full finite data spaces retained under bounded local memory |
| Recovered scoring was incomplete and slow | Shared-basis scorer and independent control/intervention/resume pilot | Missing refinement completed with unchanged quadrature conventions |
| New density driver expected20 instead of36 refinement keys | `tests/test_local_finite_gate_audit.py` | Exact coverage-key validation catches missing and duplicate controls; original negative results retained |
| Full GKP metadata retained pilot wording/runtime label | `tests/test_local_gkp_gate_pilot.py` | Mode-specific counts and actual/projection fields; frozen evidence preserved with a separate correction |
| New mechanism auditor could accept empty or duplicate design coverage | `tests/test_coupled_mechanisms.py` and independent read-only review | Exact approved16-cell coverage, unique IDs and budgets required before array comparison; corrected before use |
| New resource wrappers could reset budgets, orphan descendants or exceed deadlines | `experiments/local_process_safety.py`, nine bounded subprocess/budget tests and independent review | Durable history, owned process groups, external work deadlines and bounded final audit; fixed before further launches |
| Quadrature deltas carried inappropriate MC uncertainty fields | New deterministic serial/shared report metadata and four regression cases | Active producer corrected after completion; frozen reports preserved with a [separate correction](../results/simulator_accuracy_completion/20261124/scoring_reporting_correction/correction_20261124.md) |
| Paired cutoff intervals had high auxiliary-sampler variance | Opt-in CDF marginal tests and completed fixed mechanism controls | All four targeted direct-CDF comparisons pass; unchanged replay/exogenous branches verified bitwise; default sampler remains unchanged |

The current [integrated suite](../results/simulator_accuracy_completion/20261124/integration_final_reporting_manifest.json) passed263tests. It covers the relevant modules and directed regressions; it is not the whole repository suite or a physical benchmark. Classical displacement-model scope, hardware calibration, a single frozen LSTM training seed, and differences between replay and quantum channels remain explicitly stated model/evidence limits.

## New local methods

`experiments.continue_local_physics` resumes only the four fixed original configurations. It checks source/config/input identities, stages sealed raw files before scoring, and enforces local time, sampled RSS and disk limits. `--no-sync --no-shutdown` are mandatory. A tiny independent control/intervention/resume pilot compared 804 raw arrays and preserved all scientific hashes on resume.

`quantum.streaming_measurement` and `quantum.spectral_circuit` retain the complete data basis while streaming outcome reductions. The reference uses QuTiP finite operators and an independently stabilized generalized-Laguerre scalar basis; it does not silently substitute the production Hermite routine. The original QuTiP5.3.1 Gaussian-seeded kernel underflows for some high-order, large-coordinate inputs.

`experiments.validate_instrument_operator` compares all quadrature eigenvalues of each declared finite data operator. Its discrete-grid Stinespring norm is `sqrt(max_lambda sum_y w_y |m_ref(y,lambda) - m_candidate(y,lambda)|^2)`. The resulting channel bound applies to that finite grid and operator. Ancilla/window/grid convergence is empirical; this is not a rigorous continuum, physical-Fock-cutoff or hardware certificate. The8192-level reference ancilla fails at data5120; that negative remains visible beside the predeclared finer12288/16384 comparison.

`quantum.coupled_measurement` provides an opt-in experimental full-outcome inverseCDF sampler. It retains the finite Kraus instrument and changes common-random-number coupling across cutoffs and the classical replay arm. The original default sampler remains unchanged. The experimental drivers require a static resource width and hash the backend, quantile stream and all numerical settings into job identity. Analytic CDF interpolation bounds are supplemented by an explicitly unproved floating-point allowance. New runs do not reproduce historical joint trajectories.

`experiments.local_finite_gate_audit` uses deterministic density controls with matrix-free Lindblad propagation, checked against small QuTiP controls. Its approximation margin and numerical convergence thresholds are distinct. `experiments.local_gkp_gate_pilot` covers a separately registered two-cycle GKP setting; a numerically valid pilot permits the fixed16/state study, not a hardware or eight-cycle nominal claim.

## Reproduction and provenance

Plans, frozen source copies, input hashes, environment/device, commands, interruptions, raw seals and metrics are under [the local run directory](../results/simulator_accuracy_completion/20261124/) and [finite-gate controls](../results/simulator_accuracy_actions/20261124/finite_gate/). Use each exact registered command and its virtualenv path. Resolving `.venv/bin/python` through its symlink can select a base interpreter without the required packages.

Do not edit frozen protocols, source snapshots, sealed results or original failed controls. New source/settings require a new run identity. The reporting-only quadrature correction after completion deliberately gives future reports a new producer hash; historical studies retain their original snapshots and numerical results. The frozen physics plan also hashes `docs/simulation_quality.md`; leave it unchanged during execution.

Logical fidelity, outcome total variation, conditional-state fidelity, instrument distance, stage populations, classical parity error and rare-event probability are separate metrics. Deterministic quadrature differences have no Monte Carlo confidence intervals. Common-random-number variance reduction does not prove physical cutoff equivalence; the registered cyclewise intervals and numerical gates still decide that question.
