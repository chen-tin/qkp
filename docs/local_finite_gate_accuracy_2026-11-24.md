# Local finite-duration GKP accuracy controls — 24 November 2026

**Result:** the new two-cycle GKP controls pass their registered numerical sample checks. The fixed-budget comparison of lumped versus explicit simultaneous gate dissipation remains **inconclusive** at a ±1 percentage-point logical-fidelity margin. These are separate conclusions.

This is a new local study. The original eight EC2 controls at data cutoffs32/48 remain invalid and unchanged.

## Practical accuracy restriction

For nonzero gate duration with loss or number dephasing, continue to label `gate_backend="lumped"` as a symmetric splitting approximation. This study does **not** establish its equivalence to the explicit backend within ±.01 logical fidelity. Use `gate_backend="qutip"` for bounded finite-duration reference calculations when the explicit resource budget permits; do not advertise the lumped approximation as validated to1pp from these results. No physical implementation defect or necessary parameter correction is established by an interval that still contains zero.

The experimental direct-CDF measurement sampler is supported here for **static, single-width smooth resources**. Its default production adoption, dynamic resource-width schedules, eight-cycle nominal runs and hardware accuracy are outside this study. Its floating-error allowance is declared engineering budget, not a proved special-function error bound.

## Fixed protocol and provenance

- Six finite-energy GKP cardinal input states, dataDelta=.30, nearest-lattice feedback, two cycles plus initialization. **16 trajectories per cardinal state ×6 =96 total trajectories per job per arm.** Four variants ×two data cutoffs =eight jobs; their paired noise streams must not be counted as eight independent repetitions.
- Stationary Gaussian AR(1) displacements: sigma_q/L=.10, rho=.30, independent q/p streams, L=sqrt(pi), [q,p]=i. The resource peak-width setting is sigma_m/L=.5/L, giving sigma_m=.5 in absolute quadrature units; it is not independent additive detector noise. Detector efficiency1 and extra detector noise0.
- Original base seed26112001, fixed nearest decoder, no learned checkpoint or training seeds. The two-trajectory diagnostic pilots and the16/state run reuse the registered base stream; this is **not independent confirmatory data**. Exact raw matching found zero complete pilot noise trajectories or cardinal-labelled shots in the full run; all12 first-round q-displacement values match the full-stream prefix. Array indexing changes with sample budget, so these are shared draws without complete-shot nesting. See the [overlap audit](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/pilot_full_overlap_audit.json).
- The ideal variant has no dissipation. Other variants use cycle_time1, each gate_duration.1, data loss_rate.01 and dephasing_rate.001 per declared time unit, with number-dephasing generator gamma D[n]. No hardware seconds are assumed. Ancillary dissipation and feedback delay are zero; feedback gain and SUM coupling are1.
- Data cutoffs128/192. Explicit ancillary/grid refinement:256 levels, radius20, dy.02 versus384 levels, radius24, dy.01 (radius and spacing in absolute quadrature units). Physical noise, rates, timing and feedback remain fixed during these refinements.
- The direct-CDF sampler uses `uniforms[1]`, matching the explicit homodyne quantile. Its entire configuration, including tolerance1e-10, table settings, static-resource restriction, driver/backend source hashes and changed coupling to classical replay, is recorded in the new job identities. Original source files were not edited to install the hook; the declared scientific child enters and restores a process-local context.

Exact configurations, source snapshots, hashes, actual derived seeds and the macOS/CPU Python/NumPy/SciPy/QuTiP/Torch environment are in the [full benchmark plan](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/full16_cdf_128_192/plans/finite_gate/plan.json), [study registration](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/full16_cdf_128_192/study.json) and [sampler/budget provenance](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/full16_cdf_128_192/sampler_compatibility_manifest.json).

## Numerical findings

All eight jobs completed. Zero jobs failed the recorded population/norm gates. All eight saved-state192/384/768 scoring refinements, all four paired data-cutoff comparisons, and both explicit ancillary/grid comparisons passed.

The unchanged gates require mean top-decile population<1e-4, every trajectory<.01, norm error<1e-7, and explicit ancillary boundary/mass checks. Recorded stages do not establish continuous-time maxima. Scoring uses representative and selected hot snapshots at every saved cycle, with absolute refinement tolerance.001; these are sample checks, not an operator-norm bound. Data-cutoff and explicit-refinement95% intervals must remain wholly inside±.01 at each cycle.

Fresh-process raw readback checked16 sealed raw arms, recomputed every recorded stage gate, rescored16 saved-state banks, and independently reconstructed eight paired statistical comparisons. Scoring metrics matched the original receipts exactly. Maximum observed norm error was1.18e-13, mean boundary population9.35e-8 and individual boundary population8.98e-6. Maximum saved-state quadrature score sensitivity was1.404e-5 fidelity units. The direct-CDF hook's16 expected calls, source identities, restoration and declared CDF budgets also passed; the largest declared residual bound was6.23e-12.

At data cutoff192, the explicit coarse-minus-refined cycle-two fidelity difference was4.79e-6, with pointwise95% interval[1.04e-6,8.55e-6]. This is a numerical refinement result, distinct from the physical approximation comparison below.

[Raw readback receipt](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/full16_cdf_128_192/independent_readback.json), [readback script](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/full16_cdf_128_192/readback_audit.py), [full analysis](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/full16_cdf_128_192/analysis.json).

## Approximation finding

Logical fidelity uses the existing modular-subsystem extraction: for input Bloch axis a and sign s, F=(1+s r_a)/2, averaged equally over the six inputs and their trajectories.

The primary physical comparison is **lumped minus refined explicit six-cardinal mean logical fidelity**, using the fixed96 trajectories/job and existing paired state-stratified normal95% intervals. Higher logical fidelity is better; equivalence requires the whole difference interval inside±.01. These intervals are pointwise, approximate Monte Carlo intervals, with16 independent trajectories within each input stratum; they are not simultaneous or training-seed intervals.

| Cycle, data cutoff192 | Mean difference | Pointwise95% interval | Registered interpretation |
| --- | ---: | --- | --- |
| 1 |−.0012243|[−.0168094,+.0143608]|Inconclusive |
| 2 |−.0121558|[−.0304651,+.0061536]|Inconclusive |

The cycle-two estimate is−1.216 percentage points, with interval[−3.047,+.615]pp. It includes zero and crosses the−1pp equivalence boundary. It establishes neither equivalence nor a material discrepancy. The128-level result gives the same interpretation. Numerical sensitivities are reported separately; no rigorous additive numerical-error bound was invented for the Monte Carlo interval.

The fixed-budget study is complete with an inconclusive approximation result. No additional trials, threshold changes, physical retuning or automatic backend switch were used to obtain agreement.

## Why the earlier paired cutoff checks were inconclusive

Higher-basis legacy-sampler pilots at128/192 and256/384 had zero stage failures and passed saved-state scoring, but their ideal/lumped paired cutoff intervals were inconclusive. The explicit backend's corresponding cutoff differences were already near1e-9.

The controlled same128/192 direct-CDF pilot then passed all numerical checks without changing the physical model or numerical bases: ideal/lumped paired cutoff differences became approximately1e-9. The legacy mixture sampler and the new sampler represent the same marginal instrument within the declared numerical sampler budget; they couple random outcomes across cutoffs differently. Syndrome wrapping and feedback amplify differences in the old auxiliary-eigenvalue coupling into noisy paired comparisons.

A saved-data control verified identical physical configurations and exogenous arrays in all16 pilot arms. All440 raw arrays in the12 branches expected to be unchanged—classical-replay arms and explicit finite-gate arms—were bitwise identical. Only the intended scalable measurement coupling changed. This supports a variance/coupling diagnosis, not a claim of improved physical decoder performance.

[Sampler control receipt](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/sampler_control_comparison_bitwise.json), [reproduction script](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/compare_sampler_controls_bitwise.py), [legacy128/192 pilot](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/pilot_128_192/analysis.json), [legacy256/384 pilot](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/pilot_256_384/analysis.json), [direct-CDF pilot](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/pilot_cdf_128_192/analysis.json).

## Execution and reproduction

The full run used local CPU execution with OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1 requested in its environment (actual Apple Accelerate thread count was not independently verified), completed in629.90s, and sampled a maximum worker-tree RSS of1.20GB decimal under its2GiB/3600s watchdog. Concurrent studies and the monitoring parent are outside that per-worker-tree RSS figure. No AWS call, remote deployment or production-source change was required.

Use a **new output directory** for a reproduction:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.local_gkp_gate_pilot plan results/local_gate_reproduction/new_full16 --per-state 16 --cutoffs 128 192 --wall-seconds 3600 --rss-gib 2 --sampler direct-cdf
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m experiments.local_gkp_gate_pilot run results/local_gate_reproduction/new_full16
```

Frozen source identities are checked before execution. The completed full report retained a generic pilot sentence and labelled its measured runtime as a projection; these metadata-only issues are corrected for future runs in the active generator. The original report, raw data and source snapshots remain unchanged, with the precise correction recorded [separately](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/full16_cdf_128_192/reporting_correction.json). The raw readback preceded this metadata-only generator change and verified the then-matching frozen source identities. [Driver](../experiments/local_gkp_gate_pilot.py), [runtime receipt](../results/simulator_accuracy_completion/20261124/gkp_finite_gate/full16_cdf_128_192/resource_usage.json), [component software checks](../results/simulator_accuracy_actions/20261124/finite_gate/software_checks_after_metadata.json).

This closes the bounded numerical-control gap left by the original small-basis diagnostics. It does not close the physical approximation's1pp equivalence question, independently validate a complete hardware channel, or transfer the result to nominal eight-cycle high-energy production states. The separate low-energy deterministic density study remains separately recorded, including its failed lower-cutoff comparisons and prospective higher-cutoff follow-up.

## Separate deterministic low-energy control

The independently implemented matrix-free Lindblad density study completed its fixed32-case/36-refinement matrix. It **supports** a CQ instrument trace distance below.01 for its declared vacuum and coherent alpha=.25+.15i inputs, q/p single extraction before feedback, sigma_m=.5, gate duration.1, data loss.01 and/or number-dephasing.001 per declared time unit. Ideal, isolated loss, isolated dephasing and combined variants share the physical settings. There is no displacement noise, ancillary dissipation, detector noise, sampled trajectory or Monte Carlo interval in this control.

The final data/ancillary ladder was24/96→32/96, with radius10 and grid spacings.08/.04 in absolute quadrature units. All32 recorded stage/mass/grid checks,32 two-arm cutoff comparisons and four explicit propagation subdivision comparisons passed. Separate actual production ideal-ket controls at64/64→96/96 passed too. The maximum final CQ distance was7.28345e-5 (coherent,p,combined), compared with the registered.01 margin. Maximum cutoff sensitivity was2.03137e-4 below the unchanged.001 cutoff tolerance; the registered empirical sensitivity envelope was4.06275e-4. The maximum distance plus this envelope was4.79109e-4, still below.01. Grid differences were at most1.50e-17 and doubled propagation subdivisions gave zero difference at saved precision. Those are consistency/refinement observations, **not** a rigorous infinite-dimensional or solver-error bound.

The earlier16/64→24/96 comparisons failed: maximum CQ cutoff distance.00371879 exceeded.001 even though recorded population/mass controls passed. That negative result remains intact, as do the two earlier memory-cap interruptions, wall-time interruption receipts and the coverage-counter bug/correction receipts. The prospectively registered higher data rung changed no physical input or tolerance. The completed study's frozen low-order measurement dependency is preserved separately from the later high-order Hermite fix; it was not silently relabelled as the active dependency.

The final resource-only continuation inherited31 sealed rows after the previous3600s wall cap, then finished32/32 rows and exactly36/36 unique refinement keys in1348.98s, with maximum sampled process RSS3.936GB decimal under its4GiB watchdog. Exact physical configuration, numerical-helper AST/dependency identities, row seals and the v7→v5→v4 receipt chain were checked before reuse. No further density evolution is queued.

[Final density result](../results/simulator_accuracy_actions/20261124/finite_gate/v8/result.json), [frozen plan](../results/simulator_accuracy_actions/20261124/finite_gate/v8/plan.json), [resume receipt](../results/simulator_accuracy_actions/20261124/finite_gate/v8/resume_receipt.json), [complete receipt lineage](../results/simulator_accuracy_actions/20261124/finite_gate/v8/previous_receipt_lineage.json), [resource receipt](../results/simulator_accuracy_actions/20261124/finite_gate/v8/resource_usage.json), [negative lower-rung result](../results/simulator_accuracy_actions/20261124/finite_gate/v5/result.json), [coverage-counter correction](../results/simulator_accuracy_actions/20261124/finite_gate/coverage_counter_correction.json), [fresh raw/CQ readback](../results/simulator_accuracy_actions/20261124/finite_gate/v8/independent_readback.json), [readback script](../results/simulator_accuracy_actions/20261124/finite_gate/readback_completed_density.py).

This weak-noise low-energy instrument result and the two-cycle GKP logical-fidelity result answer different questions. It supplies no equivalence claim for the higher-energy GKP protocol whose fixed-budget physical comparison remains inconclusive above.

The fresh-process density readback passed in360.685s:32 sealed density rows and eight ideal raw controls verified; all32 primary/coarse CQ metrics and32 cutoff comparisons reproduced exactly (maximum discrepancy0). Its separately enforced1GiB/1200s read-only guard sampled at most758,923,264 bytes RSS and exited without interruption. No trajectories or evolution were rerun. Intermediate-time diagnostics and tighter-propagation states are receipt checks because only the final density matrices were saved. The frozen v7 checkpoint still says `running`; its terminal resource receipt records the clean wall timeout and is authoritative. All density/readback reservations and future queues are [released](../results/simulator_accuracy_actions/20261124/finite_gate/density_final_release.json).
