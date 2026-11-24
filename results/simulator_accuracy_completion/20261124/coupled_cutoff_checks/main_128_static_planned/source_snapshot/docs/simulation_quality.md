# Simulation-quality infrastructure

Status: implementation and bounded software validation; production comparisons remain Planned.
This infrastructure extends the [quality audit](../RESEARCH_NOTES.md). It preserves historical results and source snapshots.

## Numerical execution contract

New ordinary reference runs use schema2 and the instrumented trajectory engine.
Every q/p displacement, measurement, feedback, rotation, loss and dephasing stage
records energy, boundary mass and norm error. The gate requires ensemble-mean
top-decile mass<1e-4, every trajectory<.01 and norm error<1e-7 throughout the
recorded stages. A failed stage invalidates that and all later cycles.
These checks are not continuous-time bounds during a gate.

Every cycle saves representative states (first fixed independent trial IDs per
cardinal input) and separately labeled high-energy/high-boundary states.
Hot-state selection is diagnostic and is never treated as a random sample.
The original simulation() Python entry point remains a compatibility control;
new CLI/run_case schema2 execution uses quantum/reference_runner.py.
Legacy schema1 data remain explicitly cycle-end-only.

## Scoring without another simulation

From qkp/:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.refine_logical_scores RUN_DIRECTORY --bins 192 384 768

This works on both normal references and the completed smooth-resource run.
It streams quadrature chunks rather than allocating new cutoff-squared scoring
matrices. It checks diagonal trace, minimum logical eigenvalue, representative
paired score changes and the maximum change including hot states. Default
absolute score tolerance=.001 and trace tolerance=1e-7. These are engineering
budgets. Outputs live under RUN_DIRECTORY/quality/ with analysis/config hashes.
Historical final-only snapshots do not certify earlier cycles. New snapshot
checks remain sample checks, not rigorous operator-norm bounds.

The extraction convention follows Pantaleoni, Baragiola and Menicucci,
[PRL125,040501(2020), Eqs.(1),(3),(4)](https://doi.org/10.1103/PhysRevLett.125.040501).
Average six-cardinal fidelity, worst-case fidelity and conditional-state
agreement are distinct metrics.

## Physical settings and units

Pass a JSON file to the ordinary CLI using --physics-json FILE. Fields:

| Field | Meaning / ideal default |
| --- | --- |
| cycle_time | Total time per cycle in declared consistent units;1 |
| gate_duration, feedback_delay | Duration of EACH of two gates/delays;0 |
| loss_rate | Rate kappa; eta=exp(-kappa*time);0 |
| dephasing_rate | Rate gamma for gamma D[n];0 |
| detector_efficiency | Homodyne efficiency eta;1 |
| detector_std | Additional independent electronic noise SD in absolute q units;0 |
| coupling | Integrated SUM gain;1; decoder record is calibrated by this gain |
| feedback_gain | Multiplier on the chosen correction;1 |
| control_rotation | Unwanted phase-space rotation after EACH feedback, radians;0 |
| cooling_transmissivity | Additional explicitly applied end-cycle pure loss;1 |
| displacement_provenance | independent or loss_equivalent; reject double-counted loss |
| gate_backend | lumped (scalable) or qutip (explicit coupled reference) |
| gate_ancilla_cutoff | Ancillary Fock dimension for explicit gates;512 |
| ancilla_loss_rate, ancilla_dephasing_rate | Explicit-gate ancillary dissipation;0 |
| homodyne_radius, homodyne_dy | Explicit-gate sampling grid;60,.01 |
| max_joint_dimension | Explicit joint-amplitude budget;2,000,000 |
| max_gate_scratch_mb | Explicit-gate scratch estimate limit;512 MB |

The idle duration is cycle_time−2*gate_duration−2*feedback_delay and must be
nonnegative. Rates are per chosen time unit; no conversion to hardware seconds
is assumed. Displacement parameters and schedule levels use units of L.

The scalable lumped backend applies half a gate's data noise before and half
after the instantaneous measurement. This is a stated approximation, not exact
simultaneous gate/dissipation. The qutip backend evolves the coupled two-mode
Hamiltonian with data AND ancillary collapse operators using mcsolve. Hidden
environmental outcomes and the latent ideal homodyne value never enter a
decoder. It requires ancillary cutoff, outcome-grid and sampling convergence;
it is intentionally expensive. The finite_gate plan compares ideal, lumped,
explicit and refined-explicit implementations on common exogenous noise.

Inefficient homodyne is a destructive measurement with the detector record
rescaled to input quadrature: y_record=y_ideal+Normal(0,(1-eta)/(2eta)+detector_std²),
vacuum variance1/2. The hidden ideal outcome unravels the mixed conditional
state; only wrap(y_record/coupling) reaches the decoder. Measurement noise
sigma_m/L continues to specify RESOURCE PEAK width, not detector_std. The
Mehler resource still ties peak width and envelope; it is not an independently
calibrated hardware measurement chain.

Pure-loss/dephasing conventions and the relevance of physical gate/readout
imperfections follow [Grimsmo and Puri2021, Sec.II C, Eq.(16), Sec.II D1](https://doi.org/10.1103/PRXQuantum.2.020101).
Explicit stochastic evolution uses [QuTiP's Monte Carlo solver](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-monte.html).
The logged homodyne density conditions on the latent trajectory/ideal outcome;
it is not the likelihood of the noisy reported detector record.
Passive cooling is not an autonomous finite-energy GKP stabilizer. Adding a
specific stabilization protocol remains a new physical model requiring its
own method and validation, rather than an implicit numerical fix.

## Noise extensions and coverage

--noise-json FILE accepts family=gaussian|student_t, student_df>2,
cross_correlation in[-1,1], and schedule using simulator/nonstationary.py's
stationary/sinusoid/ramp/one_over_f/step/telegraph/burst contract. All schedule
standard deviations/levels use L units. Cross-correlation couples the AR
innovations; additional burst kicks are independent. Student-t innovations
are variance-normalized; the AR process is not claimed to have exact Student-t
marginals. Dynamic readout widths create fresh resources at the scheduled
widths, not an extra Gaussian disturbance. Negative rho is supported.
Schedules reaching the existing rho clip at±.995 are rejected unless
allow_schedule_clipping=true is explicit; actual schedules are stored.

The classical heatmap runner now supports:

    .venv/bin/python -m experiments.noise_heatmaps --coverage cube --split test
    .venv/bin/python -m experiments.noise_heatmaps --coverage cube --rho-values -.5 0 .3 .6 .9 .97

Default five-level axes give125 points; the original planes give61. It writes
new result identities and only updates the viewer with --publish-viewer.
The same training checkpoints and noise split are documented; calling a
development sweep a test split retrospectively does not make it held out.

## Independent supplied-state validation

    .venv/bin/python -m experiments.validate_reference_states --states RAW_FILE.npz --cycle 8 --columns 0 1 2 --ancilla-cutoffs 512 768 --output OUTPUT_DIRECTORY

Without --states, --data-cutoff and --fock-levels create explicitly declared
high-energy Fock inputs. Saved states retain their FULL data cutoff. There is
balanced default selection across the six cardinal input labels where known;
explicit --columns can select hot states. Provenance includes original trial
IDs, logical input labels and each state's mean photon number. There is
no silent truncation to96 levels. The command compares reduced conditional
states against a separate two-mode QuTiP circuit, with ancillary refinement,
outcome mass, TV, squared conditional fidelity, norm and grid checks.
--coupling exercises nonunit SUM gains. Joint-state and scratch budgets reject
an oversized job before setup. This tests the ideal instrument on the supplied
states, not the full noisy multicycle channel.
Input states, joint states/density curves, source snapshots and hashes are
saved. Recheck saved scalar metrics using --output OUTPUT_DIRECTORY --audit-only.

## Effective-noise calibration contract

    .venv/bin/python -m experiments.calibrate_reference_noise CALIBRATION.npz --output calibration.json

The input contains train_displacement, train_readout, test_displacement and
test_readout, each shaped (2, independent trajectories, rounds), in absolute
q units, plus disjoint train_ids/test_ids arrays. These must come from an
independent characterization; wrapped syndromes alone do not identify both
latent process noise and readout noise. The tool fits a zero-mean isotropic
Gaussian AR model only on training records and checks held-out innovations,
means, variance, fourth moments and q/p correlation. It records data hashes.
It does not silently equate fitted classical readout variance with a quantum
ancilla width or claim hardware validation.

## Benchmark planning and execution

Plan generation is cheap and does not execute trajectories:

    .venv/bin/python -m experiments.quality_benchmark plan results/quality_smoke --preset smoke
    .venv/bin/python -m experiments.quality_benchmark run results/quality_smoke
    .venv/bin/python -m experiments.quality_benchmark score results/quality_smoke
    .venv/bin/python -m experiments.quality_benchmark analyze results/quality_smoke

Available presets: numerical, physics, finite_gate, coverage, noise and rare.
Production presets default to4096/5120,512 trajectories/cardinal state,
eight cycles and all three controllers; every budget is in plan.json.
The smoke preset fixes a tiny budget. The finite_gate preset uses small
cutoffs/resources; use --per-state 2 --rounds 2 for bounded software checks
and a separately planned larger budget for statistical comparisons.
Physics settings are project-selected
interventions, not literature-prescribed optima or calibrated hardware.

--per-state, --rounds, --cutoffs, --seeds, --decoders and --chunks specify
fixed budgets. Numerical tolerances and diagnostic energy-event definitions
are frozen in the plan. Source snapshots, environment, checkpoint hashes,
plan hash and per-job raw hashes are retained. Execution rejects source
changes and damaged completed observations; reusing an unchanged plan resumes
completed jobs. --max-jobs bounds a work session without declaring the
remaining jobs complete. Output-space estimates are checked before running.
An OS lock prevents duplicate writers; completed raw arrays have integrity
seals and partial writes are atomic.

--matrix MATRIX.json supplies custom points and variants. Its points object
maps a safe point ID to [sigma_q/L,sigma_m/L,rho]; variants maps a safe variant
ID to an object with physics and noise fields. It replaces the corresponding
preset axes and is frozen with the plan; no benchmark settings are inferred
from observed test performance.

The analyzer aggregates paired cutoff differences and pointwise fidelity
intervals across independent chunks. It reads matching192/384/768 scoring
reports and keeps missing refinements visible. Attach an independently
validated instrument report with:

    .venv/bin/python -m experiments.quality_benchmark attach-instrument PLAN_DIRECTORY INSTRUMENT_REPORT_DIRECTORY

Attached controls are hash-audited and matched by data cutoff, resource width
and coupling. They remain supplied-state single-instrument checks, not an
all-state channel bound or proof of the lumped finite-duration approximation.
The complete empirical-check status is separate from rigorous certification.

Rare-event plans use --rare-target P and a precomputed fixed sample budget.
They do not stop when an attractive error count appears. Event summaries
report counts and state-stratified Clopper–Pearson bounds with Bonferroni
correction over states/events/configurations. Energy limit/jump events and
numerical-boundary failures remain separate; zero events never imply zero
probability. Interrupted groups get interim descriptive labels. Independent
chunks/seeds increase noise samples; training seeds are not substituted for
new trajectories. Mean fidelity, numerical status and event rates are
separately reported in summary.json and index.html.
Scientific plots are exported as PNG/SVG with counts/intervals in events.csv.
The interval construction follows [Clopper and Pearson1934](https://doi.org/10.1093/biomet/26.4.404);
the state/configuration Bonferroni allocation is our analysis choice.

Ordinary configurable reference example:

    .venv/bin/python -m experiments.qutip_reference --smoke --decoders nearest --physics-json physics.json --params .20 .05 .90

Read the saved plan and run scoped convergence checks before starting the
expensive presets. Completion of software or jobs does not establish hardware
validity, all-state channel accuracy, or a successful published reproduction.
