# Job parallelism for the existing EC2 run

Engineering change, 20 November 2026. This changes scheduling and thread limits;
the frozen scientific sources, plans, seeds, controller, raw observations and
acceptance criteria remain unchanged. The original OS shutdown deadline remains.
No new instance or longer compute window is requested by these commands.

## Design and decision rule

The original supervisor runs at most two suites and one job per suite. Each
job uses two BLAS threads, leaving roughly half of eight vCPUs unused during
scoring and more idle after one suite completes. The replacement runs up to
four distinct jobs with two threads each. It retains numerical-baseline,
physics/noise, and coverage/rare phase barriers. It interleaves the two suites
within a phase, sharing all four slots when one suite is finished.

The original simulation and scoring functions are called directly, with their
registered budgets and 192/384/768 quadrature. There is no new mathematical
scorer in this change. Frozen source/checkpoint hashes are verified before
handoff. A parent owns all selected plan locks; each child owns one job lock
and writes only that job. There is one serialized full-report writer. Full
reports refresh at phase boundaries and at most every15minutes while jobs
finish; lightweight status and the independent live exporter continue every30s.
Each completed job is checkpointed separately to S3 before its slot is reused.

Resume skips sealed completed observations and already checked refinements,
including checked negative numerical outcomes. Every scheduling epoch saves
its source, hash, thread settings, original plan hashes, per-job receipts and
handoff record under `execution_epochs/`. Original environment/source records
are preserved; the added execution settings are not silently merged into them.

Before deployment, the control and parallel pilot below must have identical
plan hashes, complete counts and scoring/gate outcomes; compare every raw
array with absolute tolerance1e-12 and no relative tolerance. Serialized NPZ
hashes and elapsed-time fields need not match because archive metadata and
runtime are not scientific observations. Resume must leave completed result,
raw and refinement files unchanged. Duplicate writer and CPU-budget tests
must pass. Pilot speed is descriptive only; its tiny matrices do not estimate
production speedup or establish finite-energy accuracy.

## Independent bounded pilot

Run from qkp/; choose a new output directory. The smoke preset declares two
small-cutoff jobs (32/48), two trajectories per cardinal state, two cycles,
nearest-lattice, with its own broad diagnostic noise. Failed population gates
are expected and must remain failures. Only job concurrency changes between
the matched arms; both use one BLAS thread per job.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python - <<'PY'
from pathlib import Path
from experiments.quality_benchmark import build_plan, write_plan
root = Path('results/simulation_quality_infrastructure/parallelism_pilot_20261120')
plan = build_plan('smoke')
for arm in ('control', 'parallel'):
    write_plan(plan, root / arm / 'plans' / 'smoke')
PY

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python ops/ec2/parallel_worker.py run --root results/simulation_quality_infrastructure/parallelism_pilot_20261120/control --run-id parallelism-control --presets smoke --workers 1 --threads 1 --no-sync --no-shutdown
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python ops/ec2/parallel_worker.py run --root results/simulation_quality_infrastructure/parallelism_pilot_20261120/parallel --run-id parallelism-parallel --presets smoke --workers 2 --threads 1 --no-sync --no-shutdown
```

Compare each job's raw arrays, results excluding timings/raw-file hashes, and
the numerical gate/refinement pass flags. Record actual commands, measured
durations and comparison results in an audit JSON beside the pilot. Repeat
the parallel command and verify that completed raw/results/refinements retain
their hashes. Do not compare hashes of generated figures or progress files.

Targeted scheduler checks:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_ec2_parallel.py
```

## Live handoff

Deploy this as a new, hashed orchestration file; do not overwrite the original
worker or any source in the frozen plan. First verify current instance identity,
worker PID/command/cwd and the existing OS deadline. The replacement validates
the old PID, pauses only the original supervisor, waits for its existing child
steps to finish, and then kills only the drained supervisor so its old shutdown
finally block does not run. It never kills an active scientific child. Timeout
or cancellation resumes the original supervisor. The bootstrap execs Python,
so the old shell error trap is no longer present. A separately managed service
runs the replacement; the existing shutdown timer is neither reset nor removed.

Use the verified original supervisor PID, not this historical example blindly:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /opt/qkp/.venv/bin/python -u /opt/qkp/ops/ec2/parallel_worker.py run --root /opt/qkp/results/ec2_production/qkp-20261120-003030-d2292a --run-id qkp-20261120-003030-d2292a --bucket qkp-bench-882574060060-20261120-003030-d2292a --workers 4 --threads 2 --drain-pid ORIGINAL_VERIFIED_PID
```

Verify the handoff receipt, four distinct active job PIDs, their thread settings,
CPU/memory use, identical original plan/source hashes, live metric updates and
the unchanged OS deadline. No measured production speedup is claimed until
completed-job timing receipts are available. A failed replacement leaves the
machine under its original OS deadline; investigate its service log and epoch
before restarting anything.
