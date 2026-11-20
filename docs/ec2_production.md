# Production benchmark execution on EC2

Launched19November2026 local time (20NovemberUTC) at the user's request.
Account882574060060, region us-east-1, profile AdministratorAccess-882574060060.

For the GitHub checkout, see the [dated production snapshot](../results/ec2_runs/production_20261119/snapshots/20261120T194213Z/README.md)
captured20November2026 at19:42UTC. It contains the environment, validation
receipt, plans, compact summaries and plots. This is partial evidence, not a
live feed or a completed full benchmark. The metrics/ links below describe
the operator's live local mirror, which is excluded from Git alongside raw
arrays and launch payloads.

## Active run

- Run: qkp-20261120-003030-d2292a.
- Instance: i-0be3c3be10368ba15, m7i.2xlarge,8vCPU/32GiB.
- Storage:400GiB encrypted gp3, plus private S3 raw/metric archives.
- Source/checkpoint bundle:5,294,028bytes; exact digest/file manifest and all
  newly created resource IDs are in
  [manifest.json](../results/ec2_runs/production_20261119/manifest.json).
- No inbound ports; Systems Manager provides management. The new instance
  role is restricted to its artifact prefix plus the standard SSM role.
- Compute quote:.4032USD/hour from AWS Pricing API, effective1November2026.
  EBS, public IPv4 and S3 have separate charges.

The worker schedules an OS shutdown after24hours and stops sooner when it
finishes, fails, or finds an invalid/inconclusive numerical baseline. Shutdown
stops the instance; it does not delete EBS or S3 data. Retained storage remains
billable. Cleanup/termination is separate from stopping and must preserve the
scientific raw archive.

## Environment and validation

Python3.13.13; NumPy2.5.5; SciPy1.18.1; QuTiP5.3.1; Torch2.15.0+cpu;
Matplotlib3.11.2; pytest9.2.0. BLAS/OpenMP threads2, Torch inference1.
The exact Linux platform/library record is in
[environment.json](../results/ec2_runs/production_20261119/metrics/environment.json).
This is a distinct execution platform from macOS; it is not silently merged
with historical runs.

The selected50-test validation suite **passed on EC2 in16.86s** before
production. See [remote validation](../results/ec2_runs/production_20261119/metrics/remote_validation.json)
and the fetched remote_validation.log under metrics/logs/.

## Execution sequence

The worker creates fresh remote plans with the actual Linux environment and
unchanged scientific source/checkpoint hashes. It runs:

1. Numerical baseline:4096/5120 levels,512 trajectories for each of six
   cardinal states,8cycles, nearest/Bayesian/frozen LSTM controllers.
2. Explicit finite-gate controls with their separately declared16/state,
   2cycle budget.
3. Saved-state192/384/768 scoring and paired-cutoff checks.
4. Physics/noise, then coverage/rare stages, **only if** the numerical baseline
   passes its registered stage/scoring/cutoff checks.

At most two suites run concurrently. Numerical failure is retained and keeps
broader sweeps queued; independent-instrument coverage remains a separately
reported limitation. Software completion is not numerical or hardware
certification.

## Metrics and local HTML

The instance retains raw arrays and checkpoints them under its private S3
run prefix. A read-only exporter publishes per-decoder curves as soon as a
decoder completes. The local fetcher downloads only metrics/figures/JSON/CSV
and logs, never the large rawNPZ arrays.

Local report:

    http://127.0.0.1:8765/results/ec2_runs/production_20261119/metrics/live/

Full suite status:

    http://127.0.0.1:8765/results/ec2_runs/production_20261119/metrics/

The local fetch process polls every30seconds. HTML also refreshes every30s.
Eight-cycle curves have nine cycle samples including initialization; the
finite-gate control curves have three. Failed stage gates produce diagnostic
curves, explicitly labeled.

## Operations from qkp/

Fetch once:

    .venv/bin/python ops/ec2/manage.py fetch results/ec2_runs/production_20261119

Fetch continuously:

    .venv/bin/python -u ops/ec2/fetch_metrics.py results/ec2_runs/production_20261119 --interval 30

Inspect instance/SSM state:

    .venv/bin/python ops/ec2/manage.py status results/ec2_runs/production_20261119

Stop this instance explicitly if needed:

    .venv/bin/python ops/ec2/manage.py stop results/ec2_runs/production_20261119

Do not redeploy the same manifest or start a duplicate worker. Source bundle,
bootstrap, launch specification, source/observer hashes, resource IDs and SSM
command IDs are preserved in the run directory. Raw data is separate from
the downloaded metric tree. The metrics/status.json and per-stage summaries
are the current status, rather than this launch-time note.

Method and metric conventions are in [simulation_quality.md](simulation_quality.md).
AWS [instance lifecycle](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)
documents stop/termination and volume retention; the
[SSM access documentation](https://docs.aws.amazon.com/cloud9/latest/user-guide/ec2-ssm.html)
describes management without inbound ports.
