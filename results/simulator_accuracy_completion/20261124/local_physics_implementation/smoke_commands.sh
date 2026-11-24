#!/bin/bash
# Independent bounded software check. Run only after all active sources are final.
# Every command below uses --software-smoke: no production trajectory is launched.
# Give a fresh output directory as the first argument; run from repository root.
set -euo pipefail
QKP_SMOKE_ROOT=${1:?Provide a fresh software-check output directory}
QKP_ORIGINAL_PLAN=results/ec2_runs/production_20261119/status_checks/20261123T222312Z/metrics_snapshot/physics/plan.json
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m experiments.continue_local_physics plan "$QKP_ORIGINAL_PLAN" "$QKP_SMOKE_ROOT/control" --software-smoke --freeze --sources-final
.venv/bin/python -m experiments.continue_local_physics plan "$QKP_ORIGINAL_PLAN" "$QKP_SMOKE_ROOT/intervention" --software-smoke --freeze --sources-final
# Control: ordinary reference runner and original single-bank scorer.
.venv/bin/python - "$QKP_SMOKE_ROOT/control" <<'PY'
import sys
from pathlib import Path
from experiments.quality_benchmark import load_plan,run_plan,analyze_plan
from experiments.refine_logical_scores import refine_run
folder=Path(sys.argv[1]);plan=load_plan(folder,current_sources=True)
run_plan(folder)
for job in plan['jobs']:
    refine_run(folder/'jobs'/job['id'],bins=[192,384,768],tolerance=.001)
analyze_plan(folder)
PY
# Intervention: new bounded local supervisor and shared-basis scorer.
.venv/bin/python -m experiments.continue_local_physics run "$QKP_SMOKE_ROOT/intervention" --max-seconds 180 --threads 1 --max-rss-gib 2 --reserve-disk-gib .1 --no-sync --no-shutdown --allow-software-smoke
# Both arms retain nominal Gaussian AR(1) sigma_q/L=.2,sigma_m/L=.05,rho=.9,
# original two physics settings/three controllers and seeds. Declared software
# departures only: per_state2,rounds1,data cutoffs16/24,snapshot budgets/horizon.
# These small bases MUST remain invalid population controls, not new accuracy claims.
.venv/bin/python - "$QKP_SMOKE_ROOT" <<'PY'
from pathlib import Path
import hashlib,json,sys
import numpy as np
from experiments.quality_benchmark import identity
root=Path(sys.argv[1]);a=root/'control';b=root/'intervention'
plans=[json.loads((p/'plan.json').read_text()) for p in [a,b]]
assert identity(plans[0])==identity(plans[1])
arrays=0;elements=0
for job in plans[0]['jobs']:
    left=a/'jobs'/job['id'];right=b/'jobs'/job['id']
    for source in left.glob('*/*.npz'):
        other=right/source.relative_to(left)
        with np.load(source,allow_pickle=False) as x,np.load(other,allow_pickle=False) as y:
            assert set(x.files)==set(y.files)
            for key in x.files:
                if x[key].dtype.kind in 'fc':np.testing.assert_allclose(x[key],y[key],atol=1e-12,rtol=0,equal_nan=True)
                else:np.testing.assert_array_equal(x[key],y[key])
                arrays+=1;elements+=x[key].size
summaries=[json.loads((p/'summary.json').read_text()) for p in [a,b]]
assert all(s['completed_jobs']==4 and s['stage_failures']>=4 for s in summaries)
assert all(not s['registered_numerical_sample_checks_pass'] and not s['numerically_certified'] for s in summaries)
assert [[v['pass'] for v in s['scoring_refinement'].values()] for s in summaries]==[[True]*4,[True]*4]
assert {k:v['pass'] for k,v in summaries[0]['cutoff_equivalence'].items()}=={k:v['pass'] for k,v in summaries[1]['cutoff_equivalence'].items()}
files={str(p.relative_to(b)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (b/'jobs').rglob('*') if p.is_file()}
report={'classification':'bounded software check; intentionally invalid population controls','raw_array_comparisons':arrays,
        'elements':elements,'atol':1e-12,'rtol':0,'completed_jobs_per_arm':4,
        'stage_failures':[s['stage_failures'] for s in summaries],
        'intervention_scientific_file_sha256':files,'note':'Pilot runtimes do not estimate production speedup.'}
(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='intervention_scientific_file_sha256'},indent=2))
PY
# Resume should reuse all observations/refinements without changing their bytes.
.venv/bin/python -m experiments.continue_local_physics run "$QKP_SMOKE_ROOT/intervention" --max-seconds 180 --threads 1 --max-rss-gib 2 --reserve-disk-gib .1 --no-sync --no-shutdown --allow-software-smoke
.venv/bin/python - "$QKP_SMOKE_ROOT" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);report=json.loads((root/'audit.json').read_text())
for rel,digest in report['intervention_scientific_file_sha256'].items():
    assert hashlib.sha256((root/'intervention'/rel).read_bytes()).hexdigest()==digest
report['resume_preserved_all_scientific_hashes']=True
(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('Resume preserved all scientific file hashes.')
PY
