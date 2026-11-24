"""Audit matched local scheduling pilots and one no-work resume."""
from pathlib import Path
import hashlib,json,os,subprocess,time
import numpy as np
from experiments.quality_benchmark import load_plan
ROOT=Path('results/simulation_quality_infrastructure/parallelism_pilot_20261120')
control=ROOT/'control/plans/smoke'; parallel=ROOT/'parallel/plans/smoke'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(value,omit):
 if isinstance(value,dict):return {k:clean(v,omit) for k,v in value.items() if k not in omit}
 if isinstance(value,list):return [clean(v,omit) for v in value]
 return value
cp=load_plan(control,current_sources=True); pp=load_plan(parallel,current_sources=True)
assert cp==pp
assert (control/'plan.sha256.json').read_bytes()==(parallel/'plan.sha256.json').read_bytes()
plan_hash=json.loads((control/'plan.sha256.json').read_text())['sha256']
report=dict(status='running',classification='bounded local scheduling validation; no AWS and no production trajectories',plan_sha256=plan_hash,raw_files=0,raw_arrays=0,raw_array_elements=0,max_absolute_difference=0.,jobs=[],atol=1e-12,rtol=0)
for job in cp['jobs']:
 jid=job['id'];a=control/'jobs'/jid;b=parallel/'jobs'/jid
 fa={p.relative_to(a):p for p in a.rglob('*.npz')};fb={p.relative_to(b):p for p in b.rglob('*.npz')}
 assert fa.keys()==fb.keys() and len(fa)==2
 for relative,path in fa.items():
  for raw in [path,fb[relative]]:
   assert sha(raw)==json.loads(raw.with_suffix('.sha256.json').read_text())['sha256']
  with np.load(path,allow_pickle=False) as ar,np.load(fb[relative],allow_pickle=False) as br:
   assert set(ar.files)==set(br.files)
   for key in ar.files:
    x,y=ar[key],br[key];assert x.shape==y.shape and x.dtype==y.dtype
    if np.issubdtype(x.dtype,np.number):
     np.testing.assert_allclose(x,y,rtol=0,atol=1e-12,equal_nan=True,err_msg=f'{jid}/{relative}/{key}')
     finite=np.isfinite(x)&np.isfinite(y)
     if np.any(finite):report['max_absolute_difference']=max(report['max_absolute_difference'],float(np.max(np.abs(x[finite]-y[finite]))))
    else:np.testing.assert_array_equal(x,y,err_msg=f'{jid}/{relative}/{key}')
    report['raw_arrays']+=1;report['raw_array_elements']+=x.size
  report['raw_files']+=1
 ra=json.loads((a/'result.json').read_text());rb=json.loads((b/'result.json').read_text())
 assert ra['complete'] and rb['complete']
 assert clean(ra,{'seconds','raw_hashes'})==clean(rb,{'seconds','raw_hashes'}),jid
 rfa=list((a/'quality').glob('logical_refinement_*.json'));rfb=list((b/'quality').glob('logical_refinement_*.json'))
 assert len(rfa)==len(rfb)==1
 da=json.loads(rfa[0].read_text());db=json.loads(rfb[0].read_text())
 assert clean(da,{'raw_file','raw_sha256','run_result_sha256'})==clean(db,{'raw_file','raw_sha256','run_result_sha256'}),jid
 assert not ra['rows'][0]['numerical_edge_gate_pass']
 report['jobs'].append(dict(id=jid,cutoff=job['cutoff'],per_state=job['config']['per_state'],rounds=job['config']['rounds'],point=job['config']['points'],seed=job['config']['seed'],decoders=job['config']['decoders'],raw_equal=True,result_equal_except_timing_and_archive_hashes=True,refinement_equal_except_paths_and_archive_hashes=True,numerical_gate_pass=False,refinement_pass=da['all_sample_checks_pass'],all_cycles_checked=da['all_cycles_checked']))
sa=json.loads((control/'summary.json').read_text());sb=json.loads((parallel/'summary.json').read_text())
for key in ['expected_jobs','completed_jobs','all_jobs_executed','classification','numerically_certified','registered_numerical_sample_checks_pass','all_registered_empirical_checks_pass']:
 assert sa[key]==sb[key]
report['summary_flags']={k:sa[k] for k in ['expected_jobs','completed_jobs','all_jobs_executed','numerically_certified','registered_numerical_sample_checks_pass','all_registered_empirical_checks_pass']}
assert sa['completed_jobs']==sa['expected_jobs']==2
files=sorted(p for p in (parallel/'jobs').rglob('*') if p.is_file() and (p.suffix=='.npz' or p.name=='result.json' or p.name.endswith('_summary.json') or p.name.startswith('logical_refinement_') or p.name.endswith('.sha256.json')))
before={str(p.relative_to(ROOT)):sha(p) for p in files}
(ROOT/'resume_hashes_before.json').write_text(json.dumps(before,indent=2)+'\n')
command=['.venv/bin/python','ops/ec2/parallel_worker.py','run','--root',str(ROOT/'parallel'),'--run-id','parallelism-parallel','--presets','smoke','--workers','2','--threads','1','--no-sync','--no-shutdown']
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
started=time.perf_counter()
with (ROOT/'parallel_resume_stdout.log').open('w') as log:res=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
elapsed=time.perf_counter()-started
assert res.returncode==0
assert before=={str(p.relative_to(ROOT)):sha(p) for p in files}
report['resume']=dict(returncode=res.returncode,seconds=elapsed,unchanged_scientific_files=len(files),hashes=before)
report['timings']=json.loads((ROOT/'timings.json').read_text())
report['orchestrator_sha256']=sha(Path('ops/ec2/parallel_worker.py'))
report['scope']='Only documented 2-job smoke arms were simulated. Resume launches no additional scientific trajectories. Cutoff 32/48 population gates remain invalid; pilot timing is descriptive and does not estimate production speedup. No AWS or external writes.'
report['status']='passed'
(ROOT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
with (ROOT/'commands.txt').open('a') as f:f.write('\n# Resume (same parallel command)\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '+' '.join(command)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='resume'},indent=2))
print('Resume unchanged files:',len(files),'seconds:',elapsed)
