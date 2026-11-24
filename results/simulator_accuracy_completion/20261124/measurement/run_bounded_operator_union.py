"""Local preregistered union-spectrum operator ladder; sampled RSS guard."""
import os,sys,subprocess,time,signal,json,re,hashlib,shutil,copy
from pathlib import Path
from datetime import datetime,timezone

root=Path.cwd();base=root/'results/simulator_accuracy_completion/20261124/measurement'
venv=str(root/'.venv/bin/python')  # lexical venv path; never resolve symlink
source_paths=['experiments/validate_instrument_operator.py','quantum/instrument_operator_bound.py','quantum/spectral_circuit.py','quantum/measurement_benchmark.py','quantum/continuous_ancilla.py']
source_hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_paths}
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def available_memory():
 text=subprocess.run(['vm_stat'],capture_output=True,text=True,check=True).stdout
 page=int(re.search(r'page size of (\d+) bytes',text).group(1))
 counts={k:int(v) for k,v in re.findall(r'([^\n:]+):\s+(\d+)\.',text)}
 return page*sum(counts.get(k,0) for k in ['Pages free','Pages inactive','Pages speculative'])

sample_audit=base/'production_sample_audit_receipt_v2.json'
wait_started=time.monotonic()
while not sample_audit.exists():
 if time.monotonic()-wait_started>1200:raise RuntimeError('Sampleaudit didnotpublishreceipt within waitcap; uniform remainsunlaunched')
 save(base/'operator_wait.json',{'status':'WaitingForCompletedSampleAudit','updated_utc':datetime.now(timezone.utc).isoformat(),'no_uniform_compute_started':True})
 time.sleep(2)
if json.loads(sample_audit.read_text())['exit_code']!=0:
 raise RuntimeError('Completed passing sealed-sample audit required before uniform operator launch')
free=available_memory()
if free<6.5*1024**3:
 save(base/'operator_launch_blocked.json',{'status':'Planned','reason':'Conservative free/reclaimable memory below6.5GiB','available_bytes':free});sys.exit(2)

plan={'registered_utc':datetime.now(timezone.utc).isoformat(),'status':'Planned','question':'Do independent finite-ancilla SUM Kraus functions agree uniformly on both complete production data-operator spectra?',
 'fixed':{'data_cutoffs':[4096,5120],'all_lambda_rows':9216,'sigma_absolute_q':.0886226925452758,'hbar':1,'L':'sqrt(pi)','coupling':1.,'radius':160.,'dy':.01,'transfer_row_chunk':128,'outcome_chunk':512,'axes':'q directly, p by exact finite-Fourier conjugacy independently controlled against small q/p sesolve','trajectories':'none; deterministic operator calculation','training_seeds':'none','threads':2},
 'hypothesis':'At adequate ancillary cutoff the diagonal finite-grid Stinespring difference is below2.5e-4 for every represented data eigenvalue at BOTH cutoffs.',
 'decision_rule':{'support':'Each final cutoff delta<=2.5e-4,masserrors<=1e-5,normerror<=1e-7,boundary<1e-4,gridnormsensitivity<=1e-6,gridmasssensitivity<=1e-5,and ancillaryrefinementdelta<=2.5e-4.','contradict':'A converged independent finite-grid calculation exceeds the error criterion.','inconclusive':'Unconvergedwindow/grid/ancilla, timeout, capacity or missingcoverage; retain failed levels and partial evidence.'},
 'conditional_refinement_ladder':{'ancilla_initial':[8192,12288],'conditional_third':16384,'third_trigger':'Onlyif8192→12288refinementfails while finalreference-candidate delta<=2.5e-4; sameδthreshold and newoutputidentity.','dy_refinement':.005,'dy_trigger':'Onlyif fixedgridnorm/masssensitivitygates fail and within remaining shared wall/RSS budget.','optional_coupling_1_01':{'status':'Planned','reason':'Mandatory bothdatacutoffs atgain1.0 takepriority; no inference to1.01.'}},
 'capacity':json.loads((base/'operator_union_preflight.json').read_text()),'resources':{'max_total_seconds':1200,'target_rss_bytes':4*1024**3,'sampled_rss_kill_trigger_bytes':5*1024**3,'rss_poll_seconds':2,'rss_limit_scope':'Sampled guard, not kernel hardlimit; transient overshoot between polls is possible.','required_prelaunch_reclaimable_bytes':6.5*1024**3,'observed_available_bytes':free,'physical_ram_bytes':32*1024**3,'local_only':True},
 'source_sha256':source_hashes,'orchestrator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'control_evidence':['operator_union_gold_tests_final.log','operator_union_tiny_v2.log','operator_union_tiny_v2_audit.log','additional_basis_reference_12288.json','high_energy_evaluator_reference.json'],
 'scope':'Computed exact operator-norm identity on the declared finite grids, evaluated in floating point. Grid/window/ancilla refinements empirical; no rigorous continuum,physical Fockcutoff orhardwarecertificate.'}
save(base/'operator_union_registration.json',plan)
start=time.monotonic();peak=0;attempts=[];last_output=None;stop_reason=None

def command_for(output,ancillas,dy):
 return [venv,'-u','-m','experiments.validate_instrument_operator','--data-cutoffs','4096','5120','--ancilla-cutoffs',*[str(x) for x in ancillas],'--sigma','.0886226925452758','--coupling','1','--radius','160','--dy',str(dy),'--chunk','128','--grid-chunk','512','--max-scratch-gib','5','--output',str(output)]

def execute(label,output,ancillas,dy):
 global peak,stop_reason
 command=command_for(output,ancillas,dy)
 with (base/(label+'.log')).open('w') as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2'))
  launch={'started_utc':datetime.now(timezone.utc).isoformat(),'pid':p.pid,'command':command,'output':str(output)}
  save(base/(label+'_launch.json'),launch)
  while p.poll() is None:
   observed=subprocess.run(['ps','-o','rss=','-p',str(p.pid)],capture_output=True,text=True)
   rss=int(observed.stdout.strip() or 0)*1024;peak=max(peak,rss);elapsed=time.monotonic()-start
   save(base/'operator_union_monitor.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),'phase':label,'pid':p.pid,'elapsed_seconds':elapsed,'sampled_rss_bytes':rss,'peak_sampled_rss_bytes':peak})
   if elapsed>1200:stop_reason='shared_wall_time_cap'
   if rss>5*1024**3:stop_reason='sampled_rss_trigger'
   if stop_reason:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
    break
   time.sleep(2)
  code=p.wait();attempt={'phase':label,'exit_code':code,'output':str(output),'elapsed_total_seconds':time.monotonic()-start,'stop_reason':stop_reason}
  attempts.append(attempt);save(base/(label+'_receipt.json'),attempt)
  return code


def seed_transfer_and_curves(parent,child,new_ancillas,new_dy,copy_curves):
 old=json.loads((parent/'result.json').read_text());config=copy.deepcopy(old['config'])
 config['ancilla_cutoffs']=new_ancillas;config['dy']=new_dy
 child.mkdir(parents=True,exist_ok=True)
 for path in parent.glob('transfer_Na*'):
  if path.suffix in ('.npy','.json'):shutil.copyfile(path,child/path.name)
 if copy_curves:
  for path in parent.glob('curves_Na*.npz'):
   shutil.copyfile(path,child/path.name)
   metadata=path.with_suffix('.json');item=json.loads(metadata.read_text())
   item['identity']['config']=config
   item['migration']={'from':str(metadata),'source_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),'reason':'Predeclared thirdancillaryrung only; underlying per-pair geometry/configuration unchanged'}
   save(child/metadata.name,item)
 save(child/'migration.json',{'parent':str(parent),'parent_result_sha256':hashlib.sha256((parent/'result.json').read_bytes()).hexdigest(),'new_ancilla_cutoffs':new_ancillas,'new_dy':new_dy,'raw_bytes_preserved':True,'curve_cache_reuse':copy_curves})

output=base/'uniform_operator_union_v1';ancillas=[8192,12288];dy=.01
code=execute('operator_union_base',output,ancillas,dy);last_output=output
if code==0:
 result=json.loads((output/'result.json').read_text());final=[r for r in result['rows'] if r['ancilla_cutoff']==ancillas[-1]]
 need_third=any(r.get('ancilla_refinement_operator_norm',0)>2.5e-4 for r in final) and all(r['finite_grid_stinespring_operator_norm']<=2.5e-4 for r in final)
 estimated_third=60.+1.5*max(60.,max((output/f'curves_Na{ancillas[-1]}_Nd{n}.npz').stat().st_mtime for n in [4096,5120])-(output/f'transfer_Na{ancillas[-1]}.json').stat().st_mtime)
 if need_third and 1200-(time.monotonic()-start)>=estimated_third and available_memory()>=6.5*1024**3:
  extended=base/'uniform_operator_union_extended_v1';new_ancillas=[8192,12288,16384]
  seed_transfer_and_curves(output,extended,new_ancillas,dy,True)
  code=execute('operator_union_extended',extended,new_ancillas,dy);last_output=extended;ancillas=new_ancillas
 elif need_third:
  save(base/'operator_third_planned.json',{'status':'Planned','reason':'Thirdrung required but remaining shared wallbudget ormemory gate prevents launch','elapsed_seconds':time.monotonic()-start,'estimated_required_seconds':estimated_third})
if code==0:
 result=json.loads((last_output/'result.json').read_text());final=[r for r in result['rows'] if r['ancilla_cutoff']==ancillas[-1]]
 need_grid=any(r['grid_norm_sensitivity']>1e-6 or r['grid_mass_sensitivity']>1e-5 for r in final)
 estimated_grid=max(120.,2*(time.monotonic()-start))
 if need_grid and 1200-(time.monotonic()-start)>=estimated_grid and available_memory()>=6.5*1024**3:
  refined=base/'uniform_operator_union_grid_refined_v1'
  seed_transfer_and_curves(last_output,refined,ancillas,.005,False)
  code=execute('operator_union_grid_refined',refined,ancillas,.005);last_output=refined
 elif need_grid:
  save(base/'operator_grid_planned.json',{'status':'Planned','reason':'Finergrid required but remaining shared wallbudget ormemory gate prevents launch'})
receipt={'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'peak_sampled_rss_bytes':peak,'rss_sampling_seconds':2,'rss_limit_scope':'Sampled trigger, not kernel-enforced bound','stop_reason':stop_reason,'attempts':attempts,'last_output':str(last_output),'last_exit_code':code}
save(base/'operator_union_receipt.json',receipt);print(json.dumps(receipt),flush=True)
