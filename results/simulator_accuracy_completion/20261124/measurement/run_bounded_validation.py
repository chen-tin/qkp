"""Local-only fixed-input instrument control with wall/RSS limits."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

root=Path.cwd()
evidence=root/'results/simulator_accuracy_completion/20261124/measurement'
output=evidence/'full4096_cycle8_stage_a'
source=root/'results/simulator_accuracy_actions/20261123/root/archive_spot_check/nearest_finite_ancilla.npz'
command=[str(root/'.venv/bin/python'),'-u','-m','experiments.validate_reference_states',
 '--states',str(source),'--cycle','8','--columns','0','8','--streaming','--chunk','64',
 '--ancilla-cutoffs','768','1024','--sigma','0.0886226925452758','--radius','60','--dy','.01',
 '--max-joint','7000000','--max-scratch-mb','3500','--output',str(output)]
code=['quantum/streaming_measurement.py','quantum/measurement_benchmark.py','quantum/continuous_ancilla.py','experiments/validate_reference_states.py']
plan={'registered_utc':datetime.now(timezone.utc).isoformat(),'status':'Planned','classification':'local production-snapshot single-instrument accuracy control, not multicycle/hardware/all-state certification','question':'Does the independent full-Fock circuit agree with the scalable ideal instrument on a saved representative and hot production state?','inputs':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cycle':8,'snapshot_columns':[0,8],'original_trial_ids':[0,211],'cardinal_ids':[0,0],'mean_photon_numbers':[196.1440220035597,1892.8939145178729],'data_cutoff':4096},'fixed':{'sigma_absolute_q':.0886226925452758,'hbar':1,'L':'sqrt(pi)','coupling':1.,'axes':['q','p'],'rounds':'one conditional extraction on each saved state','random_seeds':'none, deterministic sesolve and quadrature','training_seeds':'none','chunk':64,'max_wall_seconds':1200,'max_rss_bytes':4*1024**3,'blas_threads':2,'omp_threads':2,'no_cloud':True},'numerical_refinement_ladder':{'ancilla_cutoffs':[768,1024,1536],'radii':[60.,120.],'dy':[.01,.005],'sequence':'StageA usesNa768/1024,radius60,dy.01. If outcome mass is incomplete, expand radius to120; comparedy.005. If ancillary convergence/boundary fails, addNa1536 only within the same wall/RSS cap. No condition or tolerance changes; all failed controls retained.'},'hypothesis':'At adequate ancillary cutoff/window/grid, representative and hot state instrument differences satisfy unchanged empirical numerical criteria. StageA may expose unsupported ancillary/window coverage.','decision_rule':{'support':'Final ancillary control passes all fixed TV<=1e-4,conditionalinfidelity<=1e-5,masserror<=1e-5,norm<=1e-7,ancillaryedge<1e-4,gridinfidelitychange<=1e-6,gridTVchange<=1e-4 and ancillary refinement discrepancies.','contradict':'Converged numerical controls show a discrepancy above these fixed tolerances.','inconclusive':'Any capacity/time limit, missing window mass, failing ancillary boundary/refinement or unconverged grid prevents support; retain exact failing check and scope.'},'command':command,'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in code}}
(evidence/'production_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
start=time.monotonic();peak=0;stop_reason=None
with (evidence/'production_stage_a.log').open('w') as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 launch={'started_utc':datetime.now(timezone.utc).isoformat(),'pid':process.pid,'command':command,'status':'Running'}
 (evidence/'production_launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 while process.poll() is None:
  observed=subprocess.run(['ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True)
  rss=int(observed.stdout.strip() or 0)*1024;peak=max(peak,rss)
  elapsed=time.monotonic()-start
  (evidence/'production_monitor.json').write_text(json.dumps({'updated_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':elapsed,'rss_bytes':rss,'peak_rss_bytes':peak,'pid':process.pid})+'\n')
  if elapsed>1200:stop_reason='wall_time_cap'
  if rss>4*1024**3:stop_reason='rss_cap'
  if stop_reason:
   os.killpg(process.pid,signal.SIGTERM)
   try:process.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
   break
  time.sleep(2)
 code=process.wait()
 receipt={'finished_utc':datetime.now(timezone.utc).isoformat(),'exit_code':code,'stop_reason':stop_reason,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':peak,'status':'Completed execution' if code==0 else 'Interrupted/inconclusive','output':str(output)}
 (evidence/'production_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt),flush=True)
