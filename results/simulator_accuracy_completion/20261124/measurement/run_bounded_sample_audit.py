import os,sys,subprocess,time,signal,json
from pathlib import Path
from datetime import datetime,timezone
base=Path('results/simulator_accuracy_completion/20261124/measurement')
command=[str(Path('.venv/bin/python').resolve()),'-u','-m','experiments.validate_reference_states','--output',str(base/'full4096_cycle8_spectral_v1'),'--audit-only']
env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
start=time.monotonic();peak=0;reason=None
with (base/'production_sample_audit.log').open('w') as log:
 p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 (base/'production_sample_audit_launch.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),'pid':p.pid,'command':command,'classification':'sealed-raw audit; no repeated solver evolution'},indent=2)+'\n')
 while p.poll() is None:
  x=subprocess.run(['ps','-o','rss=','-p',str(p.pid)],capture_output=True,text=True)
  rss=int(x.stdout.strip() or 0)*1024;peak=max(peak,rss);elapsed=time.monotonic()-start
  (base/'production_sample_audit_monitor.json').write_text(json.dumps({'elapsed_seconds':elapsed,'rss_bytes':rss,'peak_rss_bytes':peak,'pid':p.pid})+'\n')
  if elapsed>1200:reason='wall_time_cap'
  if rss>4*1024**3:reason='rss_cap'
  if reason:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
   break
  time.sleep(2)
 code=p.wait()
 result={'finished_utc':datetime.now(timezone.utc).isoformat(),'exit_code':code,'stop_reason':reason,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':peak}
 (base/'production_sample_audit_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
