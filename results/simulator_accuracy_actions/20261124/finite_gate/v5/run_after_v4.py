"""Extend only a resource-interrupted v4 using sealed unchanged numerical rows."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,signal,subprocess,time
folder=Path(__file__).resolve().parent
previous=folder.parent/'v4'
while not (previous/'resource_usage.json').exists():
    time.sleep(5)
prior=json.loads((previous/'resource_usage.json').read_text())
if prior['returncode']==0:
    (folder/'not_launched.json').write_text(json.dumps({'reason':'v4 completed without resource interruption','previous':str(previous)},indent=2)+'\n')
    raise SystemExit(0)
if prior.get('interruption_reason') not in ('wall_time_cap','hard_rss_cap'):
    raise RuntimeError('v4 failed for a non-resource reason; investigate before continuation')
plan=json.loads((folder/'plan.json').read_text());runtime=folder/'runtime'
for name,expected in plan['source_sha256'].items():
    assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==expected
    dst=folder/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(runtime/name,dst)
repo=folder.parents[4]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(repo))
interpreter=repo/'.venv/bin/python'
command=[str(interpreter),'-u','-m','experiments.local_finite_gate_audit','run',str(folder)]
start=time.monotonic();samples=[];reason=None
with (folder/'run.log').open('w') as log:
    process=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT)
    print('PID',process.pid,flush=True)
    while process.poll() is None:
        stat=subprocess.run(['ps','-o','rss=,%cpu=','-p',str(process.pid)],capture_output=True,text=True)
        vals=stat.stdout.split()
        if vals:
            rss=int(vals[0])*1024;samples.append({'seconds':time.monotonic()-start,'rss_bytes':rss,'cpu_percent':float(vals[1])})
            if rss>plan['limits']['hard_rss_bytes']:reason='hard_rss_cap'
        if time.monotonic()-start>plan['limits']['wall_seconds']:reason='wall_time_cap'
        if reason:
            process.send_signal(signal.SIGINT)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            break
        time.sleep(1)
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'pid':process.pid,'returncode':process.returncode,
        'interruption_reason':reason,'elapsed_seconds':time.monotonic()-start,
        'max_sampled_rss_bytes':max((x['rss_bytes'] for x in samples),default=0),'samples':samples,'command':command,
        'cwd':str(runtime),'thread_environment':{k:env[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH']}}
    (folder/'resource_usage.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='samples'},indent=2),flush=True)
