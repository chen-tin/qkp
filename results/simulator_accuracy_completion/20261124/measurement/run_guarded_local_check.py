"""Local command guard; lexical venv paths and sampled RSS are explicit."""
import argparse,os,subprocess,time,signal,json,re,sys,hashlib
from pathlib import Path
from datetime import datetime,timezone


def available():
 text=subprocess.run(['vm_stat'],capture_output=True,text=True,check=True).stdout
 size=int(re.search(r'page size of (\d+) bytes',text).group(1))
 counts={k:int(v) for k,v in re.findall(r'([^\n:]+):\s+(\d+)\.',text)}
 return size*sum(counts.get(k,0) for k in ['Pages free','Pages inactive','Pages speculative'])


def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--label',required=True);ap.add_argument('--directory',type=Path,required=True)
 ap.add_argument('--max-seconds',type=float,default=1200);ap.add_argument('--max-rss-gib',type=float,default=5.)
 ap.add_argument('--target-rss-gib',type=float,default=4.);ap.add_argument('--min-available-gib',type=float,default=6.5)
 ap.add_argument('--threads',type=int,choices=[1,2],default=2);ap.add_argument('command',nargs=argparse.REMAINDER)
 args=ap.parse_args();command=args.command
 if command and command[0]=='--':command=command[1:]
 if not command:ap.error('Command required after --')
 command[0]=os.path.abspath(command[0])  # do NOT resolve venv interpreter symlink
 args.directory.mkdir(parents=True,exist_ok=True);prefix=args.directory/args.label
 free=available()
 if free<args.min_available_gib*1024**3:
  save(prefix.with_name(prefix.name+'_receipt.json'),{'status':'PlannedMemoryGate','observed_available_bytes':free,'required_available_bytes':args.min_available_gib*1024**3,'compute_started':False});sys.exit(2)
 env=dict(os.environ,OPENBLAS_NUM_THREADS=str(args.threads),OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),VECLIB_MAXIMUM_THREADS=str(args.threads),PYTHONPATH=os.getcwd())
 start=time.monotonic();peak=0;reason=None
 with prefix.with_suffix('.log').open('w') as log:
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  save(prefix.with_name(prefix.name+'_launch.json'),{'started_utc':datetime.now(timezone.utc).isoformat(),'pid':process.pid,'command':command,'requested_threads':args.threads,'observed_prelaunch_available_bytes':free,'target_rss_bytes':args.target_rss_gib*1024**3,'sampled_rss_trigger_bytes':args.max_rss_gib*1024**3,'max_seconds':args.max_seconds,'guard_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
  while process.poll() is None:
   observed=subprocess.run(['ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True)
   rss=int(observed.stdout.strip() or 0)*1024;peak=max(peak,rss);elapsed=time.monotonic()-start
   save(prefix.with_name(prefix.name+'_monitor.json'),{'updated_utc':datetime.now(timezone.utc).isoformat(),'pid':process.pid,'elapsed_seconds':elapsed,'sampled_rss_bytes':rss,'peak_sampled_rss_bytes':peak})
   if elapsed>args.max_seconds:reason='wall_time_cap'
   if rss>args.max_rss_gib*1024**3:reason='sampled_rss_trigger'
   if reason:
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
    break
   time.sleep(2)
  code=process.wait()
 receipt={'finished_utc':datetime.now(timezone.utc).isoformat(),'exit_code':code,'stop_reason':reason,'elapsed_seconds':time.monotonic()-start,'peak_sampled_rss_bytes':peak,'rss_poll_seconds':2,'rss_scope':'Sampled trigger, not kernel-enforced peak bound; transient sampling overshoot possible.'}
 save(prefix.with_name(prefix.name+'_receipt.json'),receipt);print(json.dumps(receipt),flush=True)
 sys.exit(code if code>=0 else 128-code)


if __name__=='__main__':main()
