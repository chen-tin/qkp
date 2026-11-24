"""Resource-only revision: <=6 launches and <=10800 cumulative parent-active seconds."""
from datetime import datetime,timezone
import fcntl,hashlib,importlib.util,json,math,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/Users/christopherbabayan/programs/QUANTUM/qkp');sys.path.insert(0,str(ROOT))
FOLDER=ROOT/'results/simulator_accuracy_completion/20261124/remaining_physics';S=FOLDER/'sessions'
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=module(S/'monitor_fixed_jobs.py','old_monitor')
guard=module(ROOT/'results/simulator_accuracy_completion/20261124/resource_coordination/scheduling_guard.py','guard')
from experiments.continue_local_physics import validate,preflight,available_memory
from experiments.local_process_safety import spawn_owned,stop_owned,supervise_owned,validated_budget_history
save=old.save;sha=old.sha;utc=old.utc
PLAN='aec015d75eb3fa95eba8ef589f9374f4c6a05d97aec3593d21f3ac422fb749d6'
HEADROOM=512*1024**2

def retryable(state,logtext):
    if state.get('status')=='interrupted' and state.get('error','').startswith("TimeoutError('Registered local deadline reached')"):return 'clean_registered_timeout'
    if (state.get('status')=='failed' and state.get('error','').startswith("ValueError('Local memory preflight failed: estimate=9730785280, cap=12884901888, available=")
        and 'preflight(folder,plan,threads,max_rss_bytes,reserve_disk_bytes)' in logtext
        and not state.get('active_pid') and not state.get('active_job')):return 'next_job_memory_preflight_before_child'
    return None

def main():
    lock=(S/'.monitor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registration=json.loads((S/'resource_revision_launch_budget.json').read_text())
    if registration['manager_sha256']!=sha(__file__) or registration['guard_sha256']!=sha(guard.__file__):raise ValueError('Unregistered manager/guard source')
    if registration['process_safety_sha256']!=sha(ROOT/'experiments/local_process_safety.py'):raise ValueError('Unregistered process-safety helper')
    target=S/'budget_registry.json'
    previous_registry=json.loads(target.read_text()) if target.exists() else None
    prior=previous_registry if previous_registry is not None else json.loads((S/'registry.json').read_text())
    if prior.get('plan_sha256')!=PLAN:raise ValueError('Budget history belongs to another frozen plan')
    records=prior['sessions'];spent=validated_budget_history(records)
    if any(old.live(r['parent_pid']) for r in records):raise ValueError('Prior parent still active; refuse duplicate ownership')
    state=json.loads((FOLDER/'local_status.json').read_text())
    if len(records)>=3 and state.get('started_utc')!=records[-1]['terminal_status'].get('started_utc'):
        raise ValueError('Durable last session and scientific status disagree')
    lastlog=Path(records[-1].get('log',S/f'session_{len(records)}.log'))
    if not retryable(state,lastlog.read_text()):raise ValueError('Prior exit is not an authorized continuation')
    old.verify_frozen(ROOT,FOLDER,PLAN)
    registry={'created_utc':utc(),'manager_pid':os.getpid(),'manager_sha256':sha(__file__),'plan_sha256':PLAN,
              'maximum_parent_launches':6,'maximum_cumulative_parent_active_seconds':10800,'per_launch_max_work_seconds':3600,
              'graceful_cleanup_allowance_seconds':50,'kill_verification_allowance_seconds':5,
              'cumulative_active_seconds':spent,'sessions':records,'status':'waiting_for_resource_capacity',
              'previous_registry_sha256':sha(target) if target.exists() else None,
              'resource_only_revision':'resource_revision_launch_budget.json'}
    target=S/'budget_registry.json';save(target,registry);child=None
    try:
        while len(registry['sessions'])<6:
            old.verify_frozen(ROOT,FOLDER,PLAN);plan=validate(FOLDER)
            if all(j['processed'] for j in old.terminal_summary(FOLDER).values()):
                registry.update(status='all_four_fixed_jobs_processed',finished_utc=utc(),job_summary=old.terminal_summary(FOLDER));save(target,registry);return
            remaining=10800-spent
            if remaining<70:registry.update(status='cumulative_active_budget_exhausted',finished_utc=utc());save(target,registry);return
            while True:
                if (S/'STOP').exists():registry.update(status='stop_requested',finished_utc=utc());save(target,registry);return
                available=available_memory();reserve=guard.density_reservation();effective=available-reserve['density_remaining_growth_bytes']
                wait={'observed_utc':utc(),'actual_available_memory_bytes':available,'effective_available_memory_bytes':effective,
                      'extra_headroom_bytes':HEADROOM,**reserve}
                try:
                    before=preflight(FOLDER,plan,2,12*1024**3,8*1024**3,free_memory=available)
                    if effective<before['estimated_peak_bytes']+HEADROOM:raise ValueError('Manager effective memory/headroom reservation not met')
                    break
                except ValueError as error:
                    if 'memory' not in str(error):raise
                    wait['error']=str(error);save(S/'budget_memory_wait.json',wait)
                    registry.update(status='waiting_for_resource_capacity',latest_resource_wait=wait);save(target,registry);time.sleep(30)
            old.verify_frozen(ROOT,FOLDER,PLAN)
            number=len(registry['sessions'])+1;seconds=min(3600,math.floor(remaining-60));previous=state['started_utc']
            argv=[str(ROOT/'.venv/bin/python'),'-u','-m','experiments.continue_local_physics','run',str(FOLDER),
                  '--max-seconds',str(max(1,seconds-60)),'--threads','2','--max-rss-gib','12','--reserve-disk-gib','8','--no-sync','--no-shutdown']
            env=dict(os.environ)
            for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
            logpath=S/f'session_{number}.log';start=time.monotonic()
            with logpath.open('x') as log:child=spawn_owned(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record={'number':number,'launched_utc':utc(),'parent_pid':child.pid,'argv':argv,'preflight':before,'resource_reservation':wait,
                    'cumulative_seconds_before_launch':spent,'parent_work_deadline_seconds':seconds,'scientific_cooperative_deadline_seconds':max(1,seconds-60),'plan_sha256':PLAN,'log':str(logpath)}
            registry['sessions'].append(record);registry.update(status='running',active_parent_pid=child.pid)
            save(S/f'session_{number}.json',record);save(target,registry);print(json.dumps({'event':'launched','session':number,'pid':child.pid}),flush=True)
            execution=supervise_owned(child,seconds,started_monotonic=start,
                                        stop_requested=lambda:(S/'STOP').exists(),grace_seconds=50)
            wall=execution['parent_wall_seconds'];spent+=wall;state=json.loads((FOLDER/'local_status.json').read_text())
            record.update(execution,terminal_observed_utc=utc(),terminal_status=state)
            if execution['external_stop_reason'] is not None:
                save(S/f'session_{number}.json',record);registry['cumulative_active_seconds']=spent
                registry.update(status='external_deadline_or_stop_no_auto_retry',finished_utc=utc());save(target,registry);return
            if state['started_utc']==previous:raise RuntimeError('No fresh terminal status; no automatic retry')
            old.verify_frozen(ROOT,FOLDER,PLAN);record['cumulative_job_summary']=old.terminal_summary(FOLDER)
            record['retry_classification']=retryable(state,logpath.read_text());registry['cumulative_active_seconds']=spent
            save(S/f'session_{number}.json',record);save(target,registry)
            if all(j['processed'] for j in record['cumulative_job_summary'].values()):
                registry.update(status='all_four_fixed_jobs_processed',finished_utc=utc(),job_summary=record['cumulative_job_summary']);save(target,registry);return
            if not record['retry_classification']:
                registry.update(status='blocked_non_authorized_error',finished_utc=utc());save(target,registry);return
        registry.update(status='six_parent_launch_budget_exhausted',finished_utc=utc());save(target,registry)
    except BaseException as error:
        if child is not None:stop_owned(child,grace_seconds=50)
        registry.update(status='manager_error',error=repr(error),finished_utc=utc());save(target,registry);raise
if __name__=='__main__':main()
