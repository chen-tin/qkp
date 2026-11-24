"""Monitor at most three one-hour local sessions of one frozen four-job plan.

Only a clean registered TimeoutError permits resumption; never retry numerical
failures to obtain a pass, and never change settings, seeds, source or job IDs.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

HEADROOM_BYTES=512*1024**2


def utc():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def save(path,value):
    p=Path(path);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)


def ps(pid,field):
    return subprocess.run(['ps','-p',str(pid),'-o',field+'='],capture_output=True,text=True).stdout.strip()


def live(pid):
    state=ps(pid,'stat');return bool(state) and not state.startswith('Z')


def checked_parent(pid,folder,repo):
    argv=shlex.split(ps(pid,'command'))
    try:
        index=argv.index('experiments.continue_local_physics')
        path=Path(argv[index+2]);path=path if path.is_absolute() else repo/path
        return argv[index+1]=='run' and path.resolve()==folder
    except (ValueError,IndexError):return False


def verify_frozen(repo,folder,plan_sha):
    if sha(folder/'plan.json')!=plan_sha:raise ValueError('Frozen plan bytes changed')
    plan=json.loads((folder/'plan.json').read_text())
    if len(plan['jobs'])!=4 or {j['variant'] for j in plan['jobs']}!={'finite_duration_lumped','passive_cooling'}:
        raise ValueError('Unexpected scientific job scope')
    for rel,digest in plan['source_hashes'].items():
        if sha(repo/rel)!=digest or sha(folder/'source_snapshot'/rel)!=digest:
            raise ValueError('Frozen source changed: '+rel)
    for raw in (folder/'jobs').glob('*/*/*.npz'):
        if not raw.name.endswith('.tmp.npz') and not raw.with_suffix('.sha256.json').exists():
            raise ValueError('Unsealed non-temporary observation blocks safe resume: '+str(raw))
    for seal in (folder/'jobs').glob('*/*/*.sha256.json'):
        raw=seal.with_name(seal.name[:-12]+'.npz')
        if not raw.exists() or sha(raw)!=json.loads(seal.read_text())['sha256']:
            raise ValueError('Partial/complete sealed observation changed: '+str(raw))
    return plan


def terminal_summary(folder):
    from experiments import quality_benchmark as qb
    plan=qb.load_plan(folder,current_sources=True);summary={}
    for job in plan['jobs']:
        result=qb.verify_job(folder,job)
        score=qb.scoring_status(folder,job,plan['metric_contract']['scoring_bins'])
        summary[job['id']]={'variant':job['variant'],'cutoff':job['cutoff'],'observations_saved':result is not None,
                            'scoring_status':score['status'],'scoring_pass':score['pass'],
                            'processed':result is not None and score['status']=='checked',
                            'row_stage_passes':None if result is None else [r['numerical_edge_gate_pass'] for r in result['rows']]}
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--initial-parent',type=int,required=True);parser.add_argument('--resume-monitor',action='store_true');parser.add_argument('--resume-after-timeout',action='store_true');args=parser.parse_args()
    repo=args.repo.resolve();folder=args.folder.resolve();sessions=folder/'sessions';sessions.mkdir(exist_ok=True)
    sys.path.insert(0,str(repo));plan_sha=sha(folder/'plan.json');stop=sessions/'STOP'
    lock=(sessions/'.monitor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registry_path=sessions/'registry.json'
    if registry_path.exists() and not (args.resume_monitor or args.resume_after_timeout):raise ValueError('Monitor registry already exists; do not duplicate session ownership')
    state=json.loads((folder/'local_status.json').read_text())
    if args.resume_after_timeout:
        if (state['status']!='interrupted' or not state.get('error','').startswith('TimeoutError(') or live(args.initial_parent)):
            raise ValueError('Resource-only restart requires a fresh clean timeout and exited original supervisor')
    elif state['status']!='running' or not checked_parent(args.initial_parent,folder,repo):
        raise ValueError('Initial root-launched successful session not active')
    verify_frozen(repo,folder,plan_sha)
    registry={'monitor_started_utc':utc(),'monitor_pid':os.getpid(),'plan_sha256':plan_sha,
              'maximum_sessions':3,'seconds_per_session':3600,'source_policy':'No source/scientific setting/job/seed changes',
              'sessions':[],'status':'monitoring'}
    parent=args.initial_parent;owned_process=None;number=1
    record={'number':number,'attached_utc':utc(),'started_utc':state['started_utc'],'parent_pid':parent,
            'observed_command':ps(parent,'command'),'scope':'Root-launched successful first session; previous rejected preflight does not count.'}
    if args.resume_monitor or args.resume_after_timeout:
        prior=json.loads(registry_path.read_text())
        if len(prior['sessions'])!=1 or prior['sessions'][0]['parent_pid']!=parent:
            raise ValueError('Resume-monitor only reattaches the same first session')
        prior['monitor_restart_utc']=utc();prior['previous_monitor_pid']=prior['monitor_pid'];prior['monitor_pid']=os.getpid()
        registry=prior;record=registry['sessions'][0]
    else:
        registry['sessions'].append(record)
    registry['resource_extra_headroom_bytes']=HEADROOM_BYTES
    registry['resource_revision']='resource_revision_512MiB.json'
    save(sessions/'session_1.json',record);save(registry_path,registry)
    def emit(event):print(json.dumps(event),flush=True)
    last=None;heartbeat=0
    try:
        while True:
            if stop.exists():
                if live(parent) and checked_parent(parent,folder,repo):os.kill(parent,signal.SIGINT)
                registry.update(status='stop_requested',finished_utc=utc());save(registry_path,registry);return
            state=json.loads((folder/'local_status.json').read_text())
            if number>1 and state.get('started_utc')==record.get('prior_started_utc') and not live(parent):
                record.update(startup_failed_without_fresh_status=True,process_returncode=None if owned_process is None else owned_process.wait())
                registry['sessions'][-1]=record
                registry.update(status='blocked_startup_without_fresh_status',finished_utc=utc())
                save(sessions/f'session_{number}.json',record);save(registry_path,registry)
                emit({'event':'no_retry','reason':registry['status']});return
            checkpoints=len(list((folder/'local_checkpoints').glob('*/*.json')))
            signature=(state['status'],state.get('active_job'),len(state['jobs']),checkpoints)
            if signature!=last or time.monotonic()-heartbeat>=300:
                emit({'observed_utc':utc(),'session':number,'status':state['status'],
                      'active_job':state.get('active_job'),'session_processed_jobs':len(state['jobs']),
                      'sealed_raw_checkpoints':checkpoints,'elapsed_seconds':state.get('elapsed_seconds'),
                      'active_rss_bytes':state.get('active_rss_bytes')})
                last=signature;heartbeat=time.monotonic()
            if state['status']=='running' or live(parent):time.sleep(15);continue
            if owned_process is not None:record['process_returncode']=owned_process.wait()
            record.update(terminal_observed_utc=utc(),terminal_status=state)
            verify_frozen(repo,folder,plan_sha)
            summary=terminal_summary(folder);record['cumulative_job_summary']=summary
            save(sessions/f'session_{number}.json',record);registry['sessions'][-1]=record;save(registry_path,registry)
            if all(j['processed'] for j in summary.values()):
                registry.update(status='all_four_fixed_jobs_processed',finished_utc=utc(),job_summary=summary)
                save(registry_path,registry);emit({'event':'completed','session':number,'job_summary':summary});return
            if state['status']!='interrupted' or not state.get('error','').startswith('TimeoutError('):
                registry.update(status='blocked_non_timeout_terminal',finished_utc=utc(),job_summary=summary)
                save(registry_path,registry);emit({'event':'no_retry','reason':registry['status'],'terminal':state});return
            if number>=3:
                registry.update(status='three_session_budget_exhausted',finished_utc=utc(),job_summary=summary)
                save(registry_path,registry);emit({'event':'no_retry','reason':registry['status']});return
            from experiments.continue_local_physics import validate,preflight,available_memory
            plan=validate(folder);wait_started=time.monotonic()
            while True:
                if stop.exists():registry.update(status='stop_requested',finished_utc=utc());save(registry_path,registry);return
                available=available_memory()
                try:
                    before=preflight(folder,plan,2,12*1024**3,8*1024**3,free_memory=available)
                    if available<before['estimated_peak_bytes']+HEADROOM_BYTES:
                        raise ValueError(f'Local memory headroom guard: actual={available}, target={before["estimated_peak_bytes"]+HEADROOM_BYTES}')
                    break
                except ValueError as error:
                    if 'memory' not in str(error):raise
                    wait={'observed_utc':utc(),'available_memory_bytes':available,'headroom_bytes':HEADROOM_BYTES,
                          'error':str(error),'wait_seconds':time.monotonic()-wait_started}
                    save(sessions/'memory_wait.json',wait)
                    if time.monotonic()-wait_started>1800:
                        registry.update(status='memory_wait_limit',finished_utc=utc());save(registry_path,registry);return
                    emit({'event':'waiting_for_memory',**wait});time.sleep(30)
            verify_frozen(repo,folder,plan_sha)
            number+=1
            argv=[str(repo/'.venv/bin/python'),'-u','-m','experiments.continue_local_physics','run',str(folder),
                  '--max-seconds','3600','--threads','2','--max-rss-gib','12','--reserve-disk-gib','8','--no-sync','--no-shutdown']
            env=dict(os.environ)
            for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
            log_path=sessions/f'session_{number}.log';log=log_path.open('w')
            owned_process=subprocess.Popen(argv,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT);log.close();parent=owned_process.pid
            record={'number':number,'launched_utc':utc(),'parent_pid':parent,'argv':argv,'log':str(log_path),
                    'before_resume_preflight':before,'available_memory_headroom_bytes':HEADROOM_BYTES,
                    'prior_terminal_session':number-1,'prior_terminal_status':'clean TimeoutError','prior_started_utc':state['started_utc'],
                    'plan_sha256':plan_sha}
            registry['sessions'].append(record);save(sessions/f'session_{number}.json',record);save(registry_path,registry)
            emit({'event':'resumed','session':number,'pid':parent,'plan_sha256':plan_sha})
            # Avoid interpreting the previous session's terminal file before the
            # new supervisor atomically publishes its running status.
            deadline=time.monotonic()+60
            while time.monotonic()<deadline and live(parent):
                fresh=json.loads((folder/'local_status.json').read_text())
                if fresh['status']=='running' and fresh['started_utc']!=state['started_utc']:break
                time.sleep(1)
    except BaseException as error:
        registry.update(status='monitor_error',error=repr(error),finished_utc=utc());save(registry_path,registry)
        raise


if __name__=='__main__':main()
