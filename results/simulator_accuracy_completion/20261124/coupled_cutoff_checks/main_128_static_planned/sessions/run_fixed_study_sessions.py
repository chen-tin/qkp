"""Bounded local execution of exactly one approved fixed16-job study.

Wait for the independent measurement worker's explicit RAM-release marker.
Only clean TimeoutError permits at most two additional one-hour sessions.
"""
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/Users/christopherbabayan/programs/QUANTUM/qkp')
FOLDER=ROOT/'results/simulator_accuracy_completion/20261124/coupled_cutoff_checks/main_128_static_planned'
SESSIONS=FOLDER/'sessions'
APPROVED='eb995962bfa7e47ca1e360fca56699e4a3c27699323a72a03792338c880a3032'


def utc():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()


def save(path,data):
    p=Path(path);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(p)


def verify(plan_sha):
    if sha(FOLDER/'plan.json')!=plan_sha:raise ValueError('Frozen plan bytes changed')
    plan=json.loads((FOLDER/'plan.json').read_text())
    identity=hashlib.sha256(json.dumps(plan,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if identity!=APPROVED or len(plan['jobs'])!=16:raise ValueError('Approved fixed study identity changed')
    for rel,digest in plan['source_hashes'].items():
        if sha(ROOT/rel)!=digest or sha(FOLDER/'source_snapshot'/rel)!=digest:raise ValueError('Study source changed: '+rel)
    for raw in (FOLDER/'jobs').glob('*/*/*.npz'):
        if not raw.name.endswith('.tmp.npz') and not raw.with_suffix('.sha256.json').exists():raise ValueError('Unsealed non-temporary observation blocks resume')
    for seal in (FOLDER/'jobs').glob('*/*/*.sha256.json'):
        raw=seal.with_name(seal.name[:-12]+'.npz')
        if not raw.exists() or sha(raw)!=json.loads(seal.read_text())['sha256']:raise ValueError('Sealed study observation changed')
    return plan


def main():
    SESSIONS.mkdir(exist_ok=True);lock=(SESSIONS/'.manager.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registry_path=SESSIONS/'registry.json'
    prior=json.loads(registry_path.read_text()) if registry_path.exists() else None
    if prior is not None and prior['sessions']:raise ValueError('Do not duplicate an active or previously launched scientific study')
    registration=json.loads((SESSIONS/'resource_manager_correction.json').read_text())
    if registration['manager_sha256']!=sha(__file__):raise ValueError('Unregistered manager revision')
    spec=importlib.util.spec_from_file_location('scheduling_guard',ROOT/'results/simulator_accuracy_completion/20261124/resource_coordination/scheduling_guard.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    if sha(guard.__file__)!=registration['guard_sha256']:raise ValueError('Unregistered scheduling guard')
    plan_sha=sha(FOLDER/'plan.json');verify(plan_sha)
    registry={'created_utc':utc(),'manager_pid':os.getpid(),'approved_canonical_plan_sha256':APPROVED,
              'plan_file_sha256':plan_sha,'status':'waiting_for_physics_density_priority_and_capacity','sessions':[],
              'max_sessions':3,'seconds_per_session':3600,'max_rss_bytes':8*1024**3,'disk_reserve_bytes':8*1024**3,
              'available_memory_headroom_bytes':1024**3,'fixed_scope':'16registeredjobs only; no new seeds/settings/shots or stopping when significance appears.'}
    registry['prior_unlaunched_registry']=prior
    registry['manager_sha256']=sha(__file__)
    registry['resource_manager_correction']='resource_manager_correction.json'
    save(registry_path,registry)
    stop=SESSIONS/'STOP';release=SESSIONS/'MEASUREMENT_RAM_RELEASED';wait_started=time.monotonic()
    # A completed prerequisite remains satisfied after its transient active phase.
    sys.path.insert(0,str(ROOT))
    from experiments import quality_benchmark as qb
    physics=ROOT/'results/simulator_accuracy_completion/20261124/remaining_physics'
    def completed_physics_priority():
        path=physics/'sessions/budget_registry.json'
        if not path.exists():return False
        receipt=json.loads(path.read_text())
        if receipt.get('status')!='all_four_fixed_jobs_processed':return False
        physical_plan=qb.load_plan(physics,current_sources=True)
        if len(physical_plan['jobs'])!=4:return False
        checked={}
        for job in physical_plan['jobs']:
            raw=qb.verify_job(physics,job);score=qb.scoring_status(physics,job,[192,384,768])
            checked[job['id']]={'raw_complete':raw is not None,'scoring_status':score['status']}
        if not all(row['raw_complete'] and row['scoring_status']=='checked' for row in checked.values()):return False
        save(SESSIONS/'automatic_physics_priority_receipt.json',{'checked_utc':utc(),'physics_registry_sha256':sha(path),'jobs':checked,
            'scope':'Allfour fixed observations and required scoringprocessed. Numerical negatives remain processed; no requirement toobtainpass.'})
        return True
    while not release.exists() and not completed_physics_priority():
        if stop.exists():registry.update(status='stop_requested',finished_utc=utc());save(registry_path,registry);return
        if time.monotonic()-wait_started>14400:registry.update(status='physics_priority_wait_limit',finished_utc=utc());save(registry_path,registry);return
        time.sleep(15)
    sys.path.insert(0,str(ROOT))
    from experiments import quality_benchmark as qb
    from experiments.continue_local_physics import preflight,available_memory
    from experiments.local_process_safety import spawn_owned,stop_owned,supervise_owned
    safety=ROOT/'experiments/local_process_safety.py'
    if sha(safety)!=registration['process_safety_sha256']:raise ValueError('Unregistered process-safety helper')
    child=None
    try:
        for number in range(1,4):
            plan=verify(plan_sha);guard_started=time.monotonic()
            while True:
                if stop.exists():registry.update(status='stop_requested',finished_utc=utc());save(registry_path,registry);return
                available=available_memory();reservation=guard.density_reservation()
                effective=available-reservation['density_remaining_growth_bytes']-reservation.get('readback_remaining_growth_bytes',0)
                try:
                    before=preflight(FOLDER,plan,2,8*1024**3,8*1024**3,free_memory=effective-1024**3);break
                except ValueError as error:
                    registry.update(status='waiting_for_resource_capacity');save(registry_path,registry)
                    if 'memory' not in str(error):raise
                    save(SESSIONS/'memory_wait.json',{'observed_utc':utc(),'available_memory_bytes':available,'effective_available_memory_bytes':effective,'density_reservation':reservation,'headroom_bytes':1024**3,'error':str(error)})
                    if time.monotonic()-guard_started>1800:registry.update(status='memory_wait_limit',finished_utc=utc());save(registry_path,registry);return
                    time.sleep(30)
            previous=None
            status_path=FOLDER/'study_status.json'
            if status_path.exists():previous=json.loads(status_path.read_text()).get('started_utc')
            argv=[str(ROOT/'.venv/bin/python'),'-u','-m','experiments.coupled_cutoff_check','run',str(FOLDER),
                  '--approved-plan-sha',APPROVED,'--max-seconds','3540','--threads','2',
                  '--max-rss-gib','8','--reserve-disk-gib','8','--no-sync','--no-shutdown']
            env=dict(os.environ)
            for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
            log_path=SESSIONS/f'session_{number}.log'
            launched_monotonic=time.monotonic()
            with log_path.open('w') as log:child=spawn_owned(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            receipt={'number':number,'launched_utc':utc(),'parent_pid':child.pid,'argv':argv,'preflight':before,
                     'plan_file_sha256':plan_sha,'approved_canonical_plan_sha256':APPROVED,'prior_started_utc':previous,
                     'available_memory_before_headroom_bytes':available,'effective_available_memory_before_headroom_bytes':effective,'density_reservation':reservation,'log':str(log_path)}
            registry['sessions'].append(receipt);registry.update(status='running',active_session=number,active_parent_pid=child.pid)
            save(SESSIONS/f'session_{number}.json',receipt);save(registry_path,registry)
            print(json.dumps({'event':'launched','session':number,'pid':child.pid,'plan':APPROVED}),flush=True)
            last=None;heartbeat=time.monotonic()
            def progress():
                nonlocal last,heartbeat
                if status_path.exists():
                    state=json.loads(status_path.read_text())
                    signature=(state['status'],state.get('active_job'),len(state['jobs']))
                    if signature!=last or time.monotonic()-heartbeat>300:
                        print(json.dumps({'event':'progress','session':number,'status':state['status'],
                                          'active_job':state.get('active_job'),'session_processed_jobs':len(state['jobs']),
                                          'elapsed_seconds':state.get('elapsed_seconds'),'active_rss_bytes':state.get('active_rss_bytes')}),flush=True)
                        last=signature;heartbeat=time.monotonic()
            execution=supervise_owned(child,3600,started_monotonic=launched_monotonic,
                                       stop_requested=lambda:stop.exists(),on_poll=progress,poll_seconds=1.,grace_seconds=50)
            receipt.update(execution,terminal_observed_utc=utc())
            if execution['external_stop_reason'] is not None:
                save(SESSIONS/f'session_{number}.json',receipt)
                registry.update(status='external_deadline_or_stop_no_auto_retry',finished_utc=utc());save(registry_path,registry);return
            if not status_path.exists():raise RuntimeError('Study supervisor exited without any status')
            terminal=json.loads(status_path.read_text());receipt['terminal_status']=terminal
            if terminal.get('started_utc')==previous:raise RuntimeError('Study startup failed without fresh terminal status')
            verify(plan_sha)
            summary={}
            for job in plan['jobs']:
                observed=qb.verify_job(FOLDER,job);score=qb.scoring_status(FOLDER,job,[192,384,768])
                summary[job['id']]={'variant':job['physical_variant'],'sampler':job['sampler'],'cutoff':job['cutoff'],
                                    'observations_saved':observed is not None,'scoring_status':score['status'],
                                    'scoring_pass':score['pass'],'processed':observed is not None and score['status']=='checked'}
            receipt['cumulative_job_summary']=summary;registry['sessions'][-1]=receipt
            save(SESSIONS/f'session_{number}.json',receipt);save(registry_path,registry)
            if stop.exists():registry.update(status='stop_requested',finished_utc=utc());save(registry_path,registry);return
            if all(j['processed'] for j in summary.values()):
                audit=ROOT/'experiments/audit_coupled_mechanisms.py'
                if sha(audit)!=registration['mechanism_auditor_sha256']:raise ValueError('Corrected mechanism auditor source changed since execution registration')
                audit_command=[str(ROOT/'.venv/bin/python'),'-m','experiments.audit_coupled_mechanisms',str(FOLDER),'--output',str(FOLDER/'mechanism_audit.json')]
                # CLI binds its embedded exact approved plan identity; it has no approved-plan argument.
                receipt['postrun_mechanism_audit_command']=audit_command
                with (SESSIONS/'mechanism_audit.log').open('w') as auditlog:
                    audit_started=time.monotonic()
                    audit_child=spawn_owned(audit_command,cwd=ROOT,env=env,stdout=auditlog,stderr=subprocess.STDOUT)
                    audit_execution=supervise_owned(audit_child,300,started_monotonic=audit_started,
                                                    stop_requested=lambda:stop.exists(),grace_seconds=50)
                receipt['postrun_mechanism_audit_execution']=audit_execution
                save(SESSIONS/f'session_{number}.json',receipt)
                if audit_execution['returncode'] or audit_execution['external_stop_reason']:
                    raise RuntimeError('Postrun fixed mechanism audit failed or exceeded bounded verification time; no scientific rerun')
                registry.update(status='all_fixed_study_jobs_processed',finished_utc=utc(),job_summary=summary)
                save(registry_path,registry);print(json.dumps({'event':'completed','sessions':number}),flush=True);return
            if terminal['status']!='interrupted' or not terminal.get('error','').startswith('TimeoutError('):
                registry.update(status='blocked_non_timeout_terminal',finished_utc=utc(),job_summary=summary)
                save(registry_path,registry);return
        registry.update(status='three_session_budget_exhausted',finished_utc=utc(),job_summary=summary);save(registry_path,registry)
    except BaseException as error:
        if child is not None:stop_owned(child,grace_seconds=50)
        registry.update(status='manager_error',error=repr(error),finished_utc=utc());save(registry_path,registry);raise


if __name__=='__main__':main()
