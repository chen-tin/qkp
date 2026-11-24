"""Local-only continuation of four previously unrun physics configurations.

Default planning is unfrozen. Freeze only after all source edits are final;
run requires explicit no-sync/no-shutdown flags and a finite local budget.
Historical plans, observations and hashes are never rewritten.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('finite_duration_lumped','passive_cooling')
SOURCE_PATH='experiments/continue_local_physics.py'
EXTRA_SOURCES=[SOURCE_PATH,'experiments/refine_logical_scores.py','quantum/scoring.py']


def utc():return datetime.now(timezone.utc).isoformat()


def semantic_config(config):
    copy=json.loads(json.dumps(config));copy.pop('source_hashes',None)
    return copy


def source_jobs(original):
    jobs=[j for j in original['jobs'] if j['variant'] in VARIANTS]
    identities={(j['variant'],j['cutoff']) for j in jobs}
    if len(jobs)!=4 or identities!={(v,n) for v in VARIANTS for n in (4096,5120)}:
        raise ValueError('Expected exactly the four registered remaining physics jobs')
    for j in jobs:
        c=j['config']
        if (j['base_seed']!=26112001 or j['chunk']!=0 or c['per_state']!=512 or c['rounds']!=8 or
                c['decoders']!=['nearest','bayes_gaussian','lstm_256x3_s0']):
            raise ValueError('Original registered seed/controller/trial/horizon contract changed')
    return jobs


def build_continuation(original_path, software_smoke=False):
    from experiments import quality_benchmark as qb
    original_path=Path(original_path).resolve();original=json.loads(original_path.read_text())
    selected=source_jobs(original)
    exemplar=selected[0]
    points=exemplar['config']['points']
    variants={v:{'physics':next(j['config']['physics'] for j in selected if j['variant']==v),
                 'noise':next(j['config']['noise_extension'] for j in selected if j['variant']==v)} for v in VARIANTS}
    kwargs={'per_state':2,'rounds':1,'cutoffs':[16,24]} if software_smoke else {'per_state':512,'rounds':8,'cutoffs':[4096,5120]}
    plan=qb.build_plan('physics',seeds=(26112001,),decoders=tuple(exemplar['config']['decoders']),
                       matrix={'points':points,'variants':variants},**kwargs)
    sources=dict(plan['source_hashes'])
    sources.update({p:qb.sha(ROOT/p) for p in EXTRA_SOURCES})
    plan['source_hashes']=sources
    comparisons=[]
    for job in plan['jobs']:
        original_cutoff={16:4096,24:5120}.get(job['cutoff'],job['cutoff']) if software_smoke else job['cutoff']
        old=next(j for j in selected if (j['variant'],j['cutoff'])==(job['variant'],original_cutoff))
        job['config']['source_hashes']=sources
        expected=semantic_config(old['config'])
        departures={}
        if software_smoke:
            replacements={'per_state':2,'rounds':1,'cutoffs':[job['cutoff']],
                          'score_snapshot_per_state':2,'snapshot_cycles':[0,1]}
            departures={key:{'original':expected[key],'software_check':value} for key,value in replacements.items()}
            expected.update(replacements)
        if semantic_config(job['config'])!=expected:
            raise ValueError('Scientific configuration changed beyond declared source hashes/software budget')
        if job['config']['checkpoint_sha256']!=old['config']['checkpoint_sha256']:
            raise ValueError('Frozen controller checkpoint changed')
        job.pop('id',None);job['id']=qb.identity(job)[:16]
        comparisons.append({'original_job_id':old['id'],'local_job_id':job['id'],'variant':job['variant'],
                            'original_cutoff':original_cutoff,'local_cutoff':job['cutoff'],
                            'semantic_config_equal_excluding_only_source_hashes':not software_smoke,
                            'declared_software_budget_departures':departures})
    if plan['metric_contract']!=original['metric_contract']:
        raise ValueError('Registered metric contract changed')
    if software_smoke:plan['preset']='smoke'
    plan['classification']='infrastructure_smoke' if software_smoke else 'planned_local_continuation'
    plan['migration']={'original_plan':str(original_path),'original_plan_sha256':qb.sha(original_path),
                       'selected_original_jobs':comparisons,'software_smoke':software_smoke,
                       'source_hash_policy':'Fresh active sources/environment; original hashes retained in source record and never substituted.',
                       'scope':'Four selected physics jobs only; does not complete original20-job plan or full accuracy audit.'}
    return plan


def prepare(original,output,freeze=False,sources_final=False,software_smoke=False):
    from experiments import quality_benchmark as qb
    output=Path(output).resolve()
    if output==Path(original).resolve().parent or output.is_relative_to(Path(original).resolve().parent):
        raise ValueError('Continuation output must be outside the historical input plan tree')
    if freeze and not sources_final:
        raise ValueError('Freeze requires explicit confirmation that source edits are final')
    plan=build_continuation(original,software_smoke)
    if not freeze:
        if (output/'plan.json').exists():raise ValueError('Existing frozen plan; validate or use a new output directory')
        qb.save(output/'draft_plan.json',{'status':'draft_sources_not_frozen','plan':plan})
        return {'status':'draft_sources_not_frozen','path':str(output/'draft_plan.json'),'expected_jobs':len(plan['jobs']),
                'estimated_uncompressed_output_bytes':plan['estimated_uncompressed_output_bytes']}
    qb.write_plan(plan,output)
    manifest={'registered_utc':utc(),'status':'planned','plan_sha256':qb.identity(plan),'plan':str(output/'plan.json'),
              'local_only':True,'remote_sync':False,'os_shutdown':False,'source_hashes':plan['source_hashes'],
              'environment':plan['environment'],'migration':plan['migration']}
    target=output/'local_manifest.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old['plan_sha256']!=manifest['plan_sha256']:raise ValueError('Different frozen local manifest')
    else:qb.save(target,manifest)
    return {'status':'planned','path':str(output/'plan.json'),'plan_sha256':qb.identity(plan),
            'expected_jobs':len(plan['jobs']),'estimated_uncompressed_output_bytes':plan['estimated_uncompressed_output_bytes']}


def validate(folder):
    from experiments import quality_benchmark as qb
    folder=Path(folder);plan=qb.load_plan(folder,current_sources=True)
    manifest=json.loads((folder/'local_manifest.json').read_text())
    if (not manifest['local_only'] or manifest['remote_sync'] or manifest['os_shutdown'] or
            manifest['plan_sha256']!=qb.identity(plan) or plan['expected_jobs']!=4):
        raise ValueError('Invalid local-only manifest')
    original=plan['migration']['original_plan']
    if qb.sha(original)!=plan['migration']['original_plan_sha256']:
        raise ValueError('Original scientific plan changed')
    if build_continuation(original,plan['migration']['software_smoke'])!=plan:
        raise ValueError('Local plan no longer matches exact original scientific configurations')
    return plan


def available_memory():
    """Conservative Linux availability / macOS reclaimable-page estimate."""
    if sys.platform=='darwin':
        result=subprocess.run(['vm_stat'],capture_output=True,text=True,check=True).stdout
        import re
        page=int(re.search(r'page size of (\d+) bytes',result).group(1))
        counts={key:int(value) for key,value in re.findall(r'([^\n:]+):\s+(\d+)\.',result)}
        return page*sum(counts.get(k,0) for k in ['Pages free','Pages inactive','Pages speculative'])
    values={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    return values['MemAvailable']


def preflight(folder,plan,threads,max_rss_bytes,reserve_disk_bytes,free_memory=None,free_disk=None):
    from experiments import quality_benchmark as qb
    if not 1<=threads<=min(os.cpu_count() or 1,4):raise ValueError('Thread budget must fit available cores and local cap4')
    if max_rss_bytes<=0 or reserve_disk_bytes<0:raise ValueError('Positive RSS cap and nonnegative disk reserve required')
    unfinished=[j for j in plan['jobs'] if qb.verify_job(folder,j) is None]
    remaining=plan['estimated_uncompressed_output_bytes']*len(unfinished)/len(plan['jobs'])
    free_disk=shutil.disk_usage(folder).free if free_disk is None else free_disk
    free_memory=available_memory() if free_memory is None else free_memory
    n=max(j['cutoff'] for j in plan['jobs']);p=max(j['config']['per_state'] for j in plan['jobs'])
    estimate=256*n*n+16*n*6*p*12
    if max_rss_bytes<estimate or free_memory<estimate:
        raise ValueError(f'Local memory preflight failed: estimate={estimate}, cap={max_rss_bytes}, available={free_memory}')
    if free_disk<remaining+reserve_disk_bytes:
        raise ValueError(f'Local disk preflight failed: remaining={remaining}, reserve={reserve_disk_bytes}, available={free_disk}')
    return {'estimated_peak_bytes':estimate,'available_memory_bytes':free_memory,'remaining_output_bytes':remaining,
            'available_disk_bytes':free_disk,'thread_environment_requested':threads,
            'scope':'Estimates/preflight only; parent also samples child RSS and enforces deadline. Accelerate thread count remains environment-requested.'}


def checkpoint_seals(folder,job,receipts):
    """A local receipt per complete atomic raw/seal pair, including partial jobs."""
    from experiments import quality_benchmark as qb
    folder=Path(folder);root=folder/'jobs'/job['id'];receipts=Path(receipts)
    for seal in root.glob('*/*.sha256.json'):
        if not seal.name.endswith('.sha256.json'):continue
        raw=seal.with_name(seal.name[:-12]+'.npz')
        if not raw.exists():continue
        relative=str(raw.relative_to(root));target=receipts/job['id']/(relative.replace('/','__')+'.json')
        expected=json.loads(seal.read_text())['sha256']
        if target.exists() and json.loads(target.read_text()).get('sha256')==expected:continue
        if qb.sha(raw)!=expected:raise ValueError(f'Sealed raw checkpoint changed: {raw}')
        qb.save(target,{'checked_utc':utc(),'path':str(raw.resolve()),'sha256':expected,'bytes':raw.stat().st_size,
                        'scope':'Local sealed observation; may precede complete job/refinement. No remote sync.'})


def execute_job(folder,job_id):
    from experiments import quality_benchmark as qb
    from experiments.refine_logical_scores import refine_run
    import torch
    torch.set_num_threads(1)  # Preserve original CPU inference setting.
    folder=Path(folder);plan=validate(folder);job=next(j for j in plan['jobs'] if j['id']==job_id)
    out=folder/'jobs'/job_id;out.mkdir(parents=True,exist_ok=True)
    if qb.verify_job(folder,job) is None:
        c=job['config']
        rows=qb.run_case(c,out,job['point'],job['cutoff'],c['per_state'],c['rounds'],None,c['decoders'])
        qb.save(out/'result.json',{'complete':True,'config':c,'rows':rows,'classification':plan['classification']})
    checkpoint_seals(folder,job,folder/'local_checkpoints')
    score=qb.scoring_status(folder,job,plan['metric_contract']['scoring_bins'])
    if score['status']!='checked':
        refine_run(out,bins=plan['metric_contract']['scoring_bins'],tolerance=plan['metric_contract']['scoring_tolerance'],
                   shared_basis=True,max_batch_bytes=256*1024**2)
    return qb.scoring_status(folder,job,plan['metric_contract']['scoring_bins'])


@contextmanager
def lease(path):
    with Path(path).open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Another local continuation owns this plan') from error
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


def stop_child(process):
    if process.poll() is not None:return
    process.send_signal(signal.SIGINT)
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()



@contextmanager
def interrupt_signals():
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Local supervisor received signal{signum}')
    previous={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:yield
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def run(folder,max_seconds,threads,max_rss_bytes,reserve_disk_bytes,no_sync,no_shutdown,allow_software_smoke=False):
    from experiments import quality_benchmark as qb
    if not no_sync or not no_shutdown or max_seconds<=0:
        raise ValueError('Require --no-sync --no-shutdown and positive --max-seconds')
    folder=Path(folder).resolve();plan=validate(folder)
    if plan['migration']['software_smoke'] and not allow_software_smoke:
        raise ValueError('Tiny negative-control plan requires --allow-software-smoke')
    environment=dict(os.environ)
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        environment[key]=str(threads)
    start=time.monotonic();process=None
    with interrupt_signals(),lease(folder/'.local-continuation.lock'):
        checks=preflight(folder,plan,threads,max_rss_bytes,reserve_disk_bytes)
        qb.save(folder/'local_preflight.json',{'checked_utc':utc(),'max_seconds':max_seconds,'max_rss_bytes':max_rss_bytes,**checks})
        state={'started_utc':utc(),'status':'running','jobs':{},'max_seconds':max_seconds,'local_only':True}
        try:
            for job in plan['jobs']:
                if time.monotonic()-start>=max_seconds:raise TimeoutError('Registered local deadline reached')
                preflight(folder,plan,threads,max_rss_bytes,reserve_disk_bytes)
                score=qb.scoring_status(folder,job,plan['metric_contract']['scoring_bins'])
                if qb.verify_job(folder,job) is not None and score['status']=='checked':
                    state['jobs'][job['id']]={'status':'reused','scoring':score};continue
                logfile=folder/'local_logs'/f'{job["id"]}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
                with logfile.open('a') as log:
                    process=subprocess.Popen([sys.executable,'-u','-m','experiments.continue_local_physics','_job',str(folder),job['id']],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        checkpoint_seals(folder,job,folder/'local_checkpoints')
                        rss=subprocess.run(['ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True)
                        observed=int(rss.stdout.strip() or '0')*1024
                        state.update(updated_utc=utc(),active_job=job['id'],active_pid=process.pid,active_rss_bytes=observed,elapsed_seconds=time.monotonic()-start)
                        qb.save(folder/'local_status.json',state)
                        if time.monotonic()-start>=max_seconds:raise TimeoutError('Registered local deadline reached')
                        if observed>max_rss_bytes:raise MemoryError('Local child RSS cap reached')
                        if shutil.disk_usage(folder).free<reserve_disk_bytes:raise OSError('Local disk reserve reached')
                        time.sleep(2)
                    if process.returncode:raise RuntimeError(f'Local job failed: {job["id"]}; see {logfile}')
                checkpoint_seals(folder,job,folder/'local_checkpoints')
                score=qb.scoring_status(folder,job,plan['metric_contract']['scoring_bins'])
                state['jobs'][job['id']]={'status':'completed','scoring':score}
                qb.analyze_plan(folder)
            state['status']='all_selected_jobs_processed'
        except BaseException as error:
            if process is not None:stop_child(process)
            state.update(status='interrupted' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error=repr(error))
            raise
        finally:
            for job in plan['jobs']:checkpoint_seals(folder,job,folder/'local_checkpoints')
            state.update(finished_utc=utc(),elapsed_seconds=time.monotonic()-start)
            state.pop('active_pid',None);state.pop('active_job',None)
            qb.save(folder/'local_status.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan');p.add_argument('original',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--freeze',action='store_true');p.add_argument('--sources-final',action='store_true');p.add_argument('--software-smoke',action='store_true')
    p=sub.add_parser('validate');p.add_argument('folder',type=Path)
    p=sub.add_parser('run');p.add_argument('folder',type=Path);p.add_argument('--max-seconds',type=int,required=True)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--max-rss-gib',type=float,default=12.)
    p.add_argument('--reserve-disk-gib',type=float,default=2.);p.add_argument('--no-sync',action='store_true');p.add_argument('--no-shutdown',action='store_true')
    p.add_argument('--allow-software-smoke',action='store_true')
    p=sub.add_parser('_job');p.add_argument('folder',type=Path);p.add_argument('job_id')
    args=parser.parse_args()
    if args.action=='plan':value=prepare(args.original,args.output,args.freeze,args.sources_final,args.software_smoke)
    elif args.action=='validate':
        plan=validate(args.folder);value={'pass':True,'jobs':len(plan['jobs']),'scope':plan['migration']['scope']}
    elif args.action=='_job':value=execute_job(args.folder,args.job_id)
    else:value=run(args.folder,args.max_seconds,args.threads,int(args.max_rss_gib*1024**3),int(args.reserve_disk_gib*1024**3),args.no_sync,args.no_shutdown,args.allow_software_smoke)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
