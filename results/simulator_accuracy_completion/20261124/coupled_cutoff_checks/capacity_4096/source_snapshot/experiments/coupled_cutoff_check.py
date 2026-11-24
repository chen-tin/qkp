"""Prospective local experiment comparing two valid cutoff-noise couplings.

Planning is cheap and launches no trajectories. This new study does not edit
or reinterpret historical tests. Sampler settings and uniform mapping enter
every hashed job identity. A full run requires an exact approved plan digest.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=('loss','loss_dephasing','coupling_error','control_rotation')
PARAMETERS={'max_step':.00025,'tolerance':1e-10,'shot_batch':64,'node_chunk':256,
            'tail_probability':1e-15,'float_budget':5e-12,'max_table_bytes':64*1024**2}
EXTRA=['quantum/coupled_measurement.py','experiments/coupled_cutoff_check.py',
       'experiments/continue_local_physics.py','tests/test_coupled_measurement.py']


def utc():return datetime.now(timezone.utc).isoformat()


def sampler_contract(name):
    if name=='legacy_mixture':
        return {'name':name,'eigenvalue_uniform_index':0,'resource_uniform_index':1,
                'historical_unused_jitter_index':2,'parameters':{},'physical_marginal':'existing finite Kraus instrument'}
    if name!='direct_cdf':raise ValueError('Unknown sampler')
    return {'name':name,'uniform_index':1,'parameters':PARAMETERS,
            'physical_marginal':'same finite Kraus instrument within declared numerical CDF budget',
            'roundoff_scope':'Explicit unproved floating allowance supplements analytic interpolation/slope/tail bounds.',
            'resource_scope':'One static resource width per scientific process; dynamic-width schedules unsupported. One cached table for this registered study.',
            'raw_inverse_residual_field':'For this new backend max_inverse_cdf_residual records the declared full-outcome CDF residual bound, not a measured resource-only inverse residual.',
            'coupling_change':'Joint common-random-number mapping changes across cutoffs, explicit QuTiP and classical replay; not exact historical joint-draw reproduction.'}


def build_plan(reference_path):
    from experiments import quality_benchmark as qb
    reference_path=Path(reference_path).resolve();reference=json.loads(reference_path.read_text())
    jobs=reference['jobs'];matrix={}
    for name in CONDITIONS:
        found=[j for j in jobs if j['variant']==name]
        if not found:raise ValueError('Reference plan lacks registered condition '+name)
        if any(j['config']['noise_extension'] for j in found):
            raise ValueError('This static-resource coupling study does not support noise extensions or width schedules')
        matrix[name]={'physics':found[0]['config']['physics'],'noise':found[0]['config']['noise_extension']}
    points=next(j['config']['points'] for j in jobs if j['variant']=='loss')
    if points!={'nominal':[.20,.05,.90]}:
        raise ValueError('This study requires one static nominal resource/noise point')
    base=qb.build_plan('physics',per_state=128,rounds=8,seeds=(26112401,),decoders=('nearest',),
                       cutoffs=[4096,5120],matrix={'points':points,'variants':matrix})
    sources=dict(base['source_hashes']);sources.update({p:qb.sha(ROOT/p) for p in EXTRA})
    expanded=[]
    for original in base['jobs']:
        for sampler in ['legacy_mixture','direct_cdf']:
            job=copy.deepcopy(original);job.pop('id')
            job['physical_variant']=job['variant'];job['variant']+='__'+sampler
            job['sampler']=sampler;job['config']['source_hashes']=sources
            job['config']['experimental_measurement_sampler']=sampler_contract(sampler)
            job['id']=qb.identity(job)[:16];expanded.append(job)
    base.update(jobs=expanded,expected_jobs=len(expanded),source_hashes=sources,
                estimated_uncompressed_output_bytes=2*base['estimated_uncompressed_output_bytes'])
    base['classification']='planned_common_quantile_coupling_experiment'
    base['coupling_study']={
        'registered_question':'Does a direct full-outcome CDF common-quantile coupling reduce paired-cutoff variance while retaining the same finite-operator measurement marginal?',
        'reference_plan':str(reference_path),'reference_plan_sha256':qb.sha(reference_path),
        'fixed_design':{'conditions':list(CONDITIONS),'samplers':['legacy_mixture','direct_cdf'],
                        'per_state':128,'cardinal_inputs':6,'trajectories_per_job':768,'cycles':8,'cutoffs':[4096,5120],
                        'controller':'nearest','base_seed':26112401,'jobs':16,'optional_stopping':False},
        'primary_rule':'Unchanged stage and saved-state scoring gates; paired cutoff95% pointwise intervals wholly inside±.01 at every cycle. Retain all failures/inconclusive outcomes; no added shots until pass.',
        'variance_rule':'Descriptive per-cycle ratio of squared SE for directCDF versus legacy paired cutoff means; no confidence claim or causal physical-bias attribution from this ratio.',
        'marginal_rule':'Deterministic exact-resourceCDF and Kraus-update tests precede scientific launch. Exact arithmetic interpolation bound is supplemented by explicitly unproved floating allowance; record all actual inverse residuals and nonpositive-norm failures.',
        'combined_comparison':'Same fixed768-trajectory budget in every sampler/condition/cutoff arm; algorithm/CRN coupling is changed, not shots. Comparing this128/state study to the historical512/state run is not an isolated sampler comparison.',
        'interpretation':'Reduced paired variance does not establish zero true cutoff bias or complete independent instrument/hardware validation. Both sampler marginals describe the same ideal instrument within numerical budgets; their joint replay mapping differs.',
        'prior_confirmation_registration':'Original640/640/2048/640per-state independent-noise registration remains Planned/unrun; this is a separate prospective fixed-budget experiment. Once26112401 is used here, a later independent confirmation must register an unused seed/stream instead of claiming this seed is still held out.'}
    return base


def prepare(reference,output):
    from experiments import quality_benchmark as qb
    output=Path(output).resolve()
    if output==Path(reference).resolve().parent or output.is_relative_to(Path(reference).resolve().parent):
        raise ValueError('Use a new output outside historical reference tree')
    plan=build_plan(reference);qb.write_plan(plan,output)
    manifest={'registered_utc':utc(),'status':'planned','plan_sha256':qb.identity(plan),'sources':plan['source_hashes'],
              'environment':plan['environment'],'local_only':True,'coupling_study':plan['coupling_study']}
    target=output/'study_manifest.json'
    if not target.exists():qb.save(target,manifest)
    elif json.loads(target.read_text())['plan_sha256']!=manifest['plan_sha256']:
        raise ValueError('Output belongs to a different planned study')
    return {'plan':str(output/'plan.json'),'plan_sha256':qb.identity(plan),'jobs':len(plan['jobs']),'status':'planned'}


def execute_job(folder,job_id):
    from experiments import quality_benchmark as qb
    from experiments.refine_logical_scores import refine_run
    from quantum.coupled_measurement import direct_cdf_backend
    import torch
    torch.set_num_threads(1)
    folder=Path(folder);plan=qb.load_plan(folder,current_sources=True)
    job=next(j for j in plan['jobs'] if j['id']==job_id)
    if job['config']['experimental_measurement_sampler']!=sampler_contract(job['sampler']):
        raise ValueError('Sampler contract differs from registered active implementation')
    if job['config']['noise_extension'] or len(job['config']['points'])!=1:
        raise ValueError('One static resource per process is required; no dynamic-width cache growth')
    out=folder/'jobs'/job_id;out.mkdir(parents=True,exist_ok=True)
    context=direct_cdf_backend(**PARAMETERS) if job['sampler']=='direct_cdf' else nullcontext(None)
    backend=None
    try:
        with context as backend:
            if qb.verify_job(folder,job) is None:
                c=job['config'];rows=qb.run_case(c,out,job['point'],job['cutoff'],c['per_state'],c['rounds'],None,c['decoders'])
                qb.save(out/'result.json',{'complete':True,'config':c,'rows':rows,'classification':plan['classification']})
    finally:
        if backend is not None and backend.calls:
            qb.save(out/'sampler_diagnostics.json',{'contract':backend.contract,'calls':backend.calls,
                    'scope':'New process-local experimental sampler; production defaults restored on context exit.'})
    score=qb.scoring_status(folder,job,plan['metric_contract']['scoring_bins'])
    if score['status']!='checked':refine_run(out,bins=[192,384,768],tolerance=.001,shared_basis=True)
    return qb.scoring_status(folder,job,[192,384,768])


def variance_summary(folder):
    from experiments import quality_benchmark as qb
    folder=Path(folder);data=json.loads((folder/'summary.json').read_text());ratios={}
    for condition in CONDITIONS:
        entries={name:data['cutoff_equivalence'].get(f'nominal/{condition}__{name}/nearest') for name in ['legacy_mixture','direct_cdf']}
        if any(not v or 'difference' not in v for v in entries.values()):
            ratios[condition]={'status':'pending'};continue
        legacy=entries['legacy_mixture']['difference']['se'];direct=entries['direct_cdf']['difference']['se']
        ratios[condition]={'status':'complete' if all(v['status']=='complete' for v in entries.values()) else 'partial',
                           'direct_over_legacy_variance_ratio':[None if a==0 else (b/a)**2 for a,b in zip(legacy,direct)],
                           'registered_cutoff_pass':{k:v['pass'] for k,v in entries.items()}}
    report={'created_utc':utc(),'summary_sha256':qb.sha(folder/'summary.json'),'ratios':ratios,
            'scope':'Descriptive paired-cutoff variance only; no variance-ratio interval or proof of absence of physical cutoff bias.'}
    qb.save(folder/'variance_comparison.json',report);return report


def run(folder,approved_plan_sha,max_seconds,threads,max_rss_bytes,reserve_disk_bytes,no_sync,no_shutdown):
    from experiments import quality_benchmark as qb
    from experiments.continue_local_physics import preflight,checkpoint_seals,lease,interrupt_signals,stop_child
    folder=Path(folder).resolve();plan=qb.load_plan(folder,current_sources=True)
    if approved_plan_sha!=qb.identity(plan):raise ValueError('Full study requires the exact root-approved registered plan digest')
    if not no_sync or not no_shutdown or max_seconds<=0:raise ValueError('Finite local deadline and --no-sync --no-shutdown required')
    env=dict(os.environ)
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]=str(threads)
    start=time.monotonic();child=None;state={'started_utc':utc(),'status':'running','jobs':{},'study_plan_sha256':qb.identity(plan),
        'max_seconds':max_seconds,'max_rss_bytes':max_rss_bytes,'reserve_disk_bytes':reserve_disk_bytes,
        'thread_environment_requested':threads,'thread_scope':'Accelerate count environment-requested, not independently verified.'}
    with interrupt_signals(),lease(folder/'.study.lock'):
        qb.save(folder/'study_preflight.json',preflight(folder,plan,threads,max_rss_bytes,reserve_disk_bytes))
        try:
            for job in plan['jobs']:
                if time.monotonic()-start>=max_seconds:raise TimeoutError('Fixed study local deadline reached')
                preflight(folder,plan,threads,max_rss_bytes,reserve_disk_bytes)
                score=qb.scoring_status(folder,job,[192,384,768])
                if qb.verify_job(folder,job) is not None and score['status']=='checked':
                    state['jobs'][job['id']]={'status':'reused','scoring':score};continue
                logpath=folder/'logs'/f'{job["id"]}.log';logpath.parent.mkdir(exist_ok=True)
                with logpath.open('a') as log:
                    child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.coupled_cutoff_check','_job',str(folder),job['id']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        checkpoint_seals(folder,job,folder/'local_checkpoints')
                        rss=subprocess.run(['ps','-p',str(child.pid),'-o','rss='],capture_output=True,text=True).stdout.strip()
                        memory=int(rss or '0')*1024
                        state.update(updated_utc=utc(),active_job=job['id'],active_pid=child.pid,active_rss_bytes=memory,elapsed_seconds=time.monotonic()-start)
                        qb.save(folder/'study_status.json',state)
                        if time.monotonic()-start>=max_seconds:raise TimeoutError('Fixed study local deadline reached')
                        if memory>max_rss_bytes:raise MemoryError('Study RSS cap reached')
                        if shutil.disk_usage(folder).free<reserve_disk_bytes:raise OSError('Study disk reserve reached')
                        time.sleep(2)
                    if child.returncode:raise RuntimeError('Experimental job failed; do not retry to obtain a pass: '+str(logpath))
                checkpoint_seals(folder,job,folder/'local_checkpoints')
                state['jobs'][job['id']]={'status':'processed','scoring':qb.scoring_status(folder,job,[192,384,768])}
                qb.analyze_plan(folder);variance_summary(folder)
            state['status']='all_fixed_study_jobs_processed'
        except BaseException as error:
            if child is not None:stop_child(child)
            state.update(status='interrupted' if isinstance(error,(KeyboardInterrupt,TimeoutError)) else 'failed',error=repr(error));raise
        finally:
            state.update(finished_utc=utc(),elapsed_seconds=time.monotonic()-start)
            state.pop('active_pid',None);state.pop('active_job',None)
            qb.save(folder/'study_status.json',state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('plan');p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    p=sub.add_parser('_job');p.add_argument('folder',type=Path);p.add_argument('job_id')
    p=sub.add_parser('summarize');p.add_argument('folder',type=Path)
    p=sub.add_parser('run');p.add_argument('folder',type=Path);p.add_argument('--approved-plan-sha',required=True)
    p.add_argument('--max-seconds',type=int,required=True);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-rss-gib',type=float,default=12.);p.add_argument('--reserve-disk-gib',type=float,default=8.)
    p.add_argument('--no-sync',action='store_true');p.add_argument('--no-shutdown',action='store_true')
    args=parser.parse_args()
    if args.action=='plan':value=prepare(args.reference,args.output)
    elif args.action=='_job':value=execute_job(args.folder,args.job_id)
    elif args.action=='summarize':value=variance_summary(args.folder)
    else:value=run(args.folder,args.approved_plan_sha,args.max_seconds,args.threads,int(args.max_rss_gib*1024**3),int(args.reserve_disk_gib*1024**3),args.no_sync,args.no_shutdown)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
