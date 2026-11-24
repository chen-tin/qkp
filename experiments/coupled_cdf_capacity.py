"""Registered local capacity check of the new sampler at full4096 cutoff.

Twelve cardinal trajectories/two cycles test capacity only. This cannot close
production statistical, cutoff-equivalence or rare-event requirements.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def utc():return datetime.now(timezone.utc).isoformat()


def prepare(output):
    from experiments import quality_benchmark as qb
    from experiments.coupled_cutoff_check import sampler_contract,EXTRA
    output=Path(output).resolve()
    plan=qb.build_plan('numerical',per_state=2,rounds=2,seeds=(26112491,),decoders=('nearest',),cutoffs=[4096])
    sources=dict(plan['source_hashes']);sources.update({p:qb.sha(ROOT/p) for p in EXTRA+['experiments/coupled_cdf_capacity.py']})
    job=plan['jobs'][0];job.pop('id');job['sampler']='direct_cdf';job['physical_variant']='baseline'
    job['variant']='baseline__direct_cdf';job['config']['source_hashes']=sources
    job['config']['experimental_measurement_sampler']=sampler_contract('direct_cdf');job['id']=qb.identity(job)[:16]
    plan['source_hashes']=sources;plan['classification']='infrastructure_capacity_pilot'
    plan['capacity_contract']={'registered_utc':utc(),'purpose':'Timing/RSS/full-width direct-CDF safety check only',
          'per_state':2,'cardinal_inputs':6,'trajectories':12,'cycles':2,'cutoff':4096,'controller':'nearest',
          'base_seed':26112491,'noise_units':'sigma_q/L=.20,sigma_m/L=.05,rho=.90,stationaryGaussianAR1; independentq/p',
          'max_seconds':180,'max_rss_bytes':4*1024**3,'threads_requested':2,
          'pass_rule':'Completed registered job/refinement,4q/pCDFcalls,finitepositiveconditionalnorms,eachdeclaredCDFresidual<=1e-10,elapsed<=180s,peakRSS<=4GiB. Primitive state gates are separately reported and do not certify fullphysics.',
          'scope':'One4096cutoff/12trajectories/2cycles cannot establish original8cycle statistical or cutoff validity.'}
    target=output/'plan.json'
    if target.exists():
        old=qb.load_plan(output,current_sources=True)
        plan['capacity_contract']['registered_utc']=old['capacity_contract']['registered_utc']
    qb.write_plan(plan,output)
    return plan


def child(folder):
    from experiments import quality_benchmark as qb
    from experiments.coupled_cutoff_check import execute_job
    folder=Path(folder);plan=qb.load_plan(folder,current_sources=True);job=plan['jobs'][0]
    started=time.monotonic();score=execute_job(folder,job['id'])
    result=qb.verify_job(folder,job);diagnostics=json.loads((folder/'jobs'/job['id']/'sampler_diagnostics.json').read_text())
    calls=diagnostics['calls'];maximum=max(c['max_declared_exact_cdf_residual_bound'] for c in calls)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    elapsed=time.monotonic()-started
    record={'finished_utc':utc(),'plan_sha256':qb.identity(plan),'job_id':job['id'],'elapsed_seconds':elapsed,
            'peak_rss_bytes':peak,'sampler_calls':len(calls),'sampler_diagnostics':diagnostics,
            'max_declared_cdf_residual_bound':maximum,'saved_state_scoring':score,
            'primitive_stage_passes':[r['numerical_edge_gate_pass'] for r in result['rows']],
            'evolution_seconds_including_both_arms':result['rows'][0]['seconds'],
            'capacity_pass':bool(len(calls)==4 and maximum<=1e-10 and all(c['minimum_conditional_norm']>0 for c in calls) and
                                 elapsed<=180 and peak<=4*1024**3 and score['status']=='checked' and score['pass']),
            'scope':plan['capacity_contract']['scope']}
    qb.save(folder/'capacity_result.json',record);return record


def run(folder):
    from experiments import quality_benchmark as qb
    from experiments.continue_local_physics import available_memory,stop_child,interrupt_signals
    folder=Path(folder).resolve();plan=qb.load_plan(folder,current_sources=True)
    if plan['capacity_contract']['cutoff']!=4096 or len(plan['jobs'])!=1:
        raise ValueError('Only registered4096tinycapacitypilot supported')
    available=available_memory()
    if available<5*1024**3:raise ValueError('Capacity pilot needs4GiBcap plus1GiBavailable-memoryheadroom')
    env=dict(os.environ)
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='2'
    argv=[str(ROOT/'.venv/bin/python'),'-u','-m','experiments.coupled_cdf_capacity','_job',str(folder)]
    qb.save(folder/'launch.json',{'launched_utc':utc(),'argv':argv,'plan_sha256':qb.identity(plan),
                                'available_memory_bytes':available,'max_seconds':180,'max_rss_bytes':4*1024**3,
                                'local_only':True,'no_sync':True,'no_shutdown':True,'thread_scope':'Environmentrequested2;Acceleratecountnotindependentlyverified'})
    started=time.monotonic();process=None
    with interrupt_signals():
        try:
            with (folder/'execution.log').open('w') as log:
                process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    rss=subprocess.run(['ps','-p',str(process.pid),'-o','rss='],capture_output=True,text=True).stdout.strip()
                    observed=int(rss or '0')*1024
                    qb.save(folder/'capacity_status.json',{'status':'running','pid':process.pid,'rss_bytes':observed,'elapsed_seconds':time.monotonic()-started})
                    if observed>4*1024**3:raise MemoryError('Registered4GiBpilotcapreached')
                    if time.monotonic()-started>180:raise TimeoutError('Registered180spilotcapreached')
                    time.sleep(1)
                if process.returncode:raise RuntimeError('Capacitychildfailed;seeexecution.log')
            result=json.loads((folder/'capacity_result.json').read_text())
            result['parent_wall_seconds']=time.monotonic()-started
            result['parent_budget_pass']=result['parent_wall_seconds']<=180
            qb.save(folder/'capacity_result.json',result)
            qb.save(folder/'capacity_status.json',{'status':'completed','elapsed_seconds':time.monotonic()-started,'capacity_pass':result['capacity_pass'] and result['parent_budget_pass']})
            return result
        except BaseException as error:
            if process is not None:stop_child(process)
            qb.save(folder/'capacity_status.json',{'status':'failed','error':repr(error),'elapsed_seconds':time.monotonic()-started});raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','run','_job']);parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    if args.action=='plan':
        from experiments.quality_benchmark import identity
        plan=prepare(args.folder);value={'status':'planned','plan_sha256':identity(plan),'jobs':1}
    elif args.action=='_job':value=child(args.folder)
    else:value=run(args.folder)
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
