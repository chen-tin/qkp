"""Bounded higher-cutoff reproduction of the original two-cycle GKP gate control."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from experiments import quality_benchmark as qb
from experiments.qutip_reference import estimate, L

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(per_state=2,cutoffs=(128,192),measurement_sampler=None):
    matrix={'points':{'broad_control':[.10,.5/L,.3]},'variants':{
        'ideal':{'physics':{}},
        'lumped':{'physics':{'gate_duration':.1,'loss_rate':.01,'dephasing_rate':.001}},
        'qutip':{'physics':{'gate_backend':'qutip','gate_duration':.1,'loss_rate':.01,'dephasing_rate':.001,
            'gate_ancilla_cutoff':256,'homodyne_radius':20.,'homodyne_dy':.02}},
        'qutip_refined':{'physics':{'gate_backend':'qutip','gate_duration':.1,'loss_rate':.01,'dephasing_rate':.001,
            'gate_ancilla_cutoff':384,'homodyne_radius':24.,'homodyne_dy':.01}}}}
    plan=qb.build_plan('finite_gate',per_state=per_state,rounds=2,seeds=(26112001,),
                         decoders=('nearest',),cutoffs=list(cutoffs),matrix=matrix)
    if measurement_sampler:
        sources=dict(plan['source_hashes'])
        for name in ['quantum/coupled_measurement.py','experiments/local_gkp_gate_pilot.py']:
            sources[name]=sha(ROOT/name)
        plan['source_hashes']=sources
        plan['measurement_sampler']=measurement_sampler
        for job in plan['jobs']:
            job['config']['source_hashes']=dict(sources)
            job['config']['measurement_sampler']=measurement_sampler
            job['id']=qb.identity({k:v for k,v in job.items() if k!='id'})[:16]
    return plan


def uncertainty_text(per_state):
    common=(f'Paired state-stratified normal95% logical-fidelity intervals; {per_state} trajectories/cardinalstate ×6 ='
            f'{6*per_state} total/job/arm, pointwise bycycle. Deterministic quadrature receives no MonteCarlointerval. ')
    if per_state==16:
        return common+'Original fixed16/state control budget on the registered base stream; not independent confirmation of the diagnosticpilot. No nested-shot claim follows from seed equality.'
    return common+'Diagnostic numerical/timingpilot with limited statisticalpower; not an independent confirmatory equivalence claim.'


def execution_metadata(per_state,elapsed,completed,expected,numerical):
    result={'actual_execution_seconds':elapsed,'trajectories_per_cardinal_state':per_state,
            'total_trajectories_per_job_per_arm':6*per_state}
    if per_state==16:
        result.update(budget_mode='original_fixed16_control',actual_fixed16_state_seconds=elapsed,
                      fixed16_budget_complete=completed==expected)
    else:
        projected=elapsed*16/per_state if completed==expected else None
        result.update(budget_mode='diagnostic_pilot',projected_full16_state_seconds=projected,
            full_budget_preflight_pass=numerical and projected is not None and projected<=3600)
    return result


def registration(plan,wall_seconds,rss_bytes):
    per=plan['fixed_budget']['per_state_per_job']
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'status':'Planned',
        'classification':'local_numerical_GKP_finite_gate_pilot' if per==2 else 'local_GKP_finite_gate_control',
        'purpose':'Resolve numerical validity of the original broad two-cycle finite-gate controls using higher data/ancillary cutoffs; preserve original8invalid controls.',
        'question':'Do higher-cutoff original GKP controls pass population, scoring, data-cutoff and explicit ancillary/grid refinement checks, and does lumped dissipation agree with refined explicit gates within1pp logical fidelity?',
        'hypothesis':'The declared higher-cutoff controls pass numerical checks; lumped-minus-refined-explicit logical-fidelity differences remain within±.01 atallthreecycle samples.',
        'fixed':{'noise_distribution':'stationary Gaussian AR(1) q/p displacement; independent axes; finite-energy resource measurement back-action',
            'sigma_q_over_L':.10,'resource_sigma_absolute_q':.5,'sigma_m_over_L':.5/L,'rho':.3,
            'data_delta':.3,'rounds':2,'cardinal_inputs':6,'controller':'nearest','base_seed':26112001,
            'loss_rate_per_time_unit':.01,'dephasing_rate_per_time_unit':.001,'each_gate_duration':.1,'cycle_time':1.,
            'ancillary_dissipation':0,'L':'sqrt(pi), [q,p]=i','training_seeds':[]},
        'departure_from_original':{'per_state':per,'original_per_state':16,'data_cutoffs':sorted(set(j['cutoff'] for j in plan['jobs'])),
            'original_data_cutoffs':[32,48],'coarse_explicit_ancilla_radius_dy':[256,20.,.02],
            'refined_explicit_ancilla_radius_dy':[384,24.,.01]},
        'primary_metric':'six-cardinal equal-weight mean logical fidelity by cycle; difference between paired arms; lower absolute difference isbetter',
        'numerical_metrics':'Unchanged stage meanedge<1e-4,individualedge<.01,norm<1e-7,gategridmass1±1e-5; saved-state192/384/768refinement absolute tolerance.001; paireddata-cutoff andqutip/refined95%intervals whollyinside±.01.',
        'decision_rule':{'supports':'All eight jobs and required numerical/refinement checks pass; lumped-minus-refined intervals wholly within±.01 everycycle atboth data cutoffs.',
            'contradicts':'All numerical checks pass and at least one lumped-minus-refined interval lies entirely outside±.01.',
            'inconclusive':'Any numerical gate/coverage/resource failure or interval overlaps an equivalence boundary. Keep failed controls; next registered numericalpair256/384 if128/192 fails.'},
        'uncertainty':uncertainty_text(per),
        'limits':{'wall_seconds':wall_seconds,'hard_rss_bytes':rss_bytes,'threads':1},
        'full_budget_gate':'Launch separate16/state8-job plan only after pilot numerical checks pass and projected runtime8×pilot wall<=3600s. Full run retains negative/inconclusive findings.',
        'measurement_sampler':plan.get('measurement_sampler',{'kind':'legacy_auxiliary_mixture','uniform_indices':[0,1,2]}),
        'sampler_error_scope':'DirectCDF table/interpolation/slope/tail components follow analytic bounds; float_budget is an explicitly assumed floating-error allowance, not a proved floatingpoint bound. Deterministic source-directed marginal tests are separate evidence.',
        'environment':plan['environment'],'driver_sha256':sha(__file__),'benchmark_plan_sha256':qb.identity(plan),
        'scope':'Two-cycle broad-noise GKP control only; not nominal8cycle memory, hardware accuracy, or all-state certification.'}


def interval_verdict(interval,tolerance=.01):
    if isinstance(interval,dict):interval=[interval["ci95_low"],interval["ci95_high"]]
    low,high=np.asarray(interval,float)
    if np.all(low>=-tolerance) and np.all(high<=tolerance):return 'supports'
    if np.any((low>tolerance)|(high<-tolerance)):return 'contradicts'
    return 'inconclusive'


def backend_evidence(folder,plan):
    spec=plan.get('measurement_sampler')
    if not spec:return {'required':False,'pass':True}
    errors=[];calls=[];files=[]
    for phase in ['population_preflight','run']:
        path=Path(folder)/(phase+'_backend.json')
        if not path.exists():errors.append('missing '+path.name);continue
        row=json.loads(path.read_text());files.append(path.name)
        if row.get('declared_sampler')!=spec:errors.append('sampler declaration mismatch')
        if not row.get('restored_original_measurement') or row.get('error') is not None:errors.append('hook lifecycle failure')
        if row.get('driver_sha256')!=plan['source_hashes']['experiments/local_gkp_gate_pilot.py']:errors.append('driver source mismatch')
        if row.get('backend_sha256')!=plan['source_hashes']['quantum/coupled_measurement.py']:errors.append('sampler source mismatch')
        if (row.get('contract') or {}).get('uniform_index')!=1:errors.append('quantile stream mismatch')
        calls+=row.get('calls',[])
    expected=Counter()
    jobs={j['id']:j for j in plan['jobs']}
    for job in plan['jobs']:
        if job['variant'] in ('ideal','lumped'):
            for axis in ['q','p']:expected[(job['id'],axis)]=job['config']['rounds']
    observed=Counter((c.get('job_id'),c.get('axis')) for c in calls)
    if observed!=expected:errors.append('incomplete or unexpected hooked call coverage')
    for call in calls:
        job=jobs.get(call.get('job_id'));bound=call.get('max_declared_exact_cdf_residual_bound',float('inf'))
        if not job or call.get('cutoff')!=job['cutoff'] or call.get('trajectories')!=6*job['config']['per_state']:
            errors.append('hooked input/cutoff coverage mismatch')
        if not np.isfinite(bound) or bound>spec['parameters']['tolerance'] or call.get('minimum_conditional_norm',0)<=0:
            errors.append('declared CDF/norm budget failed')
    return {'required':True,'pass':not errors,'errors':errors,'calls':len(calls),'records':files,
        'max_declared_cdf_residual_bound':max((c['max_declared_exact_cdf_residual_bound'] for c in calls),default=None),
        'floating_error_status':'Declared engineering allowance; special-function floatingerror is not rigorously certified.'}


def analyze(folder):
    folder=Path(folder);study=json.loads((folder/'study.json').read_text())
    path=folder/'plans/finite_gate';plan=qb.load_plan(path,current_sources=True)
    summary=qb.analyze_plan(path)
    pairs={};invalid=[]
    for job in plan['jobs']:
        result=qb.verify_job(path,job)
        if result is None:
            invalid.append(job['id']);continue
        case=next((path/'jobs'/job['id']).glob('*_smooth'))
        raw=case/'nearest_finite_ancilla.npz'
        with np.load(raw) as z:pairs[(job['cutoff'],job['variant'])]=z['fidelity'].copy()
    comparisons=[]
    for cutoff in sorted(set(j['cutoff'] for j in plan['jobs'])):
        for candidate,reference,kind in [('qutip','qutip_refined','explicit_refinement'),('lumped','qutip_refined','approximation')]:
            if (cutoff,candidate) not in pairs or (cutoff,reference) not in pairs:continue
            value=estimate(pairs[(cutoff,candidate)]-pairs[(cutoff,reference)],study['departure_from_original']['per_state'])
            comparisons.append({'cutoff':cutoff,'candidate':candidate,'reference':reference,'kind':kind,
                'paired_difference':value,'verdict':interval_verdict(value)})
    refinement=[x for x in comparisons if x['kind']=='explicit_refinement']
    approximation=[x for x in comparisons if x['kind']=='approximation']
    backend=backend_evidence(folder,plan)
    numerical=(backend['pass'] and summary['all_jobs_executed'] and summary['registered_numerical_sample_checks_pass'] and
               len(refinement)==2 and all(x['verdict']=='supports' for x in refinement))
    decision='inconclusive'
    if numerical:
        decision='supports' if all(x['verdict']=='supports' for x in approximation) else (
            'contradicts' if any(x['verdict']=='contradicts' for x in approximation) else 'inconclusive')
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'classification':study['classification'],
        'completed_jobs':summary['completed_jobs'],'expected_jobs':summary['expected_jobs'],'stage_failures':summary['stage_failures'],
        'registered_numerical_sample_checks_pass':summary['registered_numerical_sample_checks_pass'],
        'explicit_refinement_pass':len(refinement)==2 and all(x['verdict']=='supports' for x in refinement),
        'numerical_checks_pass':numerical,'measurement_backend_evidence':backend,'decision':decision,'comparisons':comparisons,'invalid_or_missing_jobs':invalid,
        'scope':study['scope'],'uncertainty':uncertainty_text(plan['fixed_budget']['per_state_per_job']),'source_sha256':plan['source_hashes']}
    save(folder/'analysis.json',result);return result


def direct_cdf_spec():
    return {'kind':'direct_conditional_cdf_experimental','uniform_index':1,
        'parameters':{'max_step':.00025,'tolerance':1e-10,'shot_batch':64,'node_chunk':256,
            'tail_probability':1e-15,'float_budget':5e-12,'max_table_bytes':67108864},
        'scope':'Same marginal conditional instrument; new cross-cutoff and replay common-random-number coupling. Legacy mixture uses auxiliaryuniform0+resourceuniform1; directCDFusesonlyuniform1 forphysicaloutcome. Explicitqutipbranch unchanged.'}


def execute(folder,preflight=False):
    """Enter the declared hook INSIDE the sole scientific child process."""
    from quantum.coupled_measurement import direct_cdf_backend
    from quantum import reference_runner
    from ops.ec2 import parallel_worker
    folder=Path(folder).resolve();path=folder/'plans/finite_gate'
    plan=qb.load_plan(path,current_sources=True);spec=plan.get('measurement_sampler')
    if not spec or spec['kind']!='direct_conditional_cdf_experimental' or spec['uniform_index']!=1:
        raise ValueError('Scientific child requires its explicitly hashed sampler declaration')
    for job in plan['jobs']:
        if job['config'].get('measurement_sampler')!=spec:raise ValueError('Sampler/job identity mismatch')
    original=reference_runner._measure;backend=None;error=None
    label='population_preflight' if preflight else 'run'
    epoch=folder/'execution_epochs'/('direct_cdf_'+label+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    epoch.mkdir(parents=True)
    try:
        with direct_cdf_backend(**spec['parameters']) as backend:
            jobs=plan['jobs'][:4] if preflight else plan['jobs']
            if preflight:
                for job in jobs:
                    before=len(backend.calls)
                    if qb.verify_job(path,job) is None:qb.run_plan(path,max_jobs=1)
                    for call in backend.calls[before:]:call.update(job_id=job['id'],variant=job['variant'])
            else:
                with parallel_worker.lease(path/'.run.lock'):
                    for job in jobs:
                        before=len(backend.calls)
                        parallel_worker.execute_job(path,job['id'],epoch)
                        for call in backend.calls[before:]:call.update(job_id=job['id'],variant=job['variant'])
                        qb.analyze_plan(path)
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        record={'declared_sampler':spec,'contract':backend.contract if backend is not None else None,
            'calls':backend.calls if backend is not None else [],'restored_original_measurement':reference_runner._measure is original,
            'error':error,'phase':label,'driver_sha256':sha(__file__),'backend_sha256':sha(ROOT/'quantum/coupled_measurement.py')}
        save(folder/(label+'_backend.json'),record)
    qb.analyze_plan(path)


def run(folder):
    folder=Path(folder).resolve();study=json.loads((folder/'study.json').read_text())
    if sha(__file__)!=study['driver_sha256']:raise ValueError('Frozen driver changed')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic();samples=[];reason=None;commands=[]
    def bounded(command,label):
        nonlocal reason
        commands.append(command)
        with (folder/(label+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            print(label,'PID',process.pid,flush=True)
            while process.poll() is None:
                data=subprocess.run(['ps','-axo','pid=,ppid=,rss='],capture_output=True,text=True).stdout.splitlines()
                entries=[tuple(map(int,line.split())) for line in data if len(line.split())==3]
                ids={process.pid}
                while True:
                    expanded=ids|{pid for pid,ppid,rss in entries if ppid in ids}
                    if expanded==ids:break
                    ids=expanded
                rss=sum(rss*1024 for pid,ppid,rss in entries if pid in ids)
                samples.append({'phase':label,'seconds':time.monotonic()-start,'tree_rss_bytes':rss,'pids':sorted(ids)})
                if rss>study['limits']['hard_rss_bytes']:reason='tree_rss_cap'
                if time.monotonic()-start>study['limits']['wall_seconds']:reason='wall_time_cap'
                if reason:
                    process.send_signal(signal.SIGINT)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        for pid in ids:
                            try:os.kill(pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                        process.wait()
                    break
                time.sleep(1)
        return process.returncode
    # Cheap ideal/lumped trajectories come first in the frozen job order.
    # Do not spend explicit-gate compute on a known invalid data cutoff.
    direct=study.get('measurement_sampler',{}).get('kind')=='direct_conditional_cdf_experimental'
    if direct:
        preflight=[sys.executable,'-u','-m','experiments.local_gkp_gate_pilot','execute',str(folder),'--preflight']
    else:
        preflight=[sys.executable,'-u','-m','experiments.quality_benchmark','run',
                   str(folder/'plans/finite_gate'),'--max-jobs','4']
    code=bounded(preflight,'population_preflight')
    preflight_pass=False
    if code==0 and not reason:
        first=json.loads((folder/'plans/finite_gate/summary.json').read_text())
        preflight_pass=first['completed_jobs']==4 and first['stage_failures']==0
        save(folder/'population_preflight.json',{'pass':preflight_pass,'completed_jobs':first['completed_jobs'],
            'stage_failures':first['stage_failures'],'scope':'Ideal/lumped population checks only; explicit jobs remain queued until this passes.'})
    if preflight_pass:
        if direct:
            command=[sys.executable,'-u','-m','experiments.local_gkp_gate_pilot','execute',str(folder)]
        else:
            command=[sys.executable,'-u',str(ROOT/'ops/ec2/parallel_worker.py'),'run','--root',str(folder),
                '--run-id',folder.name,'--presets','finite_gate','--workers','1','--threads','1','--no-sync','--no-shutdown']
        code=bounded(command,'run')
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'returncode':code,'interruption_reason':reason,
        'population_preflight_pass':preflight_pass,'elapsed_seconds':time.monotonic()-start,
        'max_tree_rss_bytes':max((x['tree_rss_bytes'] for x in samples),default=0),
        'samples':samples,'commands':commands,'cwd':str(ROOT),
        'thread_environment':{k:env[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}}
    save(folder/'resource_usage.json',receipt)
    if not reason and code==0:
        result=analyze(folder);result['resource_usage']=receipt|{'samples':[]}
        result.update(execution_metadata(study['departure_from_original']['per_state'],receipt['elapsed_seconds'],
            result['completed_jobs'],result['expected_jobs'],result['numerical_checks_pass']))
        save(folder/'analysis.json',result)
        print(json.dumps({k:v for k,v in result.items() if k in ['completed_jobs','stage_failures','numerical_checks_pass','decision',
            'actual_execution_seconds','budget_mode','projected_full16_state_seconds','full_budget_preflight_pass','actual_fixed16_state_seconds','fixed16_budget_complete']},indent=2))
    else:
        print(json.dumps(receipt|{'samples':[]},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    plan=sub.add_parser('plan');plan.add_argument('folder',type=Path);plan.add_argument('--per-state',type=int,default=2)
    plan.add_argument('--cutoffs',type=int,nargs=2,default=[128,192]);plan.add_argument('--wall-seconds',type=int,default=600)
    plan.add_argument('--rss-gib',type=float,default=2.);plan.add_argument('--sampler',choices=['legacy','direct-cdf'],default='legacy')
    for action in ['run','analyze']:sub.add_parser(action).add_argument('folder',type=Path)
    executor=sub.add_parser('execute');executor.add_argument('folder',type=Path);executor.add_argument('--preflight',action='store_true')
    args=p.parse_args()
    if args.action=='plan':
        if (args.folder/'study.json').exists():raise ValueError('Use new immutable study folder')
        benchmark=build_plan(args.per_state,args.cutoffs,direct_cdf_spec() if args.sampler=='direct-cdf' else None);qb.write_plan(benchmark,args.folder/'plans/finite_gate')
        save(args.folder/'study.json',registration(benchmark,args.wall_seconds,int(args.rss_gib*1024**3)))
        print(args.folder/'study.json')
    elif args.action=='run':run(args.folder)
    elif args.action=='execute':execute(args.folder,args.preflight)
    else:print(json.dumps(analyze(args.folder),indent=2))


if __name__=='__main__':main()
