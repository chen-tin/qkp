"""Read-only fixed mechanism controls for the completed two-sampler study."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_array_pair(left,right,fields=None):
    """Report every requested field, including floating NaN bit patterns."""
    report={}
    with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
        selected=sorted(set(a.files)|set(b.files)) if fields is None else sorted(fields)
        for key in selected:
            if key not in a or key not in b:
                report[key]={'bitwise_equal':False,'missing_left':key not in a,'missing_right':key not in b};continue
            x,y=a[key],b[key]
            structural=x.shape==y.shape and x.dtype==y.dtype
            record={'bitwise_equal':bool(structural and x.tobytes(order='C')==y.tobytes(order='C')),
                    'left_shape':list(x.shape),'right_shape':list(y.shape),
                    'left_dtype':str(x.dtype),'right_dtype':str(y.dtype),'elements':int(x.size)}
            if structural and x.dtype.kind in 'fc':
                finite=np.isfinite(x)&np.isfinite(y)
                record.update(numeric_equal_including_nan=bool(np.array_equal(x,y,equal_nan=True)),
                              nan_patterns_equal=bool(np.array_equal(np.isnan(x),np.isnan(y))),
                              max_absolute_finite_difference=float(np.max(np.abs(x[finite]-y[finite]))) if np.any(finite) else 0.)
            report[key]=record
    return {'left':str(left),'right':str(right),'left_sha256':sha(left),'right_sha256':sha(right),
            'all_requested_arrays_bitwise_equal':bool(report) and all(x['bitwise_equal'] for x in report.values()),
            'arrays':report}



EXPECTED_APPROVED_PLAN_SHA256='eb995962bfa7e47ca1e360fca56699e4a3c27699323a72a03792338c880a3032'
EXPECTED_CONDITIONS=('loss','loss_dephasing','coupling_error','control_rotation')
EXPECTED_SAMPLERS=('legacy_mixture','direct_cdf')
EXPECTED_CUTOFFS=(4096,5120)
EXPECTED_FIXED_DESIGN={'conditions':list(EXPECTED_CONDITIONS),'samplers':list(EXPECTED_SAMPLERS),
                       'per_state':128,'cardinal_inputs':6,'trajectories_per_job':768,'cycles':8,
                       'cutoffs':list(EXPECTED_CUTOFFS),'controller':'nearest','base_seed':26112401,
                       'jobs':16,'optional_stopping':False}


def validate_registered_design(plan):
    """Reject missing/extra/duplicate cells before constructing any lookup dict."""
    if plan.get('coupling_study',{}).get('fixed_design')!=EXPECTED_FIXED_DESIGN:
        raise ValueError('Mechanism audit fixed_design differs from the exact registered study')
    jobs=plan.get('jobs')
    if not isinstance(jobs,list) or len(jobs)!=16 or plan.get('expected_jobs')!=16 or any(not isinstance(j,dict) for j in jobs):
        raise ValueError('Mechanism audit requires all16 registered jobs')
    identities=[job.get('id') for job in jobs]
    if any(not isinstance(identity,str) or not identity for identity in identities) or len(set(identities))!=16:
        raise ValueError('Mechanism audit requires16 unique job IDs')
    expected={(condition,cutoff,sampler) for condition in EXPECTED_CONDITIONS
              for cutoff in EXPECTED_CUTOFFS for sampler in EXPECTED_SAMPLERS}
    indexed={}
    expected_seed=int(np.random.SeedSequence([26112401,0]).generate_state(1)[0])
    for job in jobs:
        cell=(job.get('physical_variant'),job.get('cutoff'),job.get('sampler'))
        if cell not in expected:raise ValueError('Mechanism audit contains an extra or unregistered condition/cutoff/sampler')
        if cell in indexed:raise ValueError('Mechanism audit contains a duplicate registered cell')
        condition,cutoff,sampler=cell;c=job.get('config',{})
        if not isinstance(c,dict):raise ValueError('Mechanism audit job config conflicts with fixed_design')
        if (job.get('point')!='nominal' or job.get('variant')!=condition+'__'+sampler or
                job.get('base_seed')!=26112401 or job.get('chunk')!=0 or c.get('seed')!=expected_seed or
                c.get('per_state')!=128 or c.get('rounds')!=8 or c.get('cutoffs')!=[cutoff] or
                c.get('decoders')!=['nearest'] or c.get('points')!={'nominal':[.2,.05,.9]} or
                c.get('measurement_resource')!='smooth' or
                c.get('experimental_measurement_sampler',{}).get('name')!=sampler):
            raise ValueError('Mechanism audit job config conflicts with fixed_design')
        indexed[cell]=job
    if set(indexed)!=expected:raise ValueError('Mechanism audit is missing registered cells')
    return indexed


def audit(folder):
    from experiments import quality_benchmark as qb
    from quantum.reference_noise import sample_noise
    folder=Path(folder).resolve();plan=qb.load_plan(folder,current_sources=True)
    indexed=validate_registered_design(plan)
    if qb.identity(plan)!=EXPECTED_APPROVED_PLAN_SHA256:
        raise ValueError('Mechanism audit requires the exact approved scientific plan identity')
    comparisons=[];reconstructions=[]
    for condition in EXPECTED_CONDITIONS:
        for cutoff in EXPECTED_CUTOFFS:
            jobs={sampler:indexed[(condition,cutoff,sampler)] for sampler in EXPECTED_SAMPLERS}
            paths={}
            for sampler,job in jobs.items():
                if qb.verify_job(folder,job) is None or qb.scoring_status(folder,job,[192,384,768])['status']!='checked':
                    raise ValueError('Mechanism audit requires all fixed observations and refinements processed')
                paths[sampler]=folder/'jobs'/job['id']/qb.case_directory(job['point'],cutoff,None,'smooth')
            a,b=jobs['legacy_mixture']['config'].copy(),jobs['direct_cdf']['config'].copy()
            a.pop('experimental_measurement_sampler');b.pop('experimental_measurement_sampler')
            if a!=b:raise ValueError('Unexpected nonsampler configuration difference')
            replay=compare_array_pair(paths['legacy_mixture']/'nearest_replayed_corrections.npz',
                                      paths['direct_cdf']/'nearest_replayed_corrections.npz')
            lp=paths['legacy_mixture']/'nearest_finite_ancilla.npz';rp=paths['direct_cdf']/'nearest_finite_ancilla.npz'
            with np.load(lp,allow_pickle=False) as x,np.load(rp,allow_pickle=False) as y:
                fields={'dq','dp','nq','np','state_ids'}|{key for key in set(x.files)|set(y.files) if key.startswith('schedule_')}
                expected_different=sorted((set(x.files)|set(y.files))-fields)
            finite=compare_array_pair(lp,rp,fields)
            comparisons.append({'condition':condition,'cutoff':cutoff,'replay_all_arrays':replay,'finite_arm_exogenous_arrays':finite,
                                'finite_arm_fields_not_claimed_unchanged':expected_different,
                                'reason':'Physical outcomes/state/feedback may differ under the new joint draw mapping, despite the same marginal instrument.'})
            c=jobs['legacy_mixture']['config'];params=c['points'][jobs['legacy_mixture']['point']]
            dq,dp,nq,np_,uq,up,schedule=sample_noise(params,6*c['per_state'],c['rounds'],c['seed'],c['noise_extension'])
            reconstructed={'dq':dq,'dp':dp,'nq':nq,'np':np_}
            checked={}
            for sampler,path in [('legacy_mixture',lp),('direct_cdf',rp)]:
                with np.load(path,allow_pickle=False) as raw:
                    checked[sampler]={key:bool(raw[key].dtype==value.dtype and raw[key].shape==value.shape and raw[key].tobytes()==value.tobytes()) for key,value in reconstructed.items()}
            reconstructions.append({'condition':condition,'cutoff':cutoff,'seed':c['seed'],'base_seed':jobs['legacy_mixture']['base_seed'],
                                    'recorded_noise_matches_reconstruction':checked,
                                    'uniform_q_sha256':hashlib.sha256(uq.tobytes()).hexdigest(),
                                    'uniform_p_sha256':hashlib.sha256(up.tobytes()).hexdigest(),
                                    'uniform_shape':list(uq.shape),'uniforms_persisted_in_original_raw':False,
                                    'scope':'Regeneration from frozen seed/config/source, not new independent noise or physical trajectories.'})
    if len(comparisons)!=8 or len(reconstructions)!=8:
        raise ValueError('Mechanism audit did not execute all8 registered condition/cutoff comparisons')
    passed=all(c[section]['all_requested_arrays_bitwise_equal'] for c in comparisons for section in ['replay_all_arrays','finite_arm_exogenous_arrays'])
    passed &= all(all(all(fields.values()) for fields in r['recorded_noise_matches_reconstruction'].values()) for r in reconstructions)
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'classification':'Read-only fixed mechanism control; no new physical trajectories',
            'plan_sha256':qb.identity(plan),'audit_source_sha256':sha(__file__),'pass':bool(passed),'comparisons':comparisons,
            'uniform_reconstruction':reconstructions,'replay_array_exclusions':[],
            'metadata_exclusions':{'NPZ_ZIP_container_bytes':'ZIP headers/timestamps can vary; every requested decoded array dtype/shape/bytes is compared.',
                                   'experimental_measurement_sampler_config':'Intentionally distinct sampler algorithm/table/uniform mapping is the registered lever; every other scientific config field must match.'},
            'limits':'Bitwise unchanged branches and frozen noise do not independently prove physical correctness or equivalence of the changed joint trajectory mapping. Any array mismatch remains explicit.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.folder)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pass':report['pass'],'condition_cutoff_pairs':len(report['comparisons']),'output':str(args.output)}))


if __name__=='__main__':main()
