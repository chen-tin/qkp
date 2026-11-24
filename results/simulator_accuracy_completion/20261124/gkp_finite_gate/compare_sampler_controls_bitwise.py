"""Check that the prospective sampler comparison changes only intended coupling."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
import numpy as np
from experiments import quality_benchmark as qb

BASE=Path(__file__).resolve().parent
names=['pilot_128_192','pilot_cdf_128_192']
folders=[BASE/name/'plans/finite_gate' for name in names]
plans=[qb.load_plan(folder,current_sources=False) for folder in folders]
key=lambda j:(j['cutoff'],j['variant'])
lookup=[{key(j):j for j in p['jobs']} for p in plans]
rows=[]
for k,j0 in lookup[0].items():
    j1=lookup[1][k]
    for folder,job in zip(folders,[j0,j1]):
        assert qb.verify_job(folder,job) is not None
    clean=lambda c:{x:v for x,v in c.items() if x not in ['source_hashes','measurement_sampler']}
    config_match=clean(j0['config'])==clean(j1['config'])
    for arm in ['replayed_corrections','finite_ancilla']:
        paths=[folder/'jobs'/job['id']/f"{job['point']}_N{job['cutoff']}_smooth"/f'nearest_{arm}.npz'
               for folder,job in zip(folders,[j0,j1])]
        with np.load(paths[0]) as a,np.load(paths[1]) as b:
            exogenous=all(np.array_equal(a[name],b[name]) and a[name].dtype==b[name].dtype and a[name].tobytes()==b[name].tobytes() for name in ['dq','dp','nq','np','state_ids'])
            unchanged=(arm=='replayed_corrections' or j0['variant'] in ['qutip','qutip_refined'])
            mismatched=[];maxdiff=0.;checked=0
            if unchanged:
                if set(a.files)!=set(b.files):mismatched.append('raw_key_coverage')
                for name in set(a.files)&set(b.files):
                    av,bv=a[name],b[name];checked+=1
                    if av.shape!=bv.shape:mismatched.append(name);continue
                    if av.dtype.kind in 'fc':
                        match=np.allclose(av,bv,atol=1e-12,rtol=0,equal_nan=True)
                        finite=np.isfinite(av)&np.isfinite(bv)
                        if finite.any():maxdiff=max(maxdiff,float(np.max(np.abs(av[finite]-bv[finite]))))
                    else:match=np.array_equal(av,bv)
                    if not match:mismatched.append(name)
                    if av.dtype!=bv.dtype or av.tobytes()!=bv.tobytes():mismatched.append(name+'_byte_identity')
            rows.append({'cutoff':k[0],'variant':k[1],'arm':arm,'physical_config_same':config_match,
                'exogenous_arrays_identical':exogenous,'expected_entire_arm_unchanged':unchanged,
                'arrays_checked':checked,'max_absolute_difference':maxdiff,'mismatched_arrays':mismatched})
common=set(plans[0]['source_hashes'])&set(plans[1]['source_hashes'])
source_match=all(plans[0]['source_hashes'][k]==plans[1]['source_hashes'][k] for k in common)
report={'created_utc':datetime.now(timezone.utc).isoformat(),'classification':'fresh strict byte-identity mechanism/control audit; original numerical comparison preserved separately',
    'pass':source_match and all(r['physical_config_same'] and r['exogenous_arrays_identical'] and not r['mismatched_arrays'] for r in rows),
    'common_original_scientific_source_hashes_match':source_match,'rows':rows,
    'sampler_comparison':'New direct-CDF physical outcome usesuniform1instead ofauxiliaryeigenvalue0+resource1; changed CRNcoupling isdeclared, marginals tested separately. Explicitgate andclassicalreplay arrays mustremainunchanged.',
    'sources':[str(f) for f in folders],'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'limitations':'Twelve trajectories/job in pilot; this audit tests unchanged draws/branches, not equivalence from empirical performance means or hardware validity.'}
(BASE/'sampler_control_comparison_bitwise.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'pass':report['pass'],'arms':len(rows),'unchanged_arms':sum(r['expected_entire_arm_unchanged'] for r in rows),
    'arrays_checked':sum(r['arrays_checked'] for r in rows),'max_absolute_difference':max(r['max_absolute_difference'] for r in rows),'failed':[r for r in rows if not r['exogenous_arrays_identical'] or r['mismatched_arrays']]},indent=2))
