"""Read exact saved exogenous draws; distinguish reused streams from nested shots."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from experiments import quality_benchmark as qb
BASE=Path(__file__).resolve().parent
folders=[BASE/n/'plans/finite_gate' for n in ['pilot_cdf_128_192','full16_cdf_128_192']]
arrays=[];sources=[]
for folder in folders:
    plan=qb.load_plan(folder,current_sources=False)
    job=next(j for j in plan['jobs'] if j['variant']=='ideal' and j['cutoff']==128)
    assert qb.verify_job(folder,job) is not None
    path=folder/'jobs'/job['id']/'broad_control_N128_smooth/nearest_finite_ancilla.npz'
    with np.load(path) as z:arrays.append({k:z[k] for k in ['dq','dp','nq','np','state_ids']})
    sources.append({'path':str(path),'sha256':qb.sha(path),'base_seed':job['base_seed'],'derived_seed':job['config']['seed'],'per_state':job['config']['per_state']})
a,b=arrays;fields=['dq','dp','nq','np'];matches=[];field_counts={}
for i in range(len(a['state_ids'])):
    full=[j for j in range(len(b['state_ids'])) if all(np.array_equal(a[k][i],b[k][j]) for k in fields)]
    matches.append({'pilot_index':i,'pilot_state_id':int(a['state_ids'][i]),'full_noise_trajectory_matches':full,
                    'same_input_state_matches':[j for j in full if b['state_ids'][j]==a['state_ids'][i]]})
for k in fields:
    field_counts[k]={'exact_row_matches':sum(any(np.array_equal(a[k][i],b[k][j]) for j in range(len(b[k]))) for i in range(len(a[k]))),
                     'same_index_prefix_equal':bool(np.array_equal(a[k],b[k][:len(a[k])]))}
initial=[]
for k in fields:
    initial.append({'field':k,'pilot_initial_scalar_values_found_in_full_same_field_initial':int(sum(bool(np.any(b[k][:,0]==value)) for value in a[k][:,0])),
        'same_index_initial_prefix_equal':bool(np.array_equal(a[k][:,0],b[k][:len(a[k]),0]))})
report={'created_utc':datetime.now(timezone.utc).isoformat(),'classification':'Exact raw stream-overlap audit; no trajectories',
    'sources':sources,'noise_fields':fields,'matched_complete_noise_trajectory_rows':sum(bool(x['full_noise_trajectory_matches']) for x in matches),
    'matched_complete_shots_including_input_label':sum(bool(x['same_input_state_matches']) for x in matches),
    'pilot_rows':len(a['state_ids']),'full_rows':len(b['state_ids']),'per_field':field_counts,'initial_round':initial,'matches':matches,
    'interpretation':'Same registered base/derived RNG stream means no independent confirmation claim. Complete-shot nesting is determined by the exact four-field two-round noise tuple plus cardinal label, not seed equality.',
    'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(BASE/'pilot_full_overlap_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['matched_complete_noise_trajectory_rows','matched_complete_shots_including_input_label','pilot_rows','full_rows','per_field','initial_round']},indent=2))
