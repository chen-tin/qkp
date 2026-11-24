"""Fresh-process readback: verify seals and recompute gates/scores/paired statistics.

No trajectories, training, instrument evolution or original report writes.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,time
import numpy as np
from experiments import quality_benchmark as qb
from experiments.refine_logical_scores import refine_raw_many
from quantum.quality import Gates,stage_gates

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'plans/finite_gate'
start=time.monotonic();plan=qb.load_plan(PLAN,current_sources=True)
analysis=json.loads((ROOT/'analysis.json').read_text());summary=json.loads((PLAN/'summary.json').read_text())
raws={};banks={};stage_rows=[];score_status=[];all_pass=True
for job in plan['jobs']:
    result=qb.verify_job(PLAN,job)
    assert result is not None
    status=qb.scoring_status(PLAN,job,[192,384,768]);score_status.append(status)
    assert status['status']=='checked' and status['pass']
    report=json.loads(Path(status['path']).read_text())
    for arm in ['replayed_corrections','finite_ancilla']:
        path=PLAN/'jobs'/job['id']/f"{job['point']}_N{job['cutoff']}_smooth"/f'nearest_{arm}.npz'
        digest=qb.sha(path)
        assert json.loads(path.with_suffix('.sha256.json').read_text())['sha256']==digest
        with np.load(path,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        assert raw['fidelity'].shape==(96,3)
        stage=stage_gates(raw,Gates(**job['config']['gates']))
        saved=result['rows'][0]['stage_gates'][arm]
        assert stage==saved and stage['pass']
        stage_rows.append({'job_id':job['id'],'variant':job['variant'],'cutoff':job['cutoff'],'arm':arm,'raw_sha256':digest,
            'stage_recomputed_pass':stage['pass'],'max_norm_error':stage['max_norm_error'],'max_mean_edge':stage['max_mean_edge'],'max_individual_edge':stage['max_edge']})
        banks.setdefault(job['cutoff'],[]).append((path,report))
        if arm=='finite_ancilla':raws[(job['variant'],job['cutoff'])]=raw['fidelity']
score_rows=[]
for cutoff,items in banks.items():
    checks=refine_raw_many([path for path,report in items],16,bins=(192,384,768),tolerance=.001,
                           trace_tolerance=1e-7,chunk=128,max_batch_bytes=64*1024**2)
    for recomputed,(path,saved) in zip(checks,items):
        old=next(c for c in saved['checks'] if Path(c['raw_file']).resolve()==path.resolve())
        assert recomputed['sample_checks_pass'] and recomputed['all_cycle_snapshots_pass']
        max_delta=0.;max_sensitivity=0.
        for fresh,prior in zip(recomputed['checks'],old['checks']):
            assert fresh['cycle']==prior['cycle'] and fresh['pass'] and prior['pass']
            for f,p in zip(fresh['refinements'],prior['refinements']):
                assert f['bins']==p['bins'] and f['pass'] and p['pass']
                for metric in ['max_absolute_difference','trace_error','minimum_logical_eigenvalue']:
                    max_delta=max(max_delta,abs(f[metric]-p[metric]))
                max_sensitivity=max(max_sensitivity,f['max_absolute_difference'])
        assert max_delta<=1e-10
        score_rows.append({'raw_file':str(path),'cutoff':cutoff,'pass':True,'max_metric_readback_difference':max_delta,
                           'max_quadrature_score_sensitivity':max_sensitivity})

def paired(values):
    a=np.asarray(values).reshape(6,16,3);mean=a.mean((0,1));se=np.sqrt(a.var(1,ddof=1).sum(0)/(36*16))
    return {'mean':mean,'se':se,'ci95_low':mean-1.95996398454*se,'ci95_high':mean+1.95996398454*se}
stat_rows=[]
for record in analysis['comparisons']:
    cutoff=record['cutoff'];delta=raws[(record['candidate'],cutoff)]-raws[(record['reference'],cutoff)]
    fresh=paired(delta);maximum=max(float(np.max(np.abs(fresh[k]-record['paired_difference'][k]))) for k in fresh)
    assert maximum<=1e-12
    stat_rows.append({'kind':record['kind'],'cutoff':cutoff,'max_statistic_difference':maximum,'verdict':record['verdict']})
for variant in ['ideal','lumped','qutip','qutip_refined']:
    fresh=paired(raws[(variant,192)]-raws[(variant,128)])
    old=summary['cutoff_equivalence'][f'broad_control/{variant}/nearest']['difference']
    maximum=max(float(np.max(np.abs(fresh[k]-old[k]))) for k in fresh)
    assert maximum<=1e-12
    stat_rows.append({'kind':'data_cutoff','variant':variant,'max_statistic_difference':maximum})
report={'created_utc':datetime.now(timezone.utc).isoformat(),'classification':'Independent fresh-process raw readback and deterministic rescoring; no new trajectories',
    'pass':True,'jobs':8,'trajectories_per_cardinal_state':16,'cardinal_states':6,'trajectories_per_job_per_arm':96,'cycles':2,
    'sealed_raw_arms':len(stage_rows),'stage_checks':stage_rows,'rescoring':score_rows,'paired_statistics':stat_rows,
    'elapsed_seconds':time.monotonic()-start,'source_sha256':plan['source_hashes'],'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope':'Stage/gate arrays, raw hashes and saved scoring samples rechecked; not all-state/hardware certification. Independent reconstruction of pairedstatistics uses fixed6×16stratification and existing normal95%rule. Representative/hot snapshots only for quadrature.'}
(ROOT/'independent_readback.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'pass':True,'raw_arms':len(stage_rows),'rescored_banks':len(score_rows),'paired_comparisons':len(stat_rows),'elapsed_seconds':report['elapsed_seconds'],
    'max_scoring_readback_delta':max(r['max_metric_readback_difference'] for r in score_rows)},indent=2))
