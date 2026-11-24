"""Read-only final synthesis; refuses incomplete fixed scientific observations."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,sys
ROOT=Path('/Users/christopherbabayan/programs/QUANTUM/qkp');sys.path.insert(0,str(ROOT))
from experiments import quality_benchmark as qb
from experiments.audit_accuracy_results import classify_interval
from experiments.audit_coupled_mechanisms import validate_registered_design,EXPECTED_APPROVED_PLAN_SHA256
FOLDER=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build():
    plan=qb.load_plan(FOLDER,current_sources=True);cells=validate_registered_design(plan)
    if qb.identity(plan)!=EXPECTED_APPROVED_PLAN_SHA256:raise ValueError('Unapproved scientific plan')
    summary=json.loads((FOLDER/'summary.json').read_text());jobs={};diagnostics={};raw_receipts=[]
    for job in plan['jobs']:
        result=qb.verify_job(FOLDER,job);score=qb.scoring_status(FOLDER,job,[192,384,768])
        if result is None or score['status']!='checked':raise ValueError('Incomplete fixedjob: '+job['id'])
        if len(result['rows'])!=1:raise ValueError('Expected nearestonly row')
        row=result['rows'][0]
        jobs[job['id']]={'condition':job['physical_variant'],'cutoff':job['cutoff'],'sampler':job['sampler'],
                         'scoring_pass':score['pass'],'numerical_stage_pass':row['numerical_edge_gate_pass'],
                         'stage_gates':row['stage_gates'],'max_edge_weight':row['max_edge_weight']}
        for seal in (FOLDER/'jobs'/job['id']).glob('*/*.sha256.json'):
            path=seal.with_name(seal.name[:-12]+'.npz');actual=sha(path)
            if actual!=json.loads(seal.read_text())['sha256']:raise ValueError('Rawseal mismatch')
            raw_receipts.append({'path':str(path.relative_to(FOLDER)),'sha256':actual,'seal_sha256':sha(seal)})
        if job['sampler']=='direct_cdf':
            path=FOLDER/'jobs'/job['id']/'sampler_diagnostics.json';d=json.loads(path.read_text());contract=d['contract'];calls=d['calls']
            contract_ok=contract['uniform_index']==1 and contract['shot_batch']==64 and contract['node_chunk']==256 and contract['max_step']==.00025 and contract['cdf_tolerance']==1e-10 and contract['tail_probability']==1e-15 and contract['float_budget']==5e-12 and contract['max_table_bytes']==67108864
            coverage_ok=len(calls)==16 and [x['axis'] for x in calls]==['q','p']*8 and all(x['cutoff']==job['cutoff'] and x['trajectories']==768 for x in calls)
            table_ids={json.dumps(x['table'],sort_keys=True) for x in calls}
            bounds=[x['max_declared_exact_cdf_residual_bound'] for x in calls];norms=[x['minimum_conditional_norm'] for x in calls]
            valid=contract_ok and coverage_ok and len(table_ids)==1 and all(math.isfinite(x) and x<=1e-10 for x in bounds) and all(math.isfinite(x) and x>0 for x in norms)
            diagnostics[job['id']]={'diagnostic_sha256':sha(path),'contract_matches_registration':contract_ok,'exact16call_coverage':coverage_ok,'unique_static_resource_table_count':len(table_ids),'maximum_declared_cdf_residual_bound':max(bounds),'minimum_conditional_norm':min(norms),'pass':bool(valid)}
    if len(raw_receipts)!=32 or len(diagnostics)!=8:raise ValueError('Incomplete raw/diagnostic coverage')
    comparisons={}
    for condition in ['loss','loss_dephasing','coupling_error','control_rotation']:
        comparisons[condition]={}
        for sampler in ['legacy_mixture','direct_cdf']:
            key=f'nominal/{condition}__{sampler}/nearest';entry=summary['cutoff_equivalence'][key]
            if entry['status']!='complete':raise ValueError('Incomplete cutoff comparison')
            paired_jobs=[jobs[cells[(condition,n,sampler)]['id']] for n in (4096,5120)]
            d=entry['difference'];extent=max(abs(x) for field in ['ci95_low','ci95_high'] for x in d[field])
            comparisons[condition][sampler]={'interval_pass':entry['pass'],'all_numerical_stage_pass':all(x['numerical_stage_pass'] for x in paired_jobs),'all_scoring_pass':all(x['scoring_pass'] for x in paired_jobs),'max_absolute_ci95_extent':extent,'failing_interval_cycles':[i for i,v in enumerate(entry['within_tolerance']) if not v],'cycle8_mean_difference':d['mean'][-1],'cycle8_ci95':[d['ci95_low'][-1],d['ci95_high'][-1]],'all_cycles':d}
            classifications=[classify_interval(lo,hi,.01) for lo,hi in zip(d['ci95_low'],d['ci95_high'])]
            comparisons[condition][sampler]['interval_classification_by_cycle']=classifications
            comparisons[condition][sampler]['interval_interpretation']=('outside_tolerance' if 'outside_tolerance' in classifications else 'inconclusive' if 'inconclusive' in classifications else 'within_tolerance')
            comparisons[condition][sampler]['combined_registered_pass']=all(comparisons[condition][sampler][key] for key in ['interval_pass','all_numerical_stage_pass','all_scoring_pass'])
    mechanism=json.loads((FOLDER/'mechanism_audit.json').read_text())
    if mechanism['plan_sha256']!=EXPECTED_APPROVED_PLAN_SHA256 or len(mechanism['comparisons'])!=8:raise ValueError('Incomplete mechanism control')
    return {'created_utc':datetime.now(timezone.utc).isoformat(),'classification':'Completed fixed-budget sampler/cutoff comparison; no added trajectories','primary_gate_scope':'combined_registered_pass combines the registered paired cutoff interval, recorded primitive-stage and selected saved-state scoring gates. CDF contract and mechanism readbacks are separate. These checks do not establish complete simulator, all-state, dissipative-channel or hardware accuracy.','plan_sha256':EXPECTED_APPROVED_PLAN_SHA256,'plan_file_sha256':sha(FOLDER/'plan.json'),'summary_sha256':sha(FOLDER/'summary.json'),'generator_sha256':sha(__file__),'processed_jobs':len(jobs),'expected_jobs':16,'metric_definition':{'quantity':'Cutoff 5120 minus 4096 mean logical fidelity, paired by trajectory within each of six cardinal states','direction':'Every interval must lie wholly within [-0.01, 0.01]','denominator':'128 independent trajectories per cardinal state; six states; 768 per job and arm. The same exogenous bank is shared across registered conditions, cutoffs and samplers.','horizon':'Cycles 0 through 8','aggregation':'Each cardinal state receives equal weight','uncertainty':'Pointwise state-stratified normal-approximation 95% CI: mean plus or minus 1.95996398454 paired SE. These are not simultaneous intervals. Separate event intervals use fixed-n Clopper-Pearson with Bonferroni correction.','base_seed':26112401,'actual_seed':855319358},'jobs':jobs,'comparisons':comparisons,'cdf_diagnostics':diagnostics,'all_cdf_diagnostics_pass':all(x['pass'] for x in diagnostics.values()),'mechanism_pass':mechanism['pass'],'mechanism_audit_sha256':sha(FOLDER/'mechanism_audit.json'),'variance_comparison':json.loads((FOLDER/'variance_comparison.json').read_text()),'raw_receipts':raw_receipts,'resource_receipt':json.loads((FOLDER/'sessions/registry.json').read_text()),'memory_observation':json.loads((FOLDER/'sessions/resource_samples_summary.json').read_text()),'interpretation_limits':['Both samplers use the same fixed budget of 768 trajectories per job and arm. Primary mean-fidelity intervals provide pointwise stratified normal-approximation 95% coverage, not simultaneous coverage.','The changed common-random-number mapping preserves the declared finite-instrument marginals within the numerical CDF budget. It does not reproduce the historical joint counterfactual draw mapping.','CDF interpolation and tail terms are analytic; the 5e-12 floating-point allowance is an unproved engineering allowance.','Variance reduction does not prove zero true cutoff bias, hardware accuracy, all-state coverage or generality beyond four registered conditions, six cardinal inputs and eight cycles.','The historical summaries with 512 trajectories per state and larger unrun grids remain unchanged.']}

if __name__=='__main__':
    report=build();target=FOLDER/'completion_report.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'jobs':report['processed_jobs'],'cdf_diagnostics_pass':report['all_cdf_diagnostics_pass'],'mechanism_pass':report['mechanism_pass'],'output':str(target)}))
