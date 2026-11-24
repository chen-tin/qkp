"""Directed negative controls for accuracy reporting; no production simulation."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import quality_benchmark as qb
from quantum.quality import Gates, stage_gates, stratified_events


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def saved_job(root, name='job', cutoff=8):
    config={'decoders':['nearest'], 'per_state':2, 'rounds':1,
            'points':{'nominal':[.2,.05,.9]}, 'physics':{'coupling':1.},
            'gates':asdict(Gates()), 'schema':2}
    job={'id':name,'point':'nominal','variant':'baseline','base_seed':1,
         'chunk':0,'cutoff':cutoff,'config':config}
    case=root/'jobs'/name/qb.case_directory('nominal',cutoff,None,'smooth')
    case.mkdir(parents=True)
    raw={'fidelity':np.ones((12,2)), 'energy':np.zeros((12,2)),
         'dq':np.zeros((12,1)), 'dp':np.zeros((12,1)), 'edge':np.zeros((12,2)),
         'stage_energy':np.zeros((12,1,1)), 'stage_edge':np.zeros((12,1,1)),
         'stage_norm_error':np.zeros((12,1,1)), 'max_norm_error':np.array(0.),
         'score_snapshot':np.eye(cutoff,dtype=complex)[:,[0]], 'score_indices':np.array([0])}
    hashes={}
    for arm in ['finite_ancilla','replayed_corrections']:
        p=case/f'nearest_{arm}.npz';np.savez(p,**raw);hashes[p.name]=qb.sha(p)
    row={'decoder':'nearest','point':'nominal','cutoff':cutoff,'raw_hashes':hashes,
         'numerical_edge_gate_pass':True}
    result={'complete':True,'config':config,'rows':[row]}
    write_json(root/'jobs'/name/'result.json',result)
    return job,result,case


@pytest.mark.parametrize('mutation',['empty_rows','duplicate_rows','missing_arm'])
def test_incomplete_decoder_or_raw_coverage_cannot_be_cached_as_complete(tmp_path,mutation):
    job,result,case=saved_job(tmp_path)
    if mutation=='empty_rows': result['rows']=[]
    elif mutation=='duplicate_rows': result['rows']*=2
    else: result['rows'][0]['raw_hashes'].pop('nearest_replayed_corrections.npz')
    write_json(tmp_path/'jobs/job/result.json',result)
    with pytest.raises(ValueError,match='coverage'):
        qb.verify_job(tmp_path,job)


def test_empty_refinement_report_cannot_pass_vacuously(tmp_path):
    job,_,_=saved_job(tmp_path)
    run=tmp_path/'jobs/job'
    write_json(run/'quality/logical_refinement_empty.json',{
        'settings':{'run_result_sha256':qb.sha(run/'result.json'),'bins':[192,384,768], 'tolerance':.001},
        'checks':[], 'all_sample_checks_pass':True,'all_cycles_checked':True})
    assert not qb.scoring_status(tmp_path,job,[192,384,768])['pass']


def test_scoring_missing_and_pending_remain_distinct(tmp_path):
    job,_,_=saved_job(tmp_path)
    assert qb.scoring_status(tmp_path,job,[192,384,768])['status']=='pending_required_refinement'
    assert qb.scoring_status(tmp_path,{'id':'missing'},[192,384,768])['status']=='pending'


def test_parameter_matched_fock_control_does_not_claim_production_state_coverage(tmp_path,monkeypatch):
    from experiments import validate_reference_states as vr
    job,_,_=saved_job(tmp_path)
    control=tmp_path/'control';control.mkdir()
    np.savez(control/'inputs.npz',states=np.eye(8,dtype=complex)[:,[0]])
    result={'config':{'sigma':.05*qb.L,'coupling':1.,'inputs':{'kind':'Fock_states'}},
            'pass':True,'input_sha256':qb.sha(control/'inputs.npz'),'scope':'supplied states'}
    write_json(control/'result.json',result)
    write_json(tmp_path/'instrument_checks/control.json',{'path':str(control),'result_sha256':qb.sha(control/'result.json')})
    monkeypatch.setattr(vr,'audit_report',lambda folder:result)
    report=qb.instrument_status(tmp_path,{'jobs':[job]})
    assert report['parameter_checks_pass']
    assert not report['production_state_checks_pass']
    assert not report['pass']
    assert 'no all-state' in report['scope']


def test_instrument_coverage_is_missing_without_reports(tmp_path):
    job,_,_=saved_job(tmp_path)
    report=qb.instrument_status(tmp_path,{'jobs':[job]})
    assert not report['pass']


@pytest.mark.parametrize('bounds,expected',[
    ((-.01,.01),'within_tolerance'),
    ((.005,.015),'inconclusive'),
    ((.0101,.02),'outside_tolerance'),
    ((-.02,-.0101),'outside_tolerance'),
])
def test_cutoff_interval_classification_at_registered_boundary(bounds,expected):
    from experiments.audit_accuracy_results import classify_interval
    assert classify_interval(*bounds,.01)==expected


def test_cutoff_status_retains_missing_pair_as_partial(tmp_path,monkeypatch):
    job,_,case=saved_job(tmp_path,'lo',8)
    high,_,_=saved_job(tmp_path,'hi',12)
    missing={**job,'id':'missing','base_seed':2}
    plan={'jobs':[job,high,missing], 'metric_contract':{'cutoff_tolerance':.01}}
    report=qb.cutoff_status(tmp_path,plan)['nominal/baseline/nearest']
    assert report['status']=='partial' and not report['pass']
    assert all(report['within_tolerance'])


def test_failed_stage_invalidates_all_later_cycles():
    raw={'edge':np.zeros((12,4)),'stage_edge':np.zeros((12,3,2)),
         'stage_norm_error':np.zeros((12,3,2))}
    raw['stage_edge'][0,1,1]=.02
    report=stage_gates(raw)
    assert report['valid_by_cycle']==[True,True,False,False]
    assert not report['pass']


def test_zero_event_samples_keep_nonzero_bound_and_six_state_scope():
    report=stratified_events(np.zeros(12,dtype=bool),2,comparisons=3)
    assert report['errors']==0 and report['trials']==12
    assert report['one_sided_upper']>0
    assert len(report['states'])==6


def test_incomplete_budget_labels_remain_interim(tmp_path,monkeypatch):
    job,_,_=saved_job(tmp_path)
    missing={**job,'id':'missing','chunk':1}
    plan={'jobs':[job,missing],'preset':'rare','metric_contract':{'scoring_bins':[192,384,768]}}
    monkeypatch.setattr(qb,'load_plan',lambda folder:plan)
    monkeypatch.setattr(qb,'cutoff_status',lambda *a:{'check':{'pass':False}})
    monkeypatch.setattr(qb,'instrument_status',lambda *a:{'pass':False})
    monkeypatch.setattr(qb,'write_plots',lambda *a:False)
    report=qb.analyze_plan(tmp_path)
    group=report['groups']['nominal/baseline/nearest/N8']
    assert not group['fixed_budget_complete']
    assert group['interval_interpretation']=='interim_descriptive_no_stopping_claim'
    assert not report['all_registered_empirical_checks_pass']


def attached_control(root,job,case,monkeypatch,*,bad_source=False,bad_states=False):
    from experiments import validate_reference_states as vr
    raw=case/'nearest_finite_ancilla.npz'
    control=root/'control';control.mkdir()
    with np.load(raw) as bank: states=bank['score_snapshot'].copy()
    if bad_states: states[:]=0;states[1]=1
    np.savez(control/'inputs.npz',states=states)
    provenance={'kind':'saved_states','sha256':'unrelated' if bad_source else qb.sha(raw),
                'key':'score_snapshot','columns':[0],'original_trial_ids':[0],'cardinal_input_ids':[0]}
    result={'config':{'sigma':.05*qb.L,'coupling':1.,'inputs':provenance},
            'pass':True,'input_sha256':qb.sha(control/'inputs.npz'),'scope':'supplied states'}
    write_json(control/'result.json',result)
    write_json(root/'instrument_checks/control.json',{'path':str(control),'result_sha256':qb.sha(control/'result.json')})
    monkeypatch.setattr(vr,'audit_report',lambda folder:result)


def test_matching_sealed_production_snapshot_has_explicit_sample_scope(tmp_path,monkeypatch):
    job,_,case=saved_job(tmp_path)
    attached_control(tmp_path,job,case,monkeypatch)
    report=qb.instrument_status(tmp_path,{'jobs':[job]})
    assert report['pass'] and report['production_state_checks_pass']
    selection=report['production_state_coverage']['job/nearest']['selections'][0]
    assert selection['columns']==[0] and selection['cardinal_input_ids']==[0]
    assert 'no all-state or all-cycle bound' in report['scope']


def test_unrelated_saved_raw_hash_does_not_fill_production_gap(tmp_path,monkeypatch):
    job,_,case=saved_job(tmp_path)
    attached_control(tmp_path,job,case,monkeypatch,bad_source=True)
    report=qb.instrument_status(tmp_path,{'jobs':[job]})
    assert report['parameter_checks_pass'] and not report['production_state_checks_pass']


def test_declared_raw_hash_with_different_actual_input_states_is_rejected(tmp_path,monkeypatch):
    job,_,case=saved_job(tmp_path)
    attached_control(tmp_path,job,case,monkeypatch,bad_states=True)
    with pytest.raises(ValueError,match='differ'):
        qb.instrument_status(tmp_path,{'jobs':[job]})


def test_partial_controller_state_coverage_does_not_fill_other_jobs(tmp_path,monkeypatch):
    job,_,case=saved_job(tmp_path)
    other,_,_=saved_job(tmp_path,'other',12)
    attached_control(tmp_path,job,case,monkeypatch)
    report=qb.instrument_status(tmp_path,{'jobs':[job,other]})
    assert report['production_state_coverage']['job/nearest']['pass']
    assert not report['production_state_coverage']['other/nearest']['pass']
    assert not report['pass']


@pytest.mark.parametrize('missing_arm',[False,True])
def test_refinement_requires_both_raw_arms_even_with_true_summary_flags(tmp_path,missing_arm):
    job,result,case=saved_job(tmp_path)
    run=tmp_path/'jobs/job'
    checks=[{'raw_file':str(case/name),'raw_sha256':sha,'sample_checks_pass':True,
             'all_cycle_snapshots_pass':True,'coverage':'all_cycles'}
            for name,sha in result['rows'][0]['raw_hashes'].items()]
    if missing_arm: checks.pop()
    write_json(run/'quality/logical_refinement_control.json',{
        'settings':{'run_result_sha256':qb.sha(run/'result.json'),'bins':[192,384,768], 'tolerance':.001},
        'checks':checks,'all_sample_checks_pass':True,'all_cycles_checked':True})
    assert qb.scoring_status(tmp_path,job,[192,384,768])['pass'] is (not missing_arm)


def test_refinement_top_level_pass_cannot_hide_selected_cycles_only(tmp_path):
    job,result,case=saved_job(tmp_path)
    run=tmp_path/'jobs/job'
    checks=[{'raw_file':str(case/name),'raw_sha256':sha,'sample_checks_pass':True,
             'all_cycle_snapshots_pass':False,'coverage':'selected_cycles_only'}
            for name,sha in result['rows'][0]['raw_hashes'].items()]
    write_json(run/'quality/logical_refinement_partial.json',{
        'settings':{'run_result_sha256':qb.sha(run/'result.json'),'bins':[192,384,768], 'tolerance':.001},
        'checks':checks,'all_sample_checks_pass':True,'all_cycles_checked':True})
    assert not qb.scoring_status(tmp_path,job,[192,384,768])['pass']


@pytest.mark.parametrize('low,high,expected',[(-.01,.01,True),(-.002,.01001,False),(-.01001,.002,False),(.011,.02,False)])
def test_cutoff_acceptance_uses_entire_interval_at_every_cycle(tmp_path,monkeypatch,low,high,expected):
    from experiments import qutip_reference as qr
    lo,_,_=saved_job(tmp_path,'lo',8)
    hi,_,_=saved_job(tmp_path,'hi',12)
    monkeypatch.setattr(qr,'estimate',lambda values,per_state:{'mean':[0.,.001],
                        'ci95_low':[0.,low],'ci95_high':[0.,high]})
    report=qb.cutoff_status(tmp_path,{'jobs':[lo,hi],'metric_contract':{'cutoff_tolerance':.01}})
    comparison=report['nominal/baseline/nearest']
    assert comparison['within_tolerance']==[True,expected]
    assert comparison['pass'] is expected
