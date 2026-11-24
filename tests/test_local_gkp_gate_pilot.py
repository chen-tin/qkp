"""Source/configuration directed checks for the actual GKP finite-gate pilot."""
import numpy as np
import pytest
from experiments import local_gkp_gate_pilot as pilot


def test_original_physics_and_seed_retained_with_explicit_numerical_departures():
    p=pilot.build_plan()
    assert len(p['jobs'])==8
    for j in p['jobs']:
        c=j['config'];assert j['base_seed']==26112001
        assert c['per_state']==2 and c['rounds']==2 and c['data_delta']==.3
        assert c['decoders']==['nearest'] and c['checkpoint_sha256'] is None
        assert c['points']['broad_control']==[.1,.5/pilot.L,.3]
        assert c['seed']==int(np.random.SeedSequence([26112001,0]).generate_state(1)[0])
        if j['variant']=='qutip':assert (c['physics']['gate_ancilla_cutoff'],c['physics']['homodyne_radius'],c['physics']['homodyne_dy'])==(256,20.,.02)
        if j['variant']=='qutip_refined':assert (c['physics']['gate_ancilla_cutoff'],c['physics']['homodyne_radius'],c['physics']['homodyne_dy'])==(384,24.,.01)
        if j['variant']!='ideal':assert (c['physics']['gate_duration'],c['physics']['loss_rate'],c['physics']['dephasing_rate'])==(.1,.01,.001)


@pytest.mark.parametrize('interval,outcome',[
    ([[-.001,0],[.001,.002]],'supports'),
    ([[.011,0],[.012,.001]],'contradicts'),
    ([[.009,0],[.012,.001]],'inconclusive')])
def test_equivalence_is_not_confused_with_overlapping_zero(interval,outcome):
    assert pilot.interval_verdict(interval)==outcome


def test_paired_estimate_interface_and_preflight_order():
    values=np.zeros((12,3))
    result=pilot.estimate(values,2)
    assert pilot.interval_verdict(result)=='supports'
    jobs=pilot.build_plan()['jobs']
    assert [j['variant'] for j in jobs[:4]]==['ideal','ideal','lumped','lumped']


def test_direct_cdf_has_distinct_hashed_job_and_source_identity():
    legacy=pilot.build_plan();spec=pilot.direct_cdf_spec()
    direct=pilot.build_plan(measurement_sampler=spec)
    assert not ({j['id'] for j in legacy['jobs']}&{j['id'] for j in direct['jobs']})
    assert direct['measurement_sampler']['uniform_index']==1
    for job in direct['jobs']:
        assert job['config']['measurement_sampler']==spec
        assert 'quantum/coupled_measurement.py' in job['config']['source_hashes']
        assert 'experiments/local_gkp_gate_pilot.py' in job['config']['source_hashes']
        assert job['id']==pilot.qb.identity({k:v for k,v in job.items() if k!='id'})[:16]


@pytest.mark.parametrize('fail',[False,True])
def test_experimental_child_owns_and_restores_hook_even_on_failure(tmp_path,monkeypatch,fail):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from quantum import coupled_measurement,reference_runner
    original=reference_runner._measure;replacement=lambda:None
    backend=SimpleNamespace(contract={'uniform_index':1},calls=[])
    @contextmanager
    def context(**kwargs):
        reference_runner._measure=replacement
        try:yield backend
        finally:reference_runner._measure=original
    monkeypatch.setattr(coupled_measurement,'direct_cdf_backend',context)
    spec=pilot.direct_cdf_spec();jobs=[{'id':str(i),'variant':'ideal','config':{'measurement_sampler':spec}} for i in range(4)]
    monkeypatch.setattr(pilot.qb,'load_plan',lambda *a,**k:{'jobs':jobs,'measurement_sampler':spec})
    monkeypatch.setattr(pilot.qb,'verify_job',lambda *a:None)
    executed=[]
    def run(*a,**k):
        assert reference_runner._measure is replacement
        executed.append(1)
        if fail:raise RuntimeError('directed child failure')
    monkeypatch.setattr(pilot.qb,'run_plan',run)
    monkeypatch.setattr(pilot.qb,'analyze_plan',lambda *a:{})
    if fail:
        with pytest.raises(RuntimeError,match='directed child failure'):pilot.execute(tmp_path,True)
    else:pilot.execute(tmp_path,True)
    assert reference_runner._measure is original
    import json
    receipt=json.loads((tmp_path/'population_preflight_backend.json').read_text())
    assert receipt['restored_original_measurement'] is True
    assert bool(receipt['error']) is fail
    assert len(executed)==(1 if fail else 4)


def test_cdf_acceptance_requires_complete_budgeted_source_bound_hook_evidence(tmp_path):
    plan=pilot.build_plan(measurement_sampler=pilot.direct_cdf_spec());spec=plan['measurement_sampler']
    assert not pilot.backend_evidence(tmp_path,plan)['pass']
    calls=[]
    for job in plan['jobs']:
        if job['variant'] not in ['ideal','lumped']:continue
        for axis in ['q','p']*2:
            calls.append({'job_id':job['id'],'axis':axis,'cutoff':job['cutoff'],'trajectories':12,
                'max_declared_exact_cdf_residual_bound':1e-11,'minimum_conditional_norm':.1})
    record={'declared_sampler':spec,'restored_original_measurement':True,'error':None,'contract':{'uniform_index':1},
        'driver_sha256':plan['source_hashes']['experiments/local_gkp_gate_pilot.py'],
        'backend_sha256':plan['source_hashes']['quantum/coupled_measurement.py'],'calls':calls}
    pilot.save(tmp_path/'population_preflight_backend.json',record)
    pilot.save(tmp_path/'run_backend.json',dict(record,calls=[]))
    assert pilot.backend_evidence(tmp_path,plan)['pass']
    record['calls']=calls[:-1];pilot.save(tmp_path/'population_preflight_backend.json',record)
    assert not pilot.backend_evidence(tmp_path,plan)['pass']


def test_full16_metadata_is_actual_runtime_and_96_total_not_pilot_projection():
    text=pilot.uncertainty_text(16)
    assert '16 trajectories/cardinalstate' in text and '96 total/job/arm' in text
    assert 'pilot2/state' not in text
    result=pilot.execution_metadata(16,629.9,8,8,True)
    assert result['actual_fixed16_state_seconds']==629.9
    assert result['total_trajectories_per_job_per_arm']==96
    assert 'projected_full16_state_seconds' not in result
    assert 'full_budget_preflight_pass' not in result
    diagnostic=pilot.execution_metadata(2,100.,8,8,True)
    assert diagnostic['projected_full16_state_seconds']==800.
    assert diagnostic['full_budget_preflight_pass'] is True
    assert 'actual_fixed16_state_seconds' not in diagnostic
