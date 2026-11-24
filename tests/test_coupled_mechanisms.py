import numpy as np
from experiments.audit_coupled_mechanisms import compare_array_pair


def test_mechanism_control_includes_nan_strings_and_every_field(tmp_path):
    left=tmp_path/'a.npz';right=tmp_path/'b.npz'
    data={'numeric':np.array([1.,np.nan]),'stage_names':np.array(['q_measurement','p_measurement']),'scalar':np.array(2)}
    np.savez(left,**data);np.savez(right,**data)
    report=compare_array_pair(left,right)
    assert report['all_requested_arrays_bitwise_equal'] and set(report['arrays'])==set(data)


def test_mechanism_control_does_not_hide_small_numeric_difference_or_missing_field(tmp_path):
    left=tmp_path/'a.npz';right=tmp_path/'b.npz'
    np.savez(left,value=np.array([1.]),missing=np.array([0]))
    np.savez(right,value=np.array([1.+1e-14]))
    report=compare_array_pair(left,right)
    assert not report['all_requested_arrays_bitwise_equal']
    assert not report['arrays']['value']['bitwise_equal']
    assert report['arrays']['value']['max_absolute_finite_difference']>0
    assert report['arrays']['missing']['missing_right']


def registered_plan():
    # Portable structural fixture: production plan identity is independently
    # sealed by audit(), and need not ship ignored run evidence with unit tests.
    import copy
    from experiments.audit_coupled_mechanisms import EXPECTED_FIXED_DESIGN
    jobs=[]
    for condition in ['loss','loss_dephasing','coupling_error','control_rotation']:
        for cutoff in [4096,5120]:
            for sampler in ['legacy_mixture','direct_cdf']:
                jobs.append({'id':f'{condition}_{cutoff}_{sampler}',
                             'physical_variant':condition,'variant':condition+'__'+sampler,
                             'cutoff':cutoff,'sampler':sampler,'point':'nominal','base_seed':26112401,'chunk':0,
                             'config':{'seed':855319358,'per_state':128,'rounds':8,'cutoffs':[cutoff],
                                       'decoders':['nearest'],'points':{'nominal':[.2,.05,.9]},
                                       'measurement_resource':'smooth','physics':{'loss_rate':.001},
                                       'experimental_measurement_sampler':{'name':sampler}}})
    return {'coupling_study':{'fixed_design':copy.deepcopy(EXPECTED_FIXED_DESIGN)},'expected_jobs':16,'jobs':jobs}


def test_exact_registered_design_has_one_job_in_every_cell():
    from experiments.audit_coupled_mechanisms import validate_registered_design
    cells=validate_registered_design(registered_plan())
    assert len(cells)==16


def test_empty_conditions_cannot_pass_vacuously():
    import pytest
    from experiments.audit_coupled_mechanisms import validate_registered_design
    plan=registered_plan();plan['coupling_study']['fixed_design']['conditions']=[]
    with pytest.raises(ValueError,match='fixed_design'):validate_registered_design(plan)


def test_missing_or_extra_jobs_are_rejected():
    import pytest,copy
    from experiments.audit_coupled_mechanisms import validate_registered_design
    for count in [0,15,17]:
        plan=registered_plan()
        plan['jobs']=(plan['jobs']+[copy.deepcopy(plan['jobs'][0])])[:count]
        with pytest.raises(ValueError,match='all16'):validate_registered_design(plan)


def test_duplicate_job_id_and_duplicate_cell_are_both_rejected():
    import pytest,copy
    from experiments.audit_coupled_mechanisms import validate_registered_design
    plan=registered_plan();plan['jobs'][1]['id']=plan['jobs'][0]['id']
    with pytest.raises(ValueError,match='unique'):validate_registered_design(plan)
    plan=registered_plan();replacement=copy.deepcopy(plan['jobs'][0]);replacement['id']='different_unique_id'
    plan['jobs'][1]=replacement
    with pytest.raises(ValueError,match='duplicate registered cell'):validate_registered_design(plan)


def test_extra_condition_and_inconsistent_fixed_budget_are_rejected():
    import pytest
    from experiments.audit_coupled_mechanisms import validate_registered_design
    plan=registered_plan();plan['jobs'][0]['physical_variant']='unexpected_extra'
    with pytest.raises(ValueError,match='extra or unregistered'):validate_registered_design(plan)
    for key,value in [('per_state',127),('rounds',7),('seed',123),('decoders',['bayes_gaussian'])]:
        plan=registered_plan();plan['jobs'][0]['config'][key]=value
        with pytest.raises(ValueError,match='conflicts'):validate_registered_design(plan)


def test_inconsistent_design_metadata_is_not_accepted_from_the_plan():
    import pytest
    from experiments.audit_coupled_mechanisms import validate_registered_design
    for key,value in [('jobs',0),('per_state',2),('cutoffs',[4096]),('samplers',['direct_cdf']),('conditions',['loss']),('conditions',['loss','loss_dephasing','coupling_error','control_rotation','extra']),('optional_stopping',True)]:
        plan=registered_plan();plan['coupling_study']['fixed_design'][key]=value
        with pytest.raises(ValueError,match='fixed_design'):validate_registered_design(plan)


def test_audit_rejects_structurally_valid_but_unregistered_scientific_plan(tmp_path,monkeypatch):
    import pytest
    from experiments import quality_benchmark as qb
    from experiments.audit_coupled_mechanisms import audit
    plan=registered_plan();plan['jobs'][0]['config']['physics']['loss_rate']=.002
    monkeypatch.setattr(qb,'load_plan',lambda *args,**kwargs:plan)
    with pytest.raises(ValueError,match='exact approved'):audit(tmp_path)
