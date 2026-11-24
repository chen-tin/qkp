"""Directed deterministic controls for the bounded local finite-gate audit."""
import numpy as np
import pytest
import qutip as qt

from experiments import local_finite_gate_audit as audit
from quantum.finite_gate import FiniteGate
from quantum.quality import Physics
from quantum.measurement_benchmark import outcome_grid


def test_density_instrument_distance_has_probability_and_conditional_sensitivity():
    nd,na=4,8
    vacuum=qt.tensor(qt.basis(nd,0),qt.basis(na,0)).proj().full()
    orthogonal=qt.tensor(qt.basis(nd,1),qt.basis(na,0)).proj().full()
    y,w,_=outcome_grid(8,.04)
    same=audit.instrument_distance(vacuum,vacuum,nd,na,'q',y,w)
    wrong=audit.instrument_distance(vacuum,orthogonal,nd,na,'q',y,w)
    assert same['instrument_trace_distance']==pytest.approx(0,abs=1e-12)
    assert same['reference_integrated_mass']==pytest.approx(1,abs=1e-8)
    assert wrong['instrument_trace_distance']==pytest.approx(1,abs=1e-8)
    assert wrong['outcome_tv_unnormalized']==pytest.approx(0,abs=1e-12)


def test_cutoff_embedding_does_not_invent_a_difference():
    small=qt.tensor(qt.basis(4,0),qt.basis(6,0)).proj().full()
    large=qt.tensor(qt.basis(8,0),qt.basis(10,0)).proj().full()
    y,w,_=outcome_grid(8,.08)
    r=audit.instrument_distance(large,small,8,10,'p',y,w,candidate_nd=4,candidate_na=6)
    assert r['instrument_trace_distance']<1e-12


def test_decision_preserves_numerically_ambiguous_threshold():
    assert audit.decide([.002],.001)=='supports'
    assert audit.decide([.02],.001)=='contradicts'
    assert audit.decide([.0098],.001)=='inconclusive'
    assert audit.decide([],0)=='inconclusive'


def test_stage_negative_control_is_not_upgraded():
    gates=audit.build_plan()['gates']
    assert audit.diagnostics_pass([{'norm_error':1e-12,'data_boundary_mass':1e-8,'ancilla_boundary_mass':1e-8}],gates)
    assert not audit.diagnostics_pass([{'norm_error':1e-12,'data_boundary_mass':.001,'ancilla_boundary_mass':1e-8}],gates)


@pytest.mark.parametrize('loss,phase',[(.4,0),(0,.3),(.4,.3)])
def test_idle_lumped_and_simultaneous_density_channels_agree(loss,phase):
    physics=Physics(gate_backend='qutip',gate_duration=.1,loss_rate=loss,dephasing_rate=phase,
                    gate_ancilla_cutoff=8,homodyne_radius=8,homodyne_dy=.2)
    gate=FiniteGate(6,.5,physics)
    gate.hamiltonians={a:0*h for a,h in gate.hamiltonians.items()}
    data=(qt.basis(6,0)+qt.basis(6,1)).unit().full()[:,0]
    opts={'atol':1e-12,'rtol':1e-11,'nsteps':100000}
    exact,_=audit.evolve_density(gate,data,'q','explicit',opts,[0,.5,1])
    lumped,_=audit.evolve_density(gate,data,'q','lumped',opts,[0,.5,1])
    assert audit.trace_distance(exact,lumped)<1e-9
    reduced=qt.Qobj(exact,dims=[[6,8],[6,8]]).ptrace(0).full()
    assert reduced[1,1].real==pytest.approx(.5*np.exp(-loss*.1),abs=1e-9)
    assert reduced[0,1].real==pytest.approx(.5*np.exp(-(.5*loss+.5*phase)*.1),abs=1e-9)


def test_matrix_free_generator_matches_explicit_liouvillian_and_adjoint():
    h=qt.tensor(qt.sigmax(),qt.qeye(3))+.2*qt.tensor(qt.sigmay(),qt.num(3))
    collapse=[.1*qt.tensor(qt.sigmax()+.3j*qt.sigmaz(),qt.qeye(3)),.2*qt.tensor(qt.qeye(2),qt.num(3))]
    operator,trace=audit.lindblad_operator(h,collapse)
    explicit=qt.liouvillian(h,collapse)
    rng=np.random.default_rng(23)
    matrix=rng.normal(size=(6,6))+1j*rng.normal(size=(6,6))
    vector=matrix.reshape(-1)
    expected=(explicit.full()@matrix.reshape(-1,order='F')).reshape(6,6,order='F')
    assert np.max(np.abs((operator@vector).reshape(6,6)-expected))<1e-12
    probe=rng.normal(size=36)+1j*rng.normal(size=36)
    assert np.vdot(probe,operator@vector)==pytest.approx(np.vdot(operator.rmatvec(probe),vector),abs=1e-11)
    assert trace==pytest.approx(np.trace(explicit.full()).real,abs=1e-12)


def test_resume_rejects_changed_scientific_settings_before_copy(tmp_path):
    import json
    old=tmp_path/'old';old.mkdir()
    original=audit.build_plan();audit.save(old/'plan.json',original)
    changed=dict(original,sigma_absolute_q=.4)
    with pytest.raises(ValueError,match='scientific settings'):
        audit.inherit_observations(tmp_path/'new',changed,old)


def test_resume_accepts_only_exact_dependencies_helpers_and_raw_bytes(tmp_path):
    import shutil
    plan=audit.build_plan();old=tmp_path/'old';new=tmp_path/'new'
    audit.save(old/'plan.json',plan)
    snapshot=old/'source_snapshot'/audit.SOURCES[0];snapshot.parent.mkdir(parents=True)
    shutil.copy2(audit.ROOT/audit.SOURCES[0],snapshot)
    raw=old/'raw/control.npz';raw.parent.mkdir();np.savez(raw,metric=np.array([.1]))
    row={'raw_path':'raw/control.npz','raw_sha256':audit.sha(raw)}
    audit.save(old/'result.json',{'rows':[row],'refinements':[]})
    rows,ref=audit.inherit_observations(new,plan,old)
    assert rows==[row] and ref==[]
    assert (new/'raw/control.npz').read_bytes()==raw.read_bytes()
    raw.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='raw observation changed'):
        audit.inherit_observations(tmp_path/'bad',plan,old)


def test_extension_requires_increasing_data_and_fixed_ancillary_rung(tmp_path):
    plan=audit.build_plan();old=tmp_path/'old';audit.save(old/'plan.json',plan)
    changed=dict(plan,numerical_extension=True,density_cutoffs=[[24,96],[32,128]])
    with pytest.raises(ValueError,match='hold ancillary cutoff'):
        audit.inherit_observations(tmp_path/'new',changed,old)


def test_lumped_joint_generator_matches_sequential_data_loss_and_number_dephasing():
    # Production applies loss then number dephasing. Their generators commute
    # because a loss jump lowers both density indices equally, preserving n-m.
    a=qt.destroy(8);number=qt.num(8)
    loss=qt.liouvillian(0*number,[.1*a]).full()
    phase=qt.liouvillian(0*number,[.03*number]).full()
    assert np.max(np.abs(loss@phase-phase@loss))<1e-14


@pytest.mark.parametrize('axis',['q','p'])
def test_actual_finite_gate_dissipative_sampler_source_path(axis):
    # A fixed-seed software spot check of the production mcsolve+homodyne path;
    # this is not an ensemble estimate or a Monte Carlo accuracy claim.
    physics=Physics(gate_backend='qutip',gate_duration=.1,loss_rate=.01,dephasing_rate=.001,
                    gate_ancilla_cutoff=64,homodyne_radius=10,homodyne_dy=.04)
    gate=FiniteGate(64,.5,physics)
    ket,y,logp,diagnostics=gate.measure(qt.basis(64,0).full()[:,0],axis,.5,23)
    assert np.isfinite(y) and np.isfinite(logp)
    assert np.linalg.norm(ket)==pytest.approx(1,abs=1e-12)
    assert diagnostics[0]<1e-7 and diagnostics[1]<1e-4 and abs(diagnostics[2]-1)<1e-5


def test_exact_refinement_coverage_requires_all_36_distinct_configurations():
    plan=audit.build_plan();rows=[]
    for label in plan['input_states']:
        for axis in plan['axes']:
            for variant in plan['rates_per_declared_time_unit']:
                for arm in ['explicit','lumped']:
                    rows.append({'kind':'cutoff','key':f'{label}_{axis}_{variant}','arm':arm,
                        'from':plan['density_cutoffs'][0],'to':plan['density_cutoffs'][1],'pass':True})
            rows.append({'kind':'solver','key':f'{label}_{axis}_combined','pass':True})
    complete=audit.refinement_coverage(plan,rows)
    assert complete['pass'] and complete['expected_records']==36
    assert not audit.refinement_coverage(plan,rows[:20])['pass']
    replaced=rows[:-1]+[rows[0]]
    result=audit.refinement_coverage(plan,replaced)
    assert not result['pass'] and result['missing'] and result['duplicate_keys']


def test_resource_resume_preserves_existing_extended_cutoff_grid(tmp_path,monkeypatch):
    import json,sys
    previous=tmp_path/'previous';target=tmp_path/'new'
    plan=audit.build_plan();plan['density_cutoffs']=[[24,96],[32,96]]
    plan['numerical_extension']='Prior separately registered data refinement'
    audit.save(previous/'plan.json',plan)
    monkeypatch.setattr(sys,'argv',['audit','plan',str(target),'--resume-from',str(previous)])
    audit.main()
    resumed=json.loads((target/'plan.json').read_text())
    assert resumed['density_cutoffs']==[[24,96],[32,96]]
    assert resumed['limits']['max_density_joint_dimension']==3072
    assert 'numerical_extension' not in resumed
