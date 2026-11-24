import numpy as np
import pytest
import qutip as qt
from quantum.instrument_operator_bound import transfer,data_eigenvalues,finite_grid_distance,reduce_transfer,capacity
from quantum.measurement_benchmark import ExplicitQutip,qutip_wave_basis,outcome_grid
from quantum.spectral_circuit import finite_q_basis
from quantum.continuous_ancilla import ContinuousAncilla


@pytest.mark.parametrize('axis',['q','p'])
def test_transfer_matches_direct_two_mode_circuit_on_every_small_eigenstate(axis):
    nd,na=6,64
    lambdas,vectors=finite_q_basis(nd)
    states=vectors.astype(complex)
    if axis=='p':states=(1j**np.arange(nd))[:,None]*states
    matrix=transfer(lambdas,na,.5,chunk=7)
    y=np.linspace(-8,8,81)
    amplitude=matrix@qutip_wave_basis(na,y)
    reference=ExplicitQutip(nd,na,.5,y,axes=(axis,))
    for k in range(nd):
        conditional=reference.conditional(states[:,k],axis)[0]
        np.testing.assert_allclose(conditional,states[:,k,None]*amplitude[k,None,:],atol=4e-10,rtol=4e-10)


def test_finite_grid_operator_norm_identity_and_random_inputs():
    rng=np.random.default_rng(26112402)
    ref=rng.normal(size=(6,17))+1j*rng.normal(size=(6,17))
    cand=rng.normal(size=(6,17))+1j*rng.normal(size=(6,17))
    weights=np.linspace(.01,.2,17)
    squared,bound=finite_grid_distance(ref,cand,weights)
    dense=np.concatenate([np.diag(ref[:,j]-cand[:,j])*np.sqrt(weights[j]) for j in range(17)],axis=0)
    assert np.linalg.svd(dense,compute_uv=False)[0]==pytest.approx(bound,abs=1e-12)
    for _ in range(20):
        state=rng.normal(size=6)+1j*rng.normal(size=6);state/=np.linalg.norm(state)
        assert np.linalg.norm(dense@state)<=bound+1e-12
    assert np.linalg.norm(dense@np.eye(6)[:,np.argmax(squared)])==pytest.approx(bound)


def test_streamed_reduction_matches_dense_finite_grid_and_preserves_all_lambdas():
    lambdas=data_eigenvalues(6);matrix=transfer(lambdas,64,.5,chunk=7)
    metrics,curves=reduce_transfer(matrix,lambdas,.5,12.,.04,chunk=7)
    y,w,_=outcome_grid(12.,.04)
    ref=matrix@qutip_wave_basis(64,y)
    local=ContinuousAncilla(.5).amplitude(y[None,:]-lambdas[:,None])
    squared,bound=finite_grid_distance(ref,local,w)
    np.testing.assert_allclose(squared,curves['squared_distance'],atol=1e-15)
    assert metrics['finite_grid_stinespring_operator_norm']==pytest.approx(bound,abs=1e-12)
    assert len(curves['squared_distance'])==6


def test_declared_largest_control_preflight_fits_four_gib_without_data_eigenvectors():
    estimate=capacity(4096,12288,160.,.01,128)
    assert estimate['estimated_scratch_bytes']<4*1024**3


def test_missing_outcome_window_cannot_pass_even_if_local_and_reference_agree():
    from quantum.instrument_operator_bound import checks_pass
    lambdas=data_eigenvalues(6);matrix=transfer(lambdas,64,.5,chunk=7)
    metrics,_=reduce_transfer(matrix,lambdas,.5,.5,.04,chunk=7)
    assert metrics['max_reference_mass_error']>.1
    assert not checks_pass(metrics,False)


def test_capacity_gate_rejects_before_constructing_data_spectrum(tmp_path,monkeypatch):
    from experiments import validate_instrument_operator as cli
    monkeypatch.setattr(cli,'data_eigenvalues',lambda *a:pytest.fail('preflight ran too late'))
    config={'data_cutoff':4096,'ancilla_cutoffs':[8192,12288],'radius':160.,'dy':.01,
            'chunk':128,'max_scratch_bytes':1}
    with pytest.raises(ValueError,match='scratch estimate exceeds'):
        cli.run(config,tmp_path)


def test_operator_audit_rejects_changed_pass_without_repeating_transfer(tmp_path,monkeypatch):
    import json
    from experiments import validate_instrument_operator as cli
    config={'data_cutoff':6,'ancilla_cutoffs':[64,96],'sigma':.5,'coupling':1.,
            'radius':12.,'dy':.04,'chunk':7,'max_scratch_bytes':512000000,'source_hashes':{}}
    result=cli.run(config,tmp_path)
    assert result['pass']
    path=tmp_path/'result.json';result['pass']=False;path.write_text(json.dumps(result))
    monkeypatch.setattr(cli,'transfer',lambda *a,**kw:pytest.fail('audit repeated transfer construction'))
    with pytest.raises(ValueError,match='pass flag differs'):
        cli.run(config,tmp_path,audit=True)


def test_row_tiled_transfer_matches_dense_and_preserves_union_rows(tmp_path):
    from quantum.instrument_operator_bound import write_transfer
    lambdas=np.concatenate([data_eigenvalues(6),data_eigenvalues(8)])
    expected=transfer(lambdas,64,.5,chunk=7)
    path=tmp_path/'tiled.npy';write_transfer(path,lambdas,64,.5,chunk=7)
    actual=np.load(path,mmap_mode='r',allow_pickle=False)
    assert actual.shape==(14,64)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)


def test_union_cli_reduction_reports_each_complete_cutoff(tmp_path):
    from experiments import validate_instrument_operator as cli
    config={'data_cutoff':8,'data_cutoffs':[6,8],'ancilla_cutoffs':[64,96],'sigma':.5,'coupling':1.,
            'radius':12.,'dy':.04,'chunk':7,'grid_chunk':64,'max_scratch_bytes':512000000,'source_hashes':{}}
    result=cli.run(config,tmp_path)
    assert result['pass']
    assert result['all_declared_lambdas']==14
    assert {(r['data_cutoff'],r['all_lambdas_tested']) for r in result['rows']}=={(6,6),(8,8)}


def test_row_tiled_transfer_resumes_sealed_row_prefix(tmp_path,monkeypatch):
    import json
    from quantum import instrument_operator_bound as bound
    lambdas=np.concatenate([data_eigenvalues(6),data_eigenvalues(8)])
    expected=transfer(lambdas,64,.5,chunk=7)
    original=bound.real_left;calls=[]
    def interrupt_second_row_block(*a,**kw):
        calls.append(1)
        if len(calls)==3:raise RuntimeError('directed interruption')
        return original(*a,**kw)
    monkeypatch.setattr(bound,'real_left',interrupt_second_row_block)
    path=tmp_path/'resumed.npy'
    with pytest.raises(RuntimeError,match='directed interruption'):
        bound.write_transfer(path,lambdas,64,.5,chunk=7)
    assert json.loads(path.with_suffix('.progress.json').read_text())['next_row']==7
    calls.clear()
    def counted(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(bound,'real_left',counted)
    bound.write_transfer(path,lambdas,64,.5,chunk=7)
    assert len(calls)==2  # ancillary coefficients and remaining row block
    np.testing.assert_allclose(np.load(path),expected,atol=1e-12,rtol=1e-12)
