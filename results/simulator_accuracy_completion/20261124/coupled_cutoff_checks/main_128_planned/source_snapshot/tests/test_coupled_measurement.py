"""Deterministic marginal/inverse-CDF controls for an opt-in coupling experiment."""
import numpy as np
import pytest
from quantum.continuous_ancilla import ContinuousAncilla
from quantum.qutip_reference import Oscillator
from quantum.coupled_measurement import (ResourceCDFTable,invert_outcome_cdf,
                                        outcome_cdf_density,DirectCDFMeasurement,direct_cdf_backend)


@pytest.fixture(scope='module',params=[.5,.05*np.sqrt(np.pi)])
def table(request):return ResourceCDFTable(ContinuousAncilla(request.param))


def test_interpolation_is_monotone_and_inside_declared_bound(table):
    rng=np.random.default_rng(26)
    x=np.sort(np.r_[rng.uniform(table.left-.1,table.right+.1,3000),table.left,table.right])
    exact,_=table.exact_cdf_density(x);approx,density=table.cdf_density(x)
    assert np.all(np.diff(approx)>=-2e-15) and density.min()>=0
    assert np.max(np.abs(exact-approx))<=table.error_bound
    assert table.error_terms['hermite_remainder']>0
    assert 'unproved' in table.diagnostics['bound_scope']


def test_deterministic_quantiles_preserve_exact_weighted_mixture_marginal(table):
    rng=np.random.default_rng(42);values=np.linspace(-5,5,24)
    u=np.r_[1e-12,1e-9,.001,.03,.2,.5,.8,.999,1-1e-9,1-1e-12]
    weights=rng.uniform(size=(24,len(u)));weights/=weights.sum(0)
    y,diagnostics=invert_outcome_cdf(weights,values,table,u,coupling=1.01)
    exact,_=outcome_cdf_density(y,weights,values,table,1.01,exact=True)
    resource=ContinuousAncilla(table.sigma)
    direct_source=(weights*resource.cdf(y[None,:]-1.01*values[:,None])).sum(0)/weights.sum(0)
    assert np.max(np.abs(direct_source-u))<=1e-10
    assert np.max(np.abs(exact-u))<=1e-10
    assert diagnostics['max_declared_exact_cdf_residual_bound']<=1e-10


def test_exact_outcome_pdf_matches_original_kraus_norm():
    from quantum.measurement_benchmark import local_conditional
    osc=Oscillator(12,logical_bins=96);resource=ContinuousAncilla(.5);tab=ResourceCDFTable(resource)
    rng=np.random.default_rng(5);psi=rng.normal(size=12)+1j*rng.normal(size=12);psi/=np.linalg.norm(psi)
    for axis in ['q','p']:
        vals,vec=(osc.qvals,osc.qvectors) if axis=='q' else (osc.pvals,osc.pvectors)
        amp=vec.conj().T@psi;y=np.linspace(-8,8,43)
        _,pdf=outcome_cdf_density(y,np.repeat((np.abs(amp)**2)[:,None],len(y),axis=1),vals,tab,1.01,exact=True)
        expected=np.sum(np.abs(local_conditional(osc,resource,psi,y,axis,1.01))**2,axis=0)
        np.testing.assert_allclose(pdf,expected,atol=1e-12,rtol=2e-11)


@pytest.mark.parametrize('axis',['q','p'])
@pytest.mark.parametrize('coupling',[1.,1.01])
def test_direct_backend_keeps_conditional_state_update_and_positive_extreme_norm(axis,coupling,table):
    osc=Oscillator(12,logical_bins=96);resource=ContinuousAncilla(table.sigma)
    psi=np.column_stack([osc.code[:,0]]*5);uniforms=np.full((3,5),.37)
    uniforms[1]=[1e-12,.1,.5,.9,1-1e-12]
    backend=DirectCDFMeasurement(shot_batch=2)
    actual,y,_=backend(osc,psi,resource,uniforms,axis,coupling)
    vals,vec=(osc.qvals,osc.qvectors) if axis=='q' else (osc.pvals,osc.pvectors)
    expected=vec.conj().T@psi;expected*=resource.amplitude(y[None,:]-coupling*vals[:,None]);expected/=np.linalg.norm(expected,axis=0)
    np.testing.assert_allclose(actual,vec@expected,atol=2e-13,rtol=2e-13)
    assert backend.calls[0]['minimum_conditional_norm']>0
    assert backend.calls[0]['max_declared_exact_cdf_residual_bound']<=1e-10


def test_context_restores_default_even_on_exception():
    from quantum import reference_runner
    before=reference_runner._measure
    with pytest.raises(RuntimeError):
        with direct_cdf_backend() as backend:
            assert reference_runner._measure is backend
            raise RuntimeError('directed cleanup control')
    assert reference_runner._measure is before


def test_invalid_probability_or_error_budget_is_rejected():
    tab=ResourceCDFTable(ContinuousAncilla(.5))
    with pytest.raises(ValueError):invert_outcome_cdf(np.zeros((3,1)),np.arange(3),tab,np.array([.5]))
    with pytest.raises(ValueError):invert_outcome_cdf(np.ones((3,1)),np.arange(3),tab,np.array([.5]),tolerance=1e-15)
    with pytest.raises(ValueError):ResourceCDFTable(ContinuousAncilla(.5),max_table_bytes=10)


def test_common_quantile_reduces_deterministic_coupling_variance_without_marginal_change():
    resource=ContinuousAncilla(.5);table=ResourceCDFTable(resource);values=np.array([-1.,1.])
    u=(np.arange(100)+.5)/100
    a=np.repeat(np.array([[.5],[.5]]),len(u),axis=1)
    b=np.repeat(np.array([[.49],[.51]]),len(u),axis=1)
    ya,_=invert_outcome_cdf(a,values,table,u);yb,_=invert_outcome_cdf(b,values,table,u)
    # Tensor-grid integration of the legacy shared latent-eigenvalue coupling.
    latent_a=np.where(u<.5,-1.,1.);latent_b=np.where(u<.49,-1.,1.)
    legacy_difference=np.repeat((latent_b-latent_a)[:,None],len(u),axis=1)
    assert np.var(yb-ya)<.5*np.var(legacy_difference)
    for y,p in [(ya,a),(yb,b)]:
        exact=(p*resource.cdf(y[None,:]-values[:,None])).sum(0)
        np.testing.assert_allclose(exact,u,atol=1e-10,rtol=0)


def test_prospective_study_hashes_sampler_and_keeps_fixed_matched_budgets():
    from pathlib import Path
    from experiments.coupled_cutoff_check import build_plan,CONDITIONS,PARAMETERS
    reference=Path('results/ec2_runs/production_20261119/status_checks/20261123T222312Z/metrics_snapshot/physics/plan.json')
    plan=build_plan(reference)
    assert plan['expected_jobs']==16 and len(plan['jobs'])==16
    assert len({j['id'] for j in plan['jobs']})==16
    for condition in CONDITIONS:
        for cutoff in [4096,5120]:
            jobs=[j for j in plan['jobs'] if j['physical_variant']==condition and j['cutoff']==cutoff]
            assert len(jobs)==2 and {j['sampler'] for j in jobs}=={'legacy_mixture','direct_cdf'}
            a,b=[j['config'] for j in jobs]
            for key in a:
                if key!='experimental_measurement_sampler':assert a[key]==b[key]
            assert a['per_state']==128 and a['rounds']==8 and a['decoders']==['nearest']
            assert all(j['base_seed']==26112401 for j in jobs)
            direct=next(j for j in jobs if j['sampler']=='direct_cdf')['config']['experimental_measurement_sampler']
            assert direct['uniform_index']==1 and direct['parameters']==PARAMETERS
            assert 'quantum/coupled_measurement.py' in a['source_hashes']
    assert plan['metric_contract']['cutoff_tolerance']==.01
    assert not plan['coupling_study']['fixed_design']['optional_stopping']


def test_full_run_refuses_wrong_plan_digest_without_creating_jobs(tmp_path):
    from pathlib import Path
    from experiments.coupled_cutoff_check import prepare,run
    reference=Path('results/ec2_runs/production_20261119/status_checks/20261123T222312Z/metrics_snapshot/physics/plan.json')
    prepare(reference,tmp_path)
    with pytest.raises(ValueError,match='approved'):
        run(tmp_path,'wrong_digest',10,1,12*1024**3,8*1024**3,True,True)
    assert not (tmp_path/'jobs').exists()


def test_failed_inverse_solver_cannot_silently_publish_an_outcome():
    table=ResourceCDFTable(ContinuousAncilla(.5))
    with pytest.raises(FloatingPointError,match='inverse CDF'):
        invert_outcome_cdf(np.array([[.7],[.3]]),np.array([-10.,10.]),table,np.array([.12345]),max_iterations=1)
