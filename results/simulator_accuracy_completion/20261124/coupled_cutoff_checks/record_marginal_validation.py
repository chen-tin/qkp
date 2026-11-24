from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,re,sys,platform
import numpy as np
from quantum.continuous_ancilla import ContinuousAncilla
from quantum.coupled_measurement import ResourceCDFTable,invert_outcome_cdf
out=Path(__file__).resolve().parent
cases=[]
for sigma in [.5,.05*math.sqrt(math.pi)]:
    resource=ContinuousAncilla(sigma);table=ResourceCDFTable(resource)
    rng=np.random.default_rng(42);values=np.linspace(-5,5,24)
    u=np.r_[1e-12,1e-9,.001,.03,.2,.5,.8,.999,1-1e-9,1-1e-12]
    probability=rng.uniform(size=(24,len(u)));probability/=probability.sum(0)
    y,detail=invert_outcome_cdf(probability,values,table,u,coupling=1.01)
    reference=(probability*resource.cdf(y[None,:]-1.01*values[:,None])).sum(0)/probability.sum(0)
    raw=out/f'marginal_sigma_{sigma:.9f}.npz'
    np.savez_compressed(raw,probabilities=probability,eigenvalues=values,uniforms=u,outcomes=y,reference_source_cdf=reference)
    cases.append({'resource_sigma_absolute_q':sigma,'quantiles_per_case':len(u),'synthetic_probability_seed':42,
                  'coupling':1.01,'metric':'maximum absolute source-CDF quantile residual; lower is better',
                  'max_reference_resource_cdf_residual':float(np.max(np.abs(reference-u))),
                  'inverse_diagnostics':detail,'table':table.diagnostics,'raw_file':raw.name,
                  'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()})
files=['quantum/coupled_measurement.py','experiments/coupled_cutoff_check.py','tests/test_coupled_measurement.py',
       'quantum/reference_runner.py','quantum/qutip_reference.py','quantum/continuous_ancilla.py']
log=(out/'unit_tests.log').read_text();match=re.search(r'(\d+) passed in ([0-9.]+)s',log)
report={'recorded_utc':datetime.now(timezone.utc).isoformat(),'classification':'Deterministic sampler-marginal and source-contract validation; no scientific trajectory benchmark',
        'tests':{'passed':int(match.group(1)),'seconds':float(match.group(2)),'log':'unit_tests.log',
                 'command':'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_coupled_measurement.py'},
        'cases':cases,'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'device':'CPU'},
        'mathematics':{'outcome_cdf':'G(y)=sum_i p_i F_resource(y-g*lambda_i), p_i=|a_i|^2/sum|a_i|^2',
                       'conditional_update':'Existing full Kraus amplitude phi(y-g*lambda_i) times every amplitude; normalization positive; no extra eigenstate projection.',
                       'gaussian_derivative_bound':'Supremum absolute fourth derivative <=(3sqrt(3)exp(-3/2)+3exp(-1/2))/sqrt(2pi)/sigma^4 <1.19/sigma^4 for the normalized common-width Gaussian mixture.',
                       'hermite_remainder':'M4*h^4/384 for exact endpoint CDF/derivatives.',
                       'slope_limiting_bound':'Endpoint derivatives capped in[0,3*adjacent_secant]; add (4h/27)*max_i(|delta_d_i|+|delta_d_(i+1)|) to account for limiter.',
                       'endpoint_projection_bound':'Add max|projected_monotone_CDF-original_CDF|; Hermite value basis is nonnegative and sums to1.',
                       'tail_bound':'Gaussian mixture means bounded; lo=minmu-zsigma,hi=maxmu+zsigma, z=-Phi^-1(tail/2), each omitted tail<=tail/2.',
                       'float_qualification':'5e-12 table allowance and8gamma_N+32epsilon conditional sum/normalization allowance are explicit conservative engineering budgets, not a formal special-function/BLAS roundoff certificate.'},
        'scope':'Both source-resource widths, deterministic probability banks/quantiles, q/p, gain1/1.01, positive tails, context restore and nonconvergence rejection. No all-state channel bound or production cutoff-equivalence result follows. No MonteCarloCI for deterministic errors.',
        'study':'main_128_static_planned/plan.json is Planned/unlaunched pending root approval and pilot; previous main_128_planned is preserved but superseded by static-resource restriction.',
        'production_source_changes':'None: new modules/new tests only; legacy4-job workers unaffected.'}
(out/'marginal_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'tests':report['tests'],'reference_cdf_residuals':[x['max_reference_resource_cdf_residual'] for x in cases]},indent=2))
