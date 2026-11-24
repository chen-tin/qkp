"""Experimental common-quantile coupling of the existing finite Kraus instrument.

This module is opt-in and changes no production default. It inverts the full
outcome CDF, sum_i |a_i|^2 F_resource(y-g lambda_i), rather than sampling a
latent eigenvalue then a resource outcome. Each marginal is the same ideal
finite-operator instrument; common random numbers across cutoffs differ.
"""
from contextlib import contextmanager
import math
import numpy as np
from scipy.special import ndtr,ndtri


# |(z^3-3z) phi(z)| <= sup |z^3 phi(z)|+3 sup |z phi(z)| <1.19.
GAUSSIAN_CDF_FOURTH_DERIVATIVE_CONSTANT=(3*math.sqrt(3)*math.exp(-1.5)+3*math.exp(-.5))/math.sqrt(2*math.pi)


class ResourceCDFTable:
    """Monotone uniform cubic Hermite interpolation with explicit error terms.

    The classical Hermite remainder is augmented by measured CDF projection
    and slope-limiting terms. The floating budget is an explicit engineering
    allowance, not a proved SciPy special-function ULP bound.
    """
    def __init__(self,resource,max_step=.00025,tail_probability=1e-15,float_budget=5e-12,max_table_bytes=64*1024**2):
        self.means=np.asarray(resource.mixture_means,dtype=float)
        weights=np.asarray(resource.mixture_weights,dtype=float)
        sigma=float(resource.sample_std)
        if (weights.ndim!=1 or self.means.shape!=weights.shape or not len(weights) or
                not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0 or
                not np.isfinite(self.means).all() or not np.isfinite(sigma) or sigma<=0 or
                not np.isfinite(max_step) or max_step<=0 or not 0<tail_probability<1e-6 or
                not np.isfinite(float_budget) or float_budget<=0):
            raise ValueError('Finite normalized Gaussian-mixture resource and positive table budgets required')
        self.weights=weights/weights.sum();self.sigma=sigma
        z=-float(ndtri(tail_probability/2))
        self.left=float(self.means.min()-sigma*z);right=float(self.means.max()+sigma*z)
        intervals=int(math.ceil((right-self.left)/max_step));self.step=(right-self.left)/intervals
        # x,F,p,secants,limits and construction scratch are sequential vectors.
        if (intervals+1)*8*10>max_table_bytes:
            raise ValueError('Resource CDF table construction exceeds explicit memory budget')
        x=self.left+self.step*np.arange(intervals+1)
        cdf=np.zeros(len(x));density=np.zeros(len(x))
        for mu,weight in zip(self.means,self.weights):
            standardized=(x-mu)/sigma
            cdf+=weight*ndtr(standardized)
            density+=weight*np.exp(-.5*standardized**2)/(math.sqrt(2*math.pi)*sigma)
        projected=np.maximum.accumulate(np.clip(cdf,0.,1.))
        projection=float(np.max(np.abs(projected-cdf)))
        secants=np.maximum(np.diff(projected)/self.step,0.)
        caps=np.r_[secants[0],np.minimum(secants[:-1],secants[1:]),secants[-1]]*3
        self.slopes=np.minimum(density,caps);adjust=np.abs(density-self.slopes)
        slope_error=float((4*self.step/27*np.max(adjust[:-1]+adjust[1:])))
        self.cdf=projected;self.right=self.left+self.step*intervals;self.intervals=intervals
        interpolation=GAUSSIAN_CDF_FOURTH_DERIVATIVE_CONSTANT/sigma**4*self.step**4/384
        self.error_terms={'hermite_remainder':float(interpolation),'monotone_slope_adjustment':slope_error,
                          'cdf_projection':projection,'outside_grid_tail':float(tail_probability/2),
                          'declared_floating_budget':float(float_budget)}
        self.error_bound=sum(self.error_terms.values())
        self.diagnostics={'step':self.step,'intervals':intervals,'left':self.left,'right':self.right,
                          'persistent_bytes':self.cdf.nbytes+self.slopes.nbytes,'error_terms':self.error_terms,
                          'declared_uniform_cdf_error_bound':self.error_bound,
                          'bound_scope':'Analytic approximation bounds plus an explicit unproved floating-point allowance.'}

    def cdf_density(self,x):
        x=np.asarray(x,dtype=float)
        if not np.isfinite(x).all():raise ValueError('Finite CDF queries required')
        index=np.floor((np.clip(x,self.left,self.right)-self.left)/self.step).astype(np.int64)
        index=np.clip(index,0,self.intervals-1)
        t=np.clip((x-(self.left+index*self.step))/self.step,0.,1.)
        f0=self.cdf[index];delta=self.cdf[index+1]-f0
        a=self.step*self.slopes[index];b=self.step*self.slopes[index+1]
        c2=3*delta-2*a-b;c3=a+b-2*delta
        value=f0+t*(a+t*(c2+t*c3))
        density=(a+t*(2*c2+3*t*c3))/self.step
        value=np.where(x<=self.left,0.,np.where(x>=self.right,1.,value))
        density=np.where((x<=self.left)|(x>=self.right),0.,np.maximum(density,0.))
        return np.clip(value,0.,1.),density

    def exact_cdf_density(self,x):
        x=np.asarray(x,float);cdf=np.zeros_like(x);density=np.zeros_like(x)
        for mu,weight in zip(self.means,self.weights):
            z=(x-mu)/self.sigma
            cdf+=weight*ndtr(z);density+=weight*np.exp(-.5*z*z)/(math.sqrt(2*math.pi)*self.sigma)
        return cdf,density


def outcome_cdf_density(y,probability,values,table,coupling=1.,node_chunk=256,exact=False):
    y=np.asarray(y,float);probability=np.asarray(probability,float);values=np.asarray(values,float)
    if (probability.ndim!=2 or probability.shape!=(len(values),len(y)) or
            not np.isfinite(probability).all() or np.any(probability<0) or
            not np.isfinite(values).all() or not np.isfinite(coupling) or coupling<=0 or node_chunk<1):
        raise ValueError('Finite positive outcome-mixture parameters required')
    mass=probability.sum(0)
    if np.any(mass<=0):raise ValueError('Positive input probability mass required')
    cdf=np.zeros(len(y));density=np.zeros(len(y))
    evaluate=table.exact_cdf_density if exact else table.cdf_density
    for start in range(0,len(values),node_chunk):
        p=probability[start:start+node_chunk]
        f,d=evaluate(y[None,:]-coupling*values[start:start+node_chunk,None])
        cdf+=np.sum(p*f,axis=0);density+=np.sum(p*d,axis=0)
    return np.clip(cdf/mass,0.,1.),density/mass


def invert_outcome_cdf(probability,values,table,uniforms,coupling=1.,tolerance=1e-10,max_iterations=96,node_chunk=256):
    probability=np.asarray(probability,float);uniforms=np.asarray(uniforms,float);values=np.asarray(values,float)
    if uniforms.ndim!=1 or not np.isfinite(uniforms).all() or np.any((uniforms<=0)|(uniforms>=1)):
        raise ValueError('Strictly interior finite uniform quantiles required')
    if (values.ndim!=1 or not len(values) or probability.shape!=(len(values),len(uniforms)) or
            not np.isfinite(values).all() or not np.isfinite(probability).all() or np.any(probability<0) or
            np.any(probability.sum(0)<=0) or not isinstance(max_iterations,int) or max_iterations<1):
        raise ValueError('Positive finite probability banks and iteration budget required')
    n=len(values);eps=np.finfo(float).eps;gamma=n*eps/(1-n*eps)
    float_sum_budget=8*gamma+32*eps
    total_bound=table.error_bound+float_sum_budget
    if tolerance<=total_bound or not np.isfinite(tolerance):
        raise ValueError('Requested CDF tolerance is below declared approximation/rounding budget')
    target=min(1e-12,.25*(tolerance-total_bound))
    mass=probability.sum(0);p=probability/mass
    center=(p*(coupling*values[:,None])).sum(0)
    variance=(p*(coupling*values[:,None]-center)**2).sum(0)
    resource_center=float(table.weights@table.means)
    variance+=float(table.weights@((table.means-resource_center)**2+table.sigma**2))
    low=np.full(len(uniforms),coupling*values.min()+table.left)
    high=np.full(len(uniforms),coupling*values.max()+table.right)
    y=np.clip(center+resource_center+np.sqrt(variance)*ndtri(uniforms),low,high)
    active=np.ones(len(y),bool)
    for iteration in range(max_iterations):
        cdf,density=outcome_cdf_density(y,probability,values,table,coupling,node_chunk)
        difference=cdf-uniforms
        active=np.abs(difference)>target
        if not active.any():break
        low=np.where(active&(difference<0),y,low);high=np.where(active&(difference>0),y,high)
        proposal=y-np.divide(difference,density,out=np.full_like(y,np.inf),where=density>1e-290)
        use_newton=np.isfinite(proposal)&(proposal>low)&(proposal<high)
        y=np.where(active,np.where(use_newton,proposal,(low+high)/2),y)
    cdf,_=outcome_cdf_density(y,probability,values,table,coupling,node_chunk)
    residual=float(np.max(np.abs(cdf-uniforms)))
    if not np.isfinite(y).all() or residual+total_bound>tolerance:
        raise FloatingPointError('Full outcome inverse CDF did not meet the declared error budget')
    return y,{'iterations':iteration+1,'max_approximate_cdf_residual':residual,
              'conditional_sum_roundoff_budget':float_sum_budget,'table_cdf_error_bound':table.error_bound,
              'max_declared_exact_cdf_residual_bound':residual+total_bound,
              'target_tolerance':tolerance,'normalization_error_before_cdf':float(np.max(np.abs(mass-1)))}


class DirectCDFMeasurement:
    def __init__(self,max_step=.00025,tolerance=1e-10,shot_batch=64,node_chunk=256,
                 tail_probability=1e-15,float_budget=5e-12,max_table_bytes=64*1024**2):
        if shot_batch<1:raise ValueError('Positive shot batch required')
        self.max_step=max_step;self.tolerance=tolerance;self.shot_batch=shot_batch;self.node_chunk=node_chunk
        self.tail_probability=tail_probability;self.float_budget=float_budget;self.max_table_bytes=max_table_bytes
        self.tables={};self.calls=[]
        self.contract={'backend':'direct_full_outcome_inverse_CDF','uniform_index':1,'quantile_stream':'uniforms[1]',
                       'unused_auxiliary_streams':['uniforms[0] eigenvalue draw','uniforms[2] historical jitter'],
                       'coupling_scope':'Common quantiles across cutoffs and explicit QuTiP; also changes coupling to classical replay, whose readout uses uniforms[1]. Marginals unchanged within declared numerical error.',
                       'max_step':max_step,'cdf_tolerance':tolerance,'shot_batch':shot_batch,'node_chunk':node_chunk,
                       'tail_probability':tail_probability,'float_budget':float_budget,'max_table_bytes':max_table_bytes,
                       'floating_bound_status':'Explicit engineering allowance; not a proved special-function ULP guarantee.'}

    def __call__(self,osc,psi,resource,uniforms,axis,coupling):
        if axis not in ('q','p'):raise ValueError('Measurement axis must be q or p')
        values,vectors=(osc.qvals,osc.qvectors) if axis=='q' else (osc.pvals,osc.pvectors)
        amplitude=vectors.conj().T@psi
        if not np.isfinite(amplitude).all():raise FloatingPointError('Nonfinite input amplitude')
        key=(resource.sample_std,resource.mixture_means.tobytes(),resource.mixture_weights.tobytes())
        if key not in self.tables:self.tables[key]=ResourceCDFTable(resource,self.max_step,self.tail_probability,self.float_budget,self.max_table_bytes)
        table=self.tables[key];outcome=np.empty(psi.shape[1]);log_density=np.empty_like(outcome)
        diagnostics=[]
        for start in range(0,psi.shape[1],self.shot_batch):
            stop=min(start+self.shot_batch,psi.shape[1]);amp=amplitude[:,start:stop]
            y,detail=invert_outcome_cdf(np.abs(amp)**2,values,table,uniforms[1,start:stop],coupling,
                                       self.tolerance,node_chunk=self.node_chunk)
            amp*=resource.amplitude(y[None,:]-coupling*values[:,None])
            norm=np.linalg.norm(amp,axis=0)
            if not np.isfinite(norm).all() or np.any(norm<=0):
                raise FloatingPointError('Direct-CDF outcome has invalid conditional Kraus norm')
            amp/=norm;outcome[start:stop]=y;log_density[start:stop]=2*np.log(norm)
            detail['minimum_conditional_norm']=float(norm.min());diagnostics.append(detail)
        bound=max(x['max_declared_exact_cdf_residual_bound'] for x in diagnostics)
        # The new experiment explicitly declares this field as the full-outcome
        # inverse-CDF budget, rather than the unused resource-only inverse.
        resource.max_quantile_residual=max(resource.max_quantile_residual,bound)
        self.calls.append({'axis':axis,'cutoff':len(values),'trajectories':psi.shape[1],
                           'max_declared_exact_cdf_residual_bound':bound,
                           'max_iterations':max(x['iterations'] for x in diagnostics),
                           'minimum_conditional_norm':min(x['minimum_conditional_norm'] for x in diagnostics),
                           'table':table.diagnostics})
        return vectors@amplitude,outcome,log_density


@contextmanager
def direct_cdf_backend(**kwargs):
    """Patch only this Python process and restore the original runner on exit."""
    from quantum import reference_runner
    previous=reference_runner._measure;backend=DirectCDFMeasurement(**kwargs)
    reference_runner._measure=backend
    try:yield backend
    finally:reference_runner._measure=previous
