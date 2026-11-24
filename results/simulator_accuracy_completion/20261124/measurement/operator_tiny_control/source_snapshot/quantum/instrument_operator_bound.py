"""Uniform finite-grid instrument comparison in the finite q eigensystem.

For diagonal Kraus functions m(y,lambda), the squared Stinespring difference
is diagonal with entries sum_y w_y |m_ref-m_local|^2. Its largest diagonal
entry is the squared operator norm on the complete finite data space. This
identity is exact for the declared grid; continuum, window and ancillary
convergence remain separate empirical checks. It is not a physical-cutoff or
hardware certificate.
"""
import math
from pathlib import Path
import gc
import numpy as np
import qutip as qt
from scipy.linalg import eigh_tridiagonal
from quantum.spectral_circuit import finite_q_basis,real_left
from quantum.measurement_benchmark import qutip_ancilla,qutip_wave_basis,outcome_grid
from quantum.continuous_ancilla import ContinuousAncilla


def data_eigenvalues(cutoff):
    a=qt.destroy(cutoff);q=(a+a.dag())/math.sqrt(2);s=q.data.as_scipy()
    return eigh_tridiagonal(np.real(s.diagonal()),np.real(s.diagonal(1)),eigvals_only=True)


def capacity(nd,na,radius,dy,chunk):
    nodes=len(outcome_grid(radius,dy)[0])
    # Clear ancillary basis caches before/after each transfer. Previous
    # transfers are unmapped while constructing a new one. The estimate
    # includes two real eigensystem arrays and two full complex transfers.
    build=16*na*na+32*nd*na+64*(nd+na)*chunk+128*nodes+64*1024**2
    grid=32*nd*na+128*(nd+na)*chunk+128*nodes+64*1024**2
    return {'estimated_scratch_bytes':int(max(build,grid)),'estimated_build_bytes':int(build),
            'estimated_grid_bytes':int(grid),'outcome_nodes':nodes,'nd':nd,'na':na,'chunk':chunk,
            'transfer_file_bytes':int(16*nd*na+128),
            'scope':'Conservative live-array/solver-free work estimate; RSS guard still required'}


def transfer(lambdas,na,sigma,coupling=1.,chunk=128):
    """Independent finite-ancilla exp(-i coupling lambda p) for ALL lambda."""
    finite_q_basis.cache_clear();gc.collect()
    values,vectors=finite_q_basis(na)
    phase=1j**np.arange(na)
    ancilla=qutip_ancilla(na,sigma)
    coefficients=real_left(vectors.T,phase.conj()*ancilla,chunk)
    result=np.empty((len(lambdas),na),complex)
    for start in range(0,len(lambdas),chunk):
        selected=slice(start,min(start+chunk,len(lambdas)))
        result[selected]=coefficients[None,:]*np.exp(-1j*coupling*lambdas[selected,None]*values[None,:])
    result=real_left(vectors,result.T,chunk).T
    result*=phase[None,:]
    del vectors,values
    finite_q_basis.cache_clear();gc.collect()
    return result


def finite_grid_distance(reference,candidate,weights):
    """Exact diagonal identity for an explicit finite-grid Kraus pair."""
    if reference.shape!=candidate.shape or reference.ndim!=2 or weights.shape!=(reference.shape[1],):
        raise ValueError('Matching (lambda,outcome) matrices and outcome weights required')
    squared=(np.abs(reference-candidate)**2)@weights
    return squared,float(np.sqrt(np.max(squared)))


def reduce_transfer(current,lambdas,sigma,radius,dy,coupling=1.,chunk=128,previous=None):
    """Stream outcome columns; retain only per-lambda reductions."""
    y,w,coarse=outcome_grid(radius,dy);resource=ContinuousAncilla(sigma)
    names=['squared_distance','coarse_squared_distance','reference_mass','candidate_mass',
           'coarse_reference_mass','coarse_candidate_mass']
    if previous is not None:names+=['refinement_squared_distance','coarse_refinement_squared_distance']
    values={name:np.zeros(len(lambdas)) for name in names}
    for start in range(0,len(y),chunk):
        sl=slice(start,min(start+chunk,len(y)))
        ref=current@qutip_wave_basis(current.shape[1],y[sl])
        candidate=resource.amplitude(y[None,sl]-coupling*lambdas[:,None])
        difference=np.abs(ref-candidate)**2
        pr,pc=np.abs(ref)**2,np.abs(candidate)**2
        for prefix,weight in [('',w[sl]),('coarse_',coarse[sl])]:
            values[prefix+'squared_distance']+=difference@weight
            values[prefix+'reference_mass']+=pr@weight
            values[prefix+'candidate_mass']+=pc@weight
        if previous is not None:
            older=previous@qutip_wave_basis(previous.shape[1],y[sl])
            delta=np.abs(ref-older)**2
            values['refinement_squared_distance']+=delta@w[sl]
            values['coarse_refinement_squared_distance']+=delta@coarse[sl]
    norm=float(np.sqrt(values['squared_distance'].max()))
    rough=float(np.sqrt(values['coarse_squared_distance'].max()))
    refnorm=float(np.sqrt(values['reference_mass'].max()))
    candnorm=float(np.sqrt(values['candidate_mass'].max()))
    max_edge=0.;max_norm_error=0.
    for start in range(0,len(lambdas),chunk):
        rows=current[start:start+chunk]
        max_norm_error=max(max_norm_error,float(np.max(np.abs((np.abs(rows)**2).sum(1)-1))))
        max_edge=max(max_edge,float(np.max((np.abs(rows[:,-max(4,current.shape[1]//10):])**2).sum(1))))
    metrics={'finite_grid_stinespring_operator_norm':norm,
             'finite_grid_channel_trace_norm_upper_bound':(refnorm+candnorm)*norm,
             'grid_norm_sensitivity':abs(norm-rough),
             'max_reference_mass_error':float(np.max(np.abs(values['reference_mass']-1))),
             'max_candidate_mass_error':float(np.max(np.abs(values['candidate_mass']-1))),
             'grid_mass_sensitivity':max(float(np.max(np.abs(values['reference_mass']-values['coarse_reference_mass']))),float(np.max(np.abs(values['candidate_mass']-values['coarse_candidate_mass'])))),
             'max_joint_norm_error':max_norm_error,'max_ancilla_boundary_mass':max_edge,
             'worst_lambda_index':int(np.argmax(values['squared_distance']))}
    if previous is not None:
        metrics['ancilla_refinement_operator_norm']=float(np.sqrt(values['refinement_squared_distance'].max()))
        metrics['ancilla_refinement_grid_sensitivity']=abs(metrics['ancilla_refinement_operator_norm']-float(np.sqrt(values['coarse_refinement_squared_distance'].max())))
    return metrics,values


def checks_pass(metrics,require_refinement=True):
    checks=(metrics['finite_grid_stinespring_operator_norm']<=2.5e-4 and
            metrics['max_reference_mass_error']<=1e-5 and metrics['max_candidate_mass_error']<=1e-5 and
            metrics['max_joint_norm_error']<=1e-7 and metrics['max_ancilla_boundary_mass']<1e-4 and
            metrics['grid_norm_sensitivity']<=1e-6 and metrics['grid_mass_sensitivity']<=1e-5)
    if require_refinement:
        checks=checks and metrics.get('ancilla_refinement_operator_norm',np.inf)<=2.5e-4 and metrics.get('ancilla_refinement_grid_sensitivity',np.inf)<=1e-6
    return bool(checks)
