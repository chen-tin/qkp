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
import hashlib
import json
import numpy as np
import qutip as qt
from scipy.linalg import eigh_tridiagonal
from quantum.spectral_circuit import finite_q_basis,real_left
from quantum.measurement_benchmark import qutip_ancilla,qutip_wave_basis,outcome_grid
from quantum.continuous_ancilla import ContinuousAncilla


def data_eigenvalues(cutoff):
    a=qt.destroy(cutoff);q=(a+a.dag())/math.sqrt(2);s=q.data.as_scipy()
    return eigh_tridiagonal(np.real(s.diagonal()),np.real(s.diagonal(1)),eigvals_only=True)


def capacity(nd,na,radius,dy,chunk,grid_chunk=None,grid_rows=None):
    nodes=len(outcome_grid(radius,dy)[0])
    grid_chunk=chunk if grid_chunk is None else grid_chunk
    grid_rows=nd if grid_rows is None else grid_rows
    # Clear ancillary basis caches before/after each transfer. Previous
    # transfers are unmapped while constructing a new one. The estimate
    # includes two real eigensystem arrays and two full complex transfers.
    # Row-tiled mmap construction closes each written view: no full transfer
    # is retained alongside the real ancillary eigensystem/workspace.
    # scipy's selected STEMR path stores one real Na*Na eigenvector output;
    # LAPACK STEMR workspace is linear (18N reals/10N integers). Include
    # 256N bytes for that workspace plus tiled complex transforms/overhead.
    build=8*na*na+256*na+96*na*chunk+128*nodes+128*1024**2
    grid=32*grid_rows*na+128*(grid_rows+na)*grid_chunk+128*nodes+128*1024**2
    return {'estimated_scratch_bytes':int(max(build,grid)),'estimated_build_bytes':int(build),
            'estimated_grid_bytes':int(grid),'outcome_nodes':nodes,'lambda_rows':nd,'resident_grid_rows':grid_rows,'na':na,'chunk':chunk,'grid_chunk':grid_chunk,
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


def write_transfer(path,lambdas,na,sigma,coupling=1.,chunk=128,identity=None):
    """Write all lambda rows without retaining a full transfer in RAM."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.tmp.npy');receipt=path.with_suffix('.progress.json')
    config={'lambda_sha256':hashlib.sha256(lambdas.tobytes()).hexdigest(),'rows':len(lambdas),
            'ancilla_cutoff':na,'sigma':sigma,'coupling':coupling,'chunk':chunk,
            'identity':identity,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    start=0
    if receipt.exists():
        record=json.loads(receipt.read_text())
        if record['config']!=config or not partial.exists():raise ValueError('Partial transfer source/configuration changed')
        start=record['next_row']
    else:
        array=np.lib.format.open_memmap(partial,mode='w+',dtype=complex,shape=(len(lambdas),na))
        array.flush();array._mmap.close();del array
    finite_q_basis.cache_clear();gc.collect()
    values,vectors=finite_q_basis(na)
    phase=1j**np.arange(na)
    coefficients=real_left(vectors.T,phase.conj()*qutip_ancilla(na,sigma),chunk)
    for begin in range(start,len(lambdas),chunk):
        end=min(begin+chunk,len(lambdas))
        block=coefficients[None,:]*np.exp(-1j*coupling*lambdas[begin:end,None]*values[None,:])
        block=real_left(vectors,block.T,chunk).T
        block*=phase[None,:]
        array=np.load(partial,mmap_mode='r+',allow_pickle=False)
        array[begin:end]=block;array.flush();array._mmap.close();del array,block
        temp_receipt=receipt.with_suffix('.tmp')
        temp_receipt.write_text(json.dumps({'config':config,'next_row':end})+'\n');temp_receipt.replace(receipt)
    del vectors,values
    finite_q_basis.cache_clear();gc.collect()
    partial.replace(path)
    receipt.unlink()
    return path


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
    values['joint_norm_error']=np.empty(len(lambdas))
    values['ancilla_boundary_mass']=np.empty(len(lambdas))
    for start in range(0,len(lambdas),chunk):
        stop=min(start+chunk,len(lambdas));rows=current[start:stop]
        values['joint_norm_error'][start:stop]=np.abs((np.abs(rows)**2).sum(1)-1)
        values['ancilla_boundary_mass'][start:stop]=(np.abs(rows[:,-max(4,current.shape[1]//10):])**2).sum(1)
    return metrics_from_curves(values),values


def metrics_from_curves(values):
    norm=float(np.sqrt(values['squared_distance'].max()))
    rough=float(np.sqrt(values['coarse_squared_distance'].max()))
    refnorm=float(np.sqrt(values['reference_mass'].max()))
    candnorm=float(np.sqrt(values['candidate_mass'].max()))
    metrics={'finite_grid_stinespring_operator_norm':norm,
             'finite_grid_channel_trace_norm_upper_bound':(refnorm+candnorm)*norm,
             'grid_norm_sensitivity':abs(norm-rough),
             'max_reference_mass_error':float(np.max(np.abs(values['reference_mass']-1))),
             'max_candidate_mass_error':float(np.max(np.abs(values['candidate_mass']-1))),
             'grid_mass_sensitivity':max(float(np.max(np.abs(values['reference_mass']-values['coarse_reference_mass']))),float(np.max(np.abs(values['candidate_mass']-values['coarse_candidate_mass'])))),
             'max_joint_norm_error':float(np.max(values['joint_norm_error'])),
             'max_ancilla_boundary_mass':float(np.max(values['ancilla_boundary_mass'])),
             'worst_lambda_index':int(np.argmax(values['squared_distance']))}
    if 'refinement_squared_distance' in values:
        metrics['ancilla_refinement_operator_norm']=float(np.sqrt(values['refinement_squared_distance'].max()))
        metrics['ancilla_refinement_grid_sensitivity']=abs(metrics['ancilla_refinement_operator_norm']-float(np.sqrt(values['coarse_refinement_squared_distance'].max())))
    return metrics



def checks_pass(metrics,require_refinement=True):
    checks=(metrics['finite_grid_stinespring_operator_norm']<=2.5e-4 and
            metrics['max_reference_mass_error']<=1e-5 and metrics['max_candidate_mass_error']<=1e-5 and
            metrics['max_joint_norm_error']<=1e-7 and metrics['max_ancilla_boundary_mass']<1e-4 and
            metrics['grid_norm_sensitivity']<=1e-6 and metrics['grid_mass_sensitivity']<=1e-5)
    if require_refinement:
        checks=checks and metrics.get('ancilla_refinement_operator_norm',np.inf)<=2.5e-4 and metrics.get('ancilla_refinement_grid_sensitivity',np.inf)<=1e-6
    return bool(checks)
