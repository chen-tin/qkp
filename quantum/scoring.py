"""Memory-bounded modular gauge trace, independently integrated per state.

Uses no precomputed logical-operator matrices or oscillator diagonalization.
Peak memory scales as cutoff * quadrature_chunk, not cutoff squared.
Pantaleoni et al., PRL 125, 040501 (2020), Eqs. (1), (3), (4).
"""
import math
import numpy as np
from quantum.qutip_reference import L, hermite_functions, CARDINAL


def modular_trace(psi, bins=384, chunk=128, backend="local"):
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 2 or not np.isfinite(psi).all() or bins < 2 or chunk < 1:
        raise ValueError("Expected finite (cutoff, shots) kets and positive quadrature sizes")
    if not np.allclose((np.abs(psi)**2).sum(0), 1, atol=1e-7, rtol=0):
        raise ValueError("Scoring requires normalized kets")
    if backend == "qutip":
        from quantum.measurement_benchmark import qutip_wave_basis
        basis = qutip_wave_basis
        evaluate = lambda x: basis(len(psi), x)
    elif backend == "local":
        evaluate = lambda x: hermite_functions(x, len(psi))
    else:
        raise ValueError("Unknown quadrature backend")
    mmax = math.ceil((math.sqrt(2*len(psi))+8)/(2*L))
    m = np.arange(-mmax, mmax+1)
    u = -L/2+(np.arange(bins)+.5)*L/bins
    x0 = (2*m[:, None]*L+u).ravel()
    diagonal = np.zeros((2, psi.shape[1]))
    overlap = np.zeros(psi.shape[1], complex)
    for start in range(0, len(x0), chunk):
        x = x0[start:start+chunk]
        w0 = evaluate(x).T@psi
        w1 = evaluate(x+L).T@psi
        diagonal[0] += (np.abs(w0)**2).sum(0)*L/bins
        diagonal[1] += (np.abs(w1)**2).sum(0)*L/bins
        overlap += (w0.conj()*w1).sum(0)*L/bins
    bloch = np.array([2*overlap.real, 2*overlap.imag, diagonal[0]-diagonal[1]])
    trace = diagonal.sum(0)
    return {"bloch": bloch, "trace": trace,
            "minimum_eigenvalue": (trace-np.linalg.norm(bloch, axis=0))/2}


def logical_fidelity(psi, state_ids, bins=384, chunk=128, backend="local"):
    state_ids = np.asarray(state_ids, dtype=int)
    if state_ids.shape != (psi.shape[1],) or np.any((state_ids < 0) | (state_ids > 5)):
        raise ValueError("Each snapshot needs its true cardinal input label")
    result = modular_trace(psi, bins, chunk, backend)
    axes = np.array([CARDINAL[k][2] for k in state_ids])
    signs = np.array([CARDINAL[k][3] for k in state_ids])
    result["fidelity"] = (result["trace"]+signs*result["bloch"][axes, np.arange(len(state_ids))])/2
    return result


def logical_fidelity_many(states, labels, bins=384, chunk=128, backend="local",
                          max_batch_bytes=256*1024**2):
    """Score banks together, sharing quadrature evaluation within a memory cap.

    The nodes, integration order per column and logical convention are those
    of logical_fidelity. The cap bounds each concatenated state matrix; input
    banks and the streamed basis/output arrays remain additional live storage.
    No cutoff-squared operator or full quadrature basis is cached.
    """
    states=list(states);labels=list(labels)
    if len(states)!=len(labels) or not states:
        raise ValueError("Nonempty matching state/label banks required")
    if not isinstance(max_batch_bytes,int) or max_batch_bytes<1:
        raise ValueError("Positive integer batch-memory budget required")
    arrays=[];ids=[]
    for bank,state_ids in zip(states,labels):
        psi=np.asarray(bank,dtype=complex)
        state_ids=np.asarray(state_ids,dtype=int)
        if psi.ndim!=2 or psi.shape[1]<1:
            raise ValueError("Nonempty two-dimensional state banks required")
        if state_ids.shape!=(psi.shape[1],) or np.any((state_ids<0)|(state_ids>5)):
            raise ValueError("Each snapshot needs its true cardinal input label")
        if psi.shape[0]*psi.dtype.itemsize>max_batch_bytes:
            raise ValueError("Batch budget cannot hold even one full-cutoff state")
        arrays.append(psi);ids.append(state_ids)
    if len({a.shape[0] for a in arrays})!=1:
        raise ValueError("Shared-basis scoring requires one common full cutoff")
    n=arrays[0].shape[0];columns=max_batch_bytes//(n*np.dtype(complex).itemsize)
    output=[{key:[] for key in ("bloch","trace","minimum_eigenvalue","fidelity")} for _ in arrays]
    pieces=[];count=0
    def flush():
        if not pieces:return
        combined=np.concatenate([arrays[i][:,lo:hi] for i,lo,hi in pieces],axis=1)
        joined=np.concatenate([ids[i][lo:hi] for i,lo,hi in pieces])
        score=logical_fidelity(combined,joined,bins,chunk,backend)
        offset=0
        for i,lo,hi in pieces:
            width=hi-lo
            for key,value in score.items():
                output[i][key].append(value[:,offset:offset+width] if key=="bloch" else value[offset:offset+width])
            offset+=width
        pieces.clear()
    for i,bank in enumerate(arrays):
        start=0
        while start<bank.shape[1]:
            take=min(columns-count,bank.shape[1]-start)
            pieces.append((i,start,start+take));count+=take;start+=take
            if count==columns:
                flush();count=0
    flush()
    return [{key:np.concatenate(value,axis=1 if key=="bloch" else 0)
             for key,value in score.items()} for score in output]
