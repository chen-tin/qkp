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
