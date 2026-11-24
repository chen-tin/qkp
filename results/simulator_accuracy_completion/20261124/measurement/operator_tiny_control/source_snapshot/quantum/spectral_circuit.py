"""Independent exact finite-operator two-mode SUM via spectral action.

The Hamiltonian is the same QuTiP-built q_data tensor p_ancilla (or its
Fourier-conjugate p extraction) used by the ODE reference. Separability lets
us apply exp(-i H) exactly in its product eigensystem, with no local Kraus
resource/update and no ODE work vectors. Tiled real/imag transforms avoid
casting a complete real eigenbasis to a complex matrix.
"""
import math
from functools import lru_cache
import numpy as np
import qutip as qt
from scipy.linalg import eigh_tridiagonal
from quantum.measurement_benchmark import qutip_ancilla


@lru_cache(maxsize=2)
def finite_q_basis(cutoff):
    if not isinstance(cutoff,(int,np.integer)) or cutoff<2:
        raise ValueError("Finite circuit cutoff must be an integer >= 2")
    a=qt.destroy(int(cutoff))
    q=(a+a.dag())/math.sqrt(2)
    sparse=q.data.as_scipy()
    diagonal=np.real(sparse.diagonal())
    off_diagonal=np.real(sparse.diagonal(1))
    values,vectors=eigh_tridiagonal(diagonal,off_diagonal,lapack_driver="stemr")
    values.flags.writeable=False;vectors.flags.writeable=False
    return values,vectors


def real_left(vectors, matrix, chunk=64):
    """Real eigenbasis times a complex array, without a full complex basis."""
    matrix=np.asarray(matrix,complex)
    vector=matrix.ndim==1
    if vector:matrix=matrix[:,None]
    result=np.empty((vectors.shape[0],matrix.shape[1]),complex)
    for start in range(0,matrix.shape[1],chunk):
        selection=slice(start,min(start+chunk,matrix.shape[1]))
        result[:,selection].real=vectors@matrix[:,selection].real
        result[:,selection].imag=vectors@matrix[:,selection].imag
    return result[:,0] if vector else result


class SpectralCircuit:
    """Exact exp[-i coupling q⊗p] in the declared finite operators."""
    def __init__(self,nd,na,sigma,*,coupling=1.,chunk=64):
        if not np.isfinite(coupling) or coupling<=0 or chunk<1:
            raise ValueError("Positive finite coupling and chunk required")
        self.nd,self.na,self.coupling,self.chunk=nd,na,coupling,chunk
        self.data_values,self.data_vectors=finite_q_basis(nd)
        self.ancilla_values,self.ancilla_vectors=finite_q_basis(na)
        self.ancilla=qutip_ancilla(na,sigma).astype(complex)
        self.data_phase=1j**np.arange(nd)
        self.ancilla_phase=1j**np.arange(na)

    def evolve(self,data,axis):
        data=np.asarray(data,complex)
        if (axis not in ("q","p") or data.shape!=(self.nd,) or
                not np.isfinite(data).all() or not np.isclose(np.vdot(data,data).real,1,atol=1e-8,rtol=0)):
            raise ValueError("Normalized full-dimensional data and q/p axis required")
        if axis=="q":
            d=real_left(self.data_vectors.T,data,self.chunk)
            a=real_left(self.ancilla_vectors.T,self.ancilla_phase.conj()*self.ancilla,self.chunk)
            sign=-1
        else:
            d=real_left(self.data_vectors.T,self.data_phase.conj()*data,self.chunk)
            a=real_left(self.ancilla_vectors.T,self.ancilla_phase*self.ancilla,self.chunk)
            sign=1
        joint=np.empty((self.nd,self.na),complex)
        for start in range(0,self.nd,self.chunk):
            selection=slice(start,min(start+self.chunk,self.nd))
            phases=np.exp(sign*1j*self.coupling*self.data_values[selection,None]*self.ancilla_values[None,:])
            joint[selection]=d[selection,None]*a[None,:]*phases
        joint=real_left(self.ancilla_vectors,joint.T,self.chunk).T
        joint=real_left(self.data_vectors,joint,self.chunk)
        if axis=="q":joint*=self.ancilla_phase[None,:]
        else:joint*=self.data_phase[:,None]
        return joint
