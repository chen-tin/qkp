"""Exact trajectory unravelings of single-mode pure loss and dephasing.

These channels are independently tested before use in the GKP reference.
Input columns are normalized oscillator kets. Environmental outcomes must
remain hidden from a syndrome-only decoder.
"""
import math

import numpy as np
from scipy.special import gammaln


def _check(psi):
    psi=np.asarray(psi,dtype=complex)
    if psi.ndim!=2 or not np.isfinite(psi).all():
        raise ValueError('Expected finite (Fock levels, trajectories) kets')
    if not np.allclose(np.sum(np.abs(psi)**2,axis=0),1,atol=1e-9,rtol=0):
        raise ValueError('Input kets must be normalized')
    return psi


def pure_loss(psi,transmissivity,rng):
    """Sample K_k|psi>, where K_k|n> = sqrt(Binomial(n,k))
    (1-eta)^(k/2) eta^((n-k)/2)|n-k> and eta is transmissivity.

    A sampled photon number is an auxiliary mixture sampler for P(k), not
    an additional projection: the output preserves the complete K_k ket.
    The returned loss count is an environmental diagnostic, not readout.
    """
    psi=_check(psi)
    eta=float(transmissivity)
    if not np.isfinite(eta) or not 0<=eta<=1:
        raise ValueError('Transmissivity must be in [0,1]')
    n,b=psi.shape
    if eta==1:return psi.copy(),np.zeros(b,dtype=int)
    probability=np.abs(psi)**2
    cdf=np.cumsum(probability,axis=0);cdf[-1]=1
    aux=(cdf<rng.random(b)[None,:]).sum(axis=0)
    lost=rng.binomial(aux,1-eta)
    if eta==0:
        out=np.zeros_like(psi);out[0]=1
        return out,lost
    output_n=np.arange(n)[:,None]
    input_n=output_n+lost[None,:]
    valid=input_n<n
    logfactor=.5*(gammaln(input_n+1)-gammaln(output_n+1)-gammaln(lost[None,:]+1)
                   +output_n*math.log(eta)+lost[None,:]*math.log1p(-eta))
    out=np.where(valid,psi[np.minimum(input_n,n-1),np.arange(b)[None,:]]*np.exp(logfactor),0)
    norms=np.linalg.norm(out,axis=0)
    if np.any(norms<=0) or not np.isfinite(norms).all():
        raise FloatingPointError('Invalid loss-trajectory normalization')
    return out/norms,lost


def dephasing(psi,strength,rng):
    """Channel exp(strength D[n]) via random rotations exp(-i theta n).

    theta~Normal(0,strength); strength means gamma_phi*time with collapse
    operator sqrt(gamma_phi)*n. Coherence nm decays by
    exp[-strength*(n-m)^2/2].
    """
    psi=_check(psi);strength=float(strength)
    if not np.isfinite(strength) or strength<0:
        raise ValueError('Dephasing strength must be finite and nonnegative')
    angles=rng.normal(0,math.sqrt(strength),psi.shape[1])
    return psi*np.exp(-1j*np.arange(psi.shape[0])[:,None]*angles[None,:])
