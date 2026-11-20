"""Finite-energy Steane syndrome extraction in a truncated oscillator.

Conventions: [q,p]=i; L=sqrt(pi); D((u+iv)/sqrt(2)).
An ideal SUM with a fresh finite-energy |+> ancilla and ideal q homodyne
has data Kraus operator M_y=phi_plus(y-q). The p extraction is its Fourier
conjugate. Sampling an eigenvalue below is an auxiliary mixture sampler for
the POVM distribution, NOT an additional projective measurement on the data.
The conditional ket is always multiplied by the full Kraus function.
"""
from dataclasses import dataclass
import math

import numpy as np
import qutip as qt
from scipy.linalg import eigh

L = math.sqrt(math.pi)


def hermite_functions(x, n):
    x = np.asarray(x)
    out = np.empty((n, x.size))
    # A plain Gaussian seed underflows beyond |x|~38 before high-order
    # Hermite functions reach their turning point. Maintain a shared column
    # log scale and rescale all accumulated orders together when necessary.
    scale = np.maximum(0.,x*x/2-200.)
    out[0] = np.pi**(-.25)*np.exp(-x*x/2+scale)
    if n > 1:
        out[1] = math.sqrt(2)*x*out[0]
    for j in range(1,n-1):
        out[j+1] = math.sqrt(2/(j+1))*x*out[j]-math.sqrt(j/(j+1))*out[j-1]
        if j%8==0:
            large=np.max(np.abs(out[j:j+2]),axis=0)>1e100
            if np.any(large):
                out[:j+2,large]*=1e-100
                scale[large]-=100*math.log(10)
    out *= np.exp(-scale)[None,:]
    return out


def raw_code(n, delta):
    kmax = math.ceil((math.sqrt(2*n)+10)/(2*L))+1
    k = np.arange(-kmax,kmax+1)
    c = np.column_stack([hermite_functions((2*k+mu)*L,n).sum(axis=1) for mu in (0,1)])
    c *= np.exp(-delta*delta*np.arange(n))[:,None]
    return c


def gkp_code(n, delta):
    """Symmetrically orthonormalized exp(-delta^2 n)|mu_ideal>, mu=0,1."""
    c = raw_code(n,delta)
    c /= np.linalg.norm(c,axis=0)
    w,u = eigh(c.T@c)
    return c@((u/np.sqrt(w))@u.T)


class Ancilla:
    """Exact Mehler-kernel wavefunction of exp(-t n)|+_ideal>.

    t=atanh(2 sigma_m^2) gives isolated probability peaks of std sigma_m.
    This matches local peak width, not the full effective measurement channel.
    Piecewise constant quadrature bins make sampling/update conventions agree.
    """
    def __init__(self,sigma_m,dx=.003):
        if not 0 < 2*sigma_m*sigma_m < 1:
            raise ValueError('Ancilla requires 0 < 2 sigma_m^2 < 1')
        self.t = np.arctanh(2*sigma_m*sigma_m)
        self.delta = math.sqrt(self.t)
        self.dx = dx
        radius = math.ceil(8/self.delta/dx)*dx
        self.x = np.arange(-radius,radius+dx/2,dx)
        self.x0 = self.x[0]
        jmax = math.ceil((radius+8*math.sqrt(math.tanh(self.t)))/L)+1
        centres = np.arange(-jmax,jmax+1)*L
        amp = np.zeros(len(self.x))
        for y in centres:
            amp += np.exp(-.5*math.tanh(self.t)*y*y
                          -(self.x-y/math.cosh(self.t))**2/(2*math.tanh(self.t)))
        amp /= math.sqrt(np.dot(amp,amp)*dx)
        self.amp = amp
        self.cdf = np.cumsum(amp*amp*dx)
        self.cdf[-1] = 1.

    def sample(self,u,jitter):
        indices = np.searchsorted(self.cdf,u,side='right').clip(0,len(self.x)-1)
        return self.x[indices]+(jitter-.5)*self.dx

    def amplitude(self,x):
        index = np.floor((x-self.x0)/self.dx+.5).astype(np.int64)
        valid = (index>=0)&(index<len(self.amp))
        return np.where(valid,self.amp[index.clip(0,len(self.amp)-1)],0.)


@dataclass
class Oscillator:
    n: int
    delta: float = .30
    logical_bins: int = 96

    def __post_init__(self):
        a = qt.destroy(self.n)
        self.q = (a+a.dag())/math.sqrt(2)
        self.p = -1j*(a-a.dag())/math.sqrt(2)
        self.qvals,self.qvectors = eigh(self.q.full())
        # Exact finite-basis Fourier conjugacy avoids a second, complex O(N^3)
        # diagonalization and fixes consistent q/p eigensystem conventions.
        self.pvals=self.qvals.copy()
        self.pvectors=(1j**np.arange(self.n))[:,None]*self.qvectors
        self.code = gkp_code(self.n,self.delta).astype(complex)
        self.number = np.arange(self.n)
        self.logical_ops = self._logical_operators()

    def _logical_operators(self):
        # Modular-subsystem extraction: pair q=(2m+l)L+u at the same m,u.
        mmax = math.ceil((math.sqrt(2*self.n)+8)/(2*L))
        m = np.arange(-mmax,mmax+1)
        du = L/self.logical_bins
        u = -L/2+(np.arange(self.logical_bins)+.5)*du
        phi0 = hermite_functions((2*m[:,None]*L+u).ravel(),self.n)
        phi1 = hermite_functions(((2*m[:,None]+1)*L+u).ravel(),self.n)
        g0,g1 = du*(phi0@phi0.T),du*(phi1@phi1.T)
        h = du*(phi0@phi1.T)
        # Frobenius norm conservatively bounds the operator norm.
        self.extraction_trace_error = float(np.linalg.norm(g0+g1-np.eye(self.n)))
        if self.extraction_trace_error > 1e-7:
            raise ValueError(f'Logical extraction quadrature not converged: {self.extraction_trace_error}')
        return [h+h.T,1j*(h.T-h),g0-g1]

    def displacement(self,psi,u,v):
        # Ordered q then p shifts differ from Weyl D only by a global phase
        # in the untruncated limit; truncation is explicitly convergence-tested.
        psi = self.pvectors@(np.exp(-1j*self.pvals[:,None]*u)* (self.pvectors.conj().T@psi))
        return self.qvectors@(np.exp(1j*self.qvals[:,None]*v)*(self.qvectors.conj().T@psi))

    def measure(self,psi,ancilla,uniforms,axis):
        vals,vec = (self.qvals,self.qvectors) if axis=='q' else (self.pvals,self.pvectors)
        amp = vec.conj().T@psi
        probability = np.abs(amp)**2
        cdf = np.cumsum(probability,axis=0)
        cdf[-1] = 1.
        sampled_index = (cdf < uniforms[0][None,:]).sum(axis=0)
        y = vals[sampled_index]+ancilla.sample(uniforms[1],uniforms[2])
        amp *= ancilla.amplitude(y[None,:]-vals[:,None])
        norm = np.linalg.norm(amp,axis=0)
        if not np.all(np.isfinite(norm)) or np.any(norm<=0):
            raise FloatingPointError('Invalid measurement likelihood')
        amp /= norm
        return vec@amp,y

    def correct(self,psi,correction,axis):
        # q displacement by -correction = exp(+i correction p), and vice versa.
        vals,vec,sgn = (self.pvals,self.pvectors,1) if axis=='q' else (self.qvals,self.qvectors,-1)
        return vec@(np.exp(sgn*1j*vals[:,None]*correction)*(vec.conj().T@psi))

    def observables(self,psi):
        probability = np.abs(psi)**2
        bloch = np.array([np.einsum('ib,ib->b',psi.conj(),op@psi).real for op in self.logical_ops])
        return {'bloch':bloch,'energy':self.number@probability,
                'edge':probability[-max(4,self.n//10):].sum(axis=0),
                'norm_error':np.abs(probability.sum(axis=0)-1)}


CARDINAL = [('Z+',np.array([1,0],complex),2,1),('Z-',np.array([0,1],complex),2,-1),
            ('X+',np.array([1,1],complex)/math.sqrt(2),0,1),
            ('X-',np.array([1,-1],complex)/math.sqrt(2),0,-1),
            ('Y+',np.array([1,1j],complex)/math.sqrt(2),1,1),
            ('Y-',np.array([1,-1j],complex)/math.sqrt(2),1,-1)]
