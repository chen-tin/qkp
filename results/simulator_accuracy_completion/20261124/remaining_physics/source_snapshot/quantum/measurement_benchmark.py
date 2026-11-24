"""Independent two-mode QuTiP measurement control and local diagnostic arms.

This module does not change quantum.qutip_reference.Ancilla or its sampler.
The QuTiP control never calls the local Kraus updater or Hermite recurrence.
Common input preparation, correction and scoring deliberately isolate the
conditional measurement instrument rather than decoder training.
"""
import math
import time

import numpy as np
import qutip as qt
from qutip._distributions import psi_n_single_fock_multiple_position_complex
from scipy.special import gammaln

from quantum.qutip_reference import L
from simulator.sim import wrap


def outcome_grid(radius, dy):
    """Nested trapezoid rules split at nearest-lattice feedback jumps.

    Boundary endpoints represent one-sided limits. Tiny inward shifts ensure
    floating point wrap chooses that side. Coarse weights use every other
    point within EACH cell, not globally across concatenated cells.
    """
    if radius <= 0 or dy <= 0:
        raise ValueError("Positive radius and spacing required")
    k = np.arange(-math.ceil(radius/L)-2, math.ceil(radius/L)+3)
    interior = (k+.5)*L
    bounds = np.r_[-radius, interior[(interior > -radius) & (interior < radius)], radius]
    yy, ww, cc = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        segments = max(2, math.ceil((b-a)/dy));segments += segments % 2
        x = np.linspace(a, b, segments+1)
        shift = 32*np.spacing(max(abs(a), abs(b), 1.))
        x[0], x[-1] = a+shift, b-shift
        step = (b-a)/segments
        fine = np.full(len(x), step);fine[[0, -1]] *= .5
        coarse = np.zeros(len(x));coarse[::2] = 2*step;coarse[[0, -1]] *= .5
        yy.append(x);ww.append(fine);cc.append(coarse)
    return np.concatenate(yy), np.concatenate(ww), np.concatenate(cc)


class SmoothMehler:
    """Continuously evaluated, analytically normalized resource; no sampler."""

    def __init__(self, sigma):
        if not 0 < 2*sigma*sigma < 1:
            raise ValueError("Resource requires 0 < 2 sigma^2 < 1")
        self.t = float(np.arctanh(2*sigma*sigma))
        self.v = math.tanh(self.t)
        jmax = math.ceil(math.sqrt(2*math.log(1e16)/self.v)/L)+2
        centres = np.arange(-jmax, jmax+1)*L
        self.means = centres/math.cosh(self.t)
        self.weights = np.exp(-.5*self.v*centres**2)
        cross = np.exp(-(self.means[:, None]-self.means[None, :])**2/(4*self.v))
        self.norm = math.sqrt(math.sqrt(math.pi*self.v)*(self.weights@cross@self.weights))

    def amplitude(self, x):
        x = np.asarray(x)
        value = np.zeros(x.shape)
        for mu, weight in zip(self.means, self.weights):
            value += weight*np.exp(-(x-mu)**2/(2*self.v))
        return value/self.norm


REFERENCE_WAVE_BASIS = "scaled_generalized_laguerre_v1"


def qutip_wave_basis(n, positions, chunk_size=256):
    """Stable independent reference basis; legacy API name is preserved.

    QuTiP's Gaussian-seeded kernel underflows at large |x| even when a
    high-order oscillator function is appreciable. Evaluate even/odd rows
    via generalized Laguerre polynomials, dynamically scaled in log space.
    This is separate from the local normalized-Hermite recurrence, and is
    tested against arbitrary-precision Hermite values. QuTiP still supplies
    the independent circuit operators/evolution, not this scalar evaluator.
    """
    x = np.asarray(positions, dtype=np.complex128)
    if (not isinstance(n,(int,np.integer)) or n<1 or chunk_size<1 or x.ndim!=1 or
            not np.isfinite(x).all() or np.any(np.abs(x.imag)>1e-13)):
        raise ValueError("Positive Fock size/chunk and finite real coordinates required")
    x=x.real
    result=np.empty((n,len(x)))
    for start in range(0,len(x),chunk_size):
        stop=min(start+chunk_size,len(x));xx=x[start:stop];z=xx*xx
        for parity,alpha in ((0,-.5),(1,.5)):
            orders=np.arange((n+1-parity)//2)
            log_coeff=(-.25*math.log(math.pi)+(orders+parity*.5)*math.log(2)+
                       gammaln(orders+1)-.5*gammaln(2*orders+parity+1))
            previous=np.zeros_like(xx);current=np.ones_like(xx);scale=np.zeros_like(xx)
            for order,coefficient in enumerate(log_coeff):
                sign=(-1 if order%2 else 1)*np.sign(current)
                if parity:sign*=np.sign(xx)
                exponent=np.log(np.maximum(np.abs(current),np.finfo(float).tiny))+scale-z/2+coefficient
                if parity:exponent+=np.log(np.maximum(np.abs(xx),np.finfo(float).tiny))
                result[2*order+parity,start:stop]=sign*np.exp(exponent)
                following=((2*order+alpha+1-z)*current-(order+alpha)*previous)/(order+1)
                previous,current=current,following
                large=np.maximum(np.abs(previous),np.abs(current))>1e100
                if np.any(large):
                    previous[large]*=1e-100;current[large]*=1e-100
                    scale[large]+=100*math.log(10)
    if not np.isfinite(result).all():
        raise FloatingPointError("Invalid scaled-Laguerre reference oscillator function")
    return result


def qutip_ancilla(n, sigma):
    """exp(-t n)|+ideal> using the independent stable reference evaluator."""
    t = float(np.arctanh(2*sigma*sigma))
    jmax = math.ceil((math.sqrt(2*n)+10)/L)+1
    coefficients = qutip_wave_basis(n, np.arange(-jmax, jmax+1)*L).sum(axis=1)
    coefficients *= np.exp(-t*np.arange(n))
    coefficients /= np.linalg.norm(coefficients)
    return coefficients


class ExplicitQutip:
    def __init__(self, nd, na, sigma, y, *, tight=False, wrong_sign=False, wave_basis=None, coupling=1., axes=("q", "p")):
        if not np.isfinite(coupling) or coupling <= 0:
            raise ValueError("Positive finite SUM coupling required")
        start = time.perf_counter()
        self.nd, self.na = nd, na
        self.ancilla = qutip_ancilla(na, sigma)
        ad, aa = qt.destroy(nd), qt.destroy(na)
        qd, pd = (ad+ad.dag())/math.sqrt(2), -1j*(ad-ad.dag())/math.sqrt(2)
        qa, pa = (aa+aa.dag())/math.sqrt(2), -1j*(aa-aa.dag())/math.sqrt(2)
        sign = (-1 if wrong_sign else 1)*coupling
        if not axes or any(axis not in ("q", "p") for axis in axes):
            raise ValueError("Explicit circuit axes must be q and/or p")
        self.hamiltonians = {}
        if "q" in axes:
            self.hamiltonians["q"] = sign*qt.tensor(qd, pa)
        if "p" in axes:
            self.hamiltonians["p"] = -sign*qt.tensor(pd, qa)
        self.phase = (1j)**np.arange(na)
        self.phi = qutip_wave_basis(na, y) if wave_basis is None else wave_basis
        self.options = {"method": "vern9", "atol": 1e-13 if tight else 1e-11,
                        "rtol": 1e-12 if tight else 1e-10, "nsteps": 100000,
                        "normalize_output": False, "store_states": False,
                        "store_final_state": True, "progress_bar": ""}
        self.setup_seconds = time.perf_counter()-start

    def evolve(self, data, axis):
        """Evolve the independent circuit without allocating an outcome grid."""
        if axis not in self.hamiltonians:
            raise ValueError(axis)
        anc = self.ancilla.astype(complex)
        if axis == "p":
            anc *= self.phase
        initial = qt.tensor(qt.Qobj(np.asarray(data)[:, None]), qt.Qobj(anc[:, None]))
        evolved = qt.sesolve(self.hamiltonians[axis], initial, [0., 1.], options=self.options).final_state
        joint = evolved.full().reshape(self.nd, self.na)
        return joint

    def conditional(self, data, axis):
        """Return unnormalized columns <y|U|data,ancilla>, via full circuit."""
        joint = self.evolve(data, axis)
        # p-space bra <p|n> = (-i)^n h_n(p); q-space h_n is real.
        weighted_joint = joint if axis == "q" else joint*self.phase.conj()[None, :]
        conditional = weighted_joint@self.phi
        norm_error = float(abs(np.vdot(joint, joint).real-1))
        edge = float((np.abs(joint[:, -max(4, self.na//10):])**2).sum())
        return conditional, joint, norm_error, edge


def local_conditional(osc, resource, data, y, axis, coupling=1.):
    vals, vec = (osc.qvals, osc.qvectors) if axis == "q" else (osc.pvals, osc.pvectors)
    return vec@((vec.conj().T@data)[:, None]*resource.amplitude(y[None, :]-coupling*vals[:, None]))


def probability(conditional):
    return np.sum(np.abs(conditional)**2, axis=0)


def comparison(reference, candidate, weights):
    pr, pc = probability(reference), probability(candidate)
    mr, mc = float(pr@weights), float(pc@weights)
    if mr <= 0 or mc <= 0:
        raise FloatingPointError("Zero probability mass")
    overlap = np.abs(np.sum(reference.conj()*candidate, axis=0))**2
    denominator = pr*pc
    infidelity = np.ones_like(pr)
    np.divide(overlap, denominator, out=infidelity, where=denominator > 1e-280)
    infidelity = np.clip(1-infidelity, 0, 1)
    # Candidate zero support at positive-reference nodes is maximally wrong.
    infidelity[(pc == 0) & (pr > 0)] = 1
    relevant = pr*weights/mr >= 1e-8
    tv = .5*float(np.abs(pr/mr-pc/mc)@weights)
    return {"outcome_tv": tv, "weighted_conditional_infidelity": float((pr*infidelity)@weights/mr),
            "worst_conditional_infidelity_relevant": float(infidelity[relevant].max()) if relevant.any() else 0.,
            "relevant_reference_probability_mass": float(pr[relevant]@weights[relevant]/mr),
            "reference_integrated_mass": mr, "candidate_integrated_mass": mc}, infidelity


def conditional_metrics(osc, ket, y, axis, state_axis, state_sign, weights):
    """Common feedback/scoring; these metrics do not independently test either."""
    p = probability(ket)
    mass = float(p@weights)
    corrected = osc.correct(ket, wrap(y), axis)
    power = np.abs(corrected)**2
    energy_density = osc.number@power
    edge_density = power[-max(4, osc.n//10):].sum(0)
    bloch_density = np.einsum("ib,ib->b", corrected.conj(), osc.logical_ops[state_axis]@corrected).real
    fidelity_density = (p+state_sign*bloch_density)/2
    return {"integrated_mass": mass,
            "expected_logical_fidelity": float(fidelity_density@weights/mass),
            "expected_energy": float(energy_density@weights/mass),
            "expected_boundary_mass": float(edge_density@weights/mass)}, {
                "probability_density": p, "energy_density": energy_density,
                "logical_fidelity_density": fidelity_density, "edge_density": edge_density}
