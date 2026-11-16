"""Decoders for one-bit (binary) GKP syndromes in the closed loop.

Measurement (PhasedBit, a PhaseBit from simulator/closed_loop.py with a phase):

    P(b = 1 | x, nu) = (1 - v cos(2 pi (x + nu) / L - phi)) / 2,  then a readout
    bit flip with probability p_flip.

phi = -pi/2 gives P(b = 1) = (1 + v sin(2 pi (x + nu) / L)) / 2: the bit says on
which side of the nearest lattice point the state sits, which is what the
feedback of small-big-small (sBs) style protocols uses (Campagne-Ibarcq et al.,
Nature 584, 368 (2020); Sivak et al., Nature 616, 50 (2023)). phi = 0 is the
plain cos-type phase-estimation bit of PhaseBit, which carries no sign.

PhasedBits generalises this to m bits per round (each with its own readout
noise and measurement phase, e.g. alternating sin and cos), packed into one
integer per round so simulator/closed_loop.py runs unchanged. Hardware takes
several ancilla bits per correction cycle; with one weak bit per round no
decoder can follow per-round displacements of sigma_q >~ 0.07 L.

Decoders (closed-loop interface: reset(trials), decide(bits) -> real correction):

SBSDecoder (memoryless baseline)
    c_t = delta * (2 b_t - 1): a fixed displacement towards the lattice point
    indicated by the latest bit only.

BinaryBayesDecoder (matched grid filter)
    Exact hidden Markov model of the closed loop, on a grid: state
    (eps_t, x mod 2L) where x is the displacement *after* all corrections so far.
    Per round: AR(1) prediction of eps, shear x <- x + eps, multiply by the bit
    likelihood (nu integrated analytically: v_eff = v exp(-2 pi^2 sigma_m^2 / L^2),
    and the grid quantisation of x adds h^2/12 to that variance), then choose
    the correction. Correction policy (greedy, explicit): pick c on the grid
    (multiples of h = 2L / n_x, in [-L, L)) minimising

        P(round((x - c)/L) is odd) + lam * E[d(x - c)^2] / L^2

    under the current posterior, where d is the distance to the nearest even
    multiple of L. The first term is the probability of a logical error if
    one read out now; the small second term breaks the ties inside a cell by
    centring the state. With lookahead=True (default) the cost is evaluated
    on the predicted next-round state x - c + eps_{t+1} instead of x - c, so
    the correction anticipates correlated drift (E[eps_{t+1}] = rho eps_t);
    with lookahead=False it centres the current state. The belief is then
    shifted by -c (exact, because c is on the grid). Corrections are
    therefore quantised to h (0.037 L at n_x = 96).
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from decoders.bayesian.adaptive import _binned_normal, _eps_grid, _stationary_kernel
from simulator.closed_loop import PhaseBit
from simulator.noise import L

SIN_PHASE = -math.pi / 2


@dataclass
class PhasedBit(PhaseBit):
    """PhaseBit with a measurement phase; visibility/p_flip may be per-trial arrays."""

    phase: float = SIN_PHASE

    def prob_one(self, u):
        return 0.5 * (1 - self.visibility * np.cos(2 * np.pi * u / L - self.phase))

    def __call__(self, x, nu, rng):
        p = self.prob_one(x + nu)
        b = (rng.random(x.shape) < p).astype(np.int8)
        pf = np.broadcast_to(np.asarray(self.p_flip, dtype=float), x.shape)
        if np.any(pf > 0):
            b ^= (rng.random(x.shape) < pf).astype(np.int8)
        return b


@dataclass
class PhasedBits:
    """m bits per round, bit j with phase phases[j]; returned packed (bit j -> 2^j).

    visibility / p_flip may be scalars or per-trial arrays. Readout noise nu is
    drawn independently per bit (the caller's nu is used for bit 0, fresh
    N(0, sigma_m_extra^2) samples are not added: pass sigma_m via nu_sd for
    the other bits).
    """

    phases: tuple = (SIN_PHASE,)
    visibility: float = 1.0
    p_flip: float = 0.0
    nu_sd: float = 0.0          # readout noise of bits 1..m-1 (bit 0 uses the simulator's nu)
    kind: str = "bit"

    @property
    def m(self):
        return len(self.phases)

    def prob_one(self, u, phase):
        return 0.5 * (1 - self.visibility * np.cos(2 * np.pi * u / L - phase))

    def __call__(self, x, nu, rng):
        packed = np.zeros(x.shape, dtype=np.int8)
        pf = np.broadcast_to(np.asarray(self.p_flip, dtype=float), x.shape)
        for j, phase in enumerate(self.phases):
            noise = nu if j == 0 else rng.normal(0.0, 1.0, x.shape) * self.nu_sd
            b = (rng.random(x.shape) < self.prob_one(x + noise, phase)).astype(np.int8)
            if np.any(pf > 0):
                b ^= (rng.random(x.shape) < pf).astype(np.int8)
            packed |= (b << j).astype(np.int8)
        return packed


def unpack(packed, m):
    """(trials,) packed bits -> (trials, m) array of 0/1."""
    packed = np.asarray(packed).astype(np.int64)
    return np.stack([(packed >> j) & 1 for j in range(m)], axis=-1)


def default_phases(m):
    """m bits alternating sin- and cos-type phases, starting with sin."""
    return tuple(SIN_PHASE if j % 2 == 0 else 0.0 for j in range(m))


def correction_costs(centres, lam=1e-2):
    """Cost matrix M[i, k]: odd-cell indicator + lam * centring term for state x_i
    after the correction c_k; c_k are the same grid points wrapped to [-L, L)."""
    x = np.asarray(centres, dtype=float)
    c = np.remainder(x + L, 2 * L) - L
    r = x[:, None] - c[None, :]
    odd = (np.rint(r / L).astype(np.int64) % 2 != 0).astype(float)
    d = np.remainder(r + L, 2 * L) - L                 # distance to nearest even multiple
    return odd + lam * (d / L) ** 2, c


class SBSDecoder:
    """Memoryless feedback: c = delta * mean over this round's sin-type bits of (2b - 1).

    With one sin-type bit per round this is c = +-delta from the latest bit.
    """

    def __init__(self, delta, phases=(SIN_PHASE,), name=None):
        self.delta = float(delta)
        self.phases = tuple(phases)
        self.sin_bits = [j for j, ph in enumerate(self.phases) if abs(ph - SIN_PHASE) < 1e-9]
        if not self.sin_bits:
            raise ValueError("the sBs baseline needs at least one sin-type bit")
        self.name = name or f"sBs delta={delta / L:.2f}L"

    def reset(self, trials):
        pass

    def decide(self, bits):
        b = unpack(bits, len(self.phases))[:, self.sin_bits]
        return self.delta * (2.0 * b - 1.0).mean(axis=1)


class BinaryBayesDecoder:
    """Matched grid filter for PhasedBit syndromes with a greedy correction policy."""

    def __init__(self, sigma_q, sigma_m, rho, visibility=1.0, p_flip=0.0, phases=(SIN_PHASE,),
                 n_x=96, lam=1e-2, lookahead=True, device=None, name="Bayes (binary, matched)"):
        self.name = name
        self.lookahead = lookahead
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_x = int(n_x)
        self.h, self.M, eps = _eps_grid(self.n_x, sigma_q)
        self.n_e = len(eps)
        self.x = np.arange(self.n_x) * self.h
        self.K = torch.tensor(_stationary_kernel(eps, self.h, sigma_q, rho),
                              device=self.device, dtype=torch.float32)
        self.prior = torch.tensor(_binned_normal(eps, self.h, sigma_q), device=self.device,
                                  dtype=torch.float32)
        shift = torch.arange(self.n_e, device=self.device) - self.M
        xs = torch.arange(self.n_x, device=self.device)
        self.shear = (xs[None, :] - shift[:, None]) % self.n_x          # (n_e, n_x)
        var = sigma_m**2 + self.h**2 / 12
        self.v_eff = float(visibility) * math.exp(-2 * math.pi**2 * var / L**2)
        self.p_flip = float(p_flip)
        self.phases = tuple(phases)
        rows = []
        for phase in self.phases:
            p1 = 0.5 * (1 - self.v_eff * np.cos(2 * math.pi * self.x / L - phase))
            rows.append(self.p_flip + (1 - 2 * self.p_flip) * p1)
        self.p_one = torch.tensor(np.stack(rows), device=self.device,
                                  dtype=torch.float32)                  # (m, n_x)
        cost, self.c_values = correction_costs(self.x, lam)
        self.cost = torch.tensor(cost, device=self.device, dtype=torch.float32)
        self.resets = 0

    def reset(self, trials):
        self.b = trials
        B = torch.zeros((self.n_e, trials, self.n_x), device=self.device, dtype=torch.float32)
        B[:, :, 0] = self.prior[:, None]
        self.B = B
        self.index = self.shear[:, None, :].expand(self.n_e, trials, self.n_x)
        self.cols = torch.arange(self.n_x, device=self.device)

    def posterior_x(self):
        return self.B.sum(dim=0)

    def decide(self, bits):
        b = torch.as_tensor(unpack(bits, len(self.phases)), device=self.device).to(torch.bool)
        n_e, trials, n_x = self.B.shape
        B = (self.K @ self.B.reshape(n_e, -1)).reshape(n_e, trials, n_x)
        B = torch.gather(B, 2, self.index)
        w = torch.ones((trials, n_x), device=self.device)
        for j in range(len(self.phases)):
            pj = self.p_one[j][None, :]
            w = w * torch.where(b[:, j:j + 1], pj, 1 - pj)
        B = B * w[None]
        Z = B.sum(dim=(0, 2))
        bad = ~((Z > 0) & torch.isfinite(Z))
        if bool(bad.any()):                       # lost support: restart from the prior
            self.resets += int(bad.sum())
            fresh = torch.zeros((n_e, int(bad.sum()), n_x), device=self.device)
            fresh[:, :, 0] = self.prior[:, None]
            B[:, bad] = fresh
            Z = B.sum(dim=(0, 2))
        B = B / Z[None, :, None]
        if self.lookahead:
            P = (self.K @ B.reshape(n_e, -1)).reshape(n_e, trials, n_x)
            px = torch.gather(P, 2, self.index).sum(dim=0)      # predicted x + eps_{t+1}
        else:
            px = B.sum(dim=0)
        k = torch.argmin(px @ self.cost, dim=1)                 # (trials,)
        # x <- x - c_k: new[i] = old[i + k]
        idx = (self.cols[None, :] + k[:, None]) % n_x
        self.B = torch.gather(B, 2, idx[None].expand(n_e, trials, n_x))
        return self.c_values[k.cpu().numpy()]


def make_measurement(visibility=1.0, p_flip=0.0, m=1, sigma_m=0.0):
    """m bits per round with alternating sin/cos phases (m = 1: one sin-type bit)."""
    return PhasedBits(phases=default_phases(m), visibility=visibility, p_flip=p_flip,
                      nu_sd=sigma_m)
