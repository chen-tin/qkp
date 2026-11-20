"""Fast finite-energy GKP encoding and Wigner function (GPU-capable).

encode_gkp_fast(delta, n_cutoff, k_max)
    Same state as quantum/encoding.py's encode_gkp: the sum over the
    lattice z = 2k alpha + ell beta (alpha = sqrt(pi/2), beta = i alpha) of
    (-1)^(k ell) |z>, damped by exp(-delta^2 n) and normalised. The
    coherent-state Fock coefficients exp(-|z|^2/2) z^n / sqrt(n!) are built
    for all lattice points at once in the log domain instead of a Python
    double loop. Returns a complex128 numpy array (Fock coefficients).

wigner_fft(coeffs, xvec)
    Wigner function W(q, p) on the square grid xvec x xvec, in QuTiP's
    convention (qutip.wigner(psi, xvec, xvec), g = sqrt(2), so q and p are
    the [q, p] = i quadratures). Computed from the position wavefunction:

        W(q, p) = (1/pi) * integral psi*(q + y) psi(q - y) exp(2 i p y) dy

    with psi(q) from the Hermite-function recursion, the y integral as a
    matrix product (a GEMM on the GPU). The integration grid is a refinement
    of xvec, extended until psi has decayed.
"""

import math

import numpy as np
import torch
from scipy.special import gammaln


def encode_gkp_fast(delta, n_cutoff, k_max, device=None):
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    alpha = math.sqrt(math.pi / 2)
    ks = np.arange(-k_max, k_max + 1)
    k, ell = np.meshgrid(ks, ks, indexing="ij")
    z = (2 * k * alpha + 1j * ell * alpha).ravel()
    sign = np.where((k * ell).ravel() % 2 == 0, 1.0, -1.0)
    n = np.arange(n_cutoff)
    # log of exp(-|z|^2/2) z^n / sqrt(n!); z = 0 contributes only n = 0
    nz = z != 0
    logz = np.zeros_like(z)
    logz[nz] = np.log(z[nz])
    logc = -np.abs(z)[:, None] ** 2 / 2 + n[None, :] * logz[:, None] - 0.5 * gammaln(n + 1)[None, :]
    c = np.exp(logc)
    c[~nz, 1:] = 0.0
    coeffs = (sign[:, None] * c).sum(axis=0) * np.exp(-delta**2 * n)
    norm = np.linalg.norm(coeffs)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Could not normalize the state")
    return coeffs / norm


def _hermite_functions(n_max, q):
    """phi_n(q) for n < n_max on points q (float64 torch), normalised recursion."""
    phi = torch.empty((n_max, q.numel()), dtype=q.dtype, device=q.device)
    phi[0] = math.pi ** (-0.25) * torch.exp(-q * q / 2)
    if n_max > 1:
        phi[1] = math.sqrt(2) * q * phi[0]
    for n in range(1, n_max - 1):
        phi[n + 1] = math.sqrt(2 / (n + 1)) * q * phi[n] - math.sqrt(n / (n + 1)) * phi[n - 1]
    return phi


def wavefunction(coeffs, q):
    """psi(q) for Fock coefficients on points q (torch tensors, complex128)."""
    phi = _hermite_functions(len(coeffs), q)
    c = torch.as_tensor(coeffs, device=q.device, dtype=torch.complex128)
    return c @ phi.to(torch.complex128)


def wigner_fft(coeffs, xvec, device=None, refine=2, tol=1e-12):
    """W(q, p) on xvec x xvec; returns numpy array indexed [p, q] like QuTiP."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.asarray(xvec, dtype=float)
    dx = (x[-1] - x[0]) / (len(x) - 1)
    if not np.allclose(np.diff(x), dx, rtol=1e-9, atol=1e-12):
        raise ValueError("xvec must be uniform")
    h = dx / refine                                   # y step; q +- y lands on a grid of step h
    # support of psi: extend until |psi|^2 < tol (bounded by the Fock cutoff)
    n = len(coeffs)
    qmax = math.sqrt(2 * n + 1) + 6.0
    J = int(math.ceil(qmax / h))                      # y in [-J h, J h]
    # fine grid covering q_i +- y_j
    q0 = x[0]
    m_lo = -J
    m_hi = (len(x) - 1) * refine + J
    m = torch.arange(m_lo, m_hi + 1, device=device, dtype=torch.float64)
    qf = q0 + m * h
    psi = wavefunction(coeffs, qf)
    # the integrand psi*(q+y) psi(q-y) vanishes once |y| exceeds the support S
    # of psi, so trim the y range to S (much smaller than the Fock bound)
    dens = (psi.abs() ** 2).cpu().numpy()
    support = np.flatnonzero(dens > tol * dens.max())
    S = float(np.max(np.abs(qf.cpu().numpy()[support]))) if len(support) else qmax
    J = min(J, int(math.ceil(S / h)) + 2)
    # rows: q_i at fine index i*refine - m_lo ; offsets j in [-J, J]
    qi = torch.arange(len(x), device=device) * refine - m_lo
    j = torch.arange(-J, J + 1, device=device)
    plus = psi[(qi[:, None] + j[None, :])]
    minus = psi[(qi[:, None] - j[None, :])]
    F = plus.conj() * minus                           # (Nq, Ny)
    y = j.to(torch.float64) * h
    p = torch.as_tensor(x, device=device, dtype=torch.float64)
    E = torch.exp(2j * y[:, None] * p[None, :]) * (h / math.pi)   # (Ny, Np)
    W = (F @ E).real                                  # (Nq, Np)
    return W.T.cpu().numpy()
