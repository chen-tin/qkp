"""HyQBench-style checks of finite-energy GKP states (quantum/encoding.py).

The finite-energy state is |psi_Delta> ~ exp(-Delta^2 n) |0_ideal>, built as
in quantum/encoding.py: a (-1)^{k l}-weighted sum of coherent states on the
lattice z = 2k sqrt(pi/2) + i l sqrt(pi/2), |k|,|l| <= k_max, truncated to
n < N Fock levels, times the envelope exp(-Delta^2 n). Conventions:
q = (a + a^dag)/sqrt(2), [q, p] = i, L = sqrt(pi).

Metrics (HyQBench, arXiv:2025 "A Benchmark Suite for Hybrid CV-DV Quantum
Computing": truncation cost, Wigner negativity, cross-check with QuTiP):

1. Truncation: <n>, weight in the top 10% of Fock levels, fidelity of the
   cutoff-N state to a reference (N = 600, k_max = 12), smallest cutoff with
   fidelity > 1 - 1e-6. |+> is the pi/2 phase-space rotation of |0>
   (c_n -> (-i)^n c_n), so its Fock populations, and all truncation
   metrics, are identical to those of |0>.
2. Wigner negativity delta_W = (int |W| dq dp - 1) / 2, on grids of three
   spacings (convergence check), with QuTiP's Clenshaw Wigner function.
3. Stabilisers <S_q> = <exp(2i sqrt(pi) q)>, <S_p> = <exp(-2i sqrt(pi) p)>
   and the effective envelope parameter
       Delta_eff^2 = -ln(|<S>|^2) / (2 pi)
   (Duivenvoorden, Terhal & Weigand, PRA 95, 012305 (2017)): for peaks of
   probability standard deviation s, |<S>| = exp(-2 pi s^2), and the
   finite-energy state has s = Delta/sqrt(2). Squeezing in dB is
   -10 log10(Delta_eff^2).
4. Shift-model cross-check: the q-marginal of |0_Delta> -- peak widths and
   positions, the probability that a q-measurement lands in an odd cell
   (round(q/L) odd: the intrinsic logical readout error), and the same after
   a random displacement u ~ N(0, sigma_q^2) -- compared with the
   simulator's Gaussian shift model p(sigma) (simulator.sim.odd_jump_probability).

Run from the repo root (CPU only):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python -m experiments.benchmark_finite_energy
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.optimize import brentq
from scipy.stats import norm

from simulator.sim import odd_jump_probability

L = math.sqrt(math.pi)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

DELTAS = (0.20, 0.25, 0.30, 0.34, 0.40)
CUTOFFS = (40, 60, 80, 100, 140, 200)
N_REF, K_REF = 600, 12
K_USED = 8                      # k_max in quantum/encoding.py
SIGMA_Q_L = (0.072, 0.10, 0.20)  # Sivak loss level, intermediate, our nominal


def gkp_zero_raw(delta, N, k_max):
    """Unnormalised Fock coefficients of exp(-Delta^2 n)|0>, as in encoding.py."""
    a = math.sqrt(math.pi / 2)
    k, l = np.meshgrid(np.arange(-k_max, k_max + 1), np.arange(-k_max, k_max + 1),
                       indexing="ij")
    z = (2 * k * a + 1j * l * a).ravel()
    sign = ((-1.0) ** (k * l)).ravel()
    c = np.zeros(N, dtype=complex)
    f = np.exp(-np.abs(z) ** 2 / 2).astype(complex)
    c[0] = (sign * f).sum()
    for n in range(1, N):
        f = f * z / math.sqrt(n)
        c[n] = (sign * f).sum()
    return c * np.exp(-delta**2 * np.arange(N))


def normalised(c):
    return c / np.linalg.norm(c)


def plus_from_zero(c):
    """|+> = pi/2 rotation of |0> for the square code."""
    return c * (-1j) ** np.arange(len(c))


def hermite_functions(x, N):
    """phi_n(x), n < N, normalised for q = (a + a^dag)/sqrt(2)."""
    phi = np.zeros((N, len(x)))
    phi[0] = np.pi ** (-0.25) * np.exp(-x**2 / 2)
    if N > 1:
        phi[1] = math.sqrt(2) * x * phi[0]
    for n in range(1, N - 1):
        phi[n + 1] = math.sqrt(2 / (n + 1)) * x * phi[n] - math.sqrt(n / (n + 1)) * phi[n - 1]
    return phi


def marginals(c, x):
    phi = hermite_functions(x, len(c))
    pq = np.abs(c @ phi) ** 2
    pp = np.abs((c * (-1j) ** np.arange(len(c))) @ phi) ** 2
    return pq, pp


def mass_cutoff(c_ref, threshold):
    cum = np.cumsum(np.abs(c_ref) ** 2)
    return int(np.searchsorted(cum, threshold) + 1)


def odd_mass(x, pq, dx):
    cells = np.rint(x / L).astype(int)
    return float(pq[cells % 2 == 1].sum() * dx)


def odd_mass_after_shift(x, pq, dx, sigma):
    """P(round((q + u)/L) odd), u ~ N(0, sigma^2), from the exact marginal."""
    if sigma == 0:
        return odd_mass(x, pq, dx)
    j = np.arange(-60, 61)
    j = j[j % 2 == 1]
    hi = (j[:, None] * L + L / 2 - x[None, :]) / sigma
    lo = (j[:, None] * L - L / 2 - x[None, :]) / sigma
    p_odd = (norm.cdf(hi) - norm.cdf(lo)).sum(axis=0)
    return float((pq * p_odd).sum() * dx)


def sigma_matching(p):
    """sigma with odd_jump_probability(sigma) = p."""
    if p <= 1e-15:
        return None
    return brentq(lambda s: odd_jump_probability(s) - p, 1e-3, 3.0)


def peak_stats(x, pq, dx, kmax=3):
    rows = []
    for k in range(-kmax, kmax + 1):
        sel = np.abs(x - 2 * k * L) < L
        m = pq[sel].sum() * dx
        if m < 1e-8:
            continue
        mu = (x[sel] * pq[sel]).sum() * dx / m
        sd = math.sqrt(((x[sel] - mu) ** 2 * pq[sel]).sum() * dx / m)
        rows.append(dict(k=k, mass=m, centre_over_L=mu / L,
                         contraction=(mu / (2 * k * L)) if k else None, sd_over_L=sd / L))
    return rows


def wigner_negativity(c, R, h):
    xs = np.arange(-R, R + h / 2, h)
    W = qt.wigner(qt.Qobj(c.reshape(-1, 1)), xs, xs, method="clenshaw")
    total = W.sum() * h * h
    absint = np.abs(W).sum() * h * h
    return dict(h=h, R=R, points=len(xs) ** 2, integral=float(total),
                negativity=float((absint - 1) / 2), negative_volume=float(-W[W < 0].sum() * h * h))


def qutip_shift_check(c, u, x, dx):
    """Apply D (q -> q + u) in a larger Hilbert space; compare odd mass with translation."""
    N = len(c)
    big = N + 300
    psi = qt.Qobj(np.pad(c, (0, big - N)).reshape(-1, 1))
    shifted = qt.displace(big, u / math.sqrt(2)) * psi
    cs = shifted.full().ravel()
    pq_s, _ = marginals(cs, x)
    pq, _ = marginals(c, x)
    pq_t = np.interp(x - u, x, pq)
    return dict(u_over_L=u / L, odd_mass_qutip=odd_mass(x, pq_s, dx),
                odd_mass_translated=odd_mass(x, pq_t, dx),
                norm_after=float(np.linalg.norm(cs) ** 2))


def main():
    t0 = time.time()
    out = dict(conventions="q=(a+a^dag)/sqrt2, [q,p]=i, L=sqrt(pi); state exp(-Delta^2 n)|0>",
               reference=dict(N=N_REF, k_max=K_REF), k_max_used=K_USED, deltas={})
    trunc_rows = []
    for delta in DELTAS:
        d = {}
        ref = normalised(gkp_zero_raw(delta, N_REF, K_REF))
        ref8 = normalised(gkp_zero_raw(delta, N_REF, K_USED))
        d["k_max_fidelity_8_vs_12"] = float(abs(np.vdot(ref, ref8)) ** 2)
        d["reference_top10pct_weight"] = float((np.abs(ref[-N_REF // 10:]) ** 2).sum())
        n_levels = np.arange(N_REF)
        d["mean_n"] = float((n_levels * np.abs(ref) ** 2).sum())
        d["cutoff_for_1e-6"] = mass_cutoff(ref, 1 - 1e-6)
        d["cutoff_for_1e-10"] = mass_cutoff(ref, 1 - 1e-10)
        d["truncation"] = []
        for N in CUTOFFS:
            psi = normalised(gkp_zero_raw(delta, N, K_USED))
            plus = plus_from_zero(psi)
            p = np.abs(psi) ** 2
            row = dict(delta=delta, N=N, mean_n=float((np.arange(N) * p).sum()),
                       top10pct_weight=float(p[-max(1, N // 10):].sum()),
                       fidelity_to_ref=float(abs(np.vdot(ref[:N], psi)) ** 2),
                       fidelity_plus_to_ref=float(
                           abs(np.vdot(plus_from_zero(ref)[:N], plus)) ** 2))
            row["enough_1e-6"] = row["fidelity_to_ref"] > 1 - 1e-6
            d["truncation"].append(row)
            trunc_rows.append(row)
        d["cutoff_100_enough"] = next(r for r in d["truncation"] if r["N"] == 100)["enough_1e-6"]

        # marginals of a converged state
        Nc = min(N_REF, d["cutoff_for_1e-10"] + 20)
        c = ref[:Nc] / np.linalg.norm(ref[:Nc])
        R = 4 / delta + 6
        dx = 0.002
        x = np.arange(-R, R + dx / 2, dx)
        pq, pp = marginals(c, x)
        d["marginal_mass_q"] = float(pq.sum() * dx)
        Sq = (pq * np.exp(2j * L * x)).sum() * dx
        Sp = (pp * np.exp(-2j * L * x)).sum() * dx
        d["S_q"] = [float(Sq.real), float(Sq.imag)]
        d["S_p"] = [float(Sp.real), float(Sp.imag)]
        for name, S in (("q", Sq), ("p", Sp)):
            de2 = -math.log(abs(S) ** 2) / (2 * math.pi)
            d[f"delta_eff_{name}"] = math.sqrt(de2)
            d[f"squeezing_dB_{name}"] = -10 * math.log10(de2)
        d["squeezing_dB_nominal"] = -10 * math.log10(delta**2)

        peaks = peak_stats(x, pq, dx)
        d["peaks"] = peaks
        centre = next(r for r in peaks if r["k"] == 0)
        d["peak_sd_centre_over_L"] = centre["sd_over_L"]
        w = np.array([r["mass"] for r in peaks])
        d["peak_sd_massweighted_over_L"] = float(
            (w * np.array([r["sd_over_L"] for r in peaks])).sum() / w.sum())
        d["delta_over_sqrt2_over_L"] = delta / math.sqrt(2) / L
        po = odd_mass(x, pq, dx)
        d["odd_cell_mass"] = po
        d["shift_model_p_delta_over_sqrt2"] = odd_jump_probability(delta / math.sqrt(2))
        s_fit = sigma_matching(po)
        d["sigma_matching_odd_mass_over_L"] = s_fit / L if s_fit else None
        d["after_displacement"] = []
        for sq_l in SIGMA_Q_L:
            sq = sq_l * L
            m_exact = odd_mass_after_shift(x, pq, dx, sq)
            row = dict(sigma_q_over_L=sq_l, odd_mass_quantum=m_exact,
                       shift_model_delta_over_sqrt2=odd_jump_probability(
                           math.sqrt(sq**2 + delta**2 / 2)),
                       shift_model_fitted_sigma=odd_jump_probability(
                           math.sqrt(sq**2 + s_fit**2)) if s_fit else None)
            row["ratio_quantum_to_shift_model"] = (m_exact / row["shift_model_delta_over_sqrt2"]
                                                  if row["shift_model_delta_over_sqrt2"] > 0 else None)
            se = sigma_matching(m_exact)
            row["total_sigma_matching_over_L"] = se / L if se else None
            row["implied_extra_sigma_over_L"] = (math.sqrt(max(se**2 - sq**2, 0)) / L
                                                 if se else None)
            d["after_displacement"].append(row)

        # Wigner negativity with grid convergence (state at 1e-8 truncation)
        Nw = min(N_REF, mass_cutoff(ref, 1 - 1e-8) + 10)
        cw = ref[:Nw] / np.linalg.norm(ref[:Nw])
        Rw = min(R, 3.2 / delta + 3)
        d["wigner"] = [wigner_negativity(cw, Rw, h) for h in (0.08, 0.05, 0.035)]
        d["wigner_N"] = Nw
        out["deltas"][f"{delta:.2f}"] = d
        print(f"Delta={delta:.2f}: <n>={d['mean_n']:.2f} N(1e-6)={d['cutoff_for_1e-6']} "
              f"sd/L={d['peak_sd_centre_over_L']:.4f} odd={po:.3e} "
              f"negW={d['wigner'][-1]['negativity']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # QuTiP cross-check of the displacement step at Delta = 0.30
    delta = 0.30
    ref = normalised(gkp_zero_raw(delta, N_REF, K_REF))
    Nc = mass_cutoff(ref, 1 - 1e-10) + 20
    c = ref[:Nc] / np.linalg.norm(ref[:Nc])
    dx = 0.002
    x = np.arange(-4 / delta - 8, 4 / delta + 8, dx)
    out["qutip_displacement_check"] = [qutip_shift_check(c, u * L, x, dx)
                                       for u in (0.1, 0.3)]
    # stabiliser via a QuTiP displacement operator (cross-check of item 3)
    big = Nc + 300
    psi = qt.Qobj(np.pad(c, (0, big - Nc)).reshape(-1, 1))
    Sq_op = qt.expect(qt.displace(big, 1j * math.sqrt(2 * math.pi)), psi)
    out["qutip_stabiliser_check_delta0.30"] = dict(
        S_q_operator=[float(np.real(Sq_op)), float(np.imag(Sq_op))],
        S_q_marginal=out["deltas"]["0.30"]["S_q"])
    out["runtime_s"] = time.time() - t0

    (OUT / "finite_energy.json").write_text(json.dumps(out, indent=1))
    keys = list(trunc_rows[0])
    with open(OUT / "finite_energy_truncation.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in trunc_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"wrote {OUT / 'finite_energy.json'} in {out['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
