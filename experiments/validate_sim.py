"""Validate simulator/sim.py against closed-form results.

Every reference curve below is either exact for the model it is applied to
or explicitly marked as an approximation / limit. The derivations are in
docs/sim_validation.md.

Run from the repo root:
    python -m experiments.validate_sim
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.stats import norm

from simulator.noise import L, NoiseModel
from simulator.sim import Simulator, NearestLattice, odd_jump_probability

seed = 2026
figure_path = Path("docs/figures/sim_validation.png")
dpi = 200


# closed forms
def gkp_bound(sigma):
    """GKP 2001, Eq. (94): p_e < P(|x| > sqrt(pi)/2) = erfc(sqrt(pi) / (2 sqrt(2) sigma))."""
    return erfc(L / (2 * np.sqrt(2) * sigma))


def parity_odd(p, n):
    """P(odd number of flips after n independent rounds with flip probability p)."""
    return (1 - (1 - 2 * p) ** n) / 2


def binomial_se(p, n):
    return np.sqrt(np.maximum(p * (1 - p), 1e-300) / n)


def z_score(sim, exact, n):
    """(sim - exact) / standard error, using the exact p for the error."""
    return (sim - exact) / binomial_se(exact, n)


def run_nearest(model, trials, rounds, seed):
    sim = Simulator(model, trials=trials, rounds=rounds)
    return sim.run_all([NearestLattice()], seed=seed)[NearestLattice.name]


def odd_jump_rate(result):
    """Per-round fraction of trials whose lattice index jumped by an odd amount."""
    return (result.lattice_jumps() % 2 == 1).mean(axis=0)


rows = []  # (check, case, simulated, reference, z, kind)


def record(check, case, sim, ref, n, kind):
    rows.append((check, case, sim, ref, z_score(sim, ref, n), kind))


# A: single round, sigma_m = 0, sweep sigma
# exact: Fukui et al. 2018 Eq. (22) at v_up = 0 (sum over odd cells)
# bound: GKP 2001 Eq. (94); literature point sigma = 0.555 -> 0.110
trials_a = 200_000
sigmas_a = np.array([0.20, 0.30, 0.40, 0.50, 0.555, 0.60, 0.70, 0.80])
sim_a, exact_a = [], []
for i, s in enumerate(sigmas_a):
    r = run_nearest(NoiseModel(sigma_q=s, sigma_m=0.0, rho=0.0),
                    trials_a, 1, seed + i)
    sim_a.append(odd_jump_rate(r)[0])
    exact_a.append(odd_jump_probability(s))
sim_a, exact_a = np.array(sim_a), np.array(exact_a)

for s, ps, pe in zip(sigmas_a, sim_a, exact_a):
    record("A single round", f"sigma={s:.3f}", ps, pe, trials_a, "exact")

i555 = int(np.argmin(np.abs(sigmas_a - 0.555)))
print(f"GKP Eq. (94) bound at sigma=0.555: {gkp_bound(0.555):.4f} (paper: 0.110)")
print(f"Exact odd-cell sum at sigma=0.555: {exact_a[i555]:.4f}")
print(f"Simulated at sigma=0.555:          {sim_a[i555]:.4f}")


# B: per-round flip rate vs round, default noise, several rho
# exact marginal for every rho: round 1 var = sq^2 + sm^2, later sq^2 + 2 sm^2
trials_b, rounds_b = 50_000, 50
base = NoiseModel()
sigma_first = np.hypot(base.sigma_q, base.sigma_m)
sigma_later = np.sqrt(base.sigma_q**2 + 2 * base.sigma_m**2)
p_first = odd_jump_probability(sigma_first)
p_later = odd_jump_probability(sigma_later)
expected_b = np.full(rounds_b, p_later)
expected_b[0] = p_first

rhos_b = [0.0, 0.9]
sim_b = {}
for rho in rhos_b:
    r = run_nearest(NoiseModel(rho=rho), trials_b, rounds_b, seed + 100)
    rate = odd_jump_rate(r)
    sim_b[rho] = rate
    record("B per-round flip", f"rho={rho}, round 1", rate[0], p_first,
           trials_b, "exact")
    # pooled over rounds 2..50; the z treats rounds as independent, which
    # they are not (shared nu_t, correlated eps_t), so it is only indicative
    record("B per-round flip", f"rho={rho}, mean of rounds 2-50",
           rate[1:].mean(), p_later, trials_b * (rounds_b - 1), "exact")
    # Bonferroni test on the most extreme single round
    z = z_score(rate[1:], p_later, trials_b)
    p_bonf = min(1.0, (rounds_b - 1) * 2 * norm.sf(np.abs(z).max()))
    print(f"B rho={rho}: max |z| over rounds 2-50 = {np.abs(z).max():.2f}, "
          f"Bonferroni p = {p_bonf:.3f}")


# C: logical error rate vs round, rho = 0
# C1: sigma_m = 0 -> jumps are i.i.d. -> parity formula is exact
# C2: sigma_m > 0 -> consecutive jumps share nu_t -> parity formula is approximate
trials_c, rounds_c = 50_000, 50
n = np.arange(1, rounds_c + 1)

model_c1 = NoiseModel(sigma_m=0.0, rho=0.0)
r_c1 = run_nearest(model_c1, trials_c, rounds_c, seed + 200)
ler_c1 = r_c1.logical_error_rate()
ref_c1 = parity_odd(odd_jump_probability(model_c1.sigma_q), n)
z = z_score(ler_c1, ref_c1, trials_c)
k = int(np.argmax(np.abs(z)))
record("C1 LER, rho=0, sm=0", f"worst round ({k+1})", ler_c1[k], ref_c1[k],
       trials_c, "exact")
record("C1 LER, rho=0, sm=0", "round 50", ler_c1[-1], ref_c1[-1],
       trials_c, "exact")

model_c2 = NoiseModel(rho=0.0)
r_c2 = run_nearest(model_c2, trials_c, rounds_c, seed + 300)
ler_c2 = r_c2.logical_error_rate()
p1 = odd_jump_probability(np.hypot(model_c2.sigma_q, model_c2.sigma_m))
p2 = odd_jump_probability(
    np.sqrt(model_c2.sigma_q**2 + 2 * model_c2.sigma_m**2))
# first round with p1, then n-1 rounds with p2 (independent-jump approximation)
ref_c2 = (1 - (1 - 2 * p1) * (1 - 2 * p2) ** (n - 1)) / 2
record("C2 LER, rho=0, sm=0.05L", "round 50", ler_c2[-1], ref_c2[-1],
       trials_c, "approx.")


# D: rho -> 1 limit, sigma_m = 0
# frozen error eps: every round jumps by round(eps/L), so the logical state
# is flipped after n rounds iff n is odd and round(eps/L) is odd:
#   P(n) = p for odd n, 0 for even n, with p = odd_jump_probability(sigma_q)
# rho must be < 1, so check convergence as 1 - rho shrinks
trials_d, rounds_d = 50_000, 50
rhos_d = [0.999, 0.9999, 0.99999, 0.999999]
ler_d_all = {}
for rho in rhos_d:
    model_d = NoiseModel(sigma_m=0.0, rho=rho)
    ler_d_all[rho] = run_nearest(model_d, trials_d, rounds_d,
                                 seed + 400).logical_error_rate()
p_d = odd_jump_probability(NoiseModel().sigma_q)
ref_d = np.where(n % 2 == 1, p_d, 0.0)
for rho in rhos_d:
    ler = ler_d_all[rho]
    record("D rho->1 limit, sm=0", f"rho={rho}, odd rounds (mean)",
           ler[0::2].mean(), p_d, trials_d, "limit")
    record("D rho->1 limit, sm=0", f"rho={rho}, even rounds (mean)",
           ler[1::2].mean(), 0.0, trials_d, "limit")
ler_d = ler_d_all[rhos_d[-1]]


# table
print()
print("| Check | Case | Simulated | Reference | z | Reference type |")
print("|---|---|---|---|---|---|")
for check, case, s, ref, z, kind in rows:
    zs = "n/a" if ref == 0 else f"{z:+.2f}"
    print(f"| {check} | {case} | {s:.5f} | {ref:.5f} | {zs} | {kind} |")


# figure
plt.rcParams.update({"font.size": 11})
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# A
ax = axes[0, 0]
grid = np.linspace(0.15, 0.85, 200)
ax.plot(grid, [odd_jump_probability(s) for s in grid], "k-",
        label="Exact: sum over odd cells\n(Fukui et al. 2018, Eq. 22)")
ax.plot(grid, gkp_bound(grid), "k--", lw=1,
        label="Upper bound (GKP 2001, Eq. 94)")
ax.errorbar(sigmas_a, sim_a, yerr=1.96 * binomial_se(sim_a, trials_a),
            fmt="o", color="C0", ms=5, capsize=3, label="Simulator (95% CI)")
ax.plot([0.555], [0.110], "*", color="C3", ms=14,
        label=r"GKP 2001: $\sigma=0.555 \to 0.110$")
ax.set_xlabel(r"$\sigma$ (q-shift std, $[q,p]=i$)")
ax.set_ylabel("Logical flip probability (one round)")
ax.set_title(r"A. One round, ideal measurement ($\sigma_m=0$)")
ax.legend(fontsize=9)

# B
ax = axes[0, 1]
t = np.arange(1, rounds_b + 1)
for rho, color in zip(rhos_b, ["C0", "C1"]):
    rate = sim_b[rho]
    ax.errorbar(t, rate, yerr=1.96 * binomial_se(expected_b, trials_b),
                fmt="o", ms=3, color=color, capsize=2, alpha=0.8,
                label=rf"Simulator, $\rho={rho}$ (95% CI)")
ax.step(t, expected_b, "k-", where="mid",
        label="Exact marginal (any $\\rho$)")
ax.set_xlabel("Round")
ax.set_ylabel("P(odd lattice jump in round)")
ax.set_title(r"B. Per-round flip rate, $\sigma_q=0.2L$, $\sigma_m=0.05L$")
ax.legend(fontsize=9)

# C
ax = axes[1, 0]
ax.plot(n, ler_c1, "o", ms=3, color="C0",
        label=r"Simulator, $\sigma_m=0$")
ax.plot(n, ref_c1, "k-", label=r"Exact parity formula, $\sigma_m=0$")
ax.plot(n, ler_c2, "s", ms=3, color="C2",
        label=r"Simulator, $\sigma_m=0.05L$")
ax.plot(n, ref_c2, "k:", label=r"Independent-jump approx., $\sigma_m=0.05L$")
ax.set_xlabel("Round $n$")
ax.set_ylabel("Logical error rate")
ax.set_title(r"C. Logical error rate, $\rho=0$, $\sigma_q=0.2L$")
ax.legend(fontsize=9)

# D
ax = axes[1, 1]
for rho, color in zip(rhos_d[:2], ["C4", "C2"]):
    ax.plot(n, ler_d_all[rho], ".", ms=4, color=color, alpha=0.6,
            label=rf"Simulator, $\rho={rho}$")
ax.plot(n, ler_d, "o", ms=4, color="C1",
        label=rf"Simulator, $\rho={rhos_d[-1]}$")
ax.axhline(p_d, color="k", ls="--", lw=1,
           label=r"$\rho\to1$ limit, odd $n$: $p$")
ax.axhline(0.0, color="k", ls="-", lw=1,
           label=r"$\rho\to1$ limit, even $n$: 0")
ax.set_xlabel("Round $n$")
ax.set_ylabel("Logical error rate")
ax.set_title(r"D. Frozen-error limit, $\sigma_q=0.2L$, $\sigma_m=0$")
ax.legend(fontsize=9, loc="center right", framealpha=0.95)

fig.tight_layout()
figure_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
print(f"\nSaved {figure_path}")
