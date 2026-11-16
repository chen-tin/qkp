"""Fresh re-run of the simulator validation and the nominal decoder curves.

Independent random streams (seed 20261113) from docs/sim_validation.md
(seed 2026). Produces the data behind the validation figures of the review
page: closed-form checks A-D, nearest-lattice curves vs rho, matched-Bayes
grid check, nominal decoder curves, and one traced trial.

Run from the repo root:
    MPLBACKEND=Agg python -m experiments.validation_rerun
Writes results/benchmarks/validation_rerun.json.
"""

import json
import time

import numpy as np

from decoders.bayesian.grid_filter import BayesGridDecoder
from experiments.common import out_dir
from simulator import nonstationary as ns
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice, NoCorrection, Simulator, lattice_index, odd_jump_probability

SEED = 20261113


def se(p, n):
    return float(np.sqrt(max(p * (1 - p), 1e-300) / n))


def parity_odd(ps):
    return 0.5 * (1 - np.cumprod(1 - 2 * np.asarray(ps)))


def main():
    out = {"seed": SEED, "L": float(L)}
    t0 = time.time()

    # A: single round, sigma_m = 0
    NA = 200000
    out["A"] = []
    for s in (0.3, 0.4, 0.5, 0.555, 0.6, 0.7, 0.8):
        sim = Simulator(NoiseModel(sigma_q=s, sigma_m=0.0, rho=0.0), trials=NA, rounds=1)
        r = sim.run_all([NearestLattice()], seed=SEED)[NearestLattice.name]
        p = float(r.logical_error_rate()[0])
        ref = odd_jump_probability(s)
        out["A"].append(dict(sigma=s, sim=p, ref=ref, z=(p - ref) / se(ref, NA)))

    # B: per-round flip marginal for several rho; NL P_L curves
    NB = 50000
    m0 = NoiseModel()
    p1 = odd_jump_probability(float(np.hypot(m0.sigma_q, m0.sigma_m)))
    p2 = odd_jump_probability(float(np.sqrt(m0.sigma_q ** 2 + 2 * m0.sigma_m ** 2)))
    out["B"], out["NL_curves"] = [], {}
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9, 0.97):
        r = Simulator(NoiseModel(rho=rho), trials=NB, rounds=50).run_all(
            [NearestLattice()], seed=SEED)[NearestLattice.name]
        odd = (r.lattice_jumps() % 2 == 1).mean(axis=0)
        pl = r.logical_error_rate()
        later = float(odd[1:].mean())
        out["B"].append(dict(rho=rho, round1=float(odd[0]), later=later, ref1=p1, ref2=p2,
                             z1=(float(odd[0]) - p1) / se(p1, NB),
                             z_later=(later - p2) / se(p2, NB * 49),
                             PL50=float(pl[-1]), PL50_se=se(float(pl[-1]), NB)))
        out["NL_curves"][str(rho)] = pl.round(5).tolist()
    out["indep_curve"] = parity_odd([p1] + [p2] * 49).round(5).tolist()
    c2ref = out["indep_curve"][-1]
    out["C2"] = dict(sim=out["NL_curves"]["0.0"][-1], approx=c2ref,
                     z=(out["NL_curves"]["0.0"][-1] - c2ref) / se(c2ref, NB))

    # C1: rho = 0, sigma_m = 0, exact over 50 rounds
    mC = NoiseModel(rho=0.0, sigma_m=0.0)
    r = Simulator(mC, trials=NB, rounds=50).run_all([NearestLattice()], seed=SEED)[NearestLattice.name]
    pl = r.logical_error_rate()
    exact = parity_odd([odd_jump_probability(mC.sigma_q)] * 50)
    z = (pl - exact) / np.sqrt(exact * (1 - exact) / NB)
    out["C1"] = dict(sim=pl.round(5).tolist(), exact=exact.round(5).tolist(),
                     zmax=float(np.abs(z).max()))

    # D: rho -> 1 frozen-error limit
    out["D"] = []
    pfro = odd_jump_probability(m0.sigma_q)
    for rho in (0.999, 0.9999, 0.99999, 0.999999):
        r = Simulator(NoiseModel(rho=rho, sigma_m=0.0), trials=NB, rounds=50).run_all(
            [NearestLattice()], seed=SEED)[NearestLattice.name]
        pl = r.logical_error_rate()
        odd_rate = float(pl[0::2].mean())
        out["D"].append(dict(rho=rho, odd=odd_rate, even=float(pl[1::2].mean()), ref_odd=pfro,
                             z=(odd_rate - pfro) / se(pfro, NB)))

    # nominal point: decoder curves, grid check, one traced trial
    NBay = 20000
    sim = Simulator(NoiseModel(), trials=NBay, rounds=50)
    eps, nu = sim.sample_noise(SEED + 7)
    dec = {}
    out["grid"] = []
    for n_x in (64, 128):
        t = time.time()
        rr = sim.run(BayesGridDecoder(m0.sigma_q, m0.sigma_m, m0.rho, n_x=n_x), eps, nu)
        dec[n_x] = rr
        out["grid"].append(dict(n_x=n_x, seconds=time.time() - t,
                                PL50=float(rr.logical_error_rate()[-1]),
                                mean=float(rr.logical_error_rate().mean())))
    out["grid_agree"] = float((dec[64].logical_errors == dec[128].logical_errors).mean())
    rNL = sim.run(NearestLattice(), eps, nu)
    rNo = sim.run(NoCorrection(), eps, nu)
    rPR = sim.run(BayesGridDecoder(m0.sigma_q, m0.sigma_m, m0.rho, per_round=True), eps, nu)
    out["nominal"] = dict(trials=NBay, curves={
        "No correction": rNo.logical_error_rate().round(5).tolist(),
        "Nearest lattice": rNL.logical_error_rate().round(5).tolist(),
        "Bayes per-round": rPR.logical_error_rate().round(5).tolist(),
        "Bayes matched": dec[128].logical_error_rate().round(5).tolist(),
    })
    kNL = lattice_index(rNL.residuals)
    kB = lattice_index(dec[128].residuals)
    cand = np.where(rNL.logical_errors[:, -1] & ~dec[128].logical_errors[:, -1])[0]
    i = int(cand[0]) if len(cand) else 0
    out["trace"] = dict(trial=i,
                        xpre=((rNL.residuals[i] + rNL.corrections[i]) / L).round(4).tolist(),
                        s=(rNL.syndromes[i] / L).round(4).tolist(),
                        eps=(eps[i] / L).round(4).tolist(),
                        kNL=(kNL[i] % 2).tolist(), kB=(kB[i] % 2).tolist())
    out["runtime_s"] = time.time() - t0
    path = out_dir() / "benchmarks" / "validation_rerun.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path} in {out['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()
