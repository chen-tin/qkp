"""E5b: Bayes model mismatch away from the nominal point.

E5 measured how a Bayes filter degrades when its assumed (rho', sigma_q',
sigma_m') differ from the truth, but only at the nominal point. E5b repeats a
compact version at three more true points:
    weak correlation    sigma_q = 0.20L, sigma_m = 0.05L, rho = 0.5
    strong noise        sigma_q = 0.26L, sigma_m = 0.05L, rho = 0.9
    hardware scale      sigma_q = 0.072L, sigma_m = 0.105L, rho = 0.9 (rho assumed;
                        see docs/benchmarks.md section H)
Assumed models: rho' in {0, 0.5, 0.9, 0.99} x sigma_q' in {0.5, 1, 2} x true, plus
sigma_m' in {0.5, 2} x true. Same machinery and caching as E5 (run_all.evaluate_point,
fixed test-set size, 50 rounds, reference decoders). Nothing has been run.

    python -m experiments.run_all --only E5b
"""

from experiments import run_all
from experiments.run_all import ar1, bayes, evaluate_point, simple
from simulator.noise import L

E5B_POINTS = {
    "rho0.5": dict(sigma_q=0.20 * L, sigma_m=0.05 * L, rho=0.5),
    "sq0.26": dict(sigma_q=0.26 * L, sigma_m=0.05 * L, rho=0.9),
    "hardware": dict(sigma_q=0.072 * L, sigma_m=0.105 * L, rho=0.9),
}
E5B_RHOS = (0.0, 0.5, 0.9, 0.99)
E5B_SQ_SCALE = (0.5, 1.0, 2.0)
E5B_SM_SCALE = (0.5, 2.0)


def decoder_set(truth):
    decs = [bayes(**truth, name="Bayes (matched)"), simple("NL")]
    for rho in E5B_RHOS:
        for k in E5B_SQ_SCALE:
            if abs(rho - truth["rho"]) < 1e-12 and k == 1.0:
                continue
            decs.append(bayes(k * truth["sigma_q"], truth["sigma_m"], rho,
                              name=f"Bayes rho'={rho} sq'x{k}"))
    for k in E5B_SM_SCALE:
        decs.append(bayes(truth["sigma_q"], k * truth["sigma_m"], truth["rho"],
                          name=f"Bayes sm'x{k}"))
    return decs


def e5b():
    for i, (pid, truth) in enumerate(E5B_POINTS.items()):
        decs = decoder_set(truth)
        evaluate_point("E5b", i, pid, ar1(**truth), decs, decs[0], cap=40_000,
                       min_events=10**9)
