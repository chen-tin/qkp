"""Nonstationary-noise experiments E9-E11 (updated proposal, "Scope and Evaluation").

E9   gradual drift          sinusoidal, ramped and 1/f changes of sigma_q or rho,
                            with a stationary control
E10  unexpected changes     unannounced steps, telegraph switching, one-round bursts
E11  unseen conditions      parameter values outside the drift-LSTM training
                            ranges (E9/E10 also contain shapes held out from
                            training: ramp, 1/f, telegraph, burst)

Every point uses the same machinery as E1-E8 (experiments/run_all.py):
shared test noise for all decoders, caching with sidecars, resumable runs.
The reference decoder is the oracle Bayes filter, which knows the true
schedule; every other decoder sees only syndromes.

Run from the repo root (nothing here has been run yet):
    python -m training.train_lstm                     # trains lstm_drift_s* too
    python -m experiments.run_nonstationary           # E9-E11
    python -m experiments.run_nonstationary --only E10
    python -m experiments.analyze_nonstationary
    QKP_MODE=quick ...                                # small smoke-test sizes
"""

import argparse
import hashlib
import time
import traceback

import matplotlib

matplotlib.use("Agg")

from decoders.bayesian.adaptive import (
    OracleBayesDecoder, RegimeBankDecoder, WindowedBayesDecoder, parameter_grid,
)
from experiments import run_all
from experiments.common import ROOT, get_budget, get_mode, log, torch_device
from experiments.run_all import NOMINAL, bayes, bayes_broad, evaluate_point, lstm_family, simple
from simulator.noise import L
from simulator.nonstationary import spec

# code that determines the noise and the new decoders; part of every point's
# noise spec, so editing it invalidates only the E9-E11 caches
NS_CODE_HASH = hashlib.sha1(b"".join(
    (ROOT / rel).read_bytes() for rel in (
        "simulator/nonstationary.py", "decoders/bayesian/adaptive.py",
        "experiments/common.py",
    ))).hexdigest()[:12]

# grid of noise models for the adaptive decoders (covers every test point
# except the deliberately extreme E11 values)
ADAPTIVE_GRID = parameter_grid(
    sigma_qs=(0.15 * L, 0.22 * L, 0.30 * L),
    rhos=(0.3, 0.7, 0.9, 0.97),
    sigma_ms=(0.05 * L,),
)
IMM_HAZARD = 0.005
WINDOWS = (50, 200)

# (point id, schedule spec, how the drift-trained LSTM relates to it)
#   "in"        shape and values inside the training mixture
#   "shape"     shape held out from training
#   "values"    shape seen in training, values outside its ranges
E9_POINTS = [
    ("control", spec("stationary"), "in"),
    ("sin_sq_P1000", spec("sinusoid", target="sigma_q", amp=0.3, period=1000.0), "in"),
    ("sin_sq_P200", spec("sinusoid", target="sigma_q", amp=0.3, period=200.0), "in"),
    ("sin_rho_P1000", spec("sinusoid", target="rho", rho=0.75, amp=0.1, period=1000.0), "in"),
    ("ramp_sq", spec("ramp", target="sigma_q", start=0.16 * L, end=0.26 * L), "shape"),
    ("onef_sq", spec("one_over_f", rel=0.25), "shape"),
]
E10_POINTS = [
    ("step_sq", spec("step", target="sigma_q", before=0.20 * L, after=0.26 * L), "in"),
    ("step_rho", spec("step", target="rho", before=0.9, after=0.3), "in"),
    ("telegraph_sq", spec("telegraph", target="sigma_q", levels=[0.20 * L, 0.26 * L],
                          p_switch=0.005), "shape"),
    ("telegraph_rho", spec("telegraph", target="rho", levels=[0.3, 0.97], p_switch=0.005),
     "shape"),
    ("burst", spec("burst", kick=0.40 * L), "shape"),
]
E11_POINTS = [
    ("sin_sq_amp0.6", spec("sinusoid", target="sigma_q", amp=0.6, period=1000.0), "values"),
    ("sin_sq_P50", spec("sinusoid", target="sigma_q", amp=0.3, period=50.0), "values"),
    ("step_sq_to0.34", spec("step", target="sigma_q", before=0.20 * L, after=0.34 * L),
     "values"),
    ("step_rho_to0", spec("step", target="rho", before=0.9, after=0.0), "values"),
    ("stationary_sq0.32", spec("stationary", sigma_q=0.32 * L), "values"),
]
POINTS = {"E9": E9_POINTS, "E10": E10_POINTS, "E11": E11_POINTS}


# decoder factories (same dict format as experiments/run_all.py)
def oracle():
    return dict(name="Bayes (oracle)", kind="bayes", spec=dict(oracle=True),
                make=lambda: OracleBayesDecoder(n_x=96, device=run_all.DEVICE))


def imm(hazard=IMM_HAZARD):
    return dict(name="Bayes-IMM", kind="bayes", spec=dict(grid=ADAPTIVE_GRID, hazard=hazard),
                make=lambda: RegimeBankDecoder(ADAPTIVE_GRID, hazard=hazard, n_x=96,
                                               device=run_all.DEVICE))


def window(w):
    return dict(name=f"Bayes-window {w}", kind="bayes",
                spec=dict(grid=ADAPTIVE_GRID, window=w),
                make=lambda: WindowedBayesDecoder(ADAPTIVE_GRID, window=w, n_x=96,
                                                  device=run_all.DEVICE))


def decoder_set(budget):
    nom = NOMINAL
    return [
        oracle(),                                   # reference (index 0)
        simple("NL"),
        bayes(nom["sigma_q"], nom["sigma_m"], nom["rho"], "Bayes (nominal)"),
        bayes_broad(),
        imm(),
        *[window(w) for w in WINDOWS],
        *lstm_family("lstm_nominal", "LSTM-nominal", budget.lstm_seeds),
        *lstm_family("lstm_broad", "LSTM-broad", budget.lstm_seeds),
        *lstm_family("lstm_drift", "LSTM-drift", budget.lstm_seeds),
    ]


def noise_spec(point_spec):
    return dict(kind="schedule", spec=point_spec, code=NS_CODE_HASH)


def run_points(exp):
    budget = get_budget()
    decs = decoder_set(budget)
    for i, (pid, point_spec, _) in enumerate(POINTS[exp]):
        evaluate_point(exp, i, pid, noise_spec(point_spec), decs, decs[0],
                       cap=budget.ns_max_test_trials, rounds=budget.ns_rounds,
                       min_events=10**9,            # fixed test-set size
                       chunk=budget.ns_chunk_trials)


def e9():
    run_points("E9")


def e10():
    run_points("E10")


def e11():
    run_points("E11")


EXPERIMENTS = dict(E9=e9, E10=e10, E11=e11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    run_all.DEVICE = torch_device()
    log(f"run_nonstationary: mode={get_mode()} device={run_all.DEVICE} "
        f"rounds={get_budget().ns_rounds} code={NS_CODE_HASH}")
    failures = []
    for name, fn in EXPERIMENTS.items():
        if args.only and name not in args.only:
            continue
        t0 = time.perf_counter()
        log(f"=== {name} ===")
        try:
            fn()
            log(f"=== {name} done in {time.perf_counter() - t0:.0f}s ===")
        except Exception:
            failures.append(name)
            log(f"=== {name} FAILED ===\n{traceback.format_exc()}")
    log(f"run_nonstationary finished; failed experiments: {failures or 'none'}; "
        f"failed decoders/points: {run_all.FAILURES or 'none'}")


if __name__ == "__main__":
    main()
