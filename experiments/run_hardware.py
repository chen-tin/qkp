"""E12: decoders at hardware-scale noise (built, not run).

The coverage map (docs/benchmarks.md section H) shows that no simulated
scenario covers the regime real cavities operate in: sigma_q ~ 0.072 L (loss
of a T1 = 610 us cavity over a 9.85 us cycle, Sivak et al. 2023) with
readout-dominated syndromes, sigma_m ~ 0.10 L. There, consecutive syndromes
are anti-correlated for any rho, so memory may help by averaging readout
noise rather than by tracking correlated drift. E12 tests every decoder there.

Points (all at sigma_q = 0.072 L unless the schedule varies it):
  stationary  sigma_m in {0.08, 0.105, 0.136} L x rho in {0, 0.5, 0.9, 0.97}
  drifting    sinusoid sigma_q (amp 0.3, P = 1000), telegraph sigma_m
              (0.08 <-> 0.136 L), step rho 0.9 -> 0.3, 1/f sigma_q (rel 0.12,
              the amplitude calibrated on measured T1 drift)
Decoders: oracle (reference), NL, Bayes fixed at the anchor (rho = 0.9),
Bayes at the nominal model (what our filters assume today), IMM and window
averages over a hardware-scale grid, LSTM-hw (trained on HW_DATASETS), and
the existing LSTM-nominal / LSTM-broad. GPU fast paths as in
run_nonstationary.py (QKP_FAST=0 for the reference implementations).

Score with experiments.common.eps_fit / paired_pl (not flip rate):
experiments/analyze_nonstationary.py works on these caches with POINTS
extended by HW_POINTS (see wiring in docs/hardware.md).

Run from the repo root (not run yet):
    python -m training.train_lstm         # after lstm_hw_s* are wired in
    python -m experiments.run_hardware
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
from experiments.run_all import NOMINAL, evaluate_point, simple
from experiments.run_nonstationary import FAST, IMPL, IMM_HAZARD, WINDOWS, lstm_fast_family
from simulator.noise import L
from simulator.nonstationary import spec

HW_CODE_HASH = hashlib.sha1(b"".join(
    (ROOT / rel).read_bytes() for rel in (
        "simulator/nonstationary.py", "decoders/bayesian/adaptive.py", "decoders/fast.py",
        "simulator/fast.py", "experiments/common.py", "experiments/run_hardware.py",
    ))).hexdigest()[:12]

# hardware anchor (docs/benchmarks.md sections B and H)
SQ_HW = 0.072 * L
SM_HW = 0.105 * L
SM_LEVELS = (0.08, 0.105, 0.136)
RHOS = (0.0, 0.5, 0.9, 0.97)


def hw(family, **kw):
    out = dict(family=family, sigma_q=SQ_HW, sigma_m=SM_HW, rho=0.9)
    out.update(kw)
    return out


HW_POINTS = (
    [(f"sm{sm}_rho{rho}", hw("stationary", sigma_m=sm * L, rho=rho), "stationary")
     for sm in SM_LEVELS for rho in RHOS]
    + [
        ("sin_sq_P1000", hw("sinusoid", target="sigma_q", amp=0.3, period=1000.0), "drift"),
        ("telegraph_sm", hw("telegraph", target="sigma_m", levels=[0.08 * L, 0.136 * L],
                            p_switch=0.005), "drift"),
        ("step_rho", hw("step", target="rho", before=0.9, after=0.3), "drift"),
        ("onef_sq", hw("one_over_f", rel=0.12), "drift"),
    ]
)
POINTS = {"E12": HW_POINTS}

# model grid for the adaptive decoders at hardware scale
HW_GRID = parameter_grid(sigma_qs=(0.05 * L, 0.072 * L, 0.10 * L), rhos=RHOS,
                         sigma_ms=(SM_HW,))
ANCHOR = dict(sigma_q=SQ_HW, sigma_m=SM_HW, rho=0.9)

# training data for the hardware-scale LSTM (same format as training/datasets.py)
_SQ = [0.05 * L, 0.10 * L]
_SM = [0.08 * L, 0.14 * L]
HW_TRAIN_MIX = [
    (2.0, dict(family="stationary", sigma_q=_SQ, rho=[0.0, 0.97], sigma_m=_SM)),
    (1.0, dict(family="sinusoid", target="sigma_q", sigma_q=_SQ, amp=[0.1, 0.4],
               period=[100.0, 1000.0], rho=[0.0, 0.97], sigma_m=_SM)),
    (1.0, dict(family="telegraph", target="sigma_m", levels=[0.08 * L, 0.14 * L],
               p_switch=0.005, sigma_q=_SQ, rho=[0.0, 0.97], sigma_m=SM_HW)),
    (1.0, dict(family="step", target="rho", before=[0.6, 0.97], after=[0.0, 0.6],
               sigma_q=_SQ, rho=0.9, sigma_m=_SM)),
]
HW_DATASETS = {"hw": dict(kind="schedule_mix", families=HW_TRAIN_MIX, long=True)}


def hw_model_list(budget):
    """(model name, dataset name, seed) entries for training/train_lstm.model_list."""
    return [(f"lstm_hw_s{s}", "hw", s) for s in range(budget.lstm_seeds)]


# decoder factories (run_all dict format)
def _fast_or_ref(name, spec_, fast, ref):
    return dict(name=name, kind="bayes", spec=dict(spec_, impl=IMPL), make=fast if FAST else ref)


def oracle():
    from decoders.fast import FastOracle
    ref = lambda: OracleBayesDecoder(n_x=96, device=run_all.DEVICE)
    fast = lambda: FastOracle(n_x=96, device=run_all.DEVICE, reference=ref, name="Bayes (oracle)")
    return _fast_or_ref("Bayes (oracle)", dict(oracle=True), fast, ref)


def fixed(model, name):
    from decoders.bayesian.grid_filter import BayesGridDecoder
    from decoders.fast import FastGridDecoder
    ref = lambda: BayesGridDecoder(**model, n_x=96, name=name, device=run_all.DEVICE)
    fast = lambda: FastGridDecoder([dict(model)], n_x=96, device=run_all.DEVICE, reference=ref,
                                   name=name)
    return _fast_or_ref(name, dict(model), fast, ref)


def imm(hazard=IMM_HAZARD):
    from decoders.fast import FastGridDecoder
    ref = lambda: RegimeBankDecoder(HW_GRID, hazard=hazard, n_x=96, device=run_all.DEVICE)
    fast = lambda: FastGridDecoder(HW_GRID, n_x=96, device=run_all.DEVICE, joint=True,
                                   hazard=hazard, reference=ref, name="Bayes-IMM (hw grid)")
    return _fast_or_ref("Bayes-IMM (hw grid)", dict(grid=HW_GRID, hazard=hazard), fast, ref)


def window(w):
    from decoders.fast import FastGridDecoder
    name = f"Bayes-window {w} (hw grid)"
    ref = lambda: WindowedBayesDecoder(HW_GRID, window=w, n_x=96, device=run_all.DEVICE)
    fast = lambda: FastGridDecoder(HW_GRID, n_x=96, device=run_all.DEVICE, window=w,
                                   reference=ref, name=name)
    return _fast_or_ref(name, dict(grid=HW_GRID, window=w), fast, ref)


def decoder_set(budget):
    return [
        oracle(),                                    # reference (index 0)
        simple("NL"),
        fixed(ANCHOR, "Bayes (hw anchor)"),
        fixed(dict(NOMINAL), "Bayes (nominal)"),
        imm(),
        *[window(w) for w in WINDOWS],
        *lstm_fast_family("lstm_hw", "LSTM-hw", budget.lstm_seeds),
        *lstm_fast_family("lstm_nominal", "LSTM-nominal", budget.lstm_seeds),
        *lstm_fast_family("lstm_broad", "LSTM-broad", budget.lstm_seeds),
    ]


def noise_spec(point_spec):
    return dict(kind="schedule", spec=point_spec, code=HW_CODE_HASH)


def e12(points=None):
    budget = get_budget()
    decs = decoder_set(budget)
    for i, (pid, point_spec, _) in enumerate(points or HW_POINTS):
        evaluate_point("E12", i, pid, noise_spec(point_spec), decs, decs[0],
                       cap=budget.ns_max_test_trials, rounds=budget.ns_rounds,
                       min_events=10**9, chunk=budget.ns_chunk_trials)


def main():
    argparse.ArgumentParser().parse_args()
    run_all.DEVICE = torch_device()
    log(f"run_hardware: mode={get_mode()} device={run_all.DEVICE} impl={IMPL} code={HW_CODE_HASH}")
    from experiments.runlog import RunLog
    with RunLog("run_hardware", args=dict(impl=IMPL, code=HW_CODE_HASH)) as rl:
        t0 = time.perf_counter()
        try:
            with rl.phase("E12"):
                e12()
            log(f"=== E12 done in {time.perf_counter() - t0:.0f}s ===")
        except Exception:
            log(f"=== E12 FAILED ===\n{traceback.format_exc()}")
        rl.note(failed_decoders=list(run_all.FAILURES))


if __name__ == "__main__":
    main()
