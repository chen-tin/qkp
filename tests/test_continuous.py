"""Continuous-correction Bayes decoder and the E15 pipeline."""

import os

os.environ.setdefault("QKP_MODE", "quick")

import numpy as np
import torch

from decoders.bayesian.continuous import ContinuousBayesDecoder
from decoders.bayesian.grid_filter import BayesGridDecoder
from experiments.common import make_rng, paired_pl
from simulator import closed_loop as cl
from simulator.noise import L, NoiseModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_parity_policy_equals_matched_bayes():
    d, n = NoiseModel().sample(make_rng("dev", "test_cont", 0), 600, 40)
    ref = cl.run(cl.ParityAdapter(BayesGridDecoder(0.2 * L, 0.05 * L, 0.9, n_x=96,
                                                   device=DEVICE)), d, n)
    res = cl.run(ContinuousBayesDecoder(0.2 * L, 0.05 * L, 0.9, policy="parity", n_x=96,
                                        device=DEVICE), d, n)
    assert np.array_equal(ref.logical_errors, res.logical_errors)
    assert np.allclose(ref.corrections, res.corrections)


def test_fine_correction_not_worse_and_centres_the_residual():
    kw = dict(sigma_q=0.072 * L, sigma_m=0.15 * L, rho=0.9)
    d, n = NoiseModel(**kw).sample(make_rng("dev", "test_cont", 1), 2000, 60)
    par = cl.run(ContinuousBayesDecoder(**kw, policy="parity", n_x=96, device=DEVICE), d, n)
    fine = cl.run(ContinuousBayesDecoder(**kw, policy="fine", n_x=96, device=DEVICE), d, n)
    diff, se = paired_pl(fine.logical_errors, par.logical_errors, 60)
    assert diff <= 3 * se + 1e-12
    # the fine policy leaves the oscillator closer to a lattice point
    off = lambda r: np.abs(r.residuals - L * np.rint(r.residuals / L)).mean()
    assert off(fine) < off(par)


def test_e15_pipeline_end_to_end(tmp_path, monkeypatch):
    from experiments import common
    from experiments import run_continuous as rc
    from rl import train as rltrain
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setitem(rc.SIZES, "quick", dict(trials=40, rounds=20))
    monkeypatch.setitem(rltrain.SIZES, "quick", dict(batch=16, rounds=20, steps=60,
                                                       batch_size=64, time_cap_s=30.0))
    monkeypatch.setattr(rc, "POINTS", {"hw": rc.POINTS["hw"]})
    rltrain.train_one("hw", rc.POINTS["hw"], "parity", device=DEVICE)
    rc.e15(DEVICE)
    rows = rc.analyze()
    names = {r["decoder"] for r in rows}
    assert {"NL", "Bayes (parity)", "Bayes (fine)", "DQN (parity)"} <= names
    assert "DQN (fine)" not in names                     # not trained: deferred
    assert all(r["trials"] == 40 for r in rows)
