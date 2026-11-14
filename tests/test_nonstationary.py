"""Tests for the nonstationary noise models, decoders and E9-E11 pipeline.

Run from the repo root:  QKP_MODE=quick .venv/bin/python -m pytest -q
"""

import os

os.environ.setdefault("QKP_MODE", "quick")
os.environ.setdefault("MPLBACKEND", "Agg")

import dataclasses

import numpy as np
import pytest
import torch

from decoders.bayesian.adaptive import (
    OracleBayesDecoder, RegimeBankDecoder, WindowedBayesDecoder, parameter_grid,
)
from decoders.bayesian.grid_filter import BayesGridDecoder
from experiments import common
from experiments.common import (
    aligned_flip_rate, flips_per_trial, make_rng, paired_difference, run_blocked,
    windowed_flip_rate,
)
from simulator import nonstationary as ns
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice, Simulator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rng(k=0):
    return make_rng("dev", "tests_ns", k)


# noise models
def test_stationary_schedule_matches_noise_model():
    d, n, sch = ns.sample(ns.spec("stationary"), rng(1), 40000, 30)
    m = NoiseModel()
    assert abs(d.std() / m.sigma_q - 1) < 0.01
    assert abs(n.std() / m.sigma_m - 1) < 0.01
    c = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
    assert abs(c - m.rho) < 0.01
    assert np.all(sch.kick == 0) and np.all(sch.tau == -1)


def test_sinusoid_variance_and_correlation_follow_the_schedule():
    sp = ns.spec("sinusoid", target="sigma_q", amp=0.4, period=40.0, phase=0.0)
    d, _, sch = ns.sample(sp, rng(2), 40000, 40)
    assert np.allclose(sch.sigma_q, sch.sigma_q[0])          # fixed phase: same for all
    for t in (5, 10, 25, 30):
        assert abs(d[:, t].std() / sch.sigma_q[0, t] - 1) < 0.03
        c = np.corrcoef(d[:, t - 1], d[:, t])[0, 1]
        assert abs(c - 0.9) < 0.02


def test_rho_sinusoid_correlation():
    sp = ns.spec("sinusoid", target="rho", rho=0.6, amp=0.3, period=20.0, phase=0.0)
    d, _, sch = ns.sample(sp, rng(3), 40000, 20)
    for t in (5, 15):
        c = np.corrcoef(d[:, t - 1], d[:, t])[0, 1]
        assert abs(c - sch.rho[0, t]) < 0.02


def test_step_telegraph_burst_shapes():
    T = 100
    _, _, s = ns.sample(ns.spec("step", target="sigma_q", before=0.2 * L, after=0.3 * L),
                        rng(4), 500, T)
    assert np.all((s.tau >= 25) & (s.tau < 75))
    for i in range(20):
        assert np.allclose(s.sigma_q[i, :s.tau[i]], 0.2 * L)
        assert np.allclose(s.sigma_q[i, s.tau[i]:], 0.3 * L)
    _, _, s = ns.sample(ns.spec("telegraph", target="rho", levels=[0.3, 0.97], p_switch=0.05),
                        rng(5), 500, T)
    assert set(np.unique(s.rho)) == {0.3, 0.97}
    _, _, s = ns.sample(ns.spec("burst", kick=0.4 * L, duration=2), rng(6), 500, T)
    assert np.all((s.kick > 0).sum(axis=1) == 2)
    assert np.all(s.kick[np.arange(500), s.tau] == 0.4 * L)


def test_ranges_are_drawn_per_trial_and_one_over_f_is_bounded():
    _, _, s = ns.sample(ns.spec("stationary", sigma_q=[0.1 * L, 0.3 * L]), rng(7), 2000, 5)
    assert s.sigma_q[:, 0].min() < 0.12 * L and s.sigma_q[:, 0].max() > 0.28 * L
    _, _, s = ns.sample(ns.spec("one_over_f", rel=0.25), rng(8), 2000, 200)
    assert s.sigma_q.max() < 0.5 * L and s.sigma_q.min() > 0.05 * L
    with pytest.raises(ValueError):
        ns.make_schedule(dict(family="nope"), rng(0), 2, 2)


def test_mixture_keeps_trial_order():
    from training.datasets import DRIFT_TRAIN_MIX
    d, n, s, which = ns.sample_mixture(DRIFT_TRAIN_MIX, rng(9), 3000, 60)
    assert d.shape == n.shape == s.sigma_q.shape == (3000, 60)
    assert set(np.unique(which)) == set(range(len(DRIFT_TRAIN_MIX)))
    steps = np.flatnonzero(which == 3)                 # step in sigma_q
    assert np.all(s.tau[steps] >= 0)
    assert np.all(s.tau[which == 0] == -1)             # stationary: no event


# decoders
def _stationary_case(n=3000, T=60, seed=10):
    return ns.sample(ns.spec("stationary"), rng(seed), n, T)


def test_oracle_with_constant_schedule_equals_matched_bayes():
    d, n, sch = _stationary_case()
    m = NoiseModel()
    sim = Simulator(trials=d.shape[0], rounds=d.shape[1])
    a = sim.run(BayesGridDecoder(m.sigma_q, m.sigma_m, m.rho, n_x=96, device=DEVICE), d, n)
    o = OracleBayesDecoder(device=DEVICE)
    o.set_schedule(sch)
    b = sim.run(o, d, n)
    assert (a.logical_errors == b.logical_errors).mean() > 0.999


def test_single_model_adaptive_decoders_equal_matched_bayes():
    d, n, _ = _stationary_case(n=1500)
    m = NoiseModel()
    reg = [dict(sigma_q=m.sigma_q, rho=m.rho, sigma_m=m.sigma_m)]
    sim = Simulator(trials=d.shape[0], rounds=d.shape[1])
    a = sim.run(BayesGridDecoder(m.sigma_q, m.sigma_m, m.rho, n_x=96, device=DEVICE), d, n)
    for dec in (RegimeBankDecoder(reg, hazard=0.0, device=DEVICE),
                RegimeBankDecoder(reg, hazard=0.3, device=DEVICE),   # one regime: no effect
                WindowedBayesDecoder(reg, window=10, device=DEVICE),
                WindowedBayesDecoder(reg, forgetting=0.9, device=DEVICE)):
        b = sim.run(dec, d, n)
        assert (a.logical_errors == b.logical_errors).mean() > 0.999, dec.name


def test_oracle_and_imm_beat_fixed_model_after_a_step():
    sp = ns.spec("step", target="sigma_q", before=0.2 * L, after=0.32 * L)
    d, n, sch = ns.sample(sp, rng(11), 2000, 150)
    m = NoiseModel()
    o = OracleBayesDecoder(device=DEVICE)
    rec_o = run_blocked(o, d, n, 2000, schedule=sch)
    rec_f = run_blocked(BayesGridDecoder(m.sigma_q, m.sigma_m, m.rho, n_x=96, device=DEVICE),
                        d, n, 2000)
    grid = parameter_grid([0.15 * L, 0.22 * L, 0.32 * L], [0.7, 0.9, 0.97], [0.05 * L])
    rec_i = run_blocked(RegimeBankDecoder(grid, hazard=0.005, device=DEVICE), d, n, 2000)
    fo, ff, fi = (flips_per_trial(r) for r in (rec_o, rec_f, rec_i))
    diff, se = paired_difference(fo, ff, 150)
    assert diff < 0 and abs(diff) > 2 * se             # oracle strictly better
    assert fi.mean() <= ff.mean() * 1.02               # IMM no worse than the fixed model


def test_oracle_handles_bursts_and_block_slicing():
    d, n, sch = ns.sample(ns.spec("burst", kick=0.4 * L), rng(12), 1200, 50)
    full = run_blocked(OracleBayesDecoder(device=DEVICE), d, n, 1200, schedule=sch)
    blocks = run_blocked(OracleBayesDecoder(device=DEVICE), d, n, 400, schedule=sch)
    assert np.array_equal(full, blocks)
    with pytest.raises(ValueError):
        run_blocked(OracleBayesDecoder(device=DEVICE), d, n, 1200)   # no schedule


def test_oracle_rejects_schedules_above_its_grid_hint():
    _, _, sch = ns.sample(ns.spec("stationary", sigma_q=0.6 * L), rng(13), 10, 5)
    with pytest.raises(ValueError):
        OracleBayesDecoder(device=DEVICE).set_schedule(sch)


# statistics helpers
def test_windowed_and_aligned_flip_rates():
    le = np.zeros((4, 20), dtype=bool)
    le[:, 10:] = True                                  # one flip at round 10
    starts, rate, _ = windowed_flip_rate(le, 5)
    assert list(starts) == [0, 5, 10, 15] and np.allclose(rate, [0, 0, 0.2, 0])
    offsets, rate, count = aligned_flip_rate(le, np.full(4, 10), 2, 3)
    assert list(offsets) == [-2, -1, 0, 1, 2] and np.allclose(rate, [0, 0, 1, 0, 0])
    assert np.all(count == 4)


# whole pipeline on a toy budget, in a temporary directory
def test_pipeline_end_to_end(tmp_path, monkeypatch):
    from experiments import analyze, run_all, run_nonstationary
    from training import train_lstm

    tiny = dataclasses.replace(
        common.BUDGETS["quick"], ns_rounds=40, ns_chunk_trials=100, ns_max_test_trials=200,
        ns_n_train=128, ns_n_val=64, ns_batch_size=32, max_epochs=1, hidden=16,
        lstm_seeds=1, train_time_cap_s=60.0,
    )
    monkeypatch.setitem(common.BUDGETS, "quick", tiny)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(run_all, "DEVICE", DEVICE)

    # train only the drift model; the others stay missing and are deferred
    path = train_lstm.train_one("lstm_drift_s0", "drift", 0, tiny, DEVICE)
    assert path is not None and path.exists()

    pid, point_spec, _ = run_nonstationary.E10_POINTS[0]          # step in sigma_q
    monkeypatch.setitem(run_nonstationary.POINTS, "E10", [run_nonstationary.E10_POINTS[0]])
    run_nonstationary.e10()
    cached = {p.stem for p in (tmp_path / "results" / "quick" / "cache" / "E10" / pid).glob("*.npz")}
    for name in ("Bayes_(oracle)", "NL", "Bayes_(nominal)", "Bayes-broad", "Bayes-IMM",
                 "Bayes-window_50", "Bayes-window_200", "LSTM-drift_s0"):
        assert name in cached, (name, cached)

    # analysis reads the cache it just wrote
    from experiments import analyze_nonstationary as an
    res = tmp_path / "results" / "quick"
    monkeypatch.setattr(analyze, "RES", res)
    monkeypatch.setattr(an, "RES", res)
    monkeypatch.setattr(an, "FIG", res / "figures")
    (res / "figures").mkdir(parents=True, exist_ok=True)
    rows = an.analyze("E10")
    names = {r["decoder"] for r in rows}
    assert {"Bayes (oracle)", "LSTM-drift"} <= names
    oracle = next(r for r in rows if r["decoder"] == "Bayes (oracle)")
    assert oracle["trials"] == 200 and oracle["rounds"] == 40
    assert (res / "figures" / "E10_aligned_on_event.png").exists()
