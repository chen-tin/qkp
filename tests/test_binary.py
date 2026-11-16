"""One-bit syndrome decoders (E14) in the closed loop."""

import os

os.environ.setdefault("QKP_MODE", "quick")
os.environ.setdefault("MPLBACKEND", "Agg")

import dataclasses

import numpy as np
import torch

from decoders.bayesian.binary_filter import (
    BinaryBayesDecoder, PhasedBits, SBSDecoder, default_phases, make_measurement, unpack,
)
from experiments import common
from experiments.common import make_rng
from simulator import closed_loop as cl
from simulator.noise import L, NoiseModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_packed_bits_statistics_and_unpacking():
    meas = PhasedBits(phases=default_phases(3), visibility=1.0)
    rng = np.random.default_rng(0)
    x = np.full(100000, 0.1 * L)                   # just above a lattice point
    b = unpack(meas(x, np.zeros_like(x), rng), 3)
    p_sin = 0.5 * (1 + np.sin(2 * np.pi * 0.1))     # P(b=1) for the sin-type bit
    p_cos = 0.5 * (1 - np.cos(2 * np.pi * 0.1))
    assert abs(b[:, 0].mean() - p_sin) < 5e-3 and abs(b[:, 2].mean() - p_sin) < 5e-3
    assert abs(b[:, 1].mean() - p_cos) < 5e-3


def test_filter_holds_the_state_at_tiny_noise():
    d, n = NoiseModel(0.01 * L, 0.0, 0.9).sample(make_rng("dev", "test_bin", 0), 500, 100)
    dec = BinaryBayesDecoder(0.01 * L, 0.0, 0.9, device=DEVICE)
    res = cl.run(dec, d, n, make_measurement(1.0, 0.0), np.random.default_rng(1))
    assert res.logical_errors.sum() == 0
    centred = np.remainder(res.residuals[:, -1] + L, 2 * L) - L
    assert np.abs(centred).max() < 0.35 * L


def test_bayes_not_worse_than_sbs_with_memory():
    sq = 0.05 * L
    d, n = NoiseModel(sq, 0.0, 0.9).sample(make_rng("dev", "test_bin", 1), 1500, 120)
    ph = default_phases(4)
    meas = make_measurement(1.0, 0.0, m=4)
    rec = {}
    for dec in (SBSDecoder(0.08 * L, ph), SBSDecoder(0.15 * L, ph),
                BinaryBayesDecoder(sq, 0.0, 0.9, phases=ph, device=DEVICE)):
        rec[dec.name] = cl.run(dec, d, n, meas, np.random.default_rng(2)).logical_errors[:, -1].mean()
    best_sbs = min(v for k, v in rec.items() if k.startswith("sBs"))
    assert rec["Bayes (binary, matched)"] <= best_sbs + 0.02, rec


def test_closed_loop_is_deterministic_with_fixed_seeds():
    d, n = NoiseModel(0.1 * L, 0.05 * L, 0.9).sample(make_rng("dev", "test_bin", 3), 200, 30)
    meas = make_measurement(0.95, 0.01, m=2, sigma_m=0.05 * L)
    runs = [cl.run(BinaryBayesDecoder(0.1 * L, 0.05 * L, 0.9, visibility=0.95, p_flip=0.01,
                                      phases=default_phases(2), device=DEVICE),
                   d, n, meas, np.random.default_rng(9)) for _ in range(2)]
    assert np.array_equal(runs[0].observations, runs[1].observations)
    assert np.array_equal(runs[0].corrections, runs[1].corrections)


def test_binary_lstm_trains_and_decodes(tmp_path, monkeypatch):
    from decoders.binary_lstm import BinaryLSTMDecoder, load
    from training import train_binary
    tiny = dataclasses.replace(common.BUDGETS["quick"], n_train=64, n_val=32, batch_size=32,
                               max_epochs=1, hidden=16)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    path = train_binary.train_one("binary_lstm_hardware_m4_s0", "hardware", 4, 0, tiny, DEVICE)
    model, ck = load(path, DEVICE)
    assert ck["m"] == 4
    d, n = NoiseModel(0.072 * L, 0.105 * L, 0.9).sample(make_rng("dev", "test_bin", 4), 50, 20)
    res = cl.run(BinaryLSTMDecoder(model, device=DEVICE), d, n,
                 make_measurement(1.0, 0.0, m=4, sigma_m=0.105 * L), np.random.default_rng(0))
    assert np.all(np.isfinite(res.corrections)) and np.all(np.abs(res.corrections) <= L)


def test_run_binary_end_to_end_one_point(tmp_path, monkeypatch):
    from experiments import run_binary
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setitem(run_binary.BINARY_BUDGET, "quick", dict(trials=40, rounds=15))
    pts = run_binary.points()
    hw = next(p for p in pts if p["id"] == "hardware_rho0.9_v1.0_pf0.0_m4")
    monkeypatch.setattr(run_binary, "points", lambda: [hw])
    rows = run_binary.e14()
    names = {r["decoder"] for r in rows}
    assert {"Bayes (binary, matched)", "Bayes (analog, matched)", "sBs delta=0.15L"} <= names
    assert (tmp_path / "results" / "quick" / "E14_summary.json").exists()
    assert len(pts) == 24
