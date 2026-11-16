"""E12 (hardware-scale) infrastructure: specs, decoders, training data, tiny pipeline."""

import os

os.environ.setdefault("QKP_MODE", "quick")
os.environ.setdefault("MPLBACKEND", "Agg")

import dataclasses

import numpy as np
import torch

from experiments import common, run_hardware as hw
from experiments.common import make_rng
from simulator import nonstationary as ns
from simulator.noise import L

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_point_specs_generate_valid_schedules():
    assert len(hw.HW_POINTS) == 16
    for k, (pid, sp, _) in enumerate(hw.HW_POINTS):
        sch = ns.make_schedule(sp, make_rng("dev", "test_hw", k), 50, 200)
        assert sch.max_sigma() < 0.5 * L                    # within the oracle's grid hint
        assert np.all(sch.sigma_m > 0)
    sch = ns.make_schedule(dict(hw.HW_POINTS[-3][1]), make_rng("dev", "test_hw", 99), 200, 400)
    assert set(np.round(np.unique(sch.sigma_m) / L, 3)) == {0.08, 0.136}   # sigma_m drift


def test_training_mixture_samples():
    d, n, sch, which = ns.sample_mixture(hw.HW_TRAIN_MIX, make_rng("dev", "test_hw_mix", 0),
                                         400, 120)
    assert d.shape == (400, 120) and set(np.unique(which)) == set(range(len(hw.HW_TRAIN_MIX)))
    assert sch.sigma_q.max() <= 0.1 * L * 1.4 + 1e-12   # sinusoid amp <= 0.4
    assert sch.sigma_m.min() >= 0.08 * L * 0.9999


def test_decoders_construct():
    from experiments import run_all
    run_all.DEVICE = DEVICE
    decs = hw.decoder_set(common.get_budget())
    names = [d["name"] for d in decs]
    assert names[0] == "Bayes (oracle)" and "Bayes (hw anchor)" in names
    for d in decs:
        if d["kind"] != "lstm":
            d["make"]()


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    from experiments import run_all
    from training import datasets, train_lstm

    tiny = dataclasses.replace(
        common.BUDGETS["quick"], ns_rounds=40, ns_chunk_trials=100, ns_max_test_trials=100,
        ns_n_train=128, ns_n_val=64, ns_batch_size=32, max_epochs=1, hidden=16,
        lstm_seeds=1, train_time_cap_s=60.0)
    monkeypatch.setitem(common.BUDGETS, "quick", tiny)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(run_all, "DEVICE", DEVICE)
    for k, v in hw.HW_DATASETS.items():
        monkeypatch.setitem(datasets.DATASETS, k, v)

    name, dataset, seed = hw.hw_model_list(tiny)[0]
    assert train_lstm.train_one(name, dataset, seed, tiny, DEVICE).exists()

    pid = "step_rho"
    point = [p for p in hw.HW_POINTS if p[0] == pid]
    hw.e12(points=point)
    cached = {p.stem for p in (tmp_path / "results" / "quick" / "cache" / "E12" / pid).glob("*.npz")}
    for expected in ("Bayes_(oracle)", "NL", "Bayes_(hw_anchor)", "Bayes_(nominal)",
                     "Bayes-IMM_(hw_grid)", "Bayes-window_50_(hw_grid)", "LSTM-hw_s0"):
        assert expected in cached, (expected, cached)
