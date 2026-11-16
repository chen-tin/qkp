"""Tests for the E13 learned-decoder infrastructure (decoders/learned.py,
training/train_variants.py, experiments/run_learned.py)."""

import os

os.environ.setdefault("QKP_MODE", "quick")
os.environ.setdefault("MPLBACKEND", "Agg")

import dataclasses

import numpy as np
import pytest
import torch

from decoders.learned import (
    FastSeq, ParityGRU, ParityTransformer, SeqDecoder, build, load_model, matched_kwargs, n_params,
)
from decoders.LSTM.model import HEADS, ParityLSTM, cumulative_from_head
from experiments.common import make_rng
from simulator.noise import NoiseModel
from simulator.sim import NearestLattice, Simulator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def noise(n=200, rounds=30, seed=0):
    return NoiseModel().sample(make_rng("dev", "test_learned", seed), n, rounds)


@pytest.mark.parametrize("model", [ParityGRU(hidden=16), ParityTransformer(hidden=16, layers=2,
                                                                             heads=4)])
def test_forward_shapes_and_incremental_state(model):
    model = model.to(DEVICE).eval()
    s = torch.randn(5, 12, device=DEVICE)
    with torch.no_grad():
        full, _ = model(s)
        assert full.shape == (5, 12, len(HEADS))
        # feeding one round at a time with state gives the same logits
        state, steps = None, []
        for t in range(12):
            out, state = model(s[:, t:t + 1], state)
            steps.append(out)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


def test_transformer_is_causal():
    model = ParityTransformer(hidden=16, layers=1, heads=4).to(DEVICE).eval()
    s = torch.randn(2, 10, device=DEVICE)
    s2 = s.clone()
    s2[:, 6:] = torch.randn(2, 4, device=DEVICE)
    with torch.no_grad():
        a, _ = model(s)
        b, _ = model(s2)
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-5)


@pytest.mark.parametrize("arch", ["gru", "transformer"])
def test_closed_loop_equals_open_loop_and_fast_path(arch):
    torch.manual_seed(0)
    model = build(arch, hidden=16, layers=2, **({"heads": 4} if arch == "transformer" else {}))
    model = model.to(DEVICE).eval()
    d, n = noise()
    sim = Simulator(trials=200, rounds=30)
    nl = sim.run(NearestLattice(), d, n)
    s = torch.tensor(nl.syndromes, dtype=torch.float32, device=DEVICE)
    K = torch.tensor(nl.logical_errors.astype(np.int64), device=DEVICE)
    with torch.no_grad():
        logits, _ = model(s)
    from decoders.fast import run as fast_run
    for i, (head, k) in enumerate(HEADS):
        w_hat = (logits[..., i] > 0).long()
        open_loop = (K ^ cumulative_from_head(w_hat, k)).bool().cpu().numpy()
        closed = sim.run(SeqDecoder(model, head=head, device=DEVICE), d, n).logical_errors
        fast = fast_run(FastSeq(model, head, device=DEVICE), d, n)
        assert np.array_equal(open_loop, closed), (arch, head)
        assert np.array_equal(closed, fast), (arch, head)


def test_parameter_matching():
    target = n_params(ParityLSTM(128, 2))
    for arch in ("gru", "transformer"):
        kw, p = matched_kwargs(arch, target)
        assert abs(p - target) / target < 0.05, (arch, kw, p)


def test_existing_lstm_checkpoint_loads():
    from experiments.common import out_dir
    paths = sorted(out_dir("checkpoints", mode="full").glob("lstm_nominal_s0.pt"))
    if not paths:
        pytest.skip("no full-mode LSTM checkpoint present")
    model, ck = load_model(paths[0], DEVICE)
    assert isinstance(model, ParityLSTM) and ck.get("arch", "lstm") == "lstm"


def test_variant_grid_and_tiny_training(tmp_path, monkeypatch):
    from experiments import common
    from training import train_variants as tv
    grid = tv.variant_grid()
    names = [v["name"] for v in grid]
    assert len(names) == len(set(names))
    assert {v["group"] for v in grid} == {"size", "data", "curriculum", "arch"}
    assert len(tv.seed_extensions()) == 6

    tiny = dataclasses.replace(common.BUDGETS["quick"], n_val=64, batch_size=64, max_epochs=2,
                               patience=2, train_time_cap_s=60.0)
    monkeypatch.setitem(common.BUDGETS, "quick", tiny)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    for v in (tv._v("transformer", dict(hidden=16, layers=1, heads=4), n_train=60_000),
              tv._v("lstm", dict(hidden=16, layers=2), n_train=60_000, curriculum="rare")):
        path = tv.train_variant(v, tiny, DEVICE)
        assert path is not None and path.exists()
        model, ck = load_model(path, DEVICE)
        assert ck["arch"] == v["arch"] and ck["chosen_head"] in dict(HEADS)
        if v["curriculum"] == "rare":
            assert ck["history"][0]["frac_elevated"] == 0.5
        # second call is a cache hit
        assert tv.train_variant(v, tiny, DEVICE) == path
