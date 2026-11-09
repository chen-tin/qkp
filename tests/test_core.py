"""Correctness tests for the decoder-comparison experiment.

Run from the repo root:  QKP_MODE=quick .venv/bin/python -m pytest -q
"""

import os

os.environ.setdefault("QKP_MODE", "quick")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
import torch

from decoders.base import ParityDecoder
from decoders.bayesian.grid_filter import BayesGridDecoder, MultiModelBayesDecoder, block_size
from decoders.LSTM.model import HEADS, LSTMDecoder, ParityLSTM, cumulative_from_head, head_labels
from experiments.common import (
    SPLITS, make_rng, paired_difference, run_blocked, seed_sequence, summarize, wilson,
)
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice, Simulator, wrap
from simulator.switching import SwitchingNoiseModel, sample_varied

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def noise(n=4000, rounds=50, seed=0, **kw):
    model = NoiseModel(**kw) if kw else NoiseModel()
    return model.sample(make_rng("dev", "tests", seed), n, rounds)


class RandomParity(ParityDecoder):
    name = "random parity"

    def _reset(self, trials):
        self.rng = np.random.default_rng(1)

    def choose(self, s):
        return self.rng.integers(0, 2, size=s.shape), np.full(s.shape, 0.5)


class CheatK(ParityDecoder):
    """Knows the true nearest-lattice logical state K_t (labels only)."""
    name = "cheat"

    def __init__(self, K):
        super().__init__()
        self.K = K

    def _reset(self, trials):
        self.t = 0

    def choose(self, s):
        j = self.K[:, self.t] ^ self.applied
        self.t += 1
        return j, j.astype(float)


# the reduction the whole design relies on
def test_syndromes_do_not_depend_on_parity_choices():
    d, n = noise()
    sim = Simulator(trials=d.shape[0], rounds=d.shape[1])
    a = sim.run(NearestLattice(), d, n)
    b = sim.run(RandomParity(), d, n)
    assert np.allclose(wrap(a.syndromes - b.syndromes), 0, atol=1e-9)


def test_logical_state_is_parity_of_mistakes():
    d, n = noise()
    sim = Simulator(trials=d.shape[0], rounds=d.shape[1])
    nl = sim.run(NearestLattice(), d, n)
    rnd = RandomParity(record=True)
    r = sim.run(rnd, d, n)
    j, _ = rnd.recorded()
    K = nl.logical_errors.astype(np.int8)
    J = np.cumsum(j, axis=1) % 2
    assert np.array_equal(r.logical_errors, (K ^ J).astype(bool))


def test_cheat_decoder_has_zero_logical_error():
    d, n = noise()
    sim = Simulator(trials=d.shape[0], rounds=d.shape[1])
    K = sim.run(NearestLattice(), d, n).logical_errors.astype(np.int8)
    r = sim.run(CheatK(K), d, n)
    assert r.logical_errors.sum() == 0


def test_all_lstm_heads_with_true_labels_give_zero_error():
    d, n = noise()
    sim = Simulator(trials=d.shape[0], rounds=d.shape[1])
    y = torch.tensor(sim.run(NearestLattice(), d, n).lattice_jumps() % 2)
    labels = head_labels(y).long()
    K = labels[..., -1]
    for i, (_, k) in enumerate(HEADS):
        assert torch.equal(cumulative_from_head(labels[..., i], k), K)


# Bayes filter
def test_bayes_matches_or_beats_nearest_lattice_at_rho0():
    d, n = noise(n=10000, rho=0.0)
    nl = run_blocked(NearestLattice(), d, n, 10000)
    m = NoiseModel(rho=0.0)
    bd = BayesGridDecoder(m.sigma_q, m.sigma_m, 0.0, n_x=96, device=DEVICE)
    by = run_blocked(bd, d, n, block_size(bd, 0.5e9))
    diff, se = paired_difference(by.sum(1), nl.sum(1), 50)   # time-averaged error
    assert diff <= 2 * se
    assert bd.resets == 0


def test_bayes_beats_nearest_lattice_when_memory_helps():
    kw = dict(sigma_q=0.2 * L, sigma_m=0.01 * L, rho=0.99)
    d, n = noise(n=4000, **kw)
    nl = run_blocked(NearestLattice(), d, n, 4000)
    bd = BayesGridDecoder(kw["sigma_q"], kw["sigma_m"], kw["rho"], n_x=96, device=DEVICE)
    by = run_blocked(bd, d, n, block_size(bd, 0.5e9))
    diff, se = paired_difference(by.sum(1), nl.sum(1), 50)
    assert diff < -5 * se
    assert by.mean() < 0.5 * nl.mean()


def test_bayes_frozen_limit_matches_closed_form():
    # sigma_m -> 0, rho -> 1: the error is frozen, so a single wrong cell
    # decision in round 1 cannot be distinguished later; LER at odd rounds is
    # P(round(eps/L) odd) and even rounds return to ~0 (see docs/sim_validation)
    from simulator.sim import odd_jump_probability
    kw = dict(sigma_q=0.2 * L, sigma_m=0.0, rho=0.9999)
    d, n = noise(n=20000, rounds=10, **kw)
    bd = BayesGridDecoder(kw["sigma_q"], 1e-3 * L, kw["rho"], n_x=96, device=DEVICE)
    by = run_blocked(bd, d, n, block_size(bd, 0.5e9))
    p = odd_jump_probability(kw["sigma_q"])
    se = np.sqrt(p * (1 - p) / 20000)
    assert np.isfinite(by).all()
    assert abs(by[:, 0].mean() - p) < 4 * se
    assert by[:, 9].mean() <= by[:, 0].mean() + 4 * se


@pytest.mark.parametrize("rho_a,sq_scale", [(0.0, 0.5), (0.99, 2.0)])
def test_bayes_stable_under_severe_mismatch(rho_a, sq_scale):
    kw = dict(sigma_q=0.2 * L, sigma_m=0.02 * L, rho=0.9)
    d, n = noise(n=2000, **kw)
    bd = BayesGridDecoder(sq_scale * kw["sigma_q"], kw["sigma_m"], rho_a, n_x=96,
                          device=DEVICE)
    by = run_blocked(bd, d, n, block_size(bd, 0.5e9))
    assert by.shape == d.shape          # finished without NaN errors


def test_bayes_grid_convergence():
    kw = dict(sigma_q=0.2 * L, sigma_m=0.02 * L, rho=0.99)
    d, n = noise(n=3000, **kw)
    out = []
    for nx in (96, 192):
        bd = BayesGridDecoder(kw["sigma_q"], kw["sigma_m"], kw["rho"], n_x=nx, device=DEVICE)
        out.append(run_blocked(bd, d, n, block_size(bd, 0.5e9)))
    assert (out[0] != out[1]).mean() < 2e-3


def test_regime_and_multimodel_filters_run():
    m = SwitchingNoiseModel()
    d, n = m.sample(make_rng("dev", "tests", 5), 1000, 50)
    rb = BayesGridDecoder(m.sigma_q, m.sigma_m, m.rho_lo, rho_hi=m.rho_hi,
                          p_switch=m.p_switch, n_x=96, device=DEVICE)
    a = run_blocked(rb, d, n, 1000)
    mm = MultiModelBayesDecoder(
        [dict(sigma_q=m.sigma_q, sigma_m=m.sigma_m, rho=r) for r in (0.3, 0.97)],
        n_x=96, device=DEVICE,
    )
    b = run_blocked(mm, d, n, 1000)
    nl = run_blocked(NearestLattice(), d, n, 1000)
    assert a.mean() < nl.mean() and b.mean() < nl.mean()


# LSTM closed loop equals open loop
def test_lstm_closed_loop_equals_open_loop():
    torch.manual_seed(0)
    model = ParityLSTM(hidden=16).to(DEVICE)
    # make predictions non-trivial
    with torch.no_grad():
        model.head.bias.fill_(0.0)
    d, n = noise(n=500)
    sim = Simulator(trials=500, rounds=50)
    nl = sim.run(NearestLattice(), d, n)
    s = torch.tensor(nl.syndromes, dtype=torch.float32, device=DEVICE)
    K = torch.tensor(nl.logical_errors.astype(np.int64), device=DEVICE)
    with torch.no_grad():
        logits, _ = model(s)
    for i, (head, k) in enumerate(HEADS):
        w_hat = (logits[..., i] > 0).long()
        open_loop = (K ^ cumulative_from_head(w_hat, k)).bool().cpu().numpy()
        closed = sim.run(LSTMDecoder(model, head=head, device=DEVICE), d, n).logical_errors
        assert np.array_equal(open_loop, closed), head


# noise models
def test_switching_noise_statistics():
    m = SwitchingNoiseModel(rho_lo=0.3, rho_hi=0.97, p_switch=0.05)
    eps, nu, reg = m.sample_with_regimes(make_rng("dev", "tests", 7), 40000, 50)
    assert abs(eps.std() / m.sigma_q - 1) < 0.01
    assert abs(nu.std() / m.sigma_m - 1) < 0.01
    for r, rho in ((0, 0.3), (1, 0.97)):
        mask = reg[:, 1:] == r
        c = np.corrcoef(eps[:, :-1][mask], eps[:, 1:][mask])[0, 1]
        assert abs(c - rho) < 0.02


def test_varied_noise_statistics():
    rng = make_rng("dev", "tests", 8)
    rho = np.full(40000, 0.6)
    eps, nu = sample_varied(rng, rho, np.full(40000, 0.3), np.full(40000, 0.1), 50)
    assert abs(eps.std() / 0.3 - 1) < 0.01
    assert abs(np.corrcoef(eps[:, :-1].ravel(), eps[:, 1:].ravel())[0, 1] - 0.6) < 0.01


# seeding
def test_split_streams_are_disjoint():
    os.environ["QKP_MODE"] = "full"
    try:
        keys = {}
        for split in SPLITS:
            for exp in ("E1", "E2", "lstm_data_nominal"):
                for point in range(3):
                    keys[(split, exp, point)] = tuple(
                        seed_sequence(split, exp, point).generate_state(4)
                    )
        test = {v for k, v in keys.items() if k[0] == "test"}
        other = {v for k, v in keys.items() if k[0] != "test"}
        assert not test & other
    finally:
        os.environ["QKP_MODE"] = "quick"


def test_quick_mode_cannot_use_test_split():
    with pytest.raises(RuntimeError):
        seed_sequence("test", "E1")


def test_statistics_helpers():
    p, lo, hi = wilson(0, 1000)
    assert p == 0 and lo == 0 and 0 < hi < 0.005
    le = np.zeros((10, 50), dtype=bool)
    le[0, 5:] = True
    s = summarize(le)
    assert s["flip_events"] == 1 and s["ler50"] == 0.1
