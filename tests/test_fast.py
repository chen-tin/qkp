"""The GPU fast paths reproduce the reference simulator, decoders and encoder."""

import os

os.environ.setdefault("QKP_MODE", "quick")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest
import torch

from experiments.common import flips_per_trial, make_rng, paired_difference
from simulator import nonstationary as ns
from simulator.noise import L
from simulator.sim import NearestLattice, Simulator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_vectorised_simulator_matches_reference_loop():
    from simulator.fast import logical_errors, syndromes_and_nl
    for spec in (ns.spec("stationary"), ns.spec("stationary", sigma_q=0.3 * L, sigma_m=0.2 * L),
                 ns.spec("burst", kick=0.4 * L)):
        d, n, _ = ns.sample(spec, make_rng("dev", "test_fast", 0), 5000, 60)
        ref = Simulator(trials=5000, rounds=60).run(NearestLattice(), d, n)
        s, K = syndromes_and_nl(d, n)
        assert np.abs(ref.syndromes - s).max() < 1e-12
        assert np.array_equal(ref.logical_errors, K)
        j = make_rng("dev", "test_fast_j", 0).integers(0, 2, size=K.shape)
        assert np.array_equal(logical_errors(K, np.zeros_like(j)), K)


@pytest.fixture(scope="module")
def decoders():
    from decoders.fast import fast_decoders
    from experiments import profile_pipeline as pp, run_all
    run_all.DEVICE = DEVICE
    return pp.reference_decoders(DEVICE), fast_decoders(DEVICE), pp


@pytest.mark.parametrize("spec", [
    ns.spec("stationary"),
    ns.spec("step", target="sigma_q", before=0.2 * L, after=0.3 * L),
    ns.spec("telegraph", target="rho", levels=[0.3, 0.97], p_switch=0.02),
    ns.spec("burst", kick=0.4 * L),
], ids=["stationary", "step", "telegraph_rho", "burst"])
def test_fast_decoders_match_reference(decoders, spec):
    ref, fast, pp = decoders
    d, n, sch = ns.sample(spec, make_rng("dev", "test_fast", 1), 600, 60)
    for name, fn in fast.items():
        a = pp.run_reference(ref[name], d, n, sch)
        b = fn(d, n, sch)
        # identical up to float32 near-ties (posterior ~ 1/2)
        assert np.mean(a != b) < 2e-3, name
        diff, se = paired_difference(flips_per_trial(b), flips_per_trial(a), 60)
        assert abs(diff) <= 3 * se + 1e-12, name


def test_cuda_graph_equals_eager_and_handles_several_batch_sizes(decoders):
    from decoders import fast as F
    _, fast, _ = decoders
    dec = fast["Bayes-window 200"].__defaults__[0]
    for b in (5, 300, 5):                                  # revisit a cached graph
        d, n, _ = ns.sample(ns.spec("step", target="sigma_q", before=0.2 * L, after=0.3 * L),
                            make_rng("dev", "test_graph", b), b, 40)
        g = F.run(dec, d, n)
        orig = dec.decide
        dec.decide = lambda s, graph=False: orig(s, graph=False)
        try:
            e = F.run(dec, d, n)
        finally:
            dec.decide = orig
        assert np.array_equal(g, e)


def test_lstm_vectorised_heads_match_recurrence():
    from decoders.LSTM.model import HEADS, cumulative_from_head
    w = torch.randint(0, 2, (7, 33))
    for _, k in HEADS:
        J = cumulative_from_head(w, k)
        if k is None:
            Jc = w
        else:
            step = k + 1
            wp = torch.nn.functional.pad(w, (0, (-33) % step))
            Jc = (torch.cumsum(wp.reshape(7, -1, step), dim=1) % 2).reshape(7, -1)[:, :33]
        assert torch.equal(J, Jc)


def test_fast_encoding_matches_reference():
    import qutip as qt
    from quantum.encoding import encode_gkp
    from quantum.fast_encoding import encode_gkp_fast, wigner_fft
    for delta, N in ((0.3, 60), (0.25, 120)):
        a = encode_gkp(delta, N, 6).full().ravel()
        b = encode_gkp_fast(delta, N, 6)
        assert np.abs(a - b).max() < 1e-12
    grid = np.linspace(-4, 4, 41)
    Wr = qt.wigner(qt.Qobj(a.reshape(-1, 1)), grid, grid, method="clenshaw")
    Wf = wigner_fft(b, grid, device=DEVICE)
    assert np.abs(Wr - Wf).max() < 1e-10
