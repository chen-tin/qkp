"""The closed-loop simulator reproduces simulator/sim.py for parity decoders."""

import os

os.environ.setdefault("QKP_MODE", "quick")

import numpy as np

from decoders.bayesian.grid_filter import BayesGridDecoder
from experiments.common import make_rng
from simulator import closed_loop as cl
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice, Simulator


def test_analog_closed_loop_matches_reference_simulator():
    d, n = NoiseModel().sample(make_rng("dev", "test_cl", 0), 800, 40)
    for dec in (NearestLattice(), BayesGridDecoder(0.2 * L, 0.05 * L, 0.9, n_x=96)):
        ref = Simulator(trials=800, rounds=40).run(dec, d, n)
        res = cl.run(cl.ParityAdapter(dec), d, n)
        assert np.array_equal(ref.logical_errors, res.logical_errors)
        assert np.allclose(ref.syndromes, res.observations)


def test_phase_bit_statistics():
    m = cl.PhaseBit(visibility=0.9, p_flip=0.01)
    rng = np.random.default_rng(1)
    u = np.zeros(200000)
    p0 = m(u, np.zeros_like(u), rng).mean()                 # at a lattice point
    uh = np.full(200000, L / 2)
    ph = m(uh, np.zeros_like(uh), rng).mean()                # half-way: P(b=1) high
    expect0 = 0.5 * (1 - 0.9) * 0.99 + 0.01 * (1 - 0.5 * (1 - 0.9))
    assert abs(p0 - expect0) < 3e-3
    assert ph > 0.9
