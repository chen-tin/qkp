"""Closed-loop simulator: the decoder's corrections feed back into later rounds.

simulator/sim.py and simulator/fast.py assume parity-only corrections
c_t = s_t + j_t L, for which syndromes do not depend on the decoder. That
breaks for two extensions:

  * continuous corrections: the decoder applies any real c_t (for example a
    fine correction that also estimates the readout error nu_t);
  * binary (one-bit) syndromes: the ancilla returns a bit whose distribution
    depends on the current displacement, and the feedback shifts that
    displacement, as in the small-big-small protocol of cavity experiments.

Here the round is

    x      <- x + d_t                        (displacement; d_t from any noise model)
    obs_t  =  measurement(x, nu_t, rng)       (analog syndrome or a bit)
    c_t    =  decoder.decide(obs_t)           (any real correction)
    x      <- x - c_t

and the logical error after round t is round(x_t / L) odd, exactly as in
simulator/sim.py: one ideal final correction removes the residual within a
cell, and the cell index sets the logical state.

Measurements (classes with __call__(x, nu, rng) -> obs and a `kind`):
    AnalogSyndrome   wrap(x + nu), identical to simulator/sim.py
    PhaseBit         one ancilla bit with P(b = 1 | x) = (1 - v cos(2 pi (x + nu) / L)) / 2,
                     the ideal phase-estimation bit of a qubit-ancilla GKP
                     stabiliser measurement (Terhal & Weigand, PRA 93, 012315
                     (2016); Campagne-Ibarcq et al., Nature 584, 368 (2020)),
                     with visibility v <= 1 and an optional bit-flip readout
                     error p_flip.

Decoders: any object with reset(trials) and decide(obs) -> c (numpy (trials,)).
With AnalogSyndrome and a parity decoder this reproduces simulator/sim.py
exactly (tests/test_closed_loop.py).
"""

from dataclasses import dataclass

import numpy as np

from simulator.noise import L
from simulator.sim import wrap


@dataclass
class AnalogSyndrome:
    kind: str = "analog"

    def __call__(self, x, nu, rng):
        return wrap(x + nu)


@dataclass
class PhaseBit:
    visibility: float = 1.0
    p_flip: float = 0.0
    kind: str = "bit"

    def prob_one(self, u):
        return 0.5 * (1 - self.visibility * np.cos(2 * np.pi * u / L))

    def __call__(self, x, nu, rng):
        p = self.prob_one(x + nu)
        b = (rng.random(x.shape) < p).astype(np.int8)
        if self.p_flip > 0:
            b ^= (rng.random(x.shape) < self.p_flip).astype(np.int8)
        return b


@dataclass
class ClosedLoopResult:
    observations: np.ndarray    # (trials, rounds) analog syndromes or bits
    corrections: np.ndarray     # (trials, rounds)
    residuals: np.ndarray       # (trials, rounds) x after correction

    @property
    def logical_errors(self):
        return (np.rint(self.residuals / L).astype(np.int64) % 2).astype(bool)


def run(decoder, displacement_noise, measurement_noise, measurement=None, rng=None):
    """Run one decoder in closed loop on pre-generated noise.

    rng is used only by stochastic measurements (PhaseBit); pass a seeded
    generator for reproducibility. Every decoder sees the same noise; with a
    stochastic measurement, pass the same seed to compare decoders on the same
    bit draws (the draws then depend on x, so they are paired only in
    distribution once two decoders' corrections differ).
    """
    measurement = measurement or AnalogSyndrome()
    rng = rng or np.random.default_rng(0)
    trials, rounds = displacement_noise.shape
    obs = np.zeros((trials, rounds), dtype=np.int8 if measurement.kind == "bit" else float)
    corr = np.zeros((trials, rounds))
    res = np.zeros((trials, rounds))
    x = np.zeros(trials)
    decoder.reset(trials)
    for t in range(rounds):
        x = x + displacement_noise[:, t]
        o = measurement(x, measurement_noise[:, t], rng)
        obs[:, t] = o
        c = np.asarray(decoder.decide(o), dtype=float)
        if c.shape != (trials,) or not np.all(np.isfinite(c)):
            raise ValueError(f"{getattr(decoder, 'name', decoder)} returned a bad correction")
        corr[:, t] = c
        x = x - c
        res[:, t] = x
    return ClosedLoopResult(obs, corr, res)


class ParityAdapter:
    """Wrap a parity decoder (decoders/base.py ParityDecoder) for the closed loop."""

    def __init__(self, parity_decoder):
        self.inner = parity_decoder
        self.name = getattr(parity_decoder, "name", "parity")

    def reset(self, trials):
        self.inner.reset(trials)

    def decide(self, syndrome):
        return self.inner.decode(np.asarray(syndrome, dtype=float))
