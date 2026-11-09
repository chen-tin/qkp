"""Noise models beyond the fixed-parameter AR(1) in noise.py.

SwitchingNoiseModel: the correlation switches between a weak and a strong
regime following a hidden two-state Markov chain (proposal, Sec. 2).

sample_varied: AR(1) noise where every trial has its own (rho, sigma_q,
sigma_m). Used for the "broad" training set.

Both keep Var(eps_t) = sigma_q^2 at every round, like NoiseModel.
"""

from dataclasses import dataclass

import numpy as np

from simulator.noise import L


@dataclass(frozen=True)
class SwitchingNoiseModel:
    sigma_q: float = 0.20 * L
    sigma_m: float = 0.05 * L
    rho_lo: float = 0.30
    rho_hi: float = 0.97
    p_switch: float = 0.05     # probability of changing regime each round

    def __post_init__(self):
        for rho in (self.rho_lo, self.rho_hi):
            if not -1 < rho < 1:
                raise ValueError("rho values must be in (-1, 1)")
        if not 0 <= self.p_switch <= 1:
            raise ValueError("p_switch must be a probability")
        if self.sigma_q < 0 or self.sigma_m < 0:
            raise ValueError("standard deviations must be nonnegative")

    @property
    def rhos(self):
        return np.array([self.rho_lo, self.rho_hi])

    @property
    def transition(self):
        """T[r, r'] = P(next regime r' | regime r)."""
        p = self.p_switch
        return np.array([[1 - p, p], [p, 1 - p]])

    def sample_with_regimes(self, rng, trials, rounds):
        """Return (displacement_noise, measurement_noise, regimes)."""
        regimes = np.zeros((trials, rounds), dtype=np.int8)
        regimes[:, 0] = rng.integers(0, 2, size=trials)   # stationary start
        switches = rng.random((trials, rounds)) < self.p_switch
        for t in range(1, rounds):
            regimes[:, t] = regimes[:, t - 1] ^ switches[:, t]

        rho = self.rhos[regimes]
        eps = np.zeros((trials, rounds))
        eps[:, 0] = rng.normal(0, self.sigma_q, size=trials)
        fresh = rng.normal(0, 1, size=(trials, rounds))
        for t in range(1, rounds):
            eps[:, t] = (
                rho[:, t] * eps[:, t - 1]
                + self.sigma_q * np.sqrt(1 - rho[:, t] ** 2) * fresh[:, t]
            )
        nu = rng.normal(0, self.sigma_m, size=(trials, rounds))
        return eps, nu, regimes

    def sample(self, rng, trials, rounds):
        eps, nu, _ = self.sample_with_regimes(rng, trials, rounds)
        return eps, nu


def sample_varied(rng, rho, sigma_q, sigma_m, rounds):
    """AR(1) noise with per-trial parameters.

    rho, sigma_q, sigma_m are arrays of shape (trials,). Returns
    (displacement_noise, measurement_noise), each (trials, rounds).
    """
    rho = np.asarray(rho, dtype=float)[:, None]
    sigma_q = np.asarray(sigma_q, dtype=float)[:, None]
    sigma_m = np.asarray(sigma_m, dtype=float)[:, None]
    trials = rho.shape[0]

    eps = np.zeros((trials, rounds))
    eps[:, 0] = rng.normal(0, 1, size=trials) * sigma_q[:, 0]
    fresh = rng.normal(0, 1, size=(trials, rounds))
    innovation = sigma_q * np.sqrt(1 - rho**2)
    for t in range(1, rounds):
        eps[:, t] = rho[:, 0] * eps[:, t - 1] + innovation[:, 0] * fresh[:, t]
    nu = rng.normal(0, 1, size=(trials, rounds)) * sigma_m
    return eps, nu
