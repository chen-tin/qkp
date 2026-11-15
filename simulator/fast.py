"""Vectorised simulator for parity decoders (no per-round Python loop).

For corrections c_t = s_t + j_t L the syndromes do not depend on the
decoder (tests/test_core.py), so the whole syndrome stream and the
nearest-lattice logical record follow in closed form from the noise
(docs/sim_validation.md, Lemma 2 and Prop. 3):

    z_t  = eps_t + nu_t - nu_{t-1}          (nu_0 = 0)
    s_t  = wrap(z_t)
    k_t  = k_{t-1} + round(z_t / L)          (k_0 = 0)
    x_t  = k_t L - nu_t                      (NL residual after round t)
    K_t  = round(x_t / L) mod 2              (NL logical error)

and any parity decoder with cumulative parity J_t = j_1 + ... + j_t mod 2
has logical error K_t XOR J_t. This replaces Simulator.run's loop over
rounds by a handful of array operations. Results agree with
simulator/sim.py except at exact floating-point ties of the wrap, which
tests/test_fast.py checks on large samples.
"""

import numpy as np

from simulator.noise import L
from simulator.sim import wrap


def syndromes_and_nl(displacement_noise, measurement_noise):
    """(syndromes, nl_logical_errors) as numpy arrays (trials, rounds)."""
    d = np.asarray(displacement_noise, dtype=float)
    n = np.asarray(measurement_noise, dtype=float)
    prev = np.zeros_like(n)
    prev[:, 1:] = n[:, :-1]
    z = d + n - prev
    s = wrap(z)
    k = np.cumsum(np.rint(z / L), axis=1)
    x = k * L - n
    K = (np.rint(x / L).astype(np.int64) % 2).astype(bool)
    return s, K


def simulate_nl(displacement_noise, measurement_noise, device=None):
    """Nearest-lattice logical-error record; device is accepted for API symmetry."""
    return syndromes_and_nl(displacement_noise, measurement_noise)[1]


def logical_errors(nl_errors, j):
    """Logical record of a parity decoder from its per-round choices j (trials, rounds)."""
    J = np.cumsum(np.asarray(j, dtype=np.int64), axis=1) % 2
    return np.asarray(nl_errors, dtype=bool) ^ J.astype(bool)
