"""Vectorised closed-loop decoding environment for reinforcement learning.

A batch of independent trials of the GKP shift model (simulator/closed_loop.py
with AnalogSyndrome). Each step:

    x <- x + d_t;  s_t = wrap(x + nu_t)       (agent observes s_t)
    action a -> (j, delta) ; c_t = s_t + j L + delta ; x <- x - c_t

Observation (float32, shape (batch, obs_dim)):
    [sin, cos of 2 pi s / L for the last `window` syndromes (oldest first),
     s / L for the same window, the agent's cumulative parity J_{t-1} as +-1,
     and its last `window` fine offsets / L]
The cumulative parity makes the state Markov for parity-only actions: the
logical state is K_t XOR J_t, so the right j_t depends on J_{t-1}.

Actions: index a in range(2 * n_fine) -> j = a // n_fine, delta = fine_bins[a % n_fine].
n_fine = 1 gives parity-only actions (delta = 0); then the problem is a
contextual bandit whose optimal policy is the matched Bayes decoder (the
sanity check in docs/continuous_rl.md).

Rewards (reward=):
    "dense"     -1 for every round the logical state is wrong after the step
    "terminal"  -1 only at the last round if the logical state is wrong
    "flip"      -1 when the logical state changes (penalises dithering; note
                it is not a logical error rate once P_L saturates)
Episodes are `rounds` long; noise is drawn per episode from a NoiseModel or a
nonstationary spec (simulator/nonstationary.py).
"""

import math

import numpy as np

from simulator import nonstationary as ns
from simulator.noise import L, NoiseModel
from simulator.sim import wrap


def features(hist_s, J, hist_f):
    """Observation vector shared by the environment and the closed-loop decoder."""
    a = 2 * math.pi * hist_s / L
    parity = (1 - 2 * np.asarray(J)).astype(np.float32)[:, None]
    return np.concatenate([np.sin(a), np.cos(a), hist_s / L, parity, hist_f / L],
                          axis=1).astype(np.float32)


class DecodingEnv:
    def __init__(self, batch=256, rounds=50, noise=None, spec=None, window=8, n_fine=1,
                 fine_max=0.25 * L, reward="dense", seed=0):
        if reward not in ("dense", "terminal", "flip"):
            raise ValueError("reward must be dense, terminal or flip")
        self.batch, self.rounds, self.window = batch, rounds, window
        self.noise = noise if noise is not None else (None if spec else NoiseModel())
        self.spec = spec
        self.n_fine = n_fine
        self.fine_bins = (np.linspace(-fine_max, fine_max, n_fine) if n_fine > 1
                          else np.zeros(1))
        self.reward_kind = reward
        self.rng = np.random.default_rng(seed)
        self.n_actions = 2 * n_fine
        self.obs_dim = 3 * window + 1 + window

    # ------------------------------------------------------------------
    def _draw(self):
        if self.spec is not None:
            d, n, _ = ns.sample(self.spec, self.rng, self.batch, self.rounds)
        else:
            d, n = self.noise.sample(self.rng, self.batch, self.rounds)
        return d, n

    def reset(self, noise=None):
        """Start a new episode (optionally with given (d, n) arrays)."""
        self.d, self.n = noise if noise is not None else self._draw()
        self.t = 0
        self.x = np.zeros(self.batch)
        self.J = np.zeros(self.batch, dtype=np.int64)
        self.hist_s = np.zeros((self.batch, self.window))
        self.hist_f = np.zeros((self.batch, self.window))
        self.wrong = np.zeros(self.batch, dtype=bool)
        self._observe()
        return self.obs()

    def _observe(self):
        self.x = self.x + self.d[:, self.t]
        self.s = wrap(self.x + self.n[:, self.t])
        self.hist_s = np.roll(self.hist_s, -1, axis=1)
        self.hist_s[:, -1] = self.s

    def obs(self):
        return features(self.hist_s, self.J, self.hist_f)

    def decode_action(self, a):
        a = np.asarray(a)
        return a // self.n_fine, self.fine_bins[a % self.n_fine]

    def step(self, a):
        """Apply actions (batch,); returns obs, reward, done, info."""
        j, delta = self.decode_action(a)
        c = self.s + j * L + delta
        self.x = self.x - c
        self.J = (self.J + j) % 2
        self.hist_f = np.roll(self.hist_f, -1, axis=1)
        self.hist_f[:, -1] = delta
        wrong = (np.rint(self.x / L).astype(np.int64) % 2).astype(bool)
        last = self.t == self.rounds - 1
        if self.reward_kind == "dense":
            r = -wrong.astype(np.float32)
        elif self.reward_kind == "terminal":
            r = -(wrong & last).astype(np.float32)
        else:
            r = -(wrong ^ self.wrong).astype(np.float32)
        self.wrong = wrong
        self.t += 1
        done = self.t >= self.rounds
        if not done:
            self._observe()
        return self.obs(), r, done, dict(wrong=wrong, correction=c)
