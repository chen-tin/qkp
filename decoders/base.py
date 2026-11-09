"""Common machinery for decoders that act on the parity of the correction.

Every decoder here applies c_t = s_t + j_t * L with j_t in {0, 1}. With that
restriction the syndromes s_t do not depend on earlier decisions, and the
logical state after round t is K_t XOR (j_1 + ... + j_t mod 2), where K_t is
the logical state the nearest-lattice decoder would be in. A decoder that
estimates K_t by K_hat_t should therefore play

    j_t = K_hat_t XOR (j_1 + ... + j_{t-1} mod 2),

which leaves the logical state equal to K_t XOR K_hat_t.
"""

import numpy as np

from simulator.noise import L
from simulator.sim import Decoder


class ParityDecoder(Decoder):
    """Base class: subclasses implement choose(s) -> (j, p_one).

    j is the parity of this round's correction (0 or 1) for every trial.
    p_one is the decoder's probability that the correct choice is j = 1 (used
    only for diagnostics). With record=True the per-round j and p_one are kept
    in self.history_j / self.history_p, shape (trials, rounds).
    """

    name = "parity decoder"

    def __init__(self, record=False):
        self.record = record

    def reset(self, trials):
        self.trials = trials
        self.applied = np.zeros(trials, dtype=np.int8)   # parity of sum of j
        self.history_j = []
        self.history_p = []
        self._reset(trials)

    def _reset(self, trials):
        pass

    def choose(self, syndrome):
        raise NotImplementedError

    def decode(self, syndrome):
        j, p_one = self.choose(syndrome)
        j = np.asarray(j, dtype=np.int8)
        self.applied ^= j
        if self.record:
            self.history_j.append(j.copy())
            self.history_p.append(np.asarray(p_one, dtype=np.float32))
        return syndrome + j * L

    def recorded(self):
        """(j, p_one) as (trials, rounds) arrays."""
        return np.stack(self.history_j, axis=1), np.stack(self.history_p, axis=1)
