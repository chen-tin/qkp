"""Bayes grid filter with real-valued (continuous) corrections, for the closed loop.

The hidden state of decoders/bayesian/grid_filter.py is the cumulative
displacement X_t (mod 2L), which does not depend on the corrections at all;
the corrections enter only through the known cumulative correction C in the
observation model

    s_t = wrap(X_t - C_{t-1} + nu_t).

So the same GridFilter is an exact HMM for *any* real corrections: nothing in
the belief has to be shifted or interpolated, C is simply a real number. The
residual oscillator displacement is x_t = X_t - C_t.

Policies (how c_t is chosen from the posterior):

  "parity"  c_t = s_t + j_t L, j_t the MAP correction class. Identical to
            BayesGridDecoder (tests/test_continuous.py checks equality in the
            closed loop).
  "fine"    c_t = s_t + j_t L + delta_t, where delta_t is the posterior mean of
            u = X_t - (s_t + C_{t-1} + j_t L), wrapped to [-L/2, L/2), over the
            chosen class. After a parity correction the residual is
            k L - nu_t (docs/sim_validation.md, Lemma 2): the readout error is
            written into the oscillator and adds to the next round's
            displacement, so each round's jump has variance
            sigma_q^2 + 2 sigma_m^2. u is (up to the cell) -nu_t, and with
            delta_t the decoder removes its estimate of it, which leaves
            sigma_q^2 + sigma_m^2 + Var(nu_t | data). This helps most when
            sigma_m is comparable to sigma_q, e.g. at hardware scale
            (sigma_q = 0.072 L, sigma_m ~ 0.1 L).

Decisions on j use the class posterior exactly as the parity decoder does.
"""

import numpy as np
import torch

from decoders.bayesian.grid_filter import GridFilter
from simulator.noise import L


class ContinuousBayesDecoder:
    """Closed-loop decoder: reset(trials), decide(syndrome) -> correction."""

    def __init__(self, sigma_q, sigma_m, rho, policy="fine", name=None, n_x=128,
                 device=None, max_fine=0.5 * L):
        if policy not in ("parity", "fine"):
            raise ValueError("policy must be 'parity' or 'fine'")
        self.policy = policy
        self.name = name or f"Bayes ({policy})"
        self.max_fine = max_fine
        self.filter = GridFilter(sigma_q, sigma_m, [rho], None, n_x=n_x, device=device)
        self.device = self.filter.device

    def reset(self, trials):
        self.filter.reset(trials)
        self.C = torch.zeros(trials, device=self.device, dtype=torch.float64)

    @property
    def resets(self):
        return self.filter.resets

    def decide(self, syndrome):
        s = torch.as_tensor(np.asarray(syndrome, dtype=float), device=self.device,
                            dtype=torch.float64)
        p_one = self.filter.step(s, self.C)
        j = (p_one > 0.5).to(torch.float64)
        c = s + j * L
        if self.policy == "fine":
            c = c + self._fine(s, j)
        self.C = torch.remainder(self.C + c, 2 * L)
        return c.cpu().numpy()

    def _fine(self, s, j):
        """Posterior mean of the within-cell offset of X from the chosen centre."""
        f = self.filter
        px = f.B.sum(dim=(0, 1)).to(torch.float64)                    # (b, n_x)
        centre = s + self.C + j * L
        u = torch.remainder(f.x[None, :] - centre[:, None] + L / 2, L) - L / 2
        # restrict to the chosen class: X within L/2 of the centre (mod 2L)
        v = torch.remainder(f.x[None, :] - centre[:, None] + L, 2 * L) - L
        inside = (v.abs() < L / 2).to(torch.float64)
        w = px * inside
        mass = w.sum(dim=1).clamp_min(1e-300)
        delta = (w * u).sum(dim=1) / mass
        return delta.clamp(-self.max_fine, self.max_fine)
