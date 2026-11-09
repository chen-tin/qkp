"""Sequential Bayesian (grid) filter decoders.

Hidden state per trial
    X_t   cumulative displacement error before correction, eps_1 + ... + eps_t,
          kept modulo 2L (shifts by 2L are stabilizers)
    eps_t this round's displacement (AR(1), possibly with a hidden regime)

What the decoder knows
    C_{t-1}  cumulative correction it has applied (mod 2L)
    s_t      syndrome, s_t = wrap(X_t - C_{t-1} + nu_t),  nu_t ~ N(0, sigma_m^2)

The true residual after round t is X_t - C_t, and the logical state is the
parity of round((X_t - C_t)/L). (X, eps) is an exact hidden Markov model for
the simulator in simulator/sim.py: the measurement noise nu_t enters only the
observation, never the state.

Per round (all on a grid with spacing h; X periodic with 2L/h points):
    1. (regimes) mix the regime probabilities with the transition matrix
    2. predict eps_t from eps_{t-1}: bin-integrated Gaussian AR(1) kernel
    3. shear: X_t = X_{t-1} + eps_t (an integer index shift per eps row)
    4. multiply by the periodic Gaussian likelihood of s_t, centred at
       s_t + C_{t-1} (evaluated analytically, so no interpolation)
    5. choose j_t in {0, 1}: the class (s_t + C_{t-1} + j L mod 2L) with
       the larger posterior mass, which is the MAP estimate of the logical
       state given all syndromes so far. Then C_t = C_{t-1} + s_t + j_t L.

Because later syndromes do not depend on earlier corrections of this form,
the greedy choice in step 5 minimises the logical error probability at every
round simultaneously, so the matched filter is the optimal syndrome-only
decoder for this action set (up to grid discretisation).
"""

import math

import numpy as np
import torch
from scipy.stats import norm

from decoders.base import ParityDecoder
from simulator.noise import L


class GridFilter:
    """Batched grid filter over (regime, eps, X mod 2L)."""

    def __init__(self, sigma_q, sigma_m, rhos, transition=None, n_x=128,
                 eps_range=None, device=None, tf32=True):
        rhos = np.atleast_1d(np.asarray(rhos, dtype=float))
        if np.any(np.abs(rhos) >= 1):
            raise ValueError("rho must be in (-1, 1)")
        if sigma_q <= 0:
            raise ValueError("sigma_q must be positive")
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float32
        self.tf32 = tf32
        self.n_reg = len(rhos)
        self.sigma_q = float(sigma_q)
        self.sigma_m = float(sigma_m)
        self.rhos = rhos

        # grids
        self.n_x = int(n_x)
        self.h = 2 * L / self.n_x
        h = self.h
        R = eps_range if eps_range is not None else max(5 * sigma_q, 0.6 * L)
        self.M = int(math.ceil(R / h))
        self.n_e = 2 * self.M + 1
        eps = (np.arange(self.n_e) - self.M) * h
        self.x = torch.arange(self.n_x, device=self.device, dtype=torch.float64) * h

        # quantisation of eps to the grid acts like extra uniform noise of
        # variance h^2/12 on the observation; include it so sigma_m -> 0
        # stays well posed
        self.sigma_eff = math.sqrt(self.sigma_m**2 + h**2 / 12)
        self.log_norm = math.log(self.sigma_eff * math.sqrt(2 * math.pi))

        # AR(1) kernels K[r][e', e] = P(eps' in bin e' | eps = eps_e, regime r)
        kernels = []
        for rho in rhos:
            sig = math.sqrt(max(sigma_q**2 * (1 - rho**2), h**2 / 12))
            hi = (eps[:, None] + h / 2 - rho * eps[None, :]) / sig
            lo = (eps[:, None] - h / 2 - rho * eps[None, :]) / sig
            K = norm.cdf(hi) - norm.cdf(lo)
            K /= K.sum(axis=0, keepdims=True)  # conserve mass at the edges
            kernels.append(K)
        self.kernels = torch.tensor(
            np.stack(kernels), device=self.device, dtype=self.dtype
        )

        # stationary prior of eps, so that eps_1 ~ N(0, sigma_q^2)
        prior = norm.cdf((eps + h / 2) / sigma_q) - norm.cdf((eps - h / 2) / sigma_q)
        prior /= prior.sum()
        self.prior_eps = torch.tensor(prior, device=self.device, dtype=self.dtype)

        if transition is None:
            transition = np.eye(self.n_reg)
        transition = np.asarray(transition, dtype=float)
        self.transition = torch.tensor(transition, device=self.device, dtype=self.dtype)
        # stationary regime distribution
        w, v = np.linalg.eig(transition.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1))])
        pi = pi / pi.sum()
        self.pi_reg = torch.tensor(pi, device=self.device, dtype=self.dtype)

        # shear: new[e, x] = old[e, (x - shift_e) mod n_x], shift_e = e - M
        shift = torch.arange(self.n_e, device=self.device) - self.M
        xs = torch.arange(self.n_x, device=self.device)
        self.shear_index = (xs[None, :] - shift[:, None]) % self.n_x

        self.resets = 0

    def bytes_per_trial(self):
        return self.n_reg * self.n_e * self.n_x * 4 * 3  # belief + temporaries

    def reset(self, trials):
        # belief layout (n_reg, n_e, trials, n_x): the AR(1) prediction is then
        # a single GEMM per regime on a contiguous (n_e, trials*n_x) view
        self.trials = trials
        B = torch.zeros(
            (self.n_reg, self.n_e, trials, self.n_x), device=self.device, dtype=self.dtype
        )
        # X_0 = 0 exactly; eps_0 from the stationary prior
        B[:, :, :, 0] = self.pi_reg[:, None, None] * self.prior_eps[None, :, None]
        self.B = B
        self.log_evidence = torch.zeros(trials, device=self.device, dtype=torch.float64)

    def _class_one(self, s, C):
        """Mask (trials, n_x): X closer to s + C + L than to s + C (mod 2L)."""
        u = torch.remainder(self.x[None, :] - (s + C)[:, None], 2 * L)
        return (u >= L / 2) & (u < 3 * L / 2)

    def step(self, s, C):
        """Advance one round with syndrome s given applied correction C.

        s, C: float64 tensors (trials,). Returns p_one, the posterior
        probability that j = 1 is the right correction class.
        """
        B = self.B
        n_reg, n_e, b, n_x = B.shape

        # 1. regime transition
        if n_reg > 1:
            B = torch.einsum("rs,rebx->sebx", self.transition, B)

        # 2. predict eps: one GEMM per regime
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        try:
            B = torch.stack(
                [(self.kernels[r] @ B[r].reshape(n_e, b * n_x)).reshape(n_e, b, n_x)
                 for r in range(n_reg)], dim=0,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

        # 3. shear X <- X + eps (row-dependent circular shift along X)
        index = self.shear_index[None, :, None, :].expand(n_reg, n_e, b, n_x)
        B = torch.gather(B, 3, index)

        # 4. likelihood of s given X (periodic Gaussian, period L)
        # offsets in float64 (C accumulates), the rest in float32
        d = torch.remainder((s + C)[:, None] - self.x[None, :] + L / 2, L) - L / 2
        d = d.to(self.dtype)
        m = torch.arange(-2, 3, device=self.device, dtype=self.dtype) * L
        ll = (torch.logsumexp(
            -((d[:, :, None] + m) ** 2) / (2 * self.sigma_eff**2), dim=2
        ) - self.log_norm).to(torch.float64)
        ll_max = ll.max(dim=1, keepdim=True).values
        w = torch.exp(ll - ll_max).to(self.dtype)          # (b, n_x), max 1
        B = B * w[None, None, :, :]

        Z = B.sum(dim=(0, 1, 3)).to(torch.float64)        # (b,)
        bad = ~torch.isfinite(Z) | (Z <= 0)
        if bool(bad.any()):
            # belief lost all support (only under severe model mismatch):
            # restart from the stationary prior times the likelihood
            self.resets += int(bad.sum())
            fresh = (
                self.pi_reg[:, None, None, None]
                * self.prior_eps[None, :, None, None]
                * w[bad][None, None, :, :]
            )
            B[:, :, bad, :] = fresh
            Z = B.sum(dim=(0, 1, 3)).to(torch.float64)
        B = B / Z.to(self.dtype)[None, None, :, None]
        self.log_evidence += torch.log(Z) + ll_max[:, 0]
        self.B = B

        # 5. posterior mass of the two correction classes
        px = B.sum(dim=(0, 1)).to(torch.float64)          # (b, n_x)
        p_one = (px * self._class_one(s, C)).sum(dim=1)
        if not bool(torch.isfinite(p_one).all()):
            raise FloatingPointError("grid filter produced non-finite values")
        return p_one

    def commit(self, j, s, C):
        """Hard decision: keep only the mass consistent with choosing j.

        Used by the 'per-round' ablation, which never revises a decision.
        """
        one = self._class_one(s, C)
        keep = torch.where(j[:, None].bool(), one, ~one).to(self.dtype)
        B = self.B * keep[None, None, :, :]
        Z = B.sum(dim=(0, 1, 3))
        Z = torch.where(Z > 0, Z, torch.ones_like(Z))
        self.B = B / Z[None, None, :, None]


class BayesGridDecoder(ParityDecoder):
    """Bayesian grid-filter decoder with an assumed AR(1) (or regime) model.

    per_round=True gives the 'no revision' ablation: after each decision the
    belief is restricted to the chosen class, so earlier decisions are never
    revisited.
    """

    def __init__(self, sigma_q, sigma_m, rho, name="Bayes", per_round=False,
                 rho_hi=None, p_switch=None, n_x=128, device=None, record=False):
        super().__init__(record=record)
        self.name = name
        self.per_round = per_round
        if rho_hi is None:
            rhos, transition = [rho], None
        else:
            rhos = [rho, rho_hi]
            transition = [[1 - p_switch, p_switch], [p_switch, 1 - p_switch]]
        self.filter = GridFilter(
            sigma_q, sigma_m, rhos, transition, n_x=n_x, device=device
        )
        self.device = self.filter.device

    def bytes_per_trial(self):
        return self.filter.bytes_per_trial()

    def _reset(self, trials):
        self.filter.reset(trials)
        self.C = torch.zeros(trials, device=self.device, dtype=torch.float64)

    def choose(self, syndrome):
        s = torch.as_tensor(syndrome, device=self.device, dtype=torch.float64)
        p_one = self.filter.step(s, self.C)
        j = (p_one > 0.5).to(torch.int64)
        if self.per_round:
            self.filter.commit(j, s, self.C)
        self.C = torch.remainder(self.C + s + j * L, 2 * L)
        return j.cpu().numpy(), p_one.cpu().numpy()

    @property
    def resets(self):
        return self.filter.resets


class MultiModelBayesDecoder(ParityDecoder):
    """Bayesian model averaging over several assumed AR(1) models.

    Each component is a grid filter; components are weighted by their
    cumulative marginal likelihood of the syndromes seen so far. Used as
    'Bayes-broad', the counterpart of an LSTM trained on a parameter box.
    """

    def __init__(self, models, name="Bayes-broad", n_x=128, device=None, record=False):
        super().__init__(record=record)
        self.name = name
        self.filters = [
            GridFilter(m["sigma_q"], m["sigma_m"], [m["rho"]], n_x=n_x, device=device)
            for m in models
        ]
        self.models = models
        self.device = self.filters[0].device

    def bytes_per_trial(self):
        return sum(f.bytes_per_trial() for f in self.filters)

    def _reset(self, trials):
        for f in self.filters:
            f.reset(trials)
        self.C = torch.zeros(trials, device=self.device, dtype=torch.float64)

    def choose(self, syndrome):
        s = torch.as_tensor(syndrome, device=self.device, dtype=torch.float64)
        p = torch.stack([f.step(s, self.C) for f in self.filters], dim=0)
        logw = torch.stack([f.log_evidence for f in self.filters], dim=0)
        w = torch.softmax(logw, dim=0)
        p_one = (w * p).sum(dim=0)
        j = (p_one > 0.5).to(torch.int64)
        self.C = torch.remainder(self.C + s + j * L, 2 * L)
        self.last_weights = w
        return j.cpu().numpy(), p_one.cpu().numpy()

    @property
    def resets(self):
        return sum(f.resets for f in self.filters)


def block_size(decoder, budget_bytes):
    """Trials per block so a decoder's belief fits in budget_bytes."""
    per = decoder.bytes_per_trial() if hasattr(decoder, "bytes_per_trial") else 0
    if per == 0:
        return 50_000
    return max(256, int(budget_bytes // per))
