"""Bayesian decoders for nonstationary noise (simulator/nonstationary.py).

All three decoders use the same hidden state as decoders/bayesian/grid_filter.py
(the displacement X mod 2L and this round's eps, on a grid). They differ in
what they assume about the noise parameters:

OracleBayesDecoder ("Bayes (oracle)")
    Knows the true schedule sigma_q(t), rho(t), sigma_m(t) and the burst
    kicks of every trial. With that knowledge (X, eps) is an exact hidden
    Markov model of the simulator, so this is the optimal syndrome-only
    decoder for the parity action set, up to grid discretisation. It is the
    reference for every nonstationary experiment, in the way Spitz et al.
    (arXiv:1712.02360) compare their adaptive decoder with one given the true
    weights at every cycle.

RegimeBankDecoder ("Bayes-IMM")
    Does not know the schedule. Treats (sigma_q, rho, sigma_m) as a hidden
    regime that takes values on a fixed grid and jumps to another grid value
    with probability `hazard` per round (an interacting-multiple-model
    filter). The regime posterior tracks drift and abrupt changes. Within a
    regime the AR(1) step is the stationary one, which is approximate when
    sigma_q changes between regimes.

WindowedBayesDecoder ("Bayes-window W")
    Does not know the schedule. Runs one fixed-model grid filter per grid
    point and weights them by their syndrome likelihood over the last W
    rounds only (or with exponential forgetting). This is the sliding-window
    estimator of Bhardwaj et al. (arXiv:2511.09491) carried over to GKP: the
    noise parameters are estimated from recent syndrome statistics, here by
    likelihood rather than by detector-event rates, because GKP syndromes are
    continuous and periodic. W = infinity is the existing Bayes-broad model
    average.
"""

import math

import numpy as np
import torch
from scipy.stats import norm

from decoders.base import ParityDecoder
from decoders.bayesian.grid_filter import GridFilter
from simulator.noise import L


def _device(device):
    return device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _class_one(x, s, C):
    """Mask (trials, n_x): X closer to s + C + L than to s + C (mod 2L)."""
    u = torch.remainder(x[None, :] - (s + C)[:, None], 2 * L)
    return (u >= L / 2) & (u < 3 * L / 2)


def _log_likelihood(x, s, C, sigma_eff):
    """log p(s | X) on the grid, periodic Gaussian; sigma_eff (trials,) or scalar.

    Returns float64 (trials, n_x).
    """
    d = torch.remainder((s + C)[:, None] - x[None, :] + L / 2, L) - L / 2
    m = torch.arange(-2, 3, device=x.device, dtype=torch.float64) * L
    se = torch.as_tensor(sigma_eff, device=x.device, dtype=torch.float64)
    if se.ndim == 0:
        se = se.expand(s.shape[0])
    q = -((d[:, :, None] + m) ** 2) / (2 * se[:, None, None] ** 2)
    return torch.logsumexp(q, dim=2) - torch.log(se * math.sqrt(2 * math.pi))[:, None]


def _eps_grid(n_x, sigma_cap):
    h = 2 * L / n_x
    R = max(5 * sigma_cap, 0.6 * L)
    M = int(math.ceil(R / h))
    eps = (np.arange(2 * M + 1) - M) * h
    return h, M, eps


def _stationary_kernel(eps, h, sigma_q, rho):
    """K[e', e] = P(eps' in bin e' | eps_e) for a stationary AR(1)."""
    sig = math.sqrt(max(sigma_q**2 * (1 - rho**2), h**2 / 12))
    hi = (eps[:, None] + h / 2 - rho * eps[None, :]) / sig
    lo = (eps[:, None] - h / 2 - rho * eps[None, :]) / sig
    K = norm.cdf(hi) - norm.cdf(lo)
    return K / K.sum(axis=0, keepdims=True)


def _binned_normal(eps, h, sigma):
    p = norm.cdf((eps + h / 2) / sigma) - norm.cdf((eps - h / 2) / sigma)
    return p / p.sum()


# ---------------------------------------------------------------------------
# oracle: known per-trial, per-round schedule
# ---------------------------------------------------------------------------
class ScheduleGridFilter:
    """Grid filter whose AR(1) parameters are given per trial and per round."""

    def __init__(self, n_x=96, device=None, tf32=True):
        self.n_x = int(n_x)
        self.device = _device(device)
        self.dtype = torch.float32
        self.tf32 = tf32
        self.h = 2 * L / self.n_x
        self.x = torch.arange(self.n_x, device=self.device, dtype=torch.float64) * self.h
        self.schedule = None
        self.resets = 0
        self.sigma_cap = None

    def set_schedule(self, schedule):
        """Load the true schedule (simulator.nonstationary.Schedule) of a block."""
        cap = schedule.max_sigma()
        if self.sigma_cap is None or cap > self.sigma_cap:
            self._build_grid(cap)
        to = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=self.device,
                                       dtype=torch.float64)
        self.schedule = schedule
        self.sq = to(schedule.sigma_q)
        self.rho = to(schedule.rho)
        self.sm = to(schedule.sigma_m)
        self.kick = to(schedule.kick)

    def _build_grid(self, sigma_cap):
        self.sigma_cap = sigma_cap
        h, self.M, eps = _eps_grid(self.n_x, sigma_cap)
        self.n_e = len(eps)
        self.eps = torch.tensor(eps, device=self.device, dtype=torch.float64)
        shift = torch.arange(self.n_e, device=self.device) - self.M
        xs = torch.arange(self.n_x, device=self.device)
        self.shear_index = (xs[None, :] - shift[:, None]) % self.n_x
        # signed offsets of the X grid on the circle of length 2L (for kicks)
        k = torch.arange(self.n_x, device=self.device, dtype=torch.float64) * h
        self.offsets = torch.remainder(k + L, 2 * L) - L

    def bytes_per_trial_for(self, sigma_cap):
        _, _, eps = _eps_grid(self.n_x, sigma_cap)
        n_e = len(eps)
        return 4 * (n_e * n_e + 4 * n_e * self.n_x)

    def reset(self, trials):
        if self.schedule is None or self.schedule.trials != trials:
            raise RuntimeError("set_schedule() with this block's schedule before reset()")
        self.trials = trials
        self.t = 0
        sd = self.sq[:, 0]
        hi = (self.eps[None, :] + self.h / 2) / sd[:, None]
        lo = (self.eps[None, :] - self.h / 2) / sd[:, None]
        prior = torch.special.ndtr(hi) - torch.special.ndtr(lo)
        prior = prior / prior.sum(dim=1, keepdim=True)
        B = torch.zeros((trials, self.n_e, self.n_x), device=self.device, dtype=self.dtype)
        B[:, :, 0] = prior.to(self.dtype)
        self.B = B
        self.log_evidence = torch.zeros(trials, device=self.device, dtype=torch.float64)

    def _kernels(self, a, sd):
        """K[b, e', e] for per-trial AR coefficients a and innovation sd."""
        e_new = self.eps[None, :, None]
        e_old = self.eps[None, None, :]
        a = a[:, None, None]
        sd = sd[:, None, None]
        hi = (e_new + self.h / 2 - a * e_old) / sd
        lo = (e_new - self.h / 2 - a * e_old) / sd
        K = torch.special.ndtr(hi) - torch.special.ndtr(lo)
        K = K / K.sum(dim=1, keepdim=True).clamp_min(1e-300)
        return K.to(self.dtype)

    def _apply_kicks(self, B, kick):
        mask = kick > 0
        if not bool(mask.any()):
            return B
        k = kick[mask]
        m = torch.arange(-2, 3, device=self.device, dtype=torch.float64) * 2 * L
        g = torch.exp(-((self.offsets[None, :, None] + m) ** 2) / (2 * k[:, None, None] ** 2)).sum(2)
        tiny = k < self.h / 2
        if bool(tiny.any()):
            delta = torch.zeros_like(g[0])
            delta[0] = 1.0
            g[tiny] = delta
        g = g / g.sum(dim=1, keepdim=True)
        sub = B[mask].to(torch.float64)
        conv = torch.fft.irfft(torch.fft.rfft(sub, dim=-1) * torch.fft.rfft(g, dim=-1)[:, None, :],
                               n=self.n_x, dim=-1)
        B = B.clone()
        B[mask] = conv.clamp_min(0).to(self.dtype)
        return B

    def step(self, s, C):
        t = self.t
        if t >= self.sq.shape[1]:
            raise RuntimeError("schedule has fewer rounds than the run")
        sq_t = self.sq[:, t]
        sq_prev = self.sq[:, t - 1] if t > 0 else sq_t
        rho_t = self.rho[:, t]
        a = rho_t * sq_t / sq_prev
        sd = torch.clamp(sq_t * torch.sqrt(1 - rho_t**2), min=self.h / math.sqrt(12))

        # predict eps (one small GEMM per trial), then shear X <- X + eps
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        try:
            B = torch.bmm(self._kernels(a, sd), self.B)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        b = B.shape[0]
        B = torch.gather(B, 2, self.shear_index[None].expand(b, -1, -1))
        B = self._apply_kicks(B, self.kick[:, t])

        # likelihood of s
        sigma_eff = torch.sqrt(self.sm[:, t] ** 2 + self.h**2 / 12)
        ll = _log_likelihood(self.x, s, C, sigma_eff)
        ll_max = ll.max(dim=1, keepdim=True).values
        w = torch.exp(ll - ll_max).to(self.dtype)
        B = B * w[:, None, :]
        Z = B.sum(dim=(1, 2)).to(torch.float64)
        bad = ~torch.isfinite(Z) | (Z <= 0)
        if bool(bad.any()):
            self.resets += int(bad.sum())
            sd0 = sq_t[bad]
            hi = (self.eps[None, :] + self.h / 2) / sd0[:, None]
            lo = (self.eps[None, :] - self.h / 2) / sd0[:, None]
            prior = (torch.special.ndtr(hi) - torch.special.ndtr(lo)).to(self.dtype)
            prior = prior / prior.sum(dim=1, keepdim=True)
            B[bad] = prior[:, :, None] * w[bad][:, None, :]
            Z = B.sum(dim=(1, 2)).to(torch.float64)
        B = B / Z.to(self.dtype)[:, None, None]
        self.log_evidence += torch.log(Z) + ll_max[:, 0]
        self.B = B
        self.t += 1

        px = B.sum(dim=1).to(torch.float64)
        p_one = (px * _class_one(self.x, s, C)).sum(dim=1)
        if not bool(torch.isfinite(p_one).all()):
            raise FloatingPointError("schedule filter produced non-finite values")
        return p_one


class OracleBayesDecoder(ParityDecoder):
    """Optimal decoder given the true per-trial schedule (call set_schedule first)."""

    needs_schedule = True

    def __init__(self, name="Bayes (oracle)", n_x=96, sigma_cap_hint=0.5 * L, device=None,
                 record=False):
        super().__init__(record=record)
        self.name = name
        self.sigma_cap_hint = sigma_cap_hint
        self.filter = ScheduleGridFilter(n_x=n_x, device=device)
        self.device = self.filter.device

    def set_schedule(self, schedule):
        if schedule.max_sigma() > self.sigma_cap_hint:
            raise ValueError("schedule exceeds sigma_cap_hint; raise the hint so blocks "
                             "are sized for the larger grid")
        self.filter.set_schedule(schedule)

    def bytes_per_trial(self):
        # sized before the schedule is known: use the grid for the hint
        return self.filter.bytes_per_trial_for(self.sigma_cap_hint)

    def _reset(self, trials):
        self.filter.reset(trials)
        self.C = torch.zeros(trials, device=self.device, dtype=torch.float64)

    def choose(self, syndrome):
        s = torch.as_tensor(syndrome, device=self.device, dtype=torch.float64)
        p_one = self.filter.step(s, self.C)
        j = (p_one > 0.5).to(torch.int64)
        self.C = torch.remainder(self.C + s + j * L, 2 * L)
        return j.cpu().numpy(), p_one.cpu().numpy()

    @property
    def resets(self):
        return self.filter.resets


# ---------------------------------------------------------------------------
# adaptive 1: regime bank with a jump hazard (IMM)
# ---------------------------------------------------------------------------
def parameter_grid(sigma_qs, rhos, sigma_ms):
    """List of regime dicts over a (sigma_q, rho, sigma_m) grid (absolute units)."""
    return [dict(sigma_q=float(sq), rho=float(r), sigma_m=float(sm))
            for sq in sigma_qs for r in rhos for sm in sigma_ms]


class RegimeBankFilter:
    """Joint grid filter over (regime, eps, X), regimes jumping with a hazard."""

    def __init__(self, regimes, hazard, n_x=96, device=None, tf32=True):
        if not regimes:
            raise ValueError("need at least one regime")
        if not 0 <= hazard < 1:
            raise ValueError("hazard must be in [0, 1)")
        self.device = _device(device)
        self.dtype = torch.float32
        self.tf32 = tf32
        self.regimes = regimes
        self.hazard = float(hazard)
        self.n_reg = len(regimes)
        self.n_x = int(n_x)
        cap = max(r["sigma_q"] for r in regimes)
        self.h, self.M, eps = _eps_grid(self.n_x, cap)
        h = self.h
        self.n_e = len(eps)
        self.x = torch.arange(self.n_x, device=self.device, dtype=torch.float64) * h

        self.kernels = torch.tensor(
            np.stack([_stationary_kernel(eps, h, r["sigma_q"], r["rho"]) for r in regimes]),
            device=self.device, dtype=self.dtype)
        priors = np.stack([_binned_normal(eps, h, r["sigma_q"]) for r in regimes])
        self.prior_eps = torch.tensor(priors, device=self.device, dtype=self.dtype)
        if self.n_reg > 1:
            T = np.full((self.n_reg, self.n_reg), self.hazard / (self.n_reg - 1))
            np.fill_diagonal(T, 1 - self.hazard)
        else:
            T = np.ones((1, 1))
        self.transition = torch.tensor(T, device=self.device, dtype=self.dtype)
        self.sigma_eff = torch.tensor(
            [math.sqrt(r["sigma_m"] ** 2 + h**2 / 12) for r in regimes],
            device=self.device, dtype=torch.float64)

        shift = torch.arange(self.n_e, device=self.device) - self.M
        xs = torch.arange(self.n_x, device=self.device)
        self.shear_index = (xs[None, :] - shift[:, None]) % self.n_x
        self.resets = 0

    def belief_bytes_per_trial(self):
        return self.n_reg * self.n_e * self.n_x * 4

    def bytes_per_trial(self):
        return 4 * self.belief_bytes_per_trial()

    def reset(self, trials):
        self.trials = trials
        B = torch.zeros((self.n_reg, self.n_e, trials, self.n_x),
                        device=self.device, dtype=self.dtype)
        B[:, :, :, 0] = self.prior_eps[:, :, None] / self.n_reg
        self.B = B
        self.log_evidence = torch.zeros(trials, device=self.device, dtype=torch.float64)

    def step(self, s, C):
        B = self.B
        n_reg, n_e, b, n_x = B.shape
        if n_reg > 1:
            B = torch.einsum("rs,rebx->sebx", self.transition, B)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        try:
            B = torch.stack(
                [(self.kernels[r] @ B[r].reshape(n_e, b * n_x)).reshape(n_e, b, n_x)
                 for r in range(n_reg)], dim=0)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        index = self.shear_index[None, :, None, :].expand(n_reg, n_e, b, n_x)
        B = torch.gather(B, 3, index)

        # per-regime likelihood (regimes may assume different sigma_m)
        uniq, inv = torch.unique(self.sigma_eff, return_inverse=True)
        lls = torch.stack([_log_likelihood(self.x, s, C, u) for u in uniq], dim=0)
        ll = lls[inv]                                       # (n_reg, b, n_x)
        ll_max = ll.amax(dim=(0, 2))                        # (b,)
        w = torch.exp(ll - ll_max[None, :, None]).to(self.dtype)
        B = B * w[:, None, :, :]
        Z = B.sum(dim=(0, 1, 3)).to(torch.float64)
        bad = ~torch.isfinite(Z) | (Z <= 0)
        if bool(bad.any()):
            self.resets += int(bad.sum())
            fresh = (self.prior_eps[:, :, None, None] / n_reg) * w[:, None, bad, :]
            B[:, :, bad, :] = fresh
            Z = B.sum(dim=(0, 1, 3)).to(torch.float64)
        B = B / Z.to(self.dtype)[None, None, :, None]
        self.log_evidence += torch.log(Z) + ll_max
        self.B = B

        px = B.sum(dim=(0, 1)).to(torch.float64)
        p_one = (px * _class_one(self.x, s, C)).sum(dim=1)
        if not bool(torch.isfinite(p_one).all()):
            raise FloatingPointError("regime-bank filter produced non-finite values")
        return p_one

    def regime_posterior(self):
        """(trials, n_reg) posterior over regimes after the last step."""
        return self.B.sum(dim=(1, 3)).T.to(torch.float64)


class RegimeBankDecoder(ParityDecoder):
    def __init__(self, regimes, hazard=0.005, name="Bayes-IMM", n_x=96, device=None,
                 record=False):
        super().__init__(record=record)
        self.name = name
        self.filter = RegimeBankFilter(regimes, hazard, n_x=n_x, device=device)
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
        self.C = torch.remainder(self.C + s + j * L, 2 * L)
        return j.cpu().numpy(), p_one.cpu().numpy()

    @property
    def resets(self):
        return self.filter.resets


# ---------------------------------------------------------------------------
# adaptive 2: sliding-window model average
# ---------------------------------------------------------------------------
class WindowedBayesDecoder(ParityDecoder):
    """Fixed-model grid filters weighted by their recent syndrome likelihood.

    window: number of most recent rounds whose evidence counts (None = all).
    forgetting: alternatively, an exponential forgetting factor lambda in
    (0, 1]; the weight of round t - k is lambda^k.
    """

    def __init__(self, models, window=None, forgetting=None, name=None, n_x=96,
                 device=None, record=False):
        super().__init__(record=record)
        if window is not None and forgetting is not None:
            raise ValueError("give window or forgetting, not both")
        if window is not None and window < 1:
            raise ValueError("window must be >= 1")
        if forgetting is not None and not 0 < forgetting <= 1:
            raise ValueError("forgetting must be in (0, 1]")
        self.window = window
        self.forgetting = forgetting
        self.name = name or (f"Bayes-window {window}" if window else
                             f"Bayes-forget {forgetting}" if forgetting else "Bayes-average")
        self.models = models
        self.filters = [GridFilter(m["sigma_q"], m["sigma_m"], [m["rho"]], n_x=n_x,
                                   device=device) for m in models]
        self.device = self.filters[0].device

    def bytes_per_trial(self):
        ring = 8 * len(self.filters) * (self.window or 1)
        return (sum(f.belief_bytes_per_trial() for f in self.filters)
                + max(f.bytes_per_trial() for f in self.filters) + ring)

    def _reset(self, trials):
        for f in self.filters:
            f.reset(trials)
        K = len(self.filters)
        self.C = torch.zeros(trials, device=self.device, dtype=torch.float64)
        self.logw = torch.zeros((K, trials), device=self.device, dtype=torch.float64)
        if self.window:
            self.ring = torch.zeros((self.window, K, trials), device=self.device,
                                    dtype=torch.float64)
        self.t = 0

    def choose(self, syndrome):
        s = torch.as_tensor(syndrome, device=self.device, dtype=torch.float64)
        before = torch.stack([f.log_evidence.clone() for f in self.filters])
        p = torch.stack([f.step(s, self.C) for f in self.filters])
        inc = torch.stack([f.log_evidence for f in self.filters]) - before
        if self.window:
            slot = self.t % self.window
            self.logw = self.logw - self.ring[slot] + inc
            self.ring[slot] = inc
        elif self.forgetting:
            self.logw = self.forgetting * self.logw + inc
        else:
            self.logw = self.logw + inc
        self.t += 1
        w = torch.softmax(self.logw, dim=0)
        p_one = (w * p).sum(dim=0)
        j = (p_one > 0.5).to(torch.int64)
        self.C = torch.remainder(self.C + s + j * L, 2 * L)
        self.last_weights = w
        return j.cpu().numpy(), p_one.cpu().numpy()

    @property
    def resets(self):
        return sum(f.resets for f in self.filters)
