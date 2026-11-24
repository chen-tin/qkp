"""Continuous Mehler GKP resource with a matching Gaussian-mixture sampler.

The historical binned Ancilla remains unchanged. The Gaussian-mixture
identity and inverse-CDF accuracy are documented in docs/smooth_ancilla_experiment.md.
"""
import math

import numpy as np
from scipy.special import ndtr, ndtri

from quantum.qutip_reference import L


class ContinuousAncilla:
    def __init__(self, sigma_m, tail_tolerance=1e-16):
        if not np.isfinite(sigma_m) or not 0 < 2*sigma_m*sigma_m < 1:
            raise ValueError("Require 0 < 2 sigma_m^2 < 1")
        if not 0 < tail_tolerance <= 1e-12:
            raise ValueError("Tail tolerance must lie in (0,1e-12]")
        self.t = float(np.arctanh(2*sigma_m*sigma_m))
        self.delta = math.sqrt(self.t)
        self.v = math.tanh(self.t)
        self.spacing = L/math.cosh(self.t)
        self.jmax = math.ceil(math.sqrt(2*math.log(1/tail_tolerance)/self.v)/L)+2
        j = np.arange(-self.jmax, self.jmax+1)
        means = j*self.spacing
        weights = np.exp(-.5*self.v*(j*L)**2)
        terms = weights[:, None]*weights[None, :]*np.exp(-(means[:, None]-means[None, :])**2/(4*self.v))
        z = float(terms.sum())
        self.normalizer = math.sqrt(math.sqrt(math.pi*self.v)*z)
        sums = (j[:, None]+j[None, :]+2*self.jmax).ravel()
        self.mixture_weights = np.bincount(sums, weights=terms.ravel(), minlength=4*self.jmax+1)/z
        self.mixture_means = np.arange(-2*self.jmax, 2*self.jmax+1)*self.spacing/2
        self.mixture_cumulative = np.cumsum(self.mixture_weights)
        self.mixture_cumulative[-1] = 1.
        self.sample_std = math.sqrt(self.v/2)
        self.neighbours = math.ceil(math.sqrt(2*self.v*math.log(1/tail_tolerance))/self.spacing)+1
        self.max_quantile_residual = 0.

    def amplitude(self, x):
        x = np.asarray(x, dtype=float)
        nearest = np.rint(x/self.spacing).astype(np.int64)
        value = np.zeros_like(x)
        for offset in range(-self.neighbours, self.neighbours+1):
            j = nearest+offset
            exponent = -.5*self.v*(j*L)**2-(x-j*self.spacing)**2/(2*self.v)
            value += np.where(np.abs(j) <= self.jmax, np.exp(exponent), 0.)
        return value/self.normalizer

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        for mu, weight in zip(self.mixture_means, self.mixture_weights):
            if weight:
                result += weight*ndtr((x-mu)/self.sample_std)
        return np.clip(result, 0., 1.)

    def sample(self, u, jitter=None):
        """Inverse CDF; jitter is accepted for historical API compatibility."""
        u = np.asarray(u, dtype=float)
        if not np.isfinite(u).all() or np.any((u <= 0) | (u >= 1)):
            raise ValueError("Sampling quantiles must lie strictly between 0 and 1")
        indices = np.searchsorted(self.mixture_cumulative, u, side="right").clip(0, len(self.mixture_weights)-1)
        before = np.r_[0., self.mixture_cumulative[:-1]][indices]
        fraction = np.clip((u-before)/self.mixture_weights[indices], 1e-15, 1-1e-15)
        x = self.mixture_means[indices]+self.sample_std*ndtri(fraction)
        z = self.sample_std*ndtri(u)
        lo, hi = self.mixture_means[0]+z, self.mixture_means[-1]+z
        for _ in range(64):
            residual = self.cdf(x)-u
            active = np.abs(residual) > 2e-14
            if not np.any(active):
                break
            lo = np.where((residual < 0) & active, x, lo)
            hi = np.where((residual > 0) & active, x, hi)
            density = self.amplitude(x)**2
            step = np.divide(residual, density, out=np.full_like(x, np.inf), where=density > 1e-290)
            proposed = x-step
            proposed = np.where(np.isfinite(proposed) & (proposed > lo) & (proposed < hi), proposed, (lo+hi)/2)
            x = np.where(active, proposed, x)
        maximum = float(np.max(np.abs(self.cdf(x)-u))) if u.size else 0.
        self.max_quantile_residual = max(self.max_quantile_residual, maximum)
        if maximum > 1e-10:
            raise FloatingPointError(f"Inverse CDF failed: residual {maximum:g}")
        return x
