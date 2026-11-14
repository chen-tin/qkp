"""Nonstationary displacement noise: drift, unannounced changes, bursts.

Every model here is an AR(1) whose parameters may change from round to
round and from trial to trial. A *schedule* holds, for every trial i and
round t,

    sigma_q[i, t]  standard deviation of eps_t
    rho[i, t]      correlation between eps_{t-1} and eps_t
    sigma_m[i, t]  measurement-noise standard deviation
    kick[i, t]     standard deviation of an extra one-round displacement
                   (0 except in burst rounds)

and the noise is

    eps_1 = sigma_1 z_1
    eps_t = rho_t (sigma_t / sigma_{t-1}) eps_{t-1} + sigma_t sqrt(1 - rho_t^2) z_t
    d_t   = eps_t + kick_t z'_t          (displacement handed to the simulator)
    nu_t  = sigma_m,t z''_t

so Var eps_t = sigma_t^2 and corr(eps_{t-1}, eps_t) = rho_t at every round,
exactly as in simulator/noise.py when the schedule is constant. The burst
kick is not part of the AR(1) state: it displaces the oscillator for one
round only (Tan, Pattison, McEwen & Preskill, arXiv:2406.18897, use a
one-round burst on the whole code block).

Schedule families (spec["family"]):

    stationary   constant parameters (the control)
    sinusoid     target = b (1 + amp sin(2 pi t / period + phase)) for sigma_q,
                 target = b + amp sin(...) for rho
                 (Bhardwaj et al., arXiv:2511.09491; Spitz et al., arXiv:1712.02360)
    ramp         target moves linearly from `start` to `end`
    one_over_f   sigma_q = b exp(rel X_t - rel^2 / 2), X a unit-variance sum of
                 OU processes with log-spaced rates (approximately 1/f between
                 the slowest and fastest rate; arXiv:2502.12296), clipped at
                 |X| <= 3
    step         target jumps from `before` to `after` at an unannounced round tau
    telegraph    target switches between `levels` with probability p_switch per
                 round (telegraphic T1 switching, Klimov et al., arXiv:1809.01043)
    burst        one burst of `duration` rounds with kick standard deviation
                 `kick` at an unannounced round tau

Any scalar parameter may be given as [lo, hi]; it is then drawn uniformly
per trial. All values are in absolute units (multiply by L yourself).
"""

from dataclasses import dataclass, field

import numpy as np

from simulator.noise import L

RHO_CLIP = 0.995
TARGETS = ("sigma_q", "rho", "sigma_m")
FAMILIES = ("stationary", "sinusoid", "ramp", "one_over_f", "step", "telegraph", "burst")


@dataclass
class Schedule:
    """Per-trial, per-round noise parameters, each of shape (trials, rounds)."""

    sigma_q: np.ndarray
    rho: np.ndarray
    sigma_m: np.ndarray
    kick: np.ndarray
    # per-trial event rounds (0-based; -1 = none), e.g. step or burst time
    tau: np.ndarray = None
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        shape = self.sigma_q.shape
        for name in ("rho", "sigma_m", "kick"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        if self.tau is None:
            self.tau = np.full(shape[0], -1, dtype=np.int64)
        if np.any(self.sigma_q <= 0) or np.any(self.sigma_m < 0) or np.any(self.kick < 0):
            raise ValueError("standard deviations must be positive (sigma_q) / nonnegative")
        if np.any(np.abs(self.rho) >= 1):
            raise ValueError("rho must be in (-1, 1)")

    @property
    def trials(self):
        return self.sigma_q.shape[0]

    @property
    def rounds(self):
        return self.sigma_q.shape[1]

    def slice(self, start, stop):
        """Schedule of trials start:stop (for blocked evaluation)."""
        return Schedule(self.sigma_q[start:stop], self.rho[start:stop],
                        self.sigma_m[start:stop], self.kick[start:stop],
                        self.tau[start:stop], dict(self.info))

    @staticmethod
    def concatenate(parts):
        cat = lambda name: np.concatenate([getattr(p, name) for p in parts], axis=0)
        return Schedule(cat("sigma_q"), cat("rho"), cat("sigma_m"), cat("kick"), cat("tau"),
                        dict(parts[0].info))

    def max_sigma(self):
        return float(self.sigma_q.max())


# parameter helpers
def _draw(value, rng, trials):
    """Scalar -> constant array; [lo, hi] -> uniform per trial."""
    if isinstance(value, (list, tuple)):
        lo, hi = value
        return rng.uniform(lo, hi, size=trials)
    return np.full(trials, float(value))


def _event_round(spec, rng, trials, rounds):
    """Unannounced event round per trial, uniform in [lo, hi] * rounds."""
    lo, hi = spec.get("tau", (0.25, 0.75))
    first = max(1, int(np.floor(lo * rounds)))
    last = max(first, min(rounds - 1, int(np.ceil(hi * rounds)) - 1))
    return rng.integers(first, last + 1, size=trials)


def make_schedule(spec, rng, trials, rounds):
    """Draw a Schedule for `trials` trials of `rounds` rounds from a spec dict."""
    family = spec.get("family", "stationary")
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    target = spec.get("target", "sigma_q")
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}")

    base = {k: _draw(spec[k], rng, trials)[:, None] * np.ones((1, rounds)) for k in TARGETS}
    kick = np.zeros((trials, rounds))
    tau = np.full(trials, -1, dtype=np.int64)
    t = np.arange(rounds)[None, :]

    if family == "stationary":
        pass
    elif family == "sinusoid":
        amp = _draw(spec["amp"], rng, trials)[:, None]
        period = _draw(spec["period"], rng, trials)[:, None]
        if spec.get("phase", "random") == "random":
            phase = rng.uniform(0, 2 * np.pi, size=(trials, 1))
        else:
            phase = np.full((trials, 1), float(spec["phase"]))
        wave = np.sin(2 * np.pi * t / period + phase)
        b = base[target]
        base[target] = b * (1 + amp * wave) if target != "rho" else b + amp * wave
    elif family == "ramp":
        start = _draw(spec["start"], rng, trials)[:, None]
        end = _draw(spec["end"], rng, trials)[:, None]
        frac = t / max(rounds - 1, 1)
        base[target] = start + (end - start) * frac
    elif family == "one_over_f":
        if target != "sigma_q":
            raise ValueError("one_over_f drives sigma_q only")
        rel = _draw(spec["rel"], rng, trials)[:, None]
        rates = np.geomspace(spec.get("rate_min", 1e-4), spec.get("rate_max", 0.3),
                             int(spec.get("n_ou", 6)))
        X = np.zeros((trials, rounds))
        state = rng.normal(size=(trials, len(rates)))          # stationary start
        noise = rng.normal(size=(trials, rounds, len(rates)))
        decay = np.exp(-rates)
        drive = np.sqrt(1 - decay**2)
        for r in range(rounds):
            if r:
                state = decay * state + drive * noise[:, r]
            X[:, r] = state.sum(axis=1) / np.sqrt(len(rates))
        # clip at 3 sd so sigma_q stays within the grids the decoders use
        X = np.clip(X, -3.0, 3.0)
        base["sigma_q"] = base["sigma_q"] * np.exp(rel * X - rel**2 / 2)
    elif family == "step":
        before = _draw(spec["before"], rng, trials)[:, None]
        after = _draw(spec["after"], rng, trials)[:, None]
        tau = _event_round(spec, rng, trials, rounds)
        base[target] = np.where(t >= tau[:, None], after, before)
    elif family == "telegraph":
        lo, hi = spec["levels"]
        p = float(spec["p_switch"])
        state = rng.integers(0, 2, size=trials)               # stationary start
        flips = rng.random((trials, rounds)) < p
        states = np.zeros((trials, rounds), dtype=np.int8)
        states[:, 0] = state
        for r in range(1, rounds):
            states[:, r] = states[:, r - 1] ^ flips[:, r]
        base[target] = np.where(states == 1, float(hi), float(lo))
    elif family == "burst":
        size = _draw(spec["kick"], rng, trials)
        duration = int(spec.get("duration", 1))
        tau = _event_round(spec, rng, trials, rounds)
        for k in range(duration):
            rows = np.arange(trials)
            cols = np.minimum(tau + k, rounds - 1)
            kick[rows, cols] = size

    rho = np.clip(base["rho"], -RHO_CLIP, RHO_CLIP)
    sigma_q = np.maximum(base["sigma_q"], 1e-6)
    sigma_m = np.maximum(base["sigma_m"], 0.0)
    return Schedule(sigma_q, rho, sigma_m, kick, tau, dict(family=family, target=target))


def sample_from_schedule(schedule, rng):
    """(displacement_noise, measurement_noise) for a schedule."""
    sq, rho = schedule.sigma_q, schedule.rho
    trials, rounds = sq.shape
    z = rng.normal(size=(trials, rounds))
    eps = np.zeros((trials, rounds))
    eps[:, 0] = sq[:, 0] * z[:, 0]
    for t in range(1, rounds):
        a = rho[:, t] * sq[:, t] / sq[:, t - 1]
        b = sq[:, t] * np.sqrt(1 - rho[:, t] ** 2)
        eps[:, t] = a * eps[:, t - 1] + b * z[:, t]
    kick = schedule.kick * rng.normal(size=(trials, rounds))
    nu = schedule.sigma_m * rng.normal(size=(trials, rounds))
    return eps + kick, nu


def sample(spec, rng, trials, rounds):
    """Return (displacement_noise, measurement_noise, schedule)."""
    schedule = make_schedule(spec, rng, trials, rounds)
    d, n = sample_from_schedule(schedule, rng)
    return d, n, schedule


def sample_mixture(families, rng, trials, rounds):
    """Noise from a weighted mixture of schedule specs (for LSTM training).

    families: list of (weight, spec). Each trial draws one spec; the result
    keeps the trial order random. Returns (d, n, schedule, family_index).
    """
    weights = np.array([w for w, _ in families], dtype=float)
    weights /= weights.sum()
    which = rng.choice(len(families), size=trials, p=weights)
    d = np.zeros((trials, rounds))
    n = np.zeros((trials, rounds))
    parts = {}
    for k, (_, spec) in enumerate(families):
        idx = np.flatnonzero(which == k)
        if len(idx) == 0:
            continue
        dk, nk, sk = sample(spec, rng, len(idx), rounds)
        d[idx], n[idx] = dk, nk
        parts[k] = (idx, sk)
    # reassemble the schedule in trial order
    fields = {name: np.zeros((trials, rounds)) for name in ("sigma_q", "rho", "sigma_m", "kick")}
    tau = np.full(trials, -1, dtype=np.int64)
    for idx, sk in parts.values():
        for name in fields:
            fields[name][idx] = getattr(sk, name)
        tau[idx] = sk.tau
    schedule = Schedule(**fields, tau=tau, info=dict(family="mixture"))
    return d, n, schedule, which


# named specs used by the experiments and training (absolute units)
def nominal_base(**overrides):
    base = dict(sigma_q=0.20 * L, sigma_m=0.05 * L, rho=0.90)
    base.update(overrides)
    return base


def spec(family, **kw):
    """Spec with the nominal base parameters filled in."""
    out = dict(family=family)
    out.update(nominal_base())
    out.update(kw)
    return out
