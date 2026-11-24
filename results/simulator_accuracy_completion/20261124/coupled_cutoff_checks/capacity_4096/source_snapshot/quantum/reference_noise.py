"""Reference noise extensions; sigma parameters and schedule values use L units."""
import numpy as np
from scipy.stats import norm
from simulator.noise import L
from simulator.nonstationary import make_schedule,RHO_CLIP


def sample_noise(params, trials, rounds, seed, extension=None):
    if not extension:
        from experiments.qutip_reference import noise
        return (*noise(params, trials, rounds, seed), {})
    allowed = {"family", "cross_correlation", "student_df", "schedule", "allow_schedule_clipping"}
    if set(extension)-allowed:
        raise ValueError(f"Unknown noise options: {set(extension)-allowed}")
    rho_qp = float(extension.get("cross_correlation", 0))
    if not np.isfinite(rho_qp) or abs(rho_qp) > 1:
        raise ValueError("Cross-quadrature correlation must lie in [-1,1]")
    family = extension.get("family", "gaussian")
    if family not in ("gaussian", "student_t"):
        raise ValueError("Noise family must be gaussian or student_t")
    df = float(extension.get("student_df", 5))
    if not np.isfinite(df) or df <= 2:
        raise ValueError("Student-t degrees of freedom must exceed two")
    rng = np.random.default_rng(seed)
    spec = {"family": "stationary", "sigma_q": params[0], "sigma_m": params[1], "rho": params[2],
            **extension.get("schedule", {})}
    schedule = make_schedule(spec, rng, trials, rounds)
    if not extension.get("allow_schedule_clipping",False) and np.any(np.abs(schedule.rho)>=RHO_CLIP):
        raise ValueError("Schedule reaches its rho clipping boundary; change ranges or explicitly allow_schedule_clipping")
    if not all(np.isfinite(getattr(schedule,k)).all() for k in ("sigma_q","sigma_m","rho","kick")):
        raise ValueError("Nonfinite noise schedule")
    # Convert dimensionless schedule outputs to physical q units.
    sq, sm, kick = schedule.sigma_q*L, schedule.sigma_m*L, schedule.kick*L
    def innovations():
        return rng.normal(size=(trials, rounds)) if family == "gaussian" else rng.standard_t(df, size=(trials, rounds))*np.sqrt((df-2)/df)
    zq, zp = innovations(), innovations()
    zp = rho_qp*zq+np.sqrt(1-rho_qp**2)*zp
    draws = []
    for z in (zq, zp):
        eps = np.empty_like(z);eps[:, 0] = sq[:, 0]*z[:, 0]
        for t in range(1, rounds):
            r = schedule.rho[:, t]
            eps[:, t] = r*sq[:, t]/sq[:, t-1]*eps[:, t-1]+sq[:, t]*np.sqrt(1-r*r)*z[:, t]
        draws.append(eps+kick*innovations())
    uq, up = rng.uniform(1e-12, 1-1e-12, (2, 3, trials, rounds))
    return (*draws, sm*norm.ppf(uq[1]), sm*norm.ppf(up[1]), uq, up,
            {"sigma_q": sq, "sigma_m": sm, "rho": schedule.rho, "kick": kick, "event_round": schedule.tau})
