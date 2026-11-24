"""Model contracts and numerical gates for new reference runs.

Rates use the same declared time unit as cycle_time. Finite gate dissipation
is a symmetric lumping around an instantaneous measurement, not a solution
of the simultaneous data/ancilla Lindblad equation. See docs/simulation_quality.md.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class Physics:
    cycle_time: float = 1.
    gate_duration: float = 0.
    feedback_delay: float = 0.
    loss_rate: float = 0.
    dephasing_rate: float = 0.
    detector_efficiency: float = 1.
    detector_std: float = 0.
    coupling: float = 1.
    feedback_gain: float = 1.
    control_rotation: float = 0.
    cooling_transmissivity: float = 1.
    displacement_provenance: str = "independent"
    gate_backend: str = "lumped"
    gate_ancilla_cutoff: int = 512
    ancilla_loss_rate: float = 0.
    ancilla_dephasing_rate: float = 0.
    homodyne_radius: float = 60.
    homodyne_dy: float = .01
    max_joint_dimension: int = 2_000_000
    max_gate_scratch_mb: float = 512.

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key not in ("displacement_provenance", "gate_backend") and not np.isfinite(value):
                raise ValueError(f"Nonfinite physics parameter: {key}")
        if self.cycle_time <= 0 or min(self.gate_duration, self.feedback_delay,
                                      self.loss_rate, self.dephasing_rate, self.detector_std) < 0:
            raise ValueError("Positive cycle time and nonnegative durations/rates required")
        if self.idle_time < -1e-12:
            raise ValueError("Two gates and two feedback delays exceed the cycle time")
        if not 0 < self.detector_efficiency <= 1 or not 0 < self.cooling_transmissivity <= 1:
            raise ValueError("Efficiencies/transmissivities must lie in (0,1]")
        if self.coupling <= 0 or self.feedback_gain < 0:
            raise ValueError("Positive coupling and nonnegative feedback gain required")
        if self.displacement_provenance not in ("independent", "loss_equivalent"):
            raise ValueError("Declare displacement_provenance independent or loss_equivalent")
        if self.displacement_provenance == "loss_equivalent" and self.loss_rate:
            raise ValueError("Loss-derived displacement plus explicit loss double-counts loss")
        if self.gate_backend not in ("lumped", "qutip"):
            raise ValueError("gate_backend must be lumped or qutip")
        if self.ancilla_loss_rate < 0 or self.ancilla_dephasing_rate < 0:
            raise ValueError("Ancilla rates must be nonnegative")
        if self.gate_backend != "qutip" and (self.ancilla_loss_rate or self.ancilla_dephasing_rate):
            raise ValueError("Ancilla dissipation requires the explicit qutip gate backend")
        if min(self.gate_ancilla_cutoff, self.max_joint_dimension, self.homodyne_radius, self.homodyne_dy, self.max_gate_scratch_mb) <= 0:
            raise ValueError("Positive explicit gate/grid budgets required")
        if not isinstance(self.gate_ancilla_cutoff,int) or not isinstance(self.max_joint_dimension,int):
            raise ValueError("Explicit dimensions must be integers")

    @property
    def idle_time(self):
        return self.cycle_time-2*self.gate_duration-2*self.feedback_delay

    @property
    def detector_variance(self):
        # Homodyne record rescaled to input quadrature, vacuum variance=1/2.
        return (1-self.detector_efficiency)/(2*self.detector_efficiency)+self.detector_std**2

    def record(self):
        return {**asdict(self), "idle_time": self.idle_time,
                "detector_variance": self.detector_variance,
                "duration_model": "explicit_coupled_Lindblad_trajectory" if self.gate_backend == "qutip" else "symmetric_lumped_noise_around_instantaneous_instrument",
                "cooling_model": "explicit_passive_loss_not_GKP_stabilization"}


@dataclass(frozen=True)
class Gates:
    mean_edge: float = 1e-4
    individual_edge: float = .01
    norm: float = 1e-7
    energy_limit: float = 1000.
    energy_jump: float = 1000.

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Gate/event thresholds must be positive and finite")
        if max(self.mean_edge, self.individual_edge) > 1:
            raise ValueError("Population thresholds cannot exceed one")


def stage_gates(raw, thresholds=None):
    """Cycle t is valid only if every logged stage through t passes.

    Legacy cycle-only data are explicitly marked incomplete, never upgraded
    to substep certification. Energy events are separate physical diagnostics.
    """
    g = thresholds or Gates()
    edge = np.asarray(raw["edge"])
    initial = bool(edge[:, 0].mean() < g.mean_edge and edge[:, 0].max() < g.individual_edge)
    if "stage_edge" not in raw or "stage_norm_error" not in raw:
        return {"coverage": "cycle_ends_only", "stage_checks_complete": False,
                "valid_through_cycle": None, "pass": False}
    stage_edge = np.asarray(raw["stage_edge"])
    norms = np.asarray(raw["stage_norm_error"])
    per_cycle = np.all((stage_edge.mean(0) < g.mean_edge) &
                       (stage_edge.max(0) < g.individual_edge) &
                       (norms.max(0) < g.norm), axis=1)
    instrument_pass = True
    if "max_inverse_cdf_residual" in raw:
        residual = float(raw["max_inverse_cdf_residual"])
        if np.isfinite(residual) and residual > 1e-10:
            per_cycle[:] = False
            instrument_pass = False
    if "gate_diagnostics" in raw and np.asarray(raw["gate_diagnostics"]).size:
        diagnostics = np.asarray(raw["gate_diagnostics"]).reshape(stage_edge.shape[1], -1, 3)
        instrument_cycles = np.all((diagnostics[:, :, 0] < g.norm) &
                                  (diagnostics[:, :, 1] < g.mean_edge) &
                                  (np.abs(diagnostics[:, :, 2]-1) < 1e-5), axis=1)
        per_cycle &= instrument_cycles
        diagnostics = diagnostics.reshape(-1, 3)
        instrument_pass &= bool(np.all(diagnostics[:, 0] < g.norm) and np.all(diagnostics[:, 1] < g.mean_edge)
                               and np.all(np.abs(diagnostics[:, 2]-1) < 1e-5))
    initial_norm = bool(norms[:, 0, 0].max() < g.norm)
    valid = np.logical_and.accumulate(np.r_[initial and initial_norm, per_cycle])
    return {"coverage": "every_recorded_primitive_stage", "stage_checks_complete": True,
            "valid_by_cycle": valid.tolist(), "valid_through_cycle": int(np.flatnonzero(valid)[-1]) if valid.any() else -1,
            "pass": bool(valid.all()) and instrument_pass, "instrument_pass": instrument_pass, "max_edge": float(stage_edge.max()),
            "max_mean_edge": float(stage_edge.mean(0).max()), "max_norm_error": float(norms.max()),
            "thresholds": asdict(g), "continuous_gate_bound": False}


def event_masks(raw, thresholds=None):
    g = thresholds or Gates()
    energy = np.asarray(raw["energy"])
    if "stage_energy" in raw:
        stages = np.asarray(raw["stage_energy"])
        path = np.concatenate([energy[:, :1], stages.reshape(len(energy), -1)], axis=1)
        edge = np.asarray(raw["stage_edge"]).reshape(len(energy), -1)
    else:
        path, edge = energy, np.asarray(raw["edge"])
    return {"energy_limit": (path > g.energy_limit).any(1),
            "energy_jump": (np.diff(path, axis=1) > g.energy_jump).any(1),
            "boundary_failure": (edge >= g.individual_edge).any(1)}


def binomial_interval(errors, trials, alpha=.05, comparisons=1):
    """Fixed-n Clopper-Pearson, Bonferroni simultaneous across comparisons.

    Also return the one-sided upper bound; zero counts never imply zero risk.
    Not a confidence sequence for optional stopping.
    """
    if not isinstance(trials, (int, np.integer)) or trials <= 0 or not 0 <= errors <= trials:
        raise ValueError("Require 0 <= errors <= positive integer trials")
    if not isinstance(errors, (int, np.integer)) or not isinstance(comparisons, int) or comparisons < 1 or not 0 < alpha < 1:
        raise ValueError("Integer counts/multiplicity and alpha in (0,1) required")
    a = alpha/comparisons
    low = 0. if errors == 0 else float(beta.ppf(a/2, errors, trials-errors+1))
    high = 1. if errors == trials else float(beta.ppf(1-a/2, errors+1, trials-errors))
    upper = 1. if errors == trials else float(beta.ppf(1-a, errors+1, trials-errors))
    return {"errors": int(errors), "trials": int(trials), "rate": errors/trials,
            "interval": [low, high], "one_sided_upper": upper, "alpha": alpha,
            "comparisons": comparisons, "method": "fixed_n_Clopper_Pearson_Bonferroni"}


def zero_event_budget(target_probability, alpha=.05, comparisons=1):
    if not 0 < target_probability < 1 or not 0 < alpha < 1 or comparisons < 1:
        raise ValueError("Invalid probability/alpha/multiplicity")
    return math.ceil(math.log(alpha/comparisons)/math.log1p(-target_probability))


def stratified_events(mask, per_state, alpha=.05, comparisons=1):
    v = np.asarray(mask, dtype=bool).reshape(6, per_state)
    rows = [binomial_interval(int(row.sum()), per_state, alpha, comparisons*6) for row in v]
    return {"errors": int(v.sum()), "trials": int(v.size), "rate": float(v.mean()),
            "states": rows, "interval": np.mean([r["interval"] for r in rows], axis=0).tolist(),
            "one_sided_upper": float(np.mean([r["one_sided_upper"] for r in rows])),
            "method": "equal_state_average_of_Bonferroni_Clopper_Pearson_bounds"}
