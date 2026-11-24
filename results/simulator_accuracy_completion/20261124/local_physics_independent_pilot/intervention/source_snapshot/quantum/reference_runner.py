"""Instrumented production trajectories, including explicitly optional physics.

Private environment draws and ideal homodyne outcomes never enter a decoder.
The replay arm remains a classical controller replay on the same physical
data channels; it is not a quantum Bayesian oracle.
"""
from dataclasses import asdict
import math
import numpy as np

from quantum.channels import pure_loss, dephasing
from quantum.continuous_ancilla import ContinuousAncilla
from quantum.qutip_reference import Ancilla, CARDINAL, L
from quantum.quality import Physics, Gates, stage_gates, event_masks
from quantum.reference_noise import sample_noise
from simulator.sim import wrap, logical_flip


def _measure(osc, psi, resource, uniforms, axis, coupling):
    vals, vec = (osc.qvals, osc.qvectors) if axis == "q" else (osc.pvals, osc.pvectors)
    amp = vec.conj().T@psi
    cdf = np.cumsum(np.abs(amp)**2, axis=0);cdf[-1] = 1.
    selected = (cdf < uniforms[0][None, :]).sum(0)
    outcome = coupling*vals[selected]+resource.sample(uniforms[1], uniforms[2])
    amp *= resource.amplitude(outcome[None, :]-coupling*vals[:, None])
    norm = np.linalg.norm(amp, axis=0)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise FloatingPointError("Invalid conditional density")
    return vec@(amp/norm), outcome, 2*np.log(norm)


def simulate(osc, ancilla, params, name, per_state, rounds, seed, arm,
             physics=None, noise_extension=None, snapshots_per_state=32,
             hot_per_state=2, snapshot_cycles=None, gates=None):
    from experiments.qutip_reference import decoder, initial
    physical = Physics(**physics) if isinstance(physics, dict) else physics or Physics()
    thresholds = Gates(**gates) if isinstance(gates, dict) else gates or Gates()
    if arm not in ("finite_ancilla", "replayed_corrections"):
        raise ValueError("Unknown reference arm")
    if per_state < 2 or rounds < 1 or snapshots_per_state < 1 or hot_per_state < 0:
        raise ValueError("Invalid trajectory/snapshot budget")
    cycles = list(range(rounds+1)) if snapshot_cycles is None else sorted(set(snapshot_cycles))
    if not cycles or min(cycles) < 0 or max(cycles) > rounds:
        raise ValueError("Snapshot cycles must be inside [0, rounds]")
    psi, axes, signs = initial(osc, per_state)
    b = len(axes)
    state_ids = np.repeat(np.arange(6), per_state)
    dq, dp, nq, np_, uq, up, schedule = sample_noise(params, b, rounds, seed, noise_extension)
    # Separate named seed streams: new channels cannot perturb exogenous draws.
    loss_rng = np.random.default_rng(np.random.SeedSequence([seed, 7351]))
    phase_rng = np.random.default_rng(np.random.SeedSequence([seed, 7352]))
    detector_rng = np.random.default_rng(np.random.SeedSequence([seed, 7353]))
    gate_rng = np.random.default_rng(np.random.SeedSequence([seed, 7354]))
    exact_gates = {}
    gate_diagnostics = []
    decq, decp = decoder(name, params, b), decoder(name, params, b)
    x, z = np.zeros(b), np.zeros(b)
    fidelity = np.empty((b, rounds+1));energy = np.empty_like(fidelity);edge = np.empty_like(fidelity)
    classical = np.empty_like(fidelity)
    outcomes = np.full((2, b, rounds), np.nan)
    records = np.empty_like(outcomes);syndromes = np.empty_like(outcomes);corrections = np.empty_like(outcomes)
    log_density = np.full_like(outcomes, np.nan)
    stage_names, all_e, all_edge, all_norm = [], [], [], []
    private_loss = []
    snapshots = {}
    norm_max = 0.
    inverse_residual = 0.
    def observe():
        p = np.abs(psi)**2
        return osc.number@p, p[-max(4, osc.n//10):].sum(0), np.abs(p.sum(0)-1)
    for t in range(rounds+1):
        obs = osc.observables(psi)
        fidelity[:, t] = (1+signs*obs["bloch"][axes, np.arange(b)])/2
        energy[:, t], edge[:, t] = obs["energy"], obs["edge"]
        norm_max = max(norm_max, float(obs["norm_error"].max()))
        fx, fz = logical_flip(x), logical_flip(z)
        classical[:, t] = 1-np.where(axes == 2, fx, np.where(axes == 0, fz, fx^fz))
        if t in cycles:
            representative = np.concatenate([np.arange(k*per_state, k*per_state+min(per_state, snapshots_per_state)) for k in range(6)])
            hot = []
            for k in range(6):
                idx = np.arange(k*per_state, (k+1)*per_state)
                for values in (energy[:, t], edge[:, t]):
                    hot.extend(idx[np.argsort(values[idx], kind="stable")[-min(hot_per_state, per_state):]].tolist() if hot_per_state else [])
            selected = np.union1d(representative, hot).astype(int)
            snapshots[f"snapshot_cycle_{t}"] = psi[:, selected].copy()
            snapshots[f"snapshot_indices_{t}"] = selected
            snapshots[f"snapshot_representative_{t}"] = np.isin(selected, representative)
        if t == rounds:
            break
        labels, cycle_e, cycle_edge, cycle_norm = [], [], [], []
        def log(label):
            nonlocal norm_max
            e, ed, nr = observe()
            labels.append(label);cycle_e.append(e);cycle_edge.append(ed);cycle_norm.append(nr)
            norm_max = max(norm_max, float(nr.max()))
        def dissipate(label, duration, cooling=False):
            nonlocal psi
            eta = physical.cooling_transmissivity if cooling else math.exp(-physical.loss_rate*duration)
            if eta != 1:
                psi, lost = pure_loss(psi, eta, loss_rng)
            else:
                lost = np.zeros(b, dtype=int)
            private_loss.append(lost)
            log(label+"_loss")
            if physical.dephasing_rate and duration:
                psi = dephasing(psi, physical.dephasing_rate*duration, phase_rng)
            log(label+"_dephasing")
        log("before_cycle")
        dissipate("idle", max(0., physical.idle_time))
        psi = osc.pvectors@(np.exp(-1j*osc.pvals[:, None]*dq[:, t])*(osc.pvectors.conj().T@psi))
        log("q_displacement")
        psi = osc.qvectors@(np.exp(1j*osc.qvals[:, None]*dp[:, t])*(osc.qvectors.conj().T@psi))
        log("p_displacement")
        x += dq[:, t];z += dp[:, t]
        for a, (axis, residual, measurement, uniforms, dec) in enumerate((
                ("q", x, nq, uq, decq), ("p", z, np_, up, decp))):
            lumped_duration = physical.gate_duration/2 if physical.gate_backend == "lumped" or arm != "finite_ancilla" else 0.
            dissipate(axis+"_gate_before", lumped_duration)
            if arm == "finite_ancilla":
                widths = schedule.get("sigma_m")
                if physical.gate_backend == "qutip":
                    from quantum.finite_gate import FiniteGate
                    y, lp = np.empty(b), np.empty(b)
                    for i in range(b):
                        width = params[1]*L if widths is None else float(widths[i, t])
                        if width not in exact_gates:
                            if len(exact_gates) >= 2:
                                exact_gates.pop(next(iter(exact_gates)))
                            exact_gates[width] = FiniteGate(osc.n, width, physical)
                        psi[:, i], y[i], lp[i], diagnostic = exact_gates[width].measure(
                            psi[:, i], axis, uniforms[1, i, t], gate_rng.integers(0, 2**32-1))
                        gate_diagnostics.append(diagnostic)
                elif widths is None:
                    psi, y, lp = _measure(osc, psi, ancilla, uniforms[:, :, t], axis, physical.coupling)
                    inverse_residual = max(inverse_residual, getattr(ancilla, "max_quantile_residual", 0.))
                else:
                    y, lp = np.empty(b), np.empty(b)
                    for width in np.unique(widths[:, t]):
                        if not 0 < 2*width*width < 1:
                            raise ValueError("Schedule's quantum resource width outside supported range")
                        idx = np.flatnonzero(widths[:, t] == width)
                        resource = ContinuousAncilla(width) if isinstance(ancilla, ContinuousAncilla) else Ancilla(width, ancilla.dx)
                        updated, y[idx], lp[idx] = _measure(osc, psi[:, idx], resource, uniforms[:, idx, t], axis, physical.coupling)
                        psi[:, idx] = updated
                        inverse_residual = max(inverse_residual, getattr(resource, "max_quantile_residual", 0.))
                reported = y+detector_rng.normal(0, math.sqrt(physical.detector_variance), b)
                # Calibrated coupling converts detector coordinate back to data units.
                s = wrap(reported/physical.coupling)
                outcomes[a, :, t], records[a, :, t], log_density[a, :, t] = y, reported, lp
            else:
                s = wrap(residual+measurement[:, t])
                records[a, :, t] = residual+measurement[:, t]
            log(axis+"_measurement")
            dissipate(axis+"_gate_after", lumped_duration)
            dissipate(axis+"_feedback_delay", physical.feedback_delay)
            c = dec.decode(s);c = ((c+L) % (2*L)-L)*physical.feedback_gain
            psi = osc.correct(psi, c, axis);residual -= c
            log(axis+"_feedback")
            if physical.control_rotation:
                psi *= np.exp(-1j*physical.control_rotation*osc.number[:, None])
            log(axis+"_control_rotation")
            syndromes[a, :, t], corrections[a, :, t] = s, c
        dissipate("explicit_cooling", 0., cooling=True)
        if not stage_names:
            stage_names = labels
        elif stage_names != labels:
            raise RuntimeError("Inconsistent stage sequence")
        all_e.append(np.stack(cycle_e, 1));all_edge.append(np.stack(cycle_edge, 1));all_norm.append(np.stack(cycle_norm, 1))
    final_indices = np.concatenate([np.arange(k*per_state, k*per_state+min(per_state, snapshots_per_state)) for k in range(6)])
    result = {"fidelity": fidelity, "energy": energy, "edge": edge, "syndromes": syndromes, "corrections": corrections,
              "outcomes": outcomes, "reported_outcomes": records, "log_outcome_density": log_density,
              "log_latent_outcome_density":log_density,
              "dq": dq, "dp": dp, "nq": nq, "np": np_, "state_ids": state_ids,
              "stage_energy": np.stack(all_e, 1), "stage_edge": np.stack(all_edge, 1),
              "stage_norm_error": np.stack(all_norm, 1), "stage_names": np.array(stage_names),
              "max_norm_error": norm_max, "max_edge_weight": max(float(edge.max()), float(np.max(all_edge))),
              "score_snapshot": psi[:, final_indices], "score_indices": final_indices,
              "snapshot_cycles": np.array(cycles), **snapshots,
              "max_inverse_cdf_residual":inverse_residual if physical.gate_backend=="lumped" else np.nan,
              "environment_loss_counts": np.array(private_loss), "schema": 2}
    result["gate_diagnostics"] = np.array(gate_diagnostics).reshape(-1, 3)
    result.update({"schedule_"+key: value for key, value in schedule.items()})
    if arm == "replayed_corrections":
        result["classical_fidelity"] = classical
    # Only numeric arrays are persisted; contracts/gates go in the summary JSON.
    return result
