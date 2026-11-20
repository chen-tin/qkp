"""Preregistered smooth-sampler validation and matched eight-cycle experiment."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import qutip as qt
import scipy
from scipy.integrate import cumulative_trapezoid

from experiments.qutip_reference import decoder, estimate, noise, simulation as historical_simulation
from quantum.continuous_ancilla import ContinuousAncilla
from quantum.measurement_benchmark import ExplicitQutip, comparison, local_conditional, outcome_grid, probability, qutip_wave_basis
from quantum.qutip_reference import Ancilla, CARDINAL, L, Oscillator
from simulator.sim import wrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ["quantum/continuous_ancilla.py", "quantum/measurement_benchmark.py", "quantum/qutip_reference.py",
                "experiments/smooth_ancilla_experiment.py", "experiments/qutip_reference.py",
                "docs/smooth_ancilla_experiment.md", "tests/test_continuous_ancilla.py", "simulator/noise.py", "simulator/sim.py"]
STAGES = ["before_displacement", "after_displacement", "after_q_measurement", "after_q_feedback", "after_p_measurement", "after_p_feedback"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def stage_observables(osc, psi):
    p = np.abs(psi)**2
    return osc.number@p, p[-max(4, osc.n//10):].sum(0), float(np.max(np.abs(p.sum(0)-1)))


def instrumented_measure(osc, psi, anc, uniforms, axis):
    vals, vec = (osc.qvals, osc.qvectors) if axis == "q" else (osc.pvals, osc.pvectors)
    amp = vec.conj().T@psi
    cumulative = np.cumsum(np.abs(amp)**2, axis=0);cumulative[-1] = 1.
    index = (cumulative < uniforms[0][None, :]).sum(0)
    y = vals[index]+anc.sample(uniforms[1], uniforms[2])
    amp *= anc.amplitude(y[None, :]-vals[:, None])
    norm = np.linalg.norm(amp, axis=0)
    if not np.isfinite(norm).all() or np.any(norm <= 0):
        raise FloatingPointError("Invalid conditional measurement density")
    return vec@(amp/norm), y, 2*np.log(norm)


def simulate(osc, anc, params, per_state, rounds, seed, indices=None):
    total = 6*per_state
    indices = np.arange(total) if indices is None else np.asarray(indices)
    state_ids = indices//per_state
    psi = np.column_stack([osc.code@CARDINAL[k][1] for k in state_ids])
    axes = np.array([CARDINAL[k][2] for k in state_ids]);signs = np.array([CARDINAL[k][3] for k in state_ids])
    dq, dp, _, _, uq, up = noise(params, total, rounds, seed)
    dq, dp, uq, up = dq[indices], dp[indices], uq[:, indices], up[:, indices]
    b = len(indices)
    decq, decp = decoder("nearest", params, b), decoder("nearest", params, b)
    fidelity = np.empty((b, rounds+1));energy = np.empty_like(fidelity);edge = np.empty_like(fidelity)
    stage_energy = np.empty((b, rounds, len(STAGES)));stage_edge = np.empty_like(stage_energy)
    outcomes = np.empty((2, b, rounds));logs = np.empty_like(outcomes)
    syndromes = np.empty_like(outcomes);corrections = np.empty_like(outcomes)
    max_norm_error = 0.
    for t in range(rounds+1):
        obs = osc.observables(psi)
        fidelity[:, t] = (1+signs*obs["bloch"][axes, np.arange(b)])/2
        energy[:, t], edge[:, t] = obs["energy"], obs["edge"]
        max_norm_error = max(max_norm_error, float(obs["norm_error"].max()))
        if t == rounds:
            break
        stage_energy[:, t, 0], stage_edge[:, t, 0] = obs["energy"], obs["edge"]
        psi = osc.displacement(psi, dq[:, t], dp[:, t])
        stage_energy[:, t, 1], stage_edge[:, t, 1], err = stage_observables(osc, psi)
        max_norm_error = max(max_norm_error, err)
        for a, (axis, uniforms, dec) in enumerate((("q", uq, decq), ("p", up, decp))):
            psi, y, log_density = instrumented_measure(osc, psi, anc, uniforms[:, :, t], axis)
            stage = 2+2*a
            stage_energy[:, t, stage], stage_edge[:, t, stage], err = stage_observables(osc, psi)
            max_norm_error = max(max_norm_error, err)
            s = wrap(y);c = dec.decode(s);c = (c+L)%(2*L)-L
            psi = osc.correct(psi, c, axis)
            stage_energy[:, t, stage+1], stage_edge[:, t, stage+1], err = stage_observables(osc, psi)
            max_norm_error = max(max_norm_error, err)
            outcomes[a, :, t], logs[a, :, t] = y, log_density
            syndromes[a, :, t], corrections[a, :, t] = s, c
        if b > 100:
            print(json.dumps({"event": "cycle", "cutoff": osc.n, "cycle": t+1,
                              "mean_energy": float(stage_energy[:, t, -1].mean()),
                              "max_edge": float(stage_edge[:, t, -1].max())}), flush=True)
    score = np.concatenate([np.flatnonzero(state_ids == k)[:32] for k in range(6)])
    return {"fidelity": fidelity, "energy": energy, "edge": edge, "stage_energy": stage_energy,
            "stage_edge": stage_edge, "outcomes": outcomes, "log_outcome_density": logs,
            "syndromes": syndromes, "corrections": corrections, "dq": dq, "dp": dp,
            "original_indices": indices, "max_norm_error": max_norm_error,
            "score_snapshot": psi[:, score], "score_indices": score,
            "max_inverse_cdf_residual": getattr(anc, "max_quantile_residual", 0.)}


def sampler_distance(samples, y, reference_cdf):
    u = np.sort(np.interp(samples, y, reference_cdf, left=0., right=1.))
    n = len(u)
    return max(float(np.max(np.arange(1, n+1)/n-u)), float(np.max(u-np.arange(n)/n)))


def validation(out, config):
    folder = out/"validation";folder.mkdir(exist_ok=True)
    nd = 12 if config["pilot"] else 96
    cutoffs = [64, 128] if config["pilot"] else [512, 768]
    resources = {"broad_control": .5} if config["pilot"] else {"nominal": .05*L, "high_readout": .16*L}
    y, w, coarse = outcome_grid(18 if config["pilot"] else 36, .02 if config["pilot"] else .005)
    phi = qutip_wave_basis(max(cutoffs), y)
    osc = Oscillator(nd, logical_bins=192)
    rng = np.random.default_rng(config["validation_seed"])
    displacements = rng.normal(0, .20*L, (6, 2))
    data = np.column_stack([osc.code@c for _, c, _, _ in CARDINAL])
    data = osc.displacement(data, displacements[:, 0], displacements[:, 1]).T
    rows, refinements = [], []
    sampler_n = 4096 if config["pilot"] else 65536
    sampler_bound = math.sqrt(math.log(2*6*2*len(resources)/.05)/(2*sampler_n))
    previous = {}
    for label, sigma in resources.items():
        anc = ContinuousAncilla(sigma)
        quantiles = np.r_[np.geomspace(1e-12, .01, 32), rng.uniform(1e-12, 1-1e-12, 16384), 1-np.geomspace(1e-12, .01, 32)]
        residual = float(np.max(np.abs(anc.cdf(anc.sample(quantiles))-quantiles)))
        for na in cutoffs:
            ref = ExplicitQutip(nd, na, sigma, y, wave_basis=phi[:na])
            for axis in ("q", "p"):
                key = f"{label}_{axis}";case = f"Nd{nd}_Na{na}_{key}"
                saved = {"y": y, "weights": w, "coarse_weights": coarse, "input_data": data, "displacements": displacements}
                ref_states = []
                for i, (state, *_rest) in enumerate(CARDINAL):
                    reference, joint, norm_error, anc_edge = ref.conditional(data[i], axis)
                    ref_states.append(reference)
                    saved[f"qutip_joint_{i}"] = joint
                    p = probability(reference);mass = float(p@w)
                    for arm, resource in (("smooth", anc), ("binned", Ancilla(sigma, .003))):
                        candidate = local_conditional(osc, resource, data[i], y, axis)
                        metrics, infid = comparison(reference, candidate, w)
                        metrics_coarse, _ = comparison(reference, candidate, coarse)
                        row = {"case": case, "data_cutoff": nd, "ancilla_cutoff": na, "resource": label,
                               "axis": axis, "state": state, "arm": arm, **metrics,
                               "accuracy": 1-metrics["weighted_conditional_infidelity"],
                               "grid_infidelity_change": metrics_coarse["weighted_conditional_infidelity"]-metrics["weighted_conditional_infidelity"],
                               "grid_tv_change": metrics_coarse["outcome_tv"]-metrics["outcome_tv"],
                               "reference_norm_error": norm_error, "reference_ancilla_edge": anc_edge,
                               "inverse_cdf_residual": residual}
                        saved[f"{arm}_probability_{i}"] = probability(candidate)
                        saved[f"{arm}_infidelity_{i}"] = infid
                        if arm == "smooth" and na == max(cutoffs):
                            vals, vec = (osc.qvals, osc.qvectors) if axis == "q" else (osc.pvals, osc.pvectors)
                            prob = np.abs(vec.conj().T@data[i])**2
                            u = rng.uniform(1e-12, 1-1e-12, (2, sampler_n))
                            selected = np.searchsorted(np.cumsum(prob), u[0]).clip(0, nd-1)
                            draws = vals[selected]+anc.sample(u[1])
                            cdf = np.r_[0., cumulative_trapezoid(p, y)];cdf /= cdf[-1]
                            distance = sampler_distance(draws, y, cdf)
                            row.update({"sampler_draws": sampler_n, "sampler_ks_distance": distance,
                                        "familywise_dkw95_bound": sampler_bound,
                                        "sampler_ks_pass": distance <= sampler_bound+1e-4})
                            saved[f"sampler_draws_{i}"] = draws
                        rows.append(row)
                    saved[f"qutip_probability_{i}"] = p
                current = np.array(ref_states)
                if key in previous:
                    for i, (state, *_rest) in enumerate(CARDINAL):
                        metrics, _ = comparison(current[i], previous[key][i], w)
                        refinements.append({"case": case, "state": state, **metrics,
                                            "pass": metrics["outcome_tv"] <= 1e-4 and metrics["weighted_conditional_infidelity"] <= 1e-5})
                previous[key] = current
                path = folder/(case+".npz");np.savez_compressed(path, **saved)
                print(json.dumps({"event": "validation", "case": case, "rows": len(rows)}), flush=True)
                save(folder/"result.json", {"complete": False, "rows": rows, "refinements": refinements})
    final = [r for r in rows if r["ancilla_cutoff"] == max(cutoffs)]
    smooth = [r for r in final if r["arm"] == "smooth"]
    accuracy_pass = all(r["accuracy"] >= .99999 and r["outcome_tv"] <= 1e-4 and r["reference_norm_error"] < 1e-7 and abs(r["reference_integrated_mass"]-1) < 1e-5 for r in smooth)
    sampling_pass = all(r["inverse_cdf_residual"] <= 1e-10 and r["sampler_ks_pass"] for r in smooth)
    result = {"complete": True, "rows": rows, "refinements": refinements,
              "primary_accuracy_pass": accuracy_pass, "sampler_pass": sampling_pass,
              "reference_refinement_pass": all(r["pass"] for r in refinements),
              "raw_hashes": {p.name: sha(p) for p in folder.glob("*.npz")}}
    save(folder/"result.json", result)
    return result


def historical_control(n, config):
    rid = "dd4e949e01da259d" if n == 5120 else "e042e0a1b30d34d0"
    run = ROOT/"results/qutip_reference/convergence"/rid
    result = json.loads((run/"result.json").read_text())
    for key in ("seed", "data_delta", "per_state", "rounds", "logical_bins"):
        if config[key] != result["config"][key]:
            raise ValueError(f"Unmatched historical control: {key}")
    row = next(r for r in result["rows"] if r["cutoff"] == n)
    path = run/f"nominal_N{n}_dx0.003/nearest_finite_ancilla.npz"
    if sha(path) != row["raw_hashes"][path.name]:
        raise ValueError("Historical raw hash mismatch")
    with np.load(path) as z:
        raw = {k: z[k] for k in ("fidelity", "energy", "edge", "dq", "dp", "syndromes", "corrections")}
    return raw, str(path.relative_to(ROOT)), row["raw_hashes"][path.name]


def run_long(out, config):
    rows, cutoffs = [], []
    previous = None
    params = (.10, .5/L, 0.) if config["pilot"] else (.20, .05, .90)
    selected = [0, 1, 2] if config["pilot"] else [1659, 1973, 2776]
    for n in config["cutoffs"]:
        start = time.perf_counter();osc = Oscillator(n, config["data_delta"], logical_bins=config["logical_bins"])
        folder = out/f"N{n}";folder.mkdir(exist_ok=True)
        path = folder/"smooth.npz"
        if path.exists():
            with np.load(path) as z:
                raw = {k: z[k] for k in z.files}
        else:
            raw = simulate(osc, ContinuousAncilla(params[1]*L), params, config["per_state"], config["rounds"], config["seed"])
            np.savez_compressed(path, **raw)
        if config["pilot"]:
            ctrl = historical_simulation(osc, Ancilla(params[1]*L), params, "nearest", config["per_state"], config["rounds"], config["seed"], "finite_ancilla")
            ctrl_path, ctrl_hash = "pilot_live_historical_simulation", None
        else:
            ctrl, ctrl_path, ctrl_hash = historical_control(n, config)
            np.testing.assert_array_equal(raw["dq"], ctrl["dq"]);np.testing.assert_array_equal(raw["dp"], ctrl["dp"])
        passed = np.logical_and.accumulate((raw["edge"].mean(0) < 1e-4) & (raw["edge"].max(0) < .01))
        row = {"cutoff": n, "smooth_fidelity": estimate(raw["fidelity"], config["per_state"]),
               "binned_fidelity": estimate(ctrl["fidelity"], config["per_state"]),
               "smooth_minus_binned": estimate(raw["fidelity"]-ctrl["fidelity"], config["per_state"]),
               "mean_energy": raw["energy"].mean(0).tolist(), "max_energy": raw["energy"].max(0).tolist(),
               "energy_q99": np.quantile(raw["energy"], .99, axis=0).tolist(),
               "mean_edge": raw["edge"].mean(0).tolist(), "max_edge": raw["edge"].max(0).tolist(),
               "population_pass": passed.tolist(), "valid_through_cycle": int(np.flatnonzero(passed)[-1]) if passed.any() else -1,
               "max_norm_error": float(raw["max_norm_error"]),
               "max_inverse_cdf_residual": float(raw["max_inverse_cdf_residual"]),
               "raw_file": str(path.relative_to(out)), "raw_sha256": sha(path),
               "control_file": ctrl_path, "control_sha256": ctrl_hash, "seconds": time.perf_counter()-start}
        forensic = {}
        for arm, resource in (("binned", Ancilla(params[1]*L)), ("smooth", ContinuousAncilla(params[1]*L))):
            observed = simulate(osc, resource, params, config["per_state"], config["rounds"], config["seed"], indices=selected)
            fpath = folder/f"forensic_{arm}.npz";np.savez_compressed(fpath, **observed)
            target = ctrl if arm == "binned" else raw
            errors = {k: float(np.max(np.abs(observed[k]-target[k][selected]))) for k in ("fidelity", "energy", "edge")}
            forensic[arm] = {"path": str(fpath.relative_to(out)), "sha256": sha(fpath), "max_replay_errors": errors}
        row["forensic"] = forensic
        if previous is not None:
            diff = estimate(raw["fidelity"]-previous[1], config["per_state"])
            diff.update({"from_cutoff": previous[0], "to_cutoff": n,
                         "within_tolerance": ((np.array(diff["ci95_low"]) >= -.01) & (np.array(diff["ci95_high"]) <= .01)).tolist()})
            cutoffs.append(diff)
        previous = n, raw["fidelity"].copy()
        rows.append(row)
        save(out/"long_result.json", {"complete": False, "rows": rows, "cutoff_comparisons": cutoffs, "stages": STAGES})
        print(json.dumps({"event": "long_case", "cutoff": n, "horizon": row["valid_through_cycle"], "max_edge": max(row["max_edge"]), "seconds": row["seconds"]}), flush=True)
        del raw, ctrl, osc
        gc.collect()
    result = {"complete": True, "rows": rows, "cutoff_comparisons": cutoffs, "stages": STAGES}
    save(out/"long_result.json", result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    config = {"pilot": args.pilot, "seed": 26111901, "validation_seed": 26111923,
              "data_delta": .30, "per_state": 4 if args.pilot else 512, "rounds": 2 if args.pilot else 8,
              "logical_bins": 192, "cutoffs": [64] if args.pilot else [3072, 4096, 5120],
              "source_hashes": {s: sha(ROOT/s) for s in SOURCE_FILES}, "python": platform.python_version(),
              "numpy": np.__version__, "scipy": scipy.__version__, "qutip": qt.__version__}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    out = ROOT/"results/smooth_ancilla_experiment"/("pilot" if args.pilot else "main")/digest;out.mkdir(parents=True, exist_ok=True)
    print("RUN", out, flush=True)
    if (out/"result.json").exists() and json.loads((out/"result.json").read_text()).get("complete"):
        print("RESULT", out/"result.json", flush=True);return
    for source in SOURCE_FILES:
        target = out/"source_snapshot"/source;target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(ROOT/source, target)
    if not (out/"plan.json").exists():
        save(out/"plan.json", {"registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "config": config})
    start = time.perf_counter()
    validated = validation(out, config)
    if not (validated["primary_accuracy_pass"] and validated["sampler_pass"] and validated["reference_refinement_pass"]):
        save(out/"result.json", {"config": config, "complete": True, "decision": "do_not_promote_validation_failed", "validation": validated})
        print("RESULT", out/"result.json", flush=True);return
    gc.collect()
    long = run_long(out, config)
    final_comparison = next((c for c in long["cutoff_comparisons"] if c["from_cutoff"] == 4096 and c["to_cutoff"] == 5120), None)
    horizon_pass = all(r["valid_through_cycle"] >= config["rounds"] and r["max_norm_error"] < 1e-7 for r in long["rows"] if r["cutoff"] in (4096, 5120))
    stable = final_comparison is not None and all(final_comparison["within_tolerance"]) and horizon_pass
    decision = "pilot_completed" if args.pilot else "adopt_smooth_for_reference" if stable else "smooth_accuracy_passes_long_horizon_inconclusive"
    save(out/"result.json", {"config": config, "complete": True, "decision": decision,
                             "validation": validated, "long": long, "wall_seconds": time.perf_counter()-start})
    print("RESULT", out/"result.json", flush=True)


if __name__ == "__main__":main()
