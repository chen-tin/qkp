"""Matched explicit QuTiP circuit versus local GKP measurement instruments.

See docs/qutip_engine_benchmark.md before running. Existing physics and saved
benchmarks are not changed. Use --pilot for independent pre-scale validation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from contextlib import nullcontext

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import qutip as qt
import qutip._distributions as qdist
import scipy
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    def threadpool_limits(limits):
        return nullcontext()

from quantum.measurement_benchmark import (
    ExplicitQutip, SmoothMehler, comparison, conditional_metrics, probability,
    qutip_wave_basis, outcome_grid,
)
from quantum.qutip_reference import Ancilla, CARDINAL, L, Oscillator


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["quantum/qutip_reference.py", "quantum/measurement_benchmark.py",
           "experiments/benchmark_qutip_engine.py", "docs/qutip_engine_benchmark.md",
           "tests/test_qutip_engine_benchmark.py", "simulator/sim.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def local_batch(osc, resource, data, y, axis):
    vals, vec = (osc.qvals, osc.qvectors) if axis == "q" else (osc.pvals, osc.pvectors)
    amplitudes = resource.amplitude(y[None, :]-vals[:, None])
    coefficients = vec.conj().T@data.T
    result = np.empty((len(data), osc.n, len(y)), dtype=complex)
    for i in range(len(data)):
        result[i] = vec@(coefficients[:, i, None]*amplitudes)
    return result


def explicit_batch(reference, data, axis):
    conditional = np.empty((len(data), reference.nd, reference.phi.shape[1]), dtype=complex)
    joint = np.empty((len(data), reference.nd, reference.na), dtype=complex)
    norms, edges = np.empty(len(data)), np.empty(len(data))
    for i, ket in enumerate(data):
        conditional[i], joint[i], norms[i], edges[i] = reference.conditional(ket, axis)
    return conditional, joint, norms, edges


def timed(call, repeats):
    times = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = call()
        times.append(time.perf_counter()-start)
    return value, {"seconds": times, "median_seconds": float(np.median(times)),
                   "min_seconds": min(times), "max_seconds": max(times), "repeats": repeats}


def reference_from_joint(joints, phi, axis):
    phase = (1j)**np.arange(phi.shape[0])
    return np.array([(joint if axis == "q" else joint*phase.conj()[None, :])@phi for joint in joints])


def evaluate_case(out, config, osc, data, label, sigma, na, axis, phi):
    y, w, coarse_w = outcome_grid(config["radius"], config["dy"])
    nd = osc.n
    name = f"Nd{nd}_Na{na}_{label}_{axis}"
    raw_path, summary_path = out/(name+".npz"), out/(name+".json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if sha(raw_path) != summary["raw_sha256"]:
            raise ValueError(f"Damaged cached case: {raw_path}")
        with np.load(raw_path) as z:
            refs = reference_from_joint(z["qutip_joint"], phi, axis)
            curves = {k: z[k] for k in z.files if k.endswith("_density")}
        print("RESUME", name, flush=True)
        return summary, refs, curves
    ref = ExplicitQutip(nd, na, sigma, y, wave_basis=phi)
    repeats = 1 if config["pilot"] or na != max(config["ancilla_cutoffs"]) else 3
    (reference, joint, norm_errors, anc_edges), ref_timing = timed(lambda: explicit_batch(ref, data, axis), repeats)
    raw = {"y": y, "weights": w, "coarse_weights": coarse_w, "qutip_joint": joint, "qutip_norm_errors": norm_errors,
           "qutip_ancilla_edges": anc_edges, "input_data": data, "ancilla_fock": ref.ancilla}
    rows = []
    arm_timings = {"qutip_explicit": ref_timing}
    setup = {"qutip_explicit": ref.setup_seconds}
    qutip_metrics = []
    resources = {"local_binned_0.003": lambda: Ancilla(sigma, dx=.003),
                 "local_binned_0.0015": lambda: Ancilla(sigma, dx=.0015),
                 "local_smooth": lambda: SmoothMehler(sigma)}
    snapshots = []
    for i in range(len(data)):
        cdf = np.cumsum(probability(reference[i])*w);cdf /= cdf[-1]
        snapshots.append(np.searchsorted(cdf, np.linspace(.001, .999, 64)))
    raw["snapshot_indices"] = np.array(snapshots)
    for arm in ["qutip_explicit", *resources]:
        if arm == "qutip_explicit":
            conditional = reference
        else:
            start = time.perf_counter();resource = resources[arm]();setup[arm] = time.perf_counter()-start
            conditional, timing = timed(lambda: local_batch(osc, resource, data, y, axis), repeats)
            arm_timings[arm] = timing
        arm_curves = {k: [] for k in ("probability_density", "energy_density", "logical_fidelity_density", "edge_density")}
        arm_snapshots = []
        for i, (state, _, state_axis, state_sign) in enumerate(CARDINAL):
            metrics, curves = conditional_metrics(osc, conditional[i], y, axis, state_axis, state_sign, w)
            coarse_mass = float(curves["probability_density"]@coarse_w)
            coarse_metrics = {"expected_energy": float(curves["energy_density"]@coarse_w/coarse_mass),
                              "expected_logical_fidelity": float(curves["logical_fidelity_density"]@coarse_w/coarse_mass)}
            diff, infidelity = comparison(reference[i], conditional[i], w)
            coarse_diff, _ = comparison(reference[i], conditional[i], coarse_w)
            row = {"case": name, "data_cutoff": nd, "ancilla_cutoff": na, "resource": label,
                   "sigma_m": sigma, "sigma_m_over_L": sigma/L, "axis": axis, "state": state,
                   "arm": arm, **metrics, **diff,
                   "grid_energy_change": coarse_metrics["expected_energy"]-metrics["expected_energy"],
                   "grid_logical_fidelity_change": coarse_metrics["expected_logical_fidelity"]-metrics["expected_logical_fidelity"],
                   "grid_tv_change": coarse_diff["outcome_tv"]-diff["outcome_tv"],
                   "grid_conditional_infidelity_change": coarse_diff["weighted_conditional_infidelity"]-diff["weighted_conditional_infidelity"]}
            if arm == "qutip_explicit":
                row["joint_norm_error"] = float(norm_errors[i])
                row["ancilla_final_boundary_mass"] = float(anc_edges[i])
                qutip_metrics.append(metrics)
            rows.append(row)
            for k, v in curves.items():
                arm_curves[k].append(v)
            raw[f"{arm}_{state}_conditional_infidelity"] = infidelity
            arm_snapshots.append(conditional[i][:, snapshots[i]])
        for k, v in arm_curves.items():
            raw[f"{arm}_{k}"] = np.array(v)
        raw[f"{arm}_conditional_snapshots"] = np.array(arm_snapshots)
        del conditional
    np.savez_compressed(raw_path, **raw)
    summary = {"name": name, "rows": rows, "timing": arm_timings, "setup_seconds": setup,
               "raw_file": raw_path.name, "raw_sha256": sha(raw_path), "qutip_expected_metrics": qutip_metrics}
    save(summary_path, summary)
    print(json.dumps({"case": name, "qutip_seconds": ref_timing["median_seconds"],
                      "max_norm_error": float(norm_errors.max()), "max_ancilla_edge": float(anc_edges.max()),
                      "smooth_max_tv": max(r["outcome_tv"] for r in rows if r["arm"] == "local_smooth"),
                      "binned_max_tv": max(r["outcome_tv"] for r in rows if r["arm"] == "local_binned_0.003")}), flush=True)
    curves = {k: v for k, v in raw.items() if k.endswith("_density")}
    return summary, reference, curves


def convergence_rows(previous, current, w, previous_summary, current_summary):
    rows = []
    for i, (state, *_rest) in enumerate(CARDINAL):
        metrics, _ = comparison(current[i], previous[i], w)
        a = previous_summary["qutip_expected_metrics"][i]
        b = current_summary["qutip_expected_metrics"][i]
        de = b["expected_energy"]-a["expected_energy"]
        df = b["expected_logical_fidelity"]-a["expected_logical_fidelity"]
        passed = (metrics["outcome_tv"] < 1e-4 and metrics["weighted_conditional_infidelity"] < 1e-5
                  and abs(de) < .01 and abs(df) < 1e-4)
        rows.append({"from_case": previous_summary["name"], "to_case": current_summary["name"], "state": state,
                     **metrics, "expected_energy_change": de, "expected_logical_fidelity_change": df,
                     "ancilla_refinement_pass": passed})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if qt.__version__ != "5.3.1":
        ap.error("This benchmark pins QuTiP 5.3.1, including its internal oscillator kernel")
    config = {"pilot": args.pilot, "seed": 26111907, "data_delta": .30, "logical_bins": 192,
              "data_cutoffs": [16] if args.pilot else [64, 96],
              "ancilla_cutoffs": [64, 128] if args.pilot else [256, 512, 768],
              "resources": {"broad_control": .5} if args.pilot else {"nominal": .05*L, "high_readout": .16*L},
              "axes": ["q", "p"], "radius": 18. if args.pilot else 36., "dy": .02 if args.pilot else .005,
              "quadrature": "cell_split_nested_trapezoid_one_sided_feedback_boundaries",
              "source_hashes": {s: sha(ROOT/s) for s in SOURCES},
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
              "qutip": qt.__version__, "qutip_wave_kernel_sha256": sha(qdist.__file__),
              "platform": platform.platform(), "processor": platform.processor(),
              "threads": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    config["outcome_nodes"] = len(outcome_grid(config["radius"], config["dy"])[0])
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    out = ROOT/"results/qutip_engine_benchmark"/("pilot" if args.pilot else "main")/digest
    out.mkdir(parents=True, exist_ok=True)
    if (out/"result.json").exists():
        existing = json.loads((out/"result.json").read_text())
        if existing["complete"]:
            for case in existing["cases"]:
                if sha(out/case["raw_file"]) != case["raw_sha256"]:
                    raise ValueError("Completed benchmark has a damaged raw file")
            print("RESULT", out/"result.json", flush=True)
            return
    for source in SOURCES:
        target = out/"source_snapshot"/source;target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/source, target)
    if not (out/"plan.json").exists():
        save(out/"plan.json", {"registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "planned", "config": config})
    print("RUN", out, flush=True)
    summaries, convergence, setup, solver_checks = [], [], {}, []
    y, w, _ = outcome_grid(config["radius"], config["dy"])
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        begin = time.perf_counter();all_phi = qutip_wave_basis(max(config["ancilla_cutoffs"]), y)
        setup["shared_wave_basis"] = time.perf_counter()-begin
        phi = {na: all_phi[:na] for na in config["ancilla_cutoffs"]}
        print("WAVE_BASIS", max(config["ancilla_cutoffs"]), setup["shared_wave_basis"], flush=True)
        rng = np.random.default_rng(config["seed"])
        displacements = rng.normal(0, .20*L, (6, 2))
        for nd in config["data_cutoffs"]:
            begin = time.perf_counter();osc = Oscillator(nd, config["data_delta"], logical_bins=config["logical_bins"])
            data = []
            for i, (_, coefficients, *_rest) in enumerate(CARDINAL):
                psi = (osc.code@coefficients)[:, None]
                psi = osc.displacement(psi, displacements[i:i+1, 0], displacements[i:i+1, 1])
                data.append(psi[:, 0])
            data = np.array(data)
            setup[f"common_input_and_score_Nd{nd}"] = time.perf_counter()-begin
            np.savez_compressed(out/f"inputs_Nd{nd}.npz", input_data=data, displacements=displacements,
                                logical_operators=osc.logical_ops)
            for label, sigma in config["resources"].items():
                previous = {}
                for na in config["ancilla_cutoffs"]:
                    for axis in config["axes"]:
                        summary, reference, curves = evaluate_case(out, config, osc, data, label, sigma, na, axis, phi[na])
                        summaries.append(summary)
                        if axis in previous:
                            prev_summary = previous[axis]
                            prev_na = prev_summary["rows"][0]["ancilla_cutoff"]
                            with np.load(out/prev_summary["raw_file"]) as z:
                                prev = reference_from_joint(z["qutip_joint"], phi[prev_na], axis)
                            convergence += convergence_rows(prev, reference, w, prev_summary, summary)
                            del prev
                        previous[axis] = summary
                        if not args.pilot and nd == max(config["data_cutoffs"]) and na == max(config["ancilla_cutoffs"]) and label == "nominal":
                            tight = ExplicitQutip(nd, na, sigma, y, tight=True, wave_basis=phi[na])
                            tighter = tight.conditional(data[0], axis)[0]
                            check, _ = comparison(reference[0], tighter, w)
                            solver_checks.append({"case": summary["name"], "state": "Z+", **check})
                        save(out/"result.json", {"config": config, "complete": False, "cases": summaries,
                                                "convergence": convergence, "setup_seconds": setup, "solver_checks": solver_checks})
    result = {"config": config, "complete": True, "cases": summaries, "convergence": convergence,
              "setup_seconds": setup, "solver_checks": solver_checks, "total_wall_seconds": time.perf_counter()-started,
              "scope": "One q or p measurement, common input/feedback/scoring; finite data operators. Not multi-cycle QEC or a long-horizon tail validation."}
    save(out/"result.json", result)
    all_rows = [r for s in summaries for r in s["rows"]]
    keys = list(dict.fromkeys(k for r in all_rows for k in r))
    with (out/"metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys);writer.writeheader();writer.writerows(all_rows)
    print("RESULT", out/"result.json", flush=True)


if __name__ == "__main__":
    main()
