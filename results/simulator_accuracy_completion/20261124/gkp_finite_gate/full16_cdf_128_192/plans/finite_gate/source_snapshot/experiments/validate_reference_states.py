"""Independent two-mode QuTiP validation on supplied or high-energy states.

No data-basis truncation/promotion is silently performed. Memory/work budgets
are explicit; single-extraction verification is not multicycle certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import qutip as qt

from quantum.measurement_benchmark import (ExplicitQutip, local_conditional, comparison, outcome_grid,
                                           REFERENCE_WAVE_BASIS)
from quantum.continuous_ancilla import ContinuousAncilla
from quantum.qutip_reference import Oscillator
from experiments.qutip_reference import ROOT, save, sha


def load_inputs(path=None, cycle=None, columns=None, cutoff=96, fock_levels=None):
    if path is None:
        levels = [0, cutoff//2, cutoff-4] if fock_levels is None else fock_levels
        if not levels or min(levels) < 0 or max(levels) >= cutoff:
            raise ValueError("Fock inputs must fit the declared cutoff")
        data = np.eye(cutoff, dtype=complex)[:, levels]
        provenance = {"kind": "Fock_states", "levels": levels, "cutoff": cutoff}
    else:
        with np.load(path, allow_pickle=False) as raw:
            key = "score_snapshot" if cycle is None else f"snapshot_cycle_{cycle}"
            if key not in raw:
                raise ValueError(f"No saved states at {key}")
            data = raw[key]
            index_key="score_indices" if cycle is None else f"snapshot_indices_{cycle}"
            original=raw[index_key] if index_key in raw else np.arange(data.shape[1])
            labels=None
            if "state_ids" in raw:
                labels=raw["state_ids"][original]
            elif "fidelity" in raw and raw["fidelity"].shape[0]%6==0:
                labels=original//(raw["fidelity"].shape[0]//6)
            if columns is None:
                if labels is not None and set(labels.tolist())==set(range(6)):
                    columns=[int(np.flatnonzero(labels==k)[0]) for k in range(6)]
                elif data.shape[1]<=6:
                    columns=list(range(data.shape[1]))
                else:
                    raise ValueError("Unlabeled state bank: select explicit --columns to avoid biased default inputs")
            data = data[:, columns]
        provenance = {"kind": "saved_states", "path": str(Path(path).resolve()),
                      "sha256": sha(path), "key": key, "columns": columns,
                      "original_trial_ids":original[columns].tolist(),
                      "cardinal_input_ids":None if labels is None else labels[columns].tolist()}
    if data.ndim != 2 or not np.isfinite(data).all() or not np.allclose((np.abs(data)**2).sum(0), 1, atol=1e-8, rtol=0):
        raise ValueError("Inputs must be complete normalized oscillator kets")
    provenance["mean_photon_number_per_input"]=(np.arange(len(data))@(np.abs(data)**2)).tolist()
    return data, provenance


def _checks_pass(rows, refinements):
    """One numerical acceptance rule shared by execution and raw-data audit."""
    if not rows or not refinements:
        return False
    final = [r for r in rows if r["ancilla_cutoff"] == max(r["ancilla_cutoff"] for r in rows)]
    return all(r["outcome_tv"] <= 1e-4 and r["weighted_conditional_infidelity"] <= 1e-5 and
               abs(r["reference_integrated_mass"]-1) <= 1e-5 and
               abs(r["candidate_integrated_mass"]-1) <= 1e-5 and
               r["joint_norm_error"] <= 1e-7 and r["ancilla_boundary_mass"] < 1e-4 and
               r["grid_infidelity_change"] <= 1e-6 and r["grid_tv_change"] <= 1e-4 for r in final) and all(
                   r["pass"] for r in refinements)


def validate(data, sigma, cutoffs, radius, dy, coupling=1., max_joint=2_000_000,
             max_output_bytes=512_000_000, output=None):
    data = np.asarray(data, complex)
    if data.ndim!=2 or min(data.shape)<1 or not np.isfinite(data).all() or not np.allclose((np.abs(data)**2).sum(0),1,atol=1e-8,rtol=0):
        raise ValueError("Complete normalized input kets required")
    if len(cutoffs) < 2 or sorted(set(cutoffs)) != list(cutoffs):
        raise ValueError("Two or more increasing ancillary cutoffs required")
    nd = len(data)
    y, w, coarse = outcome_grid(radius, dy)
    estimated = (16*nd*len(y)*(2*data.shape[1]+3)+8*max(cutoffs)*len(y)
                 +128*nd*nd+32*nd*max(cutoffs))
    if nd*max(cutoffs) > max_joint or estimated > max_output_bytes:
        raise ValueError(f"Explicit validation budget exceeded: joint={nd*max(cutoffs)}, scratch~{estimated} bytes")
    osc = Oscillator(nd, logical_bins=192)
    resource = ContinuousAncilla(sigma)
    rows, previous, refinements, raw_hashes = [], {}, [], {}
    for na in cutoffs:
        ref = ExplicitQutip(nd, na, sigma, y, coupling=coupling)
        for axis in ("q", "p"):
            for column in range(data.shape[1]):
                candidate = local_conditional(osc, resource, data[:, column], y, axis, coupling)
                explicit, joint, norm_error, anc_edge = ref.conditional(data[:, column], axis)
                metrics, infidelity = comparison(explicit, candidate, w)
                coarse_metrics, _ = comparison(explicit, candidate, coarse)
                key = (axis, column)
                if key in previous:
                    delta, _ = comparison(explicit, previous[key], w)
                    refinements.append({"ancilla_cutoff": na, "axis": axis, "column": column, **delta,
                                        "pass": delta["outcome_tv"] <= 1e-4 and delta["weighted_conditional_infidelity"] <= 1e-5})
                previous[key] = explicit
                rows.append({"ancilla_cutoff": na, "axis": axis, "column": column, **metrics,
                             "joint_norm_error": norm_error, "ancilla_boundary_mass": anc_edge,
                             "grid_infidelity_change": abs(coarse_metrics["weighted_conditional_infidelity"]-metrics["weighted_conditional_infidelity"]),
                             "grid_tv_change": abs(coarse_metrics["outcome_tv"]-metrics["outcome_tv"])})
                if output is not None:
                    path=Path(output)/f"Na{na}_{axis}_input{column}.npz"
                    np.savez_compressed(path,y=y,weights=w,coarse_weights=coarse,joint=joint,
                                        reference_probability=(np.abs(explicit)**2).sum(0),
                                        candidate_probability=(np.abs(candidate)**2).sum(0),infidelity=infidelity)
                    raw_hashes[path.name]=sha(path)
    return {"rows": rows, "refinements": refinements, "raw_hashes":raw_hashes,
            "reference_wave_basis":REFERENCE_WAVE_BASIS,
            "pass": _checks_pass(rows, refinements),
            "scope": "single extraction on supplied finite data operators; common preparation/feedback/scoring",
            "scratch_bytes_estimate": estimated}


def audit_report(folder):
    folder=Path(folder);result=json.loads((folder/"result.json").read_text())
    if not result.get("complete") or sha(folder/"inputs.npz")!=result["input_sha256"]:
        raise ValueError("Incomplete validation or changed inputs")
    if not result.get("raw_hashes"):
        raise ValueError("Validation report has no auditable raw observations")
    if result.get("validation_backend") == "streamed_instrument_v1":
        from quantum.streaming_measurement import audit_streamed_report
        return audit_streamed_report(folder, result)
    from quantum.measurement_benchmark import qutip_wave_basis
    with np.load(folder/"inputs.npz", allow_pickle=False) as inputs:
        input_count = inputs["states"].shape[1]
        data_cutoff = inputs["states"].shape[0]
    rows = result["rows"]
    cutoffs = sorted({r["ancilla_cutoff"] for r in rows})
    if ("ancilla_cutoffs" in result["config"] and
            cutoffs != result["config"]["ancilla_cutoffs"]):
        raise ValueError("Instrument rows do not cover declared ancillary refinements")
    expected = {(na, axis, column) for na in cutoffs for axis in ("q", "p") for column in range(input_count)}
    actual = [(r["ancilla_cutoff"], r["axis"], r["column"]) for r in rows]
    if len(cutoffs) < 2 or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete instrument row/refinement coverage")
    previous, recomputed_refinements = {}, []
    basis_cutoff, basis, grid = None, None, None
    for row in sorted(rows, key=lambda r: (r["ancilla_cutoff"], r["axis"], r["column"])):
        name=f'Na{row["ancilla_cutoff"]}_{row["axis"]}_input{row["column"]}.npz'
        path=folder/name
        if sha(path)!=result["raw_hashes"][name]:
            raise ValueError("Instrument raw hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            if grid is None:
                grid = z["y"].copy()
            else:
                np.testing.assert_array_equal(z["y"], grid, err_msg="Instrument outcome grids differ")
            pr,pc,w=z["reference_probability"],z["candidate_probability"],z["weights"]
            mr,mc=float(pr@w),float(pc@w)
            coarse, joint = z["coarse_weights"], z["joint"]
            if joint.shape != (data_cutoff, row["ancilla_cutoff"]):
                raise ValueError("Instrument joint state dimensions differ from inputs/configuration")
            mrc, mcc = float(pr@coarse), float(pc@coarse)
            if min(mr, mc, mrc, mcc) <= 0 or not np.isfinite([mr, mc, mrc, mcc]).all():
                raise ValueError("Invalid audited probability mass")
            weighted_infidelity = float((pr*z["infidelity"])@w/mr)
            tv = float(.5*np.abs(pr/mr-pc/mc)@w)
            checked = {"weighted_conditional_infidelity": weighted_infidelity, "outcome_tv": tv,
                       "reference_integrated_mass": mr, "candidate_integrated_mass": mc,
                       "joint_norm_error": float(abs(np.vdot(joint, joint).real-1)),
                       "ancilla_boundary_mass": float((np.abs(joint[:, -max(4, joint.shape[1]//10):])**2).sum()),
                       "grid_infidelity_change": abs(float((pr*z["infidelity"])@coarse/mrc)-weighted_infidelity),
                       "grid_tv_change": abs(float(.5*np.abs(pr/mrc-pc/mcc)@coarse)-tv)}
            for key, value in checked.items():
                np.testing.assert_allclose(value,row[key],rtol=1e-10,atol=1e-13,err_msg=key)
            weighted = joint if row["axis"] == "q" else joint*(1j**np.arange(joint.shape[1])).conj()[None, :]
            if basis_cutoff != row["ancilla_cutoff"]:
                basis_cutoff = row["ancilla_cutoff"]
                basis = qutip_wave_basis(basis_cutoff, z["y"])
            explicit = weighted@basis
            np.testing.assert_allclose((np.abs(explicit)**2).sum(0),pr,rtol=1e-10,atol=1e-13)
            key = (row["axis"], row["column"])
            if key in previous:
                delta, _ = comparison(explicit, previous[key], w)
                recomputed_refinements.append({"ancilla_cutoff":row["ancilla_cutoff"],
                    "axis":row["axis"], "column":row["column"], **delta,
                    "pass":delta["outcome_tv"] <= 1e-4 and delta["weighted_conditional_infidelity"] <= 1e-5})
            previous[key] = explicit
    if len(result["refinements"]) != len(recomputed_refinements):
        raise ValueError("Incomplete ancillary refinement evidence")
    for actual, expected in zip(sorted(result["refinements"],key=lambda r:(r["ancilla_cutoff"],r["axis"],r["column"])), recomputed_refinements):
        for key, value in expected.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                np.testing.assert_allclose(actual[key],value,rtol=1e-10,atol=1e-13,err_msg="refinement "+key)
            elif actual[key] != value:
                raise ValueError("Changed ancillary refinement "+key)
    if result["pass"] != _checks_pass(rows, recomputed_refinements):
        raise ValueError("Instrument pass flag disagrees with raw numerical checks")
    for path,digest in result["config"]["source_hashes"].items():
        if sha(folder/"source_snapshot"/path)!=digest:
            raise ValueError("Instrument source snapshot mismatch")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", type=Path);ap.add_argument("--cycle", type=int)
    ap.add_argument("--columns", type=int, nargs="+")
    ap.add_argument("--data-cutoff", type=int, default=96)
    ap.add_argument("--fock-levels", type=int, nargs="+")
    ap.add_argument("--ancilla-cutoffs", type=int, nargs="+", default=[512, 768])
    ap.add_argument("--sigma", type=float, default=.05*np.sqrt(np.pi), help="Absolute quadrature units")
    ap.add_argument("--radius", type=float, default=60.)
    ap.add_argument("--dy", type=float, default=.005)
    ap.add_argument("--coupling", type=float, default=1.)
    ap.add_argument("--max-joint", type=int, default=2_000_000)
    ap.add_argument("--max-scratch-mb", type=float, default=512)
    ap.add_argument("--streaming", action="store_true", help="Stream conditional grid blocks; preserve full Fock data cutoff")
    ap.add_argument("--chunk", type=int, default=64, help="Maximum conditional outcome columns retained in streaming mode")
    ap.add_argument("--circuit-backend", choices=["sesolve","spectral"],default="sesolve",
                    help="Independent two-mode circuit evolution for streaming validation")
    ap.add_argument("--preflight-only", action="store_true", help="Report streaming capacity without state evolution")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    if not args.streaming and args.circuit_backend!="sesolve":
        ap.error("--circuit-backend spectral requires --streaming")
    if args.audit_only:
        result=audit_report(args.output)
        print(json.dumps({"raw_and_sources_verified":True,"pass":result["pass"]}));return
    data, provenance = load_inputs(args.states, args.cycle, args.columns, args.data_cutoff, args.fock_levels)
    if args.preflight_only:
        if not args.streaming:
            ap.error("--preflight-only requires --streaming")
        from quantum.streaming_measurement import capacity
        print(json.dumps({"inputs":provenance, **capacity(len(data), args.ancilla_cutoffs,
                         args.radius, args.dy, args.chunk, data.shape[1], args.circuit_backend)}));return
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config["inputs"] = provenance
    config["reference_wave_basis"] = REFERENCE_WAVE_BASIS
    config["source_hashes"] = {p:sha(ROOT/p) for p in ["experiments/validate_reference_states.py", "quantum/measurement_benchmark.py", "quantum/continuous_ancilla.py", "quantum/qutip_reference.py"]}
    if args.streaming:
        config["source_hashes"]["quantum/streaming_measurement.py"] = sha(ROOT/"quantum/streaming_measurement.py")
        if args.circuit_backend=="spectral":
            config["source_hashes"]["quantum/spectral_circuit.py"] = sha(ROOT/"quantum/spectral_circuit.py")
    config["python"], config["qutip"] = platform.python_version(), qt.__version__
    args.output.mkdir(parents=True, exist_ok=True)
    plan = args.output/"plan.json"
    if plan.exists() and json.loads(plan.read_text()) != config:
        raise ValueError("Output belongs to another validation configuration")
    save(plan, config)
    for path,digest in config["source_hashes"].items():
        if sha(ROOT/path)!=digest:raise ValueError("Source changed during launch")
        target=args.output/"source_snapshot"/path
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/path).read_bytes())
    input_path=args.output/"inputs.npz";np.savez_compressed(input_path,states=data)
    save(args.output/"inputs.json", provenance)
    if args.streaming:
        from quantum.streaming_measurement import validate_streamed
        result = validate_streamed(data, args.sigma, args.ancilla_cutoffs, args.radius, args.dy,
                        args.coupling, args.max_joint, int(args.max_scratch_mb*1e6),args.output,args.chunk,args.circuit_backend)
    else:
        result = validate(data, args.sigma, args.ancilla_cutoffs, args.radius, args.dy,
                          args.coupling, args.max_joint, int(args.max_scratch_mb*1e6),args.output)
    save(args.output/"result.json", {"complete": True, "config": config,
                                   "input_sha256":sha(input_path), **result})
    print(json.dumps({"result":str(args.output/"result.json"),"pass":result["pass"]}))


if __name__ == "__main__":
    main()
