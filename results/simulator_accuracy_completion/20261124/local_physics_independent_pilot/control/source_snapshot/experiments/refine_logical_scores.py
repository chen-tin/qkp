"""Refine saved-state logical scores without generating trajectories.

Supports ordinary references and the completed smooth-resource experiment.
Older final-only snapshots never certify intermediate-cycle scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from quantum.scoring import logical_fidelity, logical_fidelity_many
from experiments.qutip_reference import ROOT, case_directory, estimate, save, sha


def refine_raw(path, per_state, bins=(192, 384, 768), tolerance=.001,
               trace_tolerance=1e-7, chunk=128, backend="local", cycles=None):
    if len(bins) < 2 or any(b < 2 for b in bins) or list(bins) != sorted(set(bins)):
        raise ValueError("At least two increasing quadrature sizes required")
    if min(tolerance, trace_tolerance) <= 0:
        raise ValueError("Positive tolerances required")
    checks = []
    with np.load(path, allow_pickle=False) as raw:
        horizon = raw["fidelity"].shape[1]-1
        available = raw["snapshot_cycles"].tolist() if "snapshot_cycles" in raw else [horizon]
        selected_cycles = available if cycles is None else cycles
        if set(selected_cycles)-set(available):
            raise ValueError("Requested cycle has no saved snapshots")
        for cycle in selected_cycles:
            if f"snapshot_cycle_{cycle}" in raw:
                psi = raw[f"snapshot_cycle_{cycle}"]
                indices = raw[f"snapshot_indices_{cycle}"]
                representative = raw[f"snapshot_representative_{cycle}"]
            else:
                psi, indices = raw["score_snapshot"], raw["score_indices"]
                representative = np.ones(len(indices), dtype=bool)
            state_ids = indices//per_state
            labels = state_ids[representative]
            counts = np.bincount(labels, minlength=6)
            if not np.all(counts == counts[0]) or counts[0] < 2:
                raise ValueError("Need at least two representative snapshots per input state")
            previous = raw["fidelity"][indices, cycle]
            rows = []
            for size in bins:
                scored = logical_fidelity(psi, state_ids, size, chunk, backend)
                delta = scored["fidelity"]-previous
                uniform = delta[representative]
                # Persistent snapshots are ordered by original independent trial ID.
                summary = estimate(uniform, int(counts[0]))
                error = float(np.max(np.abs(scored["trace"]-1)))
                minimum = float(scored["minimum_eigenvalue"].min())
                rows.append({"bins": size, "difference_from": "saved_operator_score" if not rows else bins[len(rows)-1],
                             "difference": summary, "max_absolute_difference": float(np.abs(delta).max()),
                             "trace_error": error, "minimum_logical_eigenvalue": minimum,
                             "pass": bool(np.abs(delta).max() <= tolerance and error <= trace_tolerance and minimum >= -trace_tolerance)})
                previous = scored["fidelity"]
            checks.append({"cycle": int(cycle), "representative_per_state": int(counts[0]),
                           "snapshots_including_hot": len(indices), "refinements": rows,
                           "pass": all(r["pass"] for r in rows)})
    return {"raw_file": str(Path(path).resolve()), "raw_sha256": sha(path),
            "bins": list(bins), "tolerance": tolerance, "trace_tolerance": trace_tolerance,
            "coverage": "all_cycles" if set(selected_cycles) == set(range(horizon+1)) else "selected_cycles_only",
            "checks": checks, "sample_checks_pass": all(r["pass"] for r in checks),
            "all_cycle_snapshots_pass": set(selected_cycles) == set(range(horizon+1)) and all(r["pass"] for r in checks),
            "note": "Representative samples and selected hot states; not an operator-norm or all-trajectory bound."}


def refine_raw_many(paths, per_state, bins=(192,384,768), tolerance=.001,
                    trace_tolerance=1e-7, chunk=128, backend="local", cycles=None,
                    max_batch_bytes=256*1024**2, progress=None):
    """Share streamed quadrature across saved files/cycles of one cutoff.

    Inputs are read only. This preserves each bank's column and quadrature
    summation order and reports the same acceptance metrics as refine_raw.
    """
    if (len(bins)<2 or any(b<2 for b in bins) or list(bins)!=sorted(set(bins)) or
            not np.isfinite(tolerance) or not np.isfinite(trace_tolerance) or
            min(tolerance,trace_tolerance)<=0):
        raise ValueError("Increasing quadrature sizes and positive finite tolerances required")
    records=[];banks=[]
    for path in paths:
        path=Path(path);checks=[]
        with np.load(path,allow_pickle=False) as raw:
            fidelity=raw["fidelity"]
            horizon=fidelity.shape[1]-1
            available=raw["snapshot_cycles"].tolist() if "snapshot_cycles" in raw else [horizon]
            selected=available if cycles is None else list(cycles)
            if not selected or len(set(selected))!=len(selected) or set(selected)-set(available):
                raise ValueError("Unique requested cycles must have saved snapshots")
            record={"raw_file":str(path.resolve()),"raw_sha256":sha(path),"bins":list(bins),
                    "tolerance":tolerance,"trace_tolerance":trace_tolerance,
                    "coverage":"all_cycles" if set(selected)==set(range(horizon+1)) else "selected_cycles_only",
                    "checks":checks}
            records.append(record)
            for cycle in selected:
                if f"snapshot_cycle_{cycle}" in raw:
                    psi=raw[f"snapshot_cycle_{cycle}"];indices=raw[f"snapshot_indices_{cycle}"]
                    representative=raw[f"snapshot_representative_{cycle}"]
                else:
                    psi=raw["score_snapshot"];indices=raw["score_indices"]
                    representative=np.ones(len(indices),dtype=bool)
                labels=indices//per_state;counts=np.bincount(labels[representative],minlength=6)
                if len(counts)!=6 or not np.all(counts==counts[0]) or counts[0]<2:
                    raise ValueError("Need at least two representative snapshots per input state")
                check={"cycle":int(cycle),"representative_per_state":int(counts[0]),
                       "snapshots_including_hot":len(indices),"refinements":[]}
                checks.append(check)
                banks.append({"psi":psi,"labels":labels,"representative":representative,
                              "previous":fidelity[indices,cycle],"counts":counts,"check":check})
    if not banks:raise ValueError("Nonempty saved-state file list required")
    for size in bins:
        if progress:progress({"phase":"scoring","bins":size,"banks":len(banks),"state_columns":sum(b["psi"].shape[1] for b in banks)})
        scores=logical_fidelity_many([b["psi"] for b in banks],[b["labels"] for b in banks],
                    size,chunk,backend,max_batch_bytes=max_batch_bytes)
        for bank,scored in zip(banks,scores):
            delta=scored["fidelity"]-bank["previous"];rows=bank["check"]["refinements"]
            error=float(np.max(np.abs(scored["trace"]-1)))
            minimum=float(scored["minimum_eigenvalue"].min())
            rows.append({"bins":size,"difference_from":"saved_operator_score" if not rows else bins[len(rows)-1],
                         "difference":estimate(delta[bank["representative"]],int(bank["counts"][0])),
                         "max_absolute_difference":float(np.abs(delta).max()),"trace_error":error,
                         "minimum_logical_eigenvalue":minimum,
                         "pass":bool(np.abs(delta).max()<=tolerance and error<=trace_tolerance and minimum>=-trace_tolerance)})
            bank["previous"]=scored["fidelity"]
        if progress:progress({"phase":"quadrature_completed","bins":size})
    for record in records:
        for check in record["checks"]:check["pass"]=all(row["pass"] for row in check["refinements"])
        record["sample_checks_pass"]=all(c["pass"] for c in record["checks"])
        record["all_cycle_snapshots_pass"]=record["coverage"]=="all_cycles" and record["sample_checks_pass"]
        record["note"]="Representative samples and selected hot states; not an operator-norm or all-trajectory bound."
    return records


def refine_run(run, bins=(192, 384, 768), tolerance=.001, chunk=128, backend="local", cycles=None,
               shared_basis=False, max_batch_bytes=256*1024**2, output_directory=None, progress=None):
    run = Path(run).resolve()
    result = json.loads((run/"result.json").read_text())
    if not result.get("complete"):
        raise ValueError("Refine a completed run")
    config = result["config"];targets = []
    if "long" in result:
        targets = [(run/r["raw_file"], r["raw_sha256"]) for r in result["long"]["rows"]]
    else:
        for row in result["rows"]:
            folder = run/case_directory(row["point"], row["cutoff"], config["dx"], config.get("measurement_resource", "binned"))
            for arm in ("finite_ancilla", "replayed_corrections"):
                path = folder/f'{row["decoder"]}_{arm}.npz'
                targets.append((path, row["raw_hashes"][path.name]))
    checks = []
    for path, digest in targets:
        if sha(path) != digest:
            raise ValueError(f"Raw hash mismatch: {path}")
        if not shared_basis:
            checks.append(refine_raw(path, config["per_state"], bins, tolerance, chunk=chunk, backend=backend, cycles=cycles))
    if shared_basis:
        checks=refine_raw_many([path for path,_ in targets],config["per_state"],bins,tolerance,
                              chunk=chunk,backend=backend,cycles=cycles,max_batch_bytes=max_batch_bytes,progress=progress)
    settings = {"bins": list(bins), "tolerance": tolerance, "chunk": chunk, "backend": backend,
                "cycles": cycles, "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                "quantum/scoring.py": sha(ROOT/"quantum/scoring.py")}, "run_result_sha256": sha(run/"result.json")}
    if shared_basis:
        settings["shared_basis"]={"enabled":True,"max_batch_bytes":max_batch_bytes}
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    report = {"classification": "saved_state_numerical_refinement", "settings": settings, "checks": checks,
              "all_sample_checks_pass": all(r["sample_checks_pass"] for r in checks),
              "all_cycles_checked": all(r["all_cycle_snapshots_pass"] for r in checks)}
    target = (run/"quality" if output_directory is None else Path(output_directory))/f"logical_refinement_{digest}.json"
    save(target, report)
    return target, report


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument("run", type=Path)
    ap.add_argument("--bins", type=int, nargs="+", default=[192, 384, 768])
    ap.add_argument("--cycles", type=int, nargs="+")
    ap.add_argument("--tolerance", type=float, default=.001)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--backend", choices=["local", "qutip"], default="local")
    args = ap.parse_args()
    path, report = refine_run(args.run, args.bins, args.tolerance, args.chunk, args.backend, args.cycles)
    print(json.dumps({"report": str(path), "sample_checks_pass": report["all_sample_checks_pass"],
                      "all_cycles_checked": report["all_cycles_checked"]}, indent=2))


if __name__ == "__main__":
    main()
