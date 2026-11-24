"""Read sealed paired trajectories; diagnose uncertainty without changing gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.qutip_reference import estimate
from quantum.quality import stage_gates


def diagnose_pair(low, high, per_state, tolerance=.01):
    if low["fidelity"].shape != high["fidelity"].shape:
        raise ValueError("Paired fidelity arrays need identical shape")
    if len(low["fidelity"]) != 6 * per_state:
        raise ValueError("Expected equal six-cardinal trial counts")
    for field in ("dq", "dp"):
        np.testing.assert_array_equal(low[field], high[field], err_msg="Unmatched exogenous noise")
    delta = high["fidelity"] - low["fidelity"]
    stats = estimate(delta, per_state)
    cycles = []
    for cycle, (mean, se, lo, hi) in enumerate(zip(stats["mean"], stats["se"], stats["ci95_low"], stats["ci95_high"])):
        values = delta[:, cycle]
        inside = lo >= -tolerance and hi <= tolerance
        outside = lo > tolerance or hi < -tolerance
        slack = tolerance - abs(mean)
        # Exploratory resource projection only: means/variances can change with fresh trials.
        multiplier = None if slack <= 0 else max(1., (1.95996398454 * se / slack)**2)
        extreme = max(1, math.ceil(.01 * len(values)))
        largest = np.argsort(np.abs(values))[-extreme:]
        cycles.append({"cycle": cycle, "difference": mean, "se": se, "ci95": [lo, hi],
                       "classification": "within_tolerance" if inside else "outside_tolerance" if outside else "inconclusive",
                       "max_absolute_trajectory_difference": float(np.abs(values).max()),
                       "fraction_absolute_difference_gt_tolerance": float((np.abs(values) > tolerance).mean()),
                       "largest_1pct_absolute_difference_share": float(np.abs(values[largest]).sum() / np.abs(values).sum()) if np.abs(values).sum() else 0.,
                       "same_mean_variance_total_sample_multiplier": multiplier,
                       "same_mean_variance_projected_per_state": None if multiplier is None else math.ceil(per_state * multiplier)})
    return {"trials": len(delta), "per_state": per_state, "cycles": cycles,
            "low_stage_checks": stage_gates(low), "high_stage_checks": stage_gates(high),
            "low_max_mean_photon_number": float(low["energy"].max()),
            "high_max_mean_photon_number": float(high["energy"].max()),
            "scope": "Paired original logical-score samples and recorded stages; no new trajectories, true-bias conclusion or simultaneous interval. Sample projection assumes unchanged mean/variance and is not a stopping rule."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    groups = {}
    for record in manifest["jobs"]:
        groups.setdefault(record["job"]["variant"], {})[record["job"]["cutoff"]] = record
    results = {}
    fields = ("fidelity", "dq", "dp", "edge", "stage_edge", "stage_norm_error", "max_norm_error", "energy", "gate_diagnostics")
    for variant, pair in groups.items():
        if set(pair) != {4096, 5120}:
            raise ValueError("Expected original 4096/5120 pair")
        arrays = []
        hashes = {}
        for cutoff in (4096, 5120):
            record = pair[cutoff]
            path = Path(record["local_raw"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if "sha256" in record and digest != record["sha256"]:
                raise ValueError("Imported raw hash changed")
            hashes[str(cutoff)] = {"path": str(path), "sha256": digest}
            with np.load(path, allow_pickle=False) as z:
                arrays.append({k: z[k] for k in fields if k in z})
        results[variant] = {"inputs": hashes, **diagnose_pair(*arrays, pair[4096]["job"]["config"]["per_state"])}
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "comparisons": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: {"unresolved_cycles": [r["cycle"] for r in v["cycles"] if r["classification"] != "within_tolerance"],
                           "projected_per_state": max(r["same_mean_variance_projected_per_state"] or 0 for r in v["cycles"])} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
