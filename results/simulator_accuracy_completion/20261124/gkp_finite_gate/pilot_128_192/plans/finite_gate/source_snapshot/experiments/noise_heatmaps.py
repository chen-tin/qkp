"""Paired, exploratory parameter heatmaps for the offline GKP walkthrough.

Run from qkp: .venv/bin/python -m experiments.noise_heatmaps
Reuses completed cells only when code, checkpoints and settings match. This
uses the dev split and never overwrites E1-E15 benchmark results.
"""
import argparse
import hashlib
import json
import platform
import time
import itertools
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from decoders.bayesian.grid_filter import BayesGridDecoder, MultiModelBayesDecoder
from decoders.fast import FastGridDecoder, FastLSTM, run as fast_run
from decoders.LSTM.model import LSTMDecoder, load_model
from experiments.common import NOMINAL, ROOT, SEED_ROOT, make_rng, run_blocked
from experiments.run_all import BROAD_COMPONENTS, SWITCHING
from simulator.fast import simulate_nl
from simulator.noise import L, NoiseModel
from simulator.sim import Simulator, NearestLattice, NoCorrection

VALUES = {"sigma_q": [.08, .14, .20, .26, .32],
          "sigma_m": [.04, .05, .08, .12, .16],
          "rho": [0., .3, .6, .9, .97]}
NOMINAL_UNITS = {"sigma_q": .20, "sigma_m": .05, "rho": .90}
PLANES = [
    dict(id="q_rho", x="sigma_q", y="rho", fixed="sigma_m", value=.05),
    dict(id="q_m", x="sigma_q", y="sigma_m", fixed="rho", value=.90),
    dict(id="rho_m", x="rho", y="sigma_m", fixed="sigma_q", value=.20),
]
SOURCE_FILES = ["experiments/noise_heatmaps.py", "experiments/common.py", "experiments/run_all.py",
                "simulator/noise.py", "simulator/sim.py", "simulator/fast.py",
                "decoders/fast.py", "decoders/base.py", "decoders/bayesian/grid_filter.py",
                "decoders/LSTM/model.py"]
LABELS = {"nominal": "LSTM · nominal", "broad": "LSTM · broad", "drift": "LSTM · drift",
          "hw": "LSTM · hardware", "switching": "LSTM · switching",
          "matched_rho0.5": "LSTM · trained at ρ = 0.5",
          "matched_sq0.26": "LSTM · trained at σq = 0.26L"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
    tmp.replace(path)


def point_id(p):
    def formatted(x):
        return f"{x:.3f}" if round(x,3)==x else f"{x:.17g}"
    return "_".join(f"{key}{formatted(p[key])}" for key in ("sigma_q", "sigma_m", "rho"))


def points(coverage="planes", values=None):
    values = VALUES if values is None else values
    if set(values) != set(VALUES) or any(not v for v in values.values()):
        raise ValueError("All three nonempty parameter axes are required")
    if coverage == "cube":
        return [(point_id(p), p) for row in itertools.product(*(values[k] for k in VALUES))
                for p in [dict(zip(VALUES, row))]]
    if coverage != "planes":
        raise ValueError("Coverage must be planes or cube")
    found = {}
    for plane in PLANES:
        for x in values[plane["x"]]:
            for y in values[plane["y"]]:
                p = {plane["x"]: x, plane["y"]: y, plane["fixed"]: plane["value"]}
                found[point_id(p)] = p
    # Put the nominal control and a strong-noise intervention first.
    priorities = [point_id(NOMINAL_UNITS), point_id(dict(sigma_q=.32, sigma_m=.05, rho=.3))]
    return sorted(found.items(), key=lambda item: (priorities.index(item[0]) if item[0] in priorities else 2, item[0]))


def model_specs():
    specs = [
        dict(id="nl", name="Nearest lattice", kind="simple", detail="Memoryless correction using the current syndrome.", primary=True),
        dict(id="bayes_matched", name="Bayes · matched", kind="bayes", detail="Uses the true σq, σm and ρ at every cell; grid approximation.", primary=True),
        dict(id="bayes_nominal", name="Bayes · fixed nominal", kind="bayes", detail="Always assumes σq = 0.20L, σm = 0.05L, ρ = 0.90.", primary=True),
        dict(id="bayes_broad", name="Bayes · broad", kind="bayes", detail="Full-history average of eight fixed models; all assume σm = 0.05L.", primary=True),
        dict(id="bayes_switching", name="Bayes · switching", kind="bayes", detail="Fixed two-regime model: ρ = 0.3 / 0.97, switch probability 0.05.", primary=True),
    ]
    for group in LABELS:
        for path in sorted((ROOT / "checkpoints").glob(f"lstm_{group}_s*.pt")):
            seed = int(path.stem.rsplit("_s", 1)[1])
            meta_path = path.with_name(path.name + ".meta.json")
            meta = json.loads(meta_path.read_text())
            specs.append(dict(id=path.stem, name=f"{LABELS[group]} · seed {seed}", kind="lstm",
                              checkpoint=str(path.relative_to(ROOT)), sha256=digest(path),
                              metadata_sha256=digest(meta_path), head=meta["chosen_head"],
                              training=meta.get("config", {}), group=group, seed=seed, primary=seed == 0,
                              detail="Frozen trained checkpoint; no retraining at individual cells."))
    specs.append(dict(id="no_correction", name="No correction", kind="simple", detail="Uncorrected reference: accumulate the physical displacement.", primary=False))
    return specs


def make_decoder(spec, truth, n_x, device):
    key = spec["id"]
    if spec["kind"] == "simple":
        return None
    if spec["kind"] == "lstm":
        model, ck = load_model(ROOT / spec["checkpoint"], device)
        if ck["chosen_head"] != spec["head"]:
            raise ValueError(f"Checkpoint and metadata head disagree: {key}")
        return FastLSTM(model, ck["chosen_head"], device=device, name=spec["name"])
    if key in ("bayes_matched", "bayes_nominal"):
        noise = truth if key == "bayes_matched" else NOMINAL
        ref = lambda: BayesGridDecoder(**noise, n_x=n_x, device=device)
        return ref()
    if key == "bayes_broad":
        return MultiModelBayesDecoder(BROAD_COMPONENTS, n_x=n_x, device=device)
    if key == "bayes_switching":
        models = [dict(sigma_q=SWITCHING["sigma_q"], sigma_m=SWITCHING["sigma_m"], rho=r)
                  for r in (SWITCHING["rho_lo"], SWITCHING["rho_hi"])]
        def ref():
            return BayesGridDecoder(SWITCHING["sigma_q"], SWITCHING["sigma_m"], SWITCHING["rho_lo"],
                                    rho_hi=SWITCHING["rho_hi"], p_switch=SWITCHING["p_switch"], n_x=n_x, device=device)
        return ref()
    raise ValueError(key)


def evaluate(spec, decoder, d, n, nl, block):
    if spec["id"] == "nl":
        return nl.copy()
    if spec["id"] == "no_correction":
        return np.rint(np.cumsum(d, axis=1) / L).astype(np.int64) % 2 == 1
    if spec["kind"] == "bayes":
        return run_blocked(decoder, d, n, block)
    return fast_run(decoder, d, n, block=block)


def summarize(record, nl, seconds, resets):
    # Trial-level discordance preserves paired uncertainty when comparing with NL.
    return dict(status="complete", counts=record.sum(axis=0).tolist(),
                better_than_nl=(~record & nl).sum(axis=0).tolist(),
                worse_than_nl=(record & ~nl).sum(axis=0).tolist(),
                seconds=seconds, fallback_trials=resets)


def validate(specs, n_x, device):
    """Check one control and one intervention against closed-loop drivers before the sweep."""
    validations = []
    for params in (NOMINAL_UNITS, dict(sigma_q=.32, sigma_m=.05, rho=.3)):
        truth = {k: v * L if k != "rho" else v for k, v in params.items()}
        d, n = NoiseModel(**truth).sample(make_rng("dev", "noise_heatmaps_validation"), 32, 50)
        nl = simulate_nl(d, n)
        np.testing.assert_array_equal(nl, Simulator(trials=32, rounds=50).run(NearestLattice(), d, n).logical_errors)
        for spec in specs:
            dec = make_decoder(spec, truth, n_x, device)
            actual = evaluate(spec, dec, d, n, nl, 16)
            if spec["kind"] == "bayes":
                expected = Simulator(trials=32, rounds=50).run(make_decoder(spec, truth, n_x, device), d, n).logical_errors
            elif spec["kind"] == "lstm":
                ref = LSTMDecoder(dec.model, head=spec["head"], device=device)
                expected = run_blocked(ref, d, n, 32)
            else:
                ref = NearestLattice() if spec["id"] == "nl" else NoCorrection()
                expected = Simulator(trials=32, rounds=50).run(ref, d, n).logical_errors
            agreement = float((actual == expected).mean())
            if agreement < .995:
                raise AssertionError(f"{spec['id']} at {params}: fast/reference agreement {agreement}")
            validations.append(dict(point=params, decoder=spec["id"], agreement=agreement, trials=32, rounds=50))
    return validations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--n-x", type=int, default=96)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Run the first N points only; remaining cells stay absent")
    parser.add_argument("--coverage", choices=["planes", "cube"], default="planes")
    parser.add_argument("--sigma-q-values", type=float, nargs="+", default=VALUES["sigma_q"])
    parser.add_argument("--sigma-m-values", type=float, nargs="+", default=VALUES["sigma_m"])
    parser.add_argument("--rho-values", type=float, nargs="+", default=VALUES["rho"])
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--publish-viewer", action="store_true", help="Explicitly update walkthrough data")
    args = parser.parse_args()
    if min(args.trials, args.rounds, args.block, args.n_x, args.threads) < 1:
        parser.error("All run sizes must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")  # existing filter uses float64, unsupported on MPS
    values=dict(sigma_q=sorted(set(args.sigma_q_values)),sigma_m=sorted(set(args.sigma_m_values)),rho=sorted(set(args.rho_values)))
    grid=points(args.coverage,values)
    for _,p in grid:
        NoiseModel(p["sigma_q"]*L,p["sigma_m"]*L,p["rho"])
    specs = model_specs()
    config = dict(schema=1, trials=args.trials, rounds=args.rounds, n_x=args.n_x, block=args.block,
                  split=args.split, seed_root=SEED_ROOT, stream="noise_heatmaps", seed_point=0,
                  coverage=args.coverage, values=values, planes=PLANES, decoders=specs,
                  source_hashes={p: digest(ROOT / p) for p in SOURCE_FILES},
                  torch=torch.__version__, numpy=np.__version__, python=platform.python_version(), device=str(device))
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    out = ROOT / "results" / "noise_heatmaps" / run_id
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "plan.json", dict(config=config, question="Where do stationary noise parameters cross each decoder's logical-error threshold?",
        scope=f"Exploratory {args.coverage} sweep; analog syndrome, parity corrections. Split={args.split}; fixed checkpoints.",
        prior_work="E2/E3/E4 are one-dimensional sweeps; E5 is model mismatch. Their summaries do not provide complete two-dimensional grids.",
        metric="P_L(t): fraction of independent trials with a logical error after round t. 95% pointwise Wilson intervals; paired differences use trial discordance.",
        decision="For the viewer-selected threshold, classify above only when the 95% lower bound exceeds it, below only when the upper bound is below it, otherwise uncertain. These are pointwise exploratory labels, not simultaneous confidence claims.",
        limitations="Finite Monte Carlo resolution and finite Bayes grid. Fixed checkpoints; training-seed variability shown separately. No composite/nonstationary, binary-readout, continuous-action or RL evaluations in this sweep.",
        started=datetime.now(timezone.utc).isoformat()))
    validation_path = out / "validation.json"
    if not validation_path.exists():
        print("Checking control and intervention against reference decoders...", flush=True)
        atomic_json(validation_path, validate(specs, args.n_x, device))
    validation = json.loads(validation_path.read_text())
    print(f"Validated {len(specs)} decoders at two points. Run {run_id}; {len(grid)} cells × {args.trials} trials × {args.rounds} rounds.", flush=True)
    fixed = {s["id"]: make_decoder(s, NOMINAL, args.n_x, device) for s in specs if s["id"] != "bayes_matched"}
    cells = {}
    target = ROOT / "writeup" / "artifact" / "gkp-walkthrough" / "data" / "decoder-heatmaps.json"

    def publish():
        payload = dict(schema=1, run_id=run_id, config=config, cells=cells,
                       validation=validation, expected_cells=len(grid),
                       completed_cells=sum(all(v["status"] == "complete" for v in cell["models"].values()) and len(cell["models"]) == len(specs) for cell in cells.values()),
                       updated=datetime.now(timezone.utc).isoformat(), artifact=str(out.relative_to(ROOT)))
        atomic_json(out / "summary.json", payload)
        if args.publish_viewer:
            atomic_json(target, payload)

    started = time.perf_counter()
    for index, (pid, params) in enumerate(grid):
        if args.limit is not None and index >= args.limit:
            break
        truth = {k: v * L if k != "rho" else v for k, v in params.items()}
        d, n = NoiseModel(**truth).sample(make_rng(args.split, "noise_heatmaps"), args.trials, args.rounds)
        nl = simulate_nl(d, n)
        cell = dict(parameters=params, models={})
        for spec in specs:
            key = spec["id"]
            cache = out / "cells" / pid / (key + ".npz")
            t0 = time.perf_counter()
            if cache.exists():
                with np.load(cache) as raw:
                    record = np.unpackbits(raw["packed"], axis=1, count=args.rounds).astype(bool)
                    seconds = float(raw["seconds"])
                    resets = int(raw["resets"])
            else:
                decoder = make_decoder(spec, truth, args.n_x, device) if key == "bayes_matched" else fixed[key]
                before = getattr(decoder, "resets", 0)
                record = evaluate(spec, decoder, d, n, nl, args.block)
                if record.shape != (args.trials, args.rounds):
                    raise ValueError(f"Unexpected record shape for {key}: {record.shape}")
                resets = getattr(decoder, "resets", 0) - before
                seconds = time.perf_counter() - t0
                cache.parent.mkdir(parents=True, exist_ok=True)
                temp = cache.with_suffix(".tmp.npz")
                np.savez_compressed(temp, packed=np.packbits(record, axis=1), seconds=seconds, resets=resets)
                temp.replace(cache)
            cell["models"][key] = summarize(record, nl, seconds, resets)
        cells[pid] = cell
        publish()
        print(f"[{index+1}/{len(grid)}] {pid}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
    publish()
    print(f"Results: {out/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
