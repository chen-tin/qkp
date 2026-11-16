"""E14: one-bit (binary) syndromes in the closed loop.

Points (absolute noise in units of L):
  * the 16-point grid  scale {nominal: sigma_q 0.20, sigma_m 0.05;
                              hardware: sigma_q 0.072, sigma_m 0.105}
                       x rho {0, 0.9} x visibility {1.0, 0.95} x p_flip {0, 0.01},
                       all with m = 4 bits per round (sin/cos alternating);
  * the bits-per-round axis: m in {1, 8} at visibility 1, p_flip 0, for each
    scale x rho (8 points). With one weak bit per round no decoder follows
    per-round displacements of sigma_q >~ 0.07 L (docs/binary.md).

Decoders: sBs baseline (delta in {0.08, 0.15, 0.25} L), binary Bayes (matched),
binary LSTM (if its checkpoint exists, else deferred), and for reference the
analog-syndrome matched Bayes filter on the same displacement noise.

Pairing: every decoder sees the same displacement and readout noise and the
same bit-draw seed (bits depend on the decoder's corrections, so pairing is in
distribution once corrections differ). Results are cached under
results/cache/E14/<point>/ with sidecars. Scoring: eps from the decay of
1 - 2 P_L(t) and paired P_L against binary Bayes at the common horizon.

    python -m experiments.run_binary [--analyze-only]
    QKP_MODE=quick python -m experiments.run_binary
Nothing has been run at full size yet.
"""

import argparse
import hashlib
import json
import time
import traceback

import numpy as np
import torch

from decoders.bayesian.binary_filter import (
    BinaryBayesDecoder, SBSDecoder, default_phases, make_measurement,
)
from experiments.common import (
    ROOT, common_horizon, eps_fit, eval_split, get_mode, is_complete, log, make_rng, out_dir,
    paired_pl, save_json, save_npz, seed_sequence, torch_device,
)
from simulator import closed_loop as cl
from simulator.noise import L, NoiseModel

BINARY_BUDGET = {
    "full": dict(trials=4000, rounds=200),
    "quick": dict(trials=300, rounds=40),
}

BINARY_CODE_HASH = hashlib.sha1(b"".join(
    (ROOT / rel).read_bytes() for rel in (
        "simulator/closed_loop.py", "decoders/bayesian/binary_filter.py",
        "decoders/binary_lstm.py", "experiments/run_binary.py",
    ))).hexdigest()[:12]

SCALES = {"nominal": (0.20, 0.05), "hardware": (0.072, 0.105)}
SBS_DELTAS = (0.08, 0.15, 0.25)
REF = "Bayes (binary, matched)"


def points():
    out = []
    for scale, (sq, sm) in SCALES.items():
        for rho in (0.0, 0.9):
            for vis in (1.0, 0.95):
                for pf in (0.0, 0.01):
                    out.append(dict(id=f"{scale}_rho{rho}_v{vis}_pf{pf}_m4", scale=scale,
                                    sigma_q=sq, sigma_m=sm, rho=rho, visibility=vis,
                                    p_flip=pf, m=4))
            for m in (1, 8):
                out.append(dict(id=f"{scale}_rho{rho}_v1.0_pf0.0_m{m}", scale=scale,
                                sigma_q=sq, sigma_m=sm, rho=rho, visibility=1.0, p_flip=0.0,
                                m=m))
    return out


def budget():
    return BINARY_BUDGET[get_mode()]


def decoders(pt, device):
    sq, sm, rho = pt["sigma_q"] * L, pt["sigma_m"] * L, pt["rho"]
    ph = default_phases(pt["m"])
    decs = {f"sBs delta={d:.2f}L": (lambda d=d: SBSDecoder(d * L, ph)) for d in SBS_DELTAS}
    decs[REF] = lambda: BinaryBayesDecoder(sq, sm, rho, visibility=pt["visibility"],
                                           p_flip=pt["p_flip"], phases=ph, device=device)
    ck = out_dir("checkpoints") / f"binary_lstm_{pt['scale']}_m{pt['m']}_s0.pt"
    if ck.exists():
        def make_lstm(ck=ck):
            from decoders.binary_lstm import BinaryLSTMDecoder, load
            model, _ = load(ck, device)
            return BinaryLSTMDecoder(model, device=device, name="LSTM (binary)")
        decs["LSTM (binary)"] = make_lstm
    return decs, ck


def run_point(index, pt, device):
    b = budget()
    pdir = out_dir() / "cache" / "E14" / pt["id"]
    pdir.mkdir(parents=True, exist_ok=True)
    rng = make_rng(eval_split(), "E14", index)
    model = NoiseModel(pt["sigma_q"] * L, pt["sigma_m"] * L, pt["rho"])
    d, n = model.sample(rng, b["trials"], b["rounds"])
    meas = make_measurement(pt["visibility"], pt["p_flip"], m=pt["m"], sigma_m=pt["sigma_m"] * L)
    bit_seed = seed_sequence(eval_split(), "E14_bits", index)
    decs, ck = decoders(pt, device)
    if "LSTM (binary)" not in decs:
        log(f"E14/{pt['id']}: LSTM (binary) deferred ({ck.name} not trained)")
    base = dict(exp="E14", point=pt, trials=b["trials"], rounds=b["rounds"],
                split=eval_split(), code=BINARY_CODE_HASH)
    for name, make in decs.items():
        path = pdir / f"{name.replace(' ', '_').replace('/', '-')}.npz"
        config = dict(base, decoder=name)
        if name == "LSTM (binary)":
            config["checkpoint"] = json.loads((ck.with_name(ck.name + ".meta.json")).read_text())["digest"]
        if is_complete(path, config):
            continue
        t0 = time.perf_counter()
        res = cl.run(make(), d, n, meas, np.random.default_rng(bit_seed))
        rec = res.logical_errors
        save_npz(path, config, extra=dict(seconds=time.perf_counter() - t0),
                 bits=np.packbits(rec, axis=1), trials=rec.shape[0])
        log(f"E14/{pt['id']}: {name:<24} P_L(T)={rec[:, -1].mean():.3f}")
    # analog reference: matched Bayes with the analog syndrome on the same noise
    path = pdir / "Bayes_(analog,_matched).npz"
    config = dict(base, decoder="Bayes (analog, matched)")
    if not is_complete(path, config):
        from decoders.fast import FastGridDecoder, run as fast_run
        rec = fast_run(FastGridDecoder([dict(sigma_q=pt["sigma_q"] * L, sigma_m=pt["sigma_m"] * L,
                                             rho=pt["rho"])], device=device), d, n)
        save_npz(path, config, bits=np.packbits(rec, axis=1), trials=rec.shape[0])


def load_point(pt):
    pdir = out_dir() / "cache" / "E14" / pt["id"]
    out = {}
    for p in sorted(pdir.glob("*.npz")):
        meta = json.loads(p.with_name(p.name + ".meta.json").read_text())
        if meta.get("mode") != get_mode():
            continue
        with np.load(p) as z:
            rec = np.unpackbits(z["bits"], axis=1, count=meta["config"]["rounds"]).astype(bool)
            rec = rec[: int(z["trials"])]
        out[meta["config"]["decoder"]] = rec
    return out


def analyze():
    rows = []
    for pt in points():
        data = load_point(pt)
        if REF not in data:
            continue
        ref = data[REF]
        h = common_horizon(ref.mean(axis=0))
        for name, rec in data.items():
            eps, used = eps_fit(rec.mean(axis=0))
            d, se = paired_pl(rec, ref, h)
            rows.append(dict(point=pt["id"], decoder=name, eps=eps, eps_rounds_used=used,
                             horizon=h, pl_h=float(rec[:, h - 1].mean()),
                             diff_vs_binary_bayes=d, diff_se=se, **{k: pt[k] for k in pt if k != "id"}))
    if rows:
        save_json(out_dir() / "E14_summary.json", dict(exp="E14", code=BINARY_CODE_HASH), rows)
    log(f"E14 analysis: {len(rows)} rows")
    return rows


def e14():
    device = torch_device()
    for i, pt in enumerate(points()):
        try:
            run_point(i, pt, device)
        except Exception:
            log(f"E14/{pt['id']} FAILED\n{traceback.format_exc()}")
    return analyze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    from experiments.runlog import RunLog
    with RunLog("run_binary", args=vars(args)) as rl:
        with rl.phase("E14"):
            analyze() if args.analyze_only else e14()


if __name__ == "__main__":
    main()
