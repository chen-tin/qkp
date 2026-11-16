"""E15: continuous corrections and an RL (DQN) decoder, in the closed loop.

Points (absolute units below; sigma in L):
    nominal        sigma_q 0.20, sigma_m 0.05,  rho 0.9
    hw             sigma_q 0.072, sigma_m 0.105, rho 0.9   (hardware scale, docs/benchmarks.md B)
    hw_sm0.05/0.10/0.15   sigma_q 0.072, rho 0.9, sigma_m swept
Decoders (all in simulator/closed_loop.py, paired noise; the analog syndrome is
deterministic given the noise, so every decoder sees the same d and nu):
    NL                    nearest lattice
    Bayes (parity)        matched grid filter, c = s + jL (= BayesGridDecoder)
    Bayes (fine)          matched grid filter, c = s + jL + posterior offset
    DQN (parity)          trained agent, actions j in {0,1}
    DQN (fine)            trained agent, actions j x 9 fine offsets in [-0.25L, 0.25L]
DQN checkpoints come from `python -m rl.train`; missing ones are deferred.

Scored with experiments.common.eps_fit (logical error per round from the
decay of 1 - 2 P_L) and paired P_L at the common horizon against Bayes (parity).

    python -m rl.train                            # train the DQN agents first
    python -m experiments.run_continuous          # evaluate (cached, resumable)
    python -m experiments.run_continuous --analyze
Nothing here has been run at full size yet.
"""

import argparse
import hashlib
import json

import numpy as np

from experiments.common import (
    ROOT, common_horizon, eps_fit, eval_split, get_mode, is_complete, log, make_rng, out_dir,
    paired_pl, save_npz, sidecar_path, torch_device,
)
from simulator import closed_loop as cl
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice

POINTS = {
    "nominal": dict(sigma_q=0.20 * L, sigma_m=0.05 * L, rho=0.9),
    "hw": dict(sigma_q=0.072 * L, sigma_m=0.105 * L, rho=0.9),
    "hw_sm0.05": dict(sigma_q=0.072 * L, sigma_m=0.05 * L, rho=0.9),
    "hw_sm0.10": dict(sigma_q=0.072 * L, sigma_m=0.10 * L, rho=0.9),
    "hw_sm0.15": dict(sigma_q=0.072 * L, sigma_m=0.15 * L, rho=0.9),
}
SIZES = {"full": dict(trials=20_000, rounds=200), "quick": dict(trials=400, rounds=60)}
REF = "Bayes (parity)"

CODE_HASH = hashlib.sha1(b"".join((ROOT / rel).read_bytes() for rel in (
    "simulator/closed_loop.py", "decoders/bayesian/continuous.py", "decoders/bayesian/grid_filter.py",
    "rl/env.py", "rl/dqn.py"))).hexdigest()[:12]


def decoders(noise, point, device):
    from decoders.bayesian.continuous import ContinuousBayesDecoder
    from rl.train import checkpoint_path
    n_x = 192 if noise["sigma_m"] < 0.035 * L else 96
    out = {
        "NL": dict(make=lambda: cl.ParityAdapter(NearestLattice()), spec={}),
        "Bayes (parity)": dict(make=lambda: ContinuousBayesDecoder(
            noise["sigma_q"], noise["sigma_m"], noise["rho"], policy="parity", n_x=n_x,
            device=device), spec=dict(policy="parity", n_x=n_x)),
        "Bayes (fine)": dict(make=lambda: ContinuousBayesDecoder(
            noise["sigma_q"], noise["sigma_m"], noise["rho"], policy="fine", n_x=n_x,
            device=device), spec=dict(policy="fine", n_x=n_x)),
    }
    for actions in ("parity", "fine"):
        path = checkpoint_path(point, actions)

        def make(path=path, actions=actions):
            from rl.dqn import DQNDecoder
            return DQNDecoder.load(path, device=device, name=f"DQN ({actions})")
        out[f"DQN ({actions})"] = dict(make=make, spec=dict(actions=actions), checkpoint=path)
    return out


def _ready(dec):
    path = dec.get("checkpoint")
    if path is None:
        return True
    meta = sidecar_path(path)
    return path.exists() and meta.exists() and json.loads(meta.read_text()).get("mode") == get_mode()


def _ckpt_identity(dec):
    if dec.get("checkpoint") is None:
        return None
    meta = json.loads(sidecar_path(dec["checkpoint"]).read_text())
    return dict(digest=meta["digest"], written=meta["written"])


def cache_path(point, name):
    return out_dir() / "cache" / "E15" / point / (name.replace(" ", "_") + ".npz")


def run_point(index, point, noise, device):
    sizes = SIZES[get_mode()]
    d, n = NoiseModel(**noise).sample(make_rng(eval_split(), "E15", index),
                                      sizes["trials"], sizes["rounds"])
    for name, dec in decoders(noise, point, device).items():
        if not _ready(dec):
            log(f"E15/{point}: {name} deferred (train it with python -m rl.train)")
            continue
        path = cache_path(point, name)
        config = dict(exp="E15", point=point, noise=noise, sizes=sizes, decoder=name,
                      spec=dec["spec"], split=eval_split(), index=index, code=CODE_HASH,
                      checkpoint=_ckpt_identity(dec))
        if is_complete(path, config):
            continue
        res = cl.run(dec["make"](), d, n)
        rec = res.logical_errors
        save_npz(path, config, extra=dict(trials=rec.shape[0]),
                 bits=np.packbits(rec, axis=1), trials=rec.shape[0])
        log(f"E15/{point}: {name:<16} P_L(final) {rec[:, -1].mean():.4f}")


def load_point(point):
    out = {}
    for p in sorted((out_dir() / "cache" / "E15" / point).glob("*.npz")):
        meta = json.loads(sidecar_path(p).read_text())
        if meta.get("mode") != get_mode():
            continue
        with np.load(p) as z:
            trials = int(z["trials"])
            rec = np.unpackbits(z["bits"], axis=1, count=meta["config"]["sizes"]["rounds"])
        out[meta["config"]["decoder"]] = rec[:trials].astype(bool)
    return out


def analyze():
    rows = []
    for point in POINTS:
        data = load_point(point)
        if REF not in data:
            continue
        ref = data[REF]
        h = common_horizon(ref.mean(axis=0))
        for name, rec in data.items():
            eps, used = eps_fit(rec.mean(axis=0))
            diff, se = paired_pl(rec, ref, h)
            rows.append(dict(point=point, decoder=name, eps=eps, eps_rounds_used=used,
                             horizon=h, pl_h=float(rec[:, h - 1].mean()),
                             diff_vs_bayes_parity=diff, diff_se=se, trials=rec.shape[0]))
    path = out_dir() / "E15_summary.json"
    path.write_text(json.dumps(rows, indent=1))
    log(f"E15: wrote {path} ({len(rows)} rows)")
    return rows


def e15(device=None):
    device = device or torch_device()
    for i, (point, noise) in enumerate(POINTS.items()):
        run_point(i, point, noise, device)
    analyze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze()
    else:
        e15()
        analyze()


if __name__ == "__main__":
    main()
