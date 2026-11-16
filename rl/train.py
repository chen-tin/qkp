"""Train the DQN decoders used by experiment E15 (experiments/run_continuous.py).

One agent per (E15 point, action set). Checkpoints go to out_dir("checkpoints")
as dqn_<point>_<actions>.pt with a sidecar (experiments.common.write_sidecar),
so run_continuous picks them up and defers DQN decoders that are not trained.

    python -m rl.train                      # all points, both action sets
    python -m rl.train --points nominal --actions parity
    QKP_MODE=quick python -m rl.train       # smoke-test sizes
"""

import argparse
import time

from experiments.common import get_mode, is_complete, log, out_dir, torch_device, write_sidecar
from rl.dqn import save, train
from rl.env import DecodingEnv
from simulator.noise import L, NoiseModel

# training sizes per mode (no Budget fields: experiments/common.py is shared)
SIZES = {
    "full": dict(batch=512, rounds=200, steps=60_000, batch_size=4096, time_cap_s=1800.0),
    "quick": dict(batch=64, rounds=40, steps=400, batch_size=256, time_cap_s=60.0),
}
ACTION_SETS = {
    "parity": dict(n_fine=1),
    "fine": dict(n_fine=9, fine_max=0.25 * L),
}
WINDOW = 8
GAMMA = 0.9


def checkpoint_path(point, actions):
    return out_dir("checkpoints") / f"dqn_{point}_{actions}.pt"


def env_config(point_noise, actions, sizes, seed):
    return dict(window=WINDOW, rounds=sizes["rounds"], batch=sizes["batch"], seed=seed,
                sigma_q=point_noise["sigma_q"], sigma_m=point_noise["sigma_m"],
                rho=point_noise["rho"], reward="dense", **ACTION_SETS[actions])


def train_one(point, point_noise, actions, seed=0, device=None):
    sizes = SIZES[get_mode()]
    path = checkpoint_path(point, actions)
    cfg = env_config(point_noise, actions, sizes, seed)
    config = dict(kind="dqn", point=point, actions=actions, env=cfg, gamma=GAMMA,
                  steps=sizes["steps"], batch_size=sizes["batch_size"])
    if is_complete(path, config):
        log(f"dqn {point}/{actions}: checkpoint up to date, skipping")
        return path
    env = DecodingEnv(batch=cfg["batch"], rounds=cfg["rounds"], window=WINDOW,
                      n_fine=cfg["n_fine"], fine_max=cfg.get("fine_max", 0.25 * L),
                      noise=NoiseModel(cfg["sigma_q"], cfg["sigma_m"], cfg["rho"]),
                      reward="dense", seed=1000 + seed)
    t0 = time.time()
    agent, hist = train(env, steps=sizes["steps"], gamma=GAMMA, batch_size=sizes["batch_size"],
                        device=device or torch_device(), seed=seed,
                        log_every=max(1, sizes["steps"] // 20), time_cap_s=sizes["time_cap_s"])
    seconds = time.time() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    save(agent, path, cfg, extra=dict(history=hist, seconds=seconds))
    write_sidecar(path, config, extra=dict(train_seconds=seconds))
    log(f"dqn {point}/{actions}: trained in {seconds:.0f}s")
    return path


def main():
    from experiments.run_continuous import POINTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", nargs="*", default=None)
    parser.add_argument("--actions", nargs="*", default=list(ACTION_SETS))
    args = parser.parse_args()
    for pid, noise in POINTS.items():
        if args.points and pid not in args.points:
            continue
        for actions in args.actions:
            train_one(pid, noise, actions)


if __name__ == "__main__":
    main()
