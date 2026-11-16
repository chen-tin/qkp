"""Train every learned decoder needed by the experiments, in dependency order.

Each step skips checkpoints that are already up to date (sidecar digests), so
this can be re-run safely and resumed after an interruption.

    1. LSTMs (training/train_lstm.py): nominal, broad, matched, switching,
       drift (E9-E11), hardware-scale (E12) and the extra seeds of the
       single-seed models (E13)
    2. E13 architecture / size / data / curriculum variants (train_variants.py)
    3. E14 binary-syndrome LSTMs (train_binary.py)
    4. E15 DQN agents (rl/train.py)

    python -m training.train_everything                 # all
    python -m training.train_everything --only lstm binary
    QKP_MODE=quick python -m training.train_everything  # smoke sizes
"""

import argparse
import time

from experiments.common import log

STEPS = ("lstm", "variants", "binary", "dqn")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=STEPS, default=None)
    args = parser.parse_args()
    for step in STEPS:
        if args.only and step not in args.only:
            continue
        t0 = time.perf_counter()
        log(f"=== train {step} ===")
        if step == "lstm":
            from training.train_lstm import train_all
            train_all()
        elif step == "variants":
            from training.train_variants import train_all
            train_all(include_seed_extensions=False)     # the LSTM step trains those
        elif step == "binary":
            from training.train_binary import train_all
            train_all()
        elif step == "dqn":
            # rl.train.main parses sys.argv, so drive its loop directly
            from experiments.run_continuous import POINTS
            from rl.train import ACTION_SETS, train_one
            for pid, noise in POINTS.items():
                for actions in ACTION_SETS:
                    train_one(pid, noise, actions)
        log(f"=== train {step} done in {time.perf_counter() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
