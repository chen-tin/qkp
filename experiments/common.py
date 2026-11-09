"""Shared infrastructure for the decoder-comparison experiments.

- non-interactive plotting (Agg) before anything imports pyplot
- run modes ("quick" for smoke tests, "full" for the real run) with
  separate output folders
- deterministic, split-separated random streams
- atomic result files with a JSON sidecar that marks completion
- blocked (memory-bounded) evaluation of any decoder
- statistics on logical-error records
"""

import matplotlib

matplotlib.use("Agg")

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from simulator.noise import L
from simulator.sim import SimResult, Simulator

ROOT = Path(__file__).resolve().parents[1]
SEED_ROOT = 20261109
SPLITS = {"train": 0, "val": 1, "dev": 2, "test": 3}

# nominal noise point of the proposal
NOMINAL = dict(sigma_q=0.20 * L, sigma_m=0.05 * L, rho=0.90)
ROUNDS = 50

# parameter box used to train the "broad" LSTM (and to build Bayes-broad)
BROAD_BOX = dict(
    rho=(0.0, 0.97),
    sigma_q=(0.12 * L, 0.28 * L),
    sigma_m=(0.02 * L, 0.08 * L),
)


# run mode
def get_mode():
    mode = os.environ.get("QKP_MODE", "full")
    if mode not in ("quick", "full"):
        raise ValueError(f"QKP_MODE must be quick or full, got {mode!r}")
    return mode


def out_dir(kind="results", mode=None):
    """results/ or checkpoints/ for the full run, <kind>/quick/ for quick mode."""
    mode = mode or get_mode()
    path = ROOT / kind
    if mode == "quick":
        path = path / "quick"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class Budget:
    """All sizes that control cost, logged with every result."""

    chunk_trials: int          # test trials per adaptive chunk
    max_test_trials: int       # cap on adaptive growth per point
    min_events: int            # stop growing when best decoder has this many flips
    diag_trials: int           # recorded subset for diagnostics
    n_train: int               # LSTM training sequences
    n_val: int                 # LSTM validation sequences
    batch_size: int
    max_epochs: int
    patience: int
    train_time_cap_s: float    # wall-clock cap per LSTM model
    hidden: int
    lstm_seeds: int
    filter_block_bytes: float  # memory per Bayes-filter block


BUDGETS = {
    "full": Budget(
        chunk_trials=20_000, max_test_trials=100_000, min_events=100,
        diag_trials=2_000, n_train=200_000, n_val=20_000, batch_size=2048,
        max_epochs=40, patience=5, train_time_cap_s=600.0, hidden=128,
        lstm_seeds=3, filter_block_bytes=0.5e9,
    ),
    "quick": Budget(
        chunk_trials=2_000, max_test_trials=4_000, min_events=100,
        diag_trials=500, n_train=8_000, n_val=2_000, batch_size=1024,
        max_epochs=3, patience=2, train_time_cap_s=60.0, hidden=64,
        lstm_seeds=1, filter_block_bytes=0.25e9,
    ),
}


def get_budget(mode=None):
    return BUDGETS[mode or get_mode()]


# devices
def torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# random streams
def _stable_int(text):
    return int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "little")


def seed_sequence(split, experiment, point=0, chunk=0):
    """Independent stream for (split, experiment, point, chunk).

    Streams from different splits can never coincide because the split id is
    part of the spawn key. Quick mode must only use the "dev" split.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if get_mode() == "quick" and split == "test":
        raise RuntimeError("quick mode must not touch the test split")
    key = (SPLITS[split], _stable_int(experiment), int(point), int(chunk))
    return np.random.SeedSequence(SEED_ROOT, spawn_key=key)


def make_rng(split, experiment, point=0, chunk=0):
    return np.random.default_rng(seed_sequence(split, experiment, point, chunk))


def eval_split():
    """Split used for evaluation: test in the full run, dev in quick mode."""
    return "dev" if get_mode() == "quick" else "test"


# files
def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def config_digest(config):
    text = json.dumps(_to_jsonable(config), sort_keys=True)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def sidecar_path(path):
    return Path(str(path) + ".meta.json")


def is_complete(path, config):
    """True if path exists with a sidecar matching this config and mode."""
    meta = sidecar_path(path)
    if not Path(path).exists() or not meta.exists():
        return False
    info = json.loads(meta.read_text())
    return info.get("digest") == config_digest(config) and info.get("mode") == get_mode()


def write_sidecar(path, config, extra=None):
    info = dict(
        mode=get_mode(), digest=config_digest(config), git=git_hash(),
        config=_to_jsonable(config), written=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    if extra:
        info.update(_to_jsonable(extra))
    tmp = sidecar_path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, sidecar_path(path))


def save_npz(path, config, extra=None, **arrays):
    """Atomic npz write, then the sidecar (the completion marker)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    write_sidecar(path, config, extra)


def save_json(path, config, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_to_jsonable(data), indent=2))
    os.replace(tmp, path)
    write_sidecar(path, config)


def log(message):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


# evaluation
def run_blocked(decoder, displacement_noise, measurement_noise, block):
    """Run a decoder over trial blocks and return the logical-error record.

    Returns a bool array (trials, rounds). Each block is an independent
    Simulator run, so decoder memory is bounded by the block size.
    """
    trials, rounds = displacement_noise.shape
    out = np.zeros((trials, rounds), dtype=bool)
    for start in range(0, trials, block):
        stop = min(start + block, trials)
        sim = Simulator(trials=stop - start, rounds=rounds)
        result = sim.run(
            decoder, displacement_noise[start:stop], measurement_noise[start:stop]
        )
        out[start:stop] = result.logical_errors
    return out


def run_blocked_full(decoder, displacement_noise, measurement_noise, block):
    """Like run_blocked but returns a full SimResult (for small runs)."""
    trials, rounds = displacement_noise.shape
    parts = []
    for start in range(0, trials, block):
        stop = min(start + block, trials)
        sim = Simulator(trials=stop - start, rounds=rounds)
        parts.append(sim.run(
            decoder, displacement_noise[start:stop], measurement_noise[start:stop]
        ))
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts], axis=0)
    return SimResult(
        decoder_name=decoder.name,
        syndromes=cat("syndromes"),
        corrections=cat("corrections"),
        residuals=cat("residuals"),
        displacement_noise=cat("displacement_noise"),
        measurement_noise=cat("measurement_noise"),
    )


# statistics on logical-error records
def flips_per_trial(logical_errors):
    """Number of logical-state changes per trial (starting from no error)."""
    prev = np.concatenate(
        [np.zeros((logical_errors.shape[0], 1), dtype=bool), logical_errors[:, :-1]],
        axis=1,
    )
    return (logical_errors ^ prev).sum(axis=1).astype(np.int32)


def flip_rate(flips, rounds):
    """Per-round logical-flip rate and its standard error (trials independent)."""
    per_trial = flips / rounds
    n = len(per_trial)
    return per_trial.mean(), per_trial.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def effective_eps(p_l, n):
    """Per-round logical error rate implied by P_L(n) for independent flips."""
    p_l = np.asarray(p_l, dtype=float)
    base = np.clip(1 - 2 * p_l, 0, None)
    return 0.5 * (1 - base ** (1.0 / n))


def paired_difference(flips_a, flips_b, rounds):
    """Mean per-round flip-rate difference a - b on common noise, with SE."""
    d = (flips_a.astype(float) - flips_b.astype(float)) / rounds
    n = len(d)
    return d.mean(), d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan


def summarize(logical_errors):
    """Compact summary of a (trials, rounds) logical-error record."""
    trials, rounds = logical_errors.shape
    flips = flips_per_trial(logical_errors)
    rate, rate_se = flip_rate(flips, rounds)
    ler = logical_errors.mean(axis=0)
    out = dict(
        trials=trials, rounds=rounds, flip_rate=rate, flip_rate_se=rate_se,
        flip_events=int(flips.sum()), ler_curve=ler,
    )
    for n in (10, 50):
        if n <= rounds:
            k = int(logical_errors[:, n - 1].sum())
            p, lo, hi = wilson(k, trials)
            out[f"ler{n}"] = p
            out[f"ler{n}_lo"] = lo
            out[f"ler{n}_hi"] = hi
            out[f"ler{n}_events"] = k
            out[f"eps{n}"] = float(effective_eps(p, n))
            out[f"saturated{n}"] = bool(p > 0.5 - 2 * np.sqrt(0.25 / trials))
    return out


def budget_dict(budget):
    return asdict(budget)
