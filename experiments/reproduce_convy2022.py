"""Independent reproduction of Convy et al. (2022), Fig. 2, Scheme A.

See docs/convy2022_reproduction.md before running. This is a three-qubit
continuous-readout experiment, completely separate from the GKP benchmark.
"""
import argparse
import copy
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reproductions/convy2022/reference"
MEANS = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1],
                  [-1, 1], [-1, -1], [1, -1], [1, 1]], dtype=np.float32)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_torch(path, obj):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def transition(gamma, dt):
    """Exact exp(Q dt), including multiple independent flips within a step."""
    p = -np.expm1(-2 * gamma * dt) / 2
    hamming = np.array([i.bit_count() for i in range(8)])
    d = hamming[np.arange(8)[:, None] ^ np.arange(8)[None, :]]
    return p ** d * (1 - p) ** (3 - d)


def simulate(n, steps, gamma, dt, gamma_m, seed):
    """Paper Scheme A: independent Poisson bit flips, then Gaussian readout.

    Bernoulli sampling with the exact odd-Poisson probability is equivalent
    to sampling full Poisson counts and retaining their parity.
    """
    if n % 8:
        raise ValueError("Use a multiple of eight for balanced initial states")
    rng = np.random.default_rng(seed)
    initial = np.tile(np.arange(8, dtype=np.uint8), n // 8)
    rng.shuffle(initial)
    p = -np.expm1(-2 * gamma * dt) / 2
    flips = np.zeros((n, steps), dtype=np.uint8)
    for mask in (4, 2, 1):
        flips ^= (rng.random((n, steps)) < p).astype(np.uint8) * mask
    labels = np.bitwise_xor.accumulate(flips, axis=1) ^ initial[:, None]
    measurements = rng.standard_normal((n, steps, 2), dtype=np.float32)
    measurements *= np.float32(1 / math.sqrt(gamma_m * dt))
    measurements += MEANS[labels]
    return measurements, labels, initial


def dataset(config, directory):
    result = {}
    for split, seed in (("train", 210378), ("val", 210379), ("test", 210380)):
        paths = {key: directory / f"{split}_{key}.npy"
                 for key in ("measurements", "labels", "initial")}
        marker = directory / f"{split}_manifest.json"
        if not marker.exists():
            arrays = simulate(config[f"n_{split}"], config["steps"], config["gamma"],
                              config["dt_us"], config["gamma_m_us"], seed)
            for (key, path), value in zip(paths.items(), arrays):
                tmp = path.with_suffix(".tmp.npy")
                np.save(tmp, value)
                tmp.replace(path)
            save_json(marker, {"seed": seed, "n": config[f"n_{split}"],
                               "sha256": {k: sha(p) for k, p in paths.items()}})
        manifest = json.loads(marker.read_text())
        for key, path in paths.items():
            if sha(path) != manifest["sha256"][key]:
                raise ValueError(f"Dataset hash mismatch: {path}")
        result[split] = tuple(np.load(p, mmap_mode="r") for p in paths.values())
    return result


def fit_bayes(data):
    """Empirical transitions and state-conditioned full 2x2 covariances.

    Matches the author's Bayesian.train depth=0 estimator, without loading
    all 62.5 million training samples into a temporary float64 array.
    """
    measurements, labels, _ = data
    counts = np.zeros((8, 8), dtype=np.int64)
    numbers = np.zeros(8, dtype=np.int64)
    sums = np.zeros((8, 2), dtype=np.float64)
    products = np.zeros((8, 2, 2), dtype=np.float64)
    for start in range(0, len(labels), 512):
        y = labels[start:start + 512]
        x = measurements[start:start + 512].astype(np.float64)
        counts += np.bincount((8 * y[:, :-1].astype(np.int64) + y[:, 1:]).ravel(),
                              minlength=64).reshape(8, 8)
        for state in range(8):
            a = x[y == state]
            numbers[state] += len(a)
            sums[state] += a.sum(axis=0)
            products[state] += a.T @ a
    means = sums / numbers[:, None]
    covs = (products - numbers[:, None, None] * means[:, :, None] * means[:, None, :])
    covs /= (numbers - 1)[:, None, None]
    return {"transition": counts / counts.sum(axis=1, keepdims=True),
            "means": means, "covs": covs, "counts": counts, "state_samples": numbers}


def bayes_predict(data, fit):
    measurements, labels, initial = data
    precision = np.linalg.inv(fit["covs"])
    logdet = np.linalg.slogdet(fit["covs"])[1]
    pred = np.zeros(labels.shape, dtype=np.uint8)
    # Small batches keep the working array in cache and avoid BLAS overhead.
    for start in range(0, len(labels), 1024):
        x = measurements[start:start + 1024]
        init = initial[start:start + 1024]
        prior = np.eye(8)[init]
        for step in range(x.shape[1]):
            residual = x[:, step, None, :] - fit["means"][None]
            loglike = -.5 * (np.einsum("bsi,sij,bsj->bs", residual, precision, residual)
                             + logdet)
            likelihood = np.exp(loglike - loglike.max(axis=1, keepdims=True))
            prior = (prior @ fit["transition"]) * likelihood
            prior /= prior.sum(axis=1, keepdims=True)
            pred[start:start + len(x), step] = prior.argmax(axis=1)
    return pred


class PaperLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 32, num_layers=2, batch_first=True)
        self.readout = nn.Linear(32, 8)

    def forward(self, x):
        hidden, _ = self.lstm(x)
        return self.readout(hidden)


def tensors(data, indices, device):
    measurements, labels, initial = data
    x = np.empty((len(indices), labels.shape[1], 3), dtype=np.float32)
    x[:, :, :2] = measurements[indices]
    x[:, :, 2] = initial[indices, None]  # Known initial state, not current label.
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(labels[indices].astype(np.int64)).to(device))


def state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def evaluate(model, data, device, batch_size):
    model.eval()
    predictions = np.empty(data[1].shape, dtype=np.uint8)
    total_loss = 0.
    with torch.inference_mode():
        for start in range(0, len(data[1]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(data[1])))
            x, y = tensors(data, indices, device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
            total_loss += loss.item() * len(indices)
            predictions[indices] = logits.argmax(-1).cpu().numpy().astype(np.uint8)
    return total_loss / len(data[1]), predictions


def train(config, data, directory, seed, device):
    final_path = directory / f"lstm_s{seed}.pt"
    if final_path.exists():
        return final_path
    torch.manual_seed(seed)
    model = PaperLSTM().to(device)
    assert sum(p.numel() for p in model.parameters()) == 13448
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=4, min_lr=1e-5)
    shuffle = np.random.default_rng(1000 + seed)
    history, best, best_epoch, stale, start_epoch = [], math.inf, 0, 0, 1
    best_weights = state_dict(model)
    resume = directory / f"lstm_s{seed}.resume.pt"
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("Resume protocol mismatch")
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, best, best_epoch, stale = (saved[k] for k in ("history", "best", "best_epoch", "stale"))
        best_weights = saved["best_weights"]
        shuffle.bit_generator.state = saved["shuffle_rng"]
        start_epoch = saved["epoch"] + 1
    for epoch in range(start_epoch, config["max_epochs"] + 1):
        tick = time.monotonic()
        model.train()
        order = shuffle.permutation(len(data["train"][1]))
        total, seen = 0., 0
        for offset in range(0, len(order), config["batch_size"]):
            indices = order[offset:offset + config["batch_size"]]
            x, y = tensors(data["train"], indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.item() * len(indices)
            seen += len(indices)
        val_loss, val_pred = evaluate(model, data["val"], device, config["batch_size"])
        if val_loss < best - config["min_delta"]:
            best, best_epoch, stale, best_weights = val_loss, epoch, 0, state_dict(model)
        else:
            stale += 1
        row = {"epoch": epoch, "train_ce": total / seen, "val_ce": val_loss,
               "val_final_fidelity": float((val_pred[:, -1] == data["val"][1][:, -1]).mean()),
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - tick}
        history.append(row)
        scheduler.step(val_loss)
        save_torch(resume, {"config": config, "seed": seed, "epoch": epoch,
                           "weights": state_dict(model), "best_weights": best_weights,
                           "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                           "history": history, "best": best, "best_epoch": best_epoch,
                           "stale": stale, "shuffle_rng": shuffle.bit_generator.state})
        save_json(directory / f"lstm_s{seed}_history.json", history)
        print(json.dumps({"seed": seed, **row}), flush=True)
        if epoch >= config["min_epochs"] and stale >= config["patience"]:
            break
    save_torch(final_path, {"config": config, "seed": seed, "weights": best_weights,
                           "best_epoch": best_epoch, "history": history,
                           "stopping_reason": "validation_patience" if stale >= config["patience"]
                           else "epoch_budget", "completed_utc": now()})
    return final_path


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center - half, center + half]


def metrics(predictions, labels):
    correct = predictions == labels
    k, n = int(correct[:, -1].sum()), len(labels)
    return {"n": n, "correct_final": k, "final_fidelity": k/n,
            "wilson95": wilson(k, n), "sem": math.sqrt(k/n*(1-k/n)/n),
            "mean_time_fidelity": float(correct.mean()),
            "fidelity_by_step": correct.mean(axis=0).tolist()}


def record_prediction(directory, name, predictions, labels):
    path = directory / f"{name}_predictions.npz"
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, predictions=predictions)
    tmp.replace(path)
    result = metrics(predictions, labels)
    result["predictions_sha256"] = sha(path)
    save_json(directory / f"{name}_metrics.json", result)
    print(json.dumps({"model": name, **{k:v for k,v in result.items()
                                       if k != "fidelity_by_step"}}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                         "mps" if torch.backends.mps.is_available() else "cpu"))
    config = {"schema": 1, "paper": "Convy et al. NJP 24 063019 (2022)",
              "target": "Figure 2 Scheme A, gamma=.04/us", "gamma": .04,
              "dt_us": .032, "gamma_m_us": 4.7, "steps": 625,
              "n_train": 512 if args.smoke else 100000,
              "n_val": 128 if args.smoke else 10000,
              "n_test": 256 if args.smoke else 30000,
              "max_epochs": 2 if args.smoke else 100,
              "min_epochs": 2 if args.smoke else 20,
              "patience": 12, "min_delta": 1e-5, "learning_rate": .001,
              "batch_size": 512, "device": str(device),
              "source_sha256": sha(Path(__file__)),
              "protocol_sha256": sha(ROOT / "docs/convy2022_reproduction.md"),
              "reference_sha256": sha(REF / "provenance.json"),
              "torch": torch.__version__, "numpy": np.__version__,
              "profile": "smoke" if args.smoke else "anchor"}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    directory = args.output or ROOT / "results/reproduce_convy2022" / config["profile"] / digest
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text())["config"] != config:
        raise ValueError("Output directory belongs to another protocol")
    if not plan_path.exists():
        save_json(plan_path, {"registered_utc": now(), "config": config,
                              "platform": platform.platform(),
                              "planned_training_seeds": [0, 1, 2],
                              "disclosure": "Independent implementation. Original LSTM notebook unavailable."})
    print(json.dumps({"directory": str(directory), "config": config}), flush=True)
    data = dataset(config, directory)
    fit_path = directory / "bayes_fit.npz"
    if not fit_path.exists():
        fit = fit_bayes(data["train"])
        np.savez(fit_path, **fit)
    with np.load(fit_path) as f:
        fit = {k: f[k] for k in f.files}
    if not (directory / "bayesian_metrics.json").exists():
        pred = bayes_predict(data["test"], fit)
        record_prediction(directory, "bayesian", pred, data["test"][1])
    if args.baselines_only:
        return
    for seed in args.seeds:
        path = train(config, data, directory, seed, device)
        if not (directory / f"lstm_s{seed}_metrics.json").exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = PaperLSTM().to(device)
            model.load_state_dict(checkpoint["weights"])
            _, pred = evaluate(model, data["test"], device, config["batch_size"])
            result = record_prediction(directory, f"lstm_s{seed}", pred, data["test"][1])
            result["checkpoint_sha256"] = sha(path)
            result["best_epoch"] = checkpoint["best_epoch"]
            save_json(directory / f"lstm_s{seed}_metrics.json", result)
    complete = all((directory / f"lstm_s{s}_metrics.json").exists() for s in (0, 1, 2))
    save_json(directory / "status.json", {"updated_utc": now(), "anchor_complete": complete,
                                         "profile": config["profile"],
                                         "full_paper_reproduced": False})


if __name__ == "__main__":
    main()
