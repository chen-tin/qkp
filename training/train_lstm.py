"""Train LSTM decoders.

The architecture and optimiser are fixed in advance: 2-layer LSTM
(hidden size from the budget), AdamW, cosine learning-rate schedule, early
stopping on validation loss. The head used at test time is the one with the
lowest validation logical error (averaged over all rounds); test data is
never used here.

Run from the repo root:
    python -m training.train_lstm            # all models in the budget
    QKP_MODE=quick python -m training.train_lstm
"""

import math
import time

import numpy as np
import torch
from torch import nn

from decoders.LSTM.model import HEADS, ParityLSTM, cumulative_from_head, head_labels
from experiments.common import (
    get_budget, get_mode, is_complete, log, out_dir, torch_device, write_sidecar,
)
from training.datasets import DATASETS, is_long, make_dataset

LR = 2e-3
WEIGHT_DECAY = 1e-4
LAYERS = 2


def model_list(budget, include_nonstationary=True):
    """(model name, dataset name, seed) to train, in priority order.

    The drift-trained models (E9-E11) come last; include_nonstationary=False
    gives the original list (E7 timing is keyed on it).
    """
    items = [(f"lstm_nominal_s{s}", "nominal", s) for s in range(budget.lstm_seeds)]
    items += [(f"lstm_broad_s{s}", "broad", s) for s in range(budget.lstm_seeds)]
    items += [
        ("lstm_matched_rho0.5_s0", "matched_rho0.5", 0),
        ("lstm_matched_sq0.26_s0", "matched_sq0.26", 0),
        ("lstm_switching_s0", "switching", 0),   # stretch goal
    ]
    if include_nonstationary:
        items += [(f"lstm_drift_s{s}", "drift", s) for s in range(budget.lstm_seeds)]
    return items


def sizes(dataset, budget):
    """(n_train, n_val, batch_size, rounds); long datasets use the ns_* budget."""
    if is_long(dataset):
        return budget.ns_n_train, budget.ns_n_val, budget.ns_batch_size, budget.ns_rounds
    return budget.n_train, budget.n_val, budget.batch_size, None


@torch.no_grad()
def evaluate_heads(model, s, y, batch=None):
    """Validation loss and per-head mean logical error (open loop).

    Open loop equals closed loop here because syndromes do not depend on
    the decoder's choices (tested in tests/test_lstm.py).
    """
    model.eval()
    if batch is None:
        # keep the LSTM activations of one batch around 1e8 values
        batch = max(64, min(8192, int(1e8 // (s.shape[1] * model.hidden))))
    labels = head_labels(y)
    K = labels[..., [h for h, _ in HEADS].index("full")].to(torch.int64)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    total, count = 0.0, 0
    logits_all = []
    for start in range(0, s.shape[0], batch):
        logits, _ = model(s[start:start + batch])
        total += loss_fn(logits, labels[start:start + batch]).item()
        count += labels[start:start + batch].numel()
        logits_all.append(logits)
    logits = torch.cat(logits_all)
    per_head = {}
    for i, (name, k) in enumerate(HEADS):
        w_hat = (logits[..., i] > 0).to(torch.int64)
        J = cumulative_from_head(w_hat, k)
        err = (K ^ J).float()
        per_head[name] = dict(mean_ler=err.mean().item(), ler50=err[:, -1].mean().item())
    model.train()
    return total / count, per_head


def train_one(name, dataset, seed, budget, device):
    path = out_dir("checkpoints") / f"{name}.pt"
    n_train, n_val, batch_size, rounds = sizes(dataset, budget)
    config = dict(name=name, dataset=dataset, seed=seed, spec=DATASETS[dataset],
                  n_train=n_train, n_val=n_val, hidden=budget.hidden,
                  layers=LAYERS, batch=batch_size, max_epochs=budget.max_epochs,
                  patience=budget.patience, lr=LR, wd=WEIGHT_DECAY,
                  time_cap=budget.train_time_cap_s)
    if rounds is not None:
        # only long datasets carry this key, so existing checkpoints keep
        # their config digest
        config["rounds"] = rounds
    data_kw = {} if rounds is None else dict(rounds=rounds)
    if is_complete(path, config):
        log(f"{name}: checkpoint up to date, skipping")
        return path
    # The wall-clock cap is a resource guard, not a hyperparameter: a model
    # trained under a different cap that never hit it is identical to one
    # trained under this cap, so keep it. Models that hit the cap retrain.
    meta_path = path.with_name(path.name + ".meta.json")
    if path.exists() and meta_path.exists():
        import json
        info = json.loads(meta_path.read_text())
        old = dict(info.get("config", {}))
        if (info.get("mode") == get_mode() and info.get("stop_reason") != "time_cap"
                and {k: v for k, v in old.items() if k != "time_cap"}
                == json.loads(json.dumps({k: v for k, v in config.items()
                                          if k != "time_cap"}, default=float))):
            log(f"{name}: checkpoint complete (time cap did not bind), skipping")
            return path
        if info.get("stop_reason") == "time_cap":
            log(f"{name}: previous training hit the time cap; retraining")

    t0 = time.time()
    train = make_dataset(dataset, "train", n_train, seed=seed, **data_kw)
    val = make_dataset(dataset, "val", n_val, seed=seed, **data_kw)
    s_tr = torch.tensor(train["syndromes"], device=device)
    y_tr = torch.tensor(train["y"], device=device)
    lab_tr = head_labels(y_tr)
    s_va = torch.tensor(val["syndromes"], device=device)
    y_va = torch.tensor(val["y"], device=device)
    t_data = time.time() - t0

    torch.manual_seed(1000 + seed)
    model = ParityLSTM(budget.hidden, LAYERS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(n_train / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=budget.max_epochs * steps_per_epoch
    )
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator(device=device).manual_seed(2000 + seed)

    best = (math.inf, None, None)
    history = []
    bad_epochs = 0
    stop_reason = "max_epochs"
    t_train = time.time()
    for epoch in range(budget.max_epochs):
        perm = torch.randperm(n_train, device=device, generator=gen)
        running = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            logits, _ = model(s_tr[idx])
            loss = loss_fn(logits, lab_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        val_loss, per_head = evaluate_heads(model, s_va, y_va)
        history.append(dict(epoch=epoch, train_loss=running / steps_per_epoch,
                            val_loss=val_loss, heads=per_head,
                            seconds=time.time() - t_train))
        log(f"{name} epoch {epoch}: train {running/steps_per_epoch:.4f} "
            f"val {val_loss:.4f} "
            + " ".join(f"{h}={v['mean_ler']:.4f}" for h, v in per_head.items()))
        if val_loss < best[0] - 1e-5:
            best = (val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()},
                    per_head)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= budget.patience:
                stop_reason = "early_stop"
                break
        if time.time() - t_train > budget.train_time_cap_s:
            stop_reason = "time_cap"
            break

    if best[1] is None:
        log(f"{name}: no finite validation loss in any epoch; not saving a checkpoint")
        return None
    model.load_state_dict(best[1])
    _, per_head = evaluate_heads(model, s_va, y_va)
    chosen = min(per_head, key=lambda h: per_head[h]["mean_ler"])
    train_seconds = time.time() - t_train
    log(f"{name}: stop={stop_reason}, chosen head={chosen}, "
        f"val mean LER {per_head[chosen]['mean_ler']:.5f}, {train_seconds:.0f}s")

    ckpt = dict(state_dict=model.state_dict(), hidden=budget.hidden, layers=LAYERS,
                chosen_head=chosen, val_heads=per_head, history=history,
                stop_reason=stop_reason, train_seconds=train_seconds,
                data_seconds=t_data, config=config,
                params_train=train["params"])
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    write_sidecar(path, config, extra=dict(chosen_head=chosen, stop_reason=stop_reason,
                                           train_seconds=train_seconds))
    return path


def train_all(names=None):
    budget = get_budget()
    device = torch_device()
    log(f"training LSTMs (mode={get_mode()}, device={device}, budget={budget})")
    paths = {}
    for name, dataset, seed in model_list(budget):
        if names is not None and name not in names:
            continue
        paths[name] = train_one(name, dataset, seed, budget, device)
    return paths


if __name__ == "__main__":
    train_all()
