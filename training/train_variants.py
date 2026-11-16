"""Train the E13 learned-decoder variants (architecture, size, data, curriculum).

Reuses the pieces of training/train_lstm.py (labels, head evaluation, head
selection, AdamW + cosine schedule, early stopping, checkpoint sidecars) with
a variant config:

    arch        lstm | gru | transformer
    arch_kwargs model size (decoders/learned.py)
    dataset     a training/datasets.py DATASETS name (noise distribution)
    n_train     training sequences (full mode; quick mode scales it down)
    curriculum  none | rare   (see below)
    seed

Curriculum "rare" (aimed at the regime where the LSTM is 3-4x worse than
Bayes because logical errors are rare, the nominal point):
  1. Noise annealing. A second pool is drawn at elevated noise (sigma_q x 1.3,
     same rho and sigma_m), where nearest-lattice jumps are ~10x more common.
     The fraction of each epoch drawn from it falls linearly from 0.5 to 0
     over the first half of training; the second half is nominal only.
  2. Oversampling. Within the nominal pool, sequences in which the nearest
     lattice makes at least one odd jump are drawn with weight 4 (others 1),
     so the rare-error sequences that carry the parity signal are seen more.
  Validation is always on the plain nominal distribution, so head selection
  and early stopping measure the target task.

Checkpoints: out_dir("checkpoints")/<name>.pt, name from variant_name(), with
`arch` and `arch_kwargs` stored so decoders/learned.load_model can rebuild
them, plus a sidecar (experiments.common.write_sidecar) like train_lstm.

    python -m training.train_variants                  # every E13 variant
    python -m training.train_variants --only var_gru_h148_n200k_none_s0
    QKP_MODE=quick python -m training.train_variants   # smoke sizes
Nothing here has been run at full size.
"""

import argparse
import math
import time

import numpy as np
import torch
from torch import nn

from decoders.learned import build, matched_kwargs, n_params
from decoders.LSTM.model import HEADS, ParityLSTM, head_labels
from experiments.common import (
    get_budget, get_mode, is_complete, log, make_rng, out_dir, torch_device, write_sidecar,
)
from simulator.sim import NearestLattice, Simulator
from training.datasets import DATASETS, sample_noise
from training.train_lstm import LAYERS, LR, WEIGHT_DECAY, evaluate_heads

NOMINAL_DATASET = "nominal"
REF_HIDDEN = 128                         # the existing LSTMs (full budget)
BASE_SECONDS = 339.0                     # measured: lstm_nominal_s0 train + data (cost.json)
BASE_N_TRAIN = 200_000


# ---------------------------------------------------------------------------
# variant registry
# ---------------------------------------------------------------------------
def variant_name(v):
    n = f"{v['n_train'] // 1000}k"
    return f"var_{v['arch']}_h{v['arch_kwargs']['hidden']}_n{n}_{v['curriculum']}_s{v['seed']}"


def _v(arch, arch_kwargs, n_train=200_000, curriculum="none", seed=0, dataset=NOMINAL_DATASET,
       group=""):
    v = dict(arch=arch, arch_kwargs=arch_kwargs, n_train=n_train, curriculum=curriculum,
             seed=seed, dataset=dataset, group=group)
    v["name"] = variant_name(v)
    return v


def lstm_kwargs(h):
    return dict(hidden=h, layers=LAYERS)


def variant_grid():
    """The E13 grid. Duplicates (same name) appear once."""
    out = []
    # (a) LSTM size at nominal
    out += [_v("lstm", lstm_kwargs(h), group="size") for h in (32, 64, 128, 256)]
    # (b) training-set size at nominal
    out += [_v("lstm", lstm_kwargs(REF_HIDDEN), n_train=n, group="data")
            for n in (25_000, 50_000, 100_000, 200_000, 400_000)]
    # (c) curriculum toward rare errors (and its control above: h128, n200k, none)
    out += [_v("lstm", lstm_kwargs(REF_HIDDEN), curriculum="rare", seed=s, group="curriculum")
            for s in (0, 1, 2)]
    out += [_v("lstm", lstm_kwargs(REF_HIDDEN), curriculum="none", seed=s, group="curriculum")
            for s in (1, 2)]
    # (d) GRU and transformer at the parameter count of the 128-unit LSTM
    target = n_params(ParityLSTM(REF_HIDDEN, LAYERS))
    for arch in ("gru", "transformer"):
        kw, _ = matched_kwargs(arch, target)
        out += [_v(arch, kw, seed=s, group="arch") for s in (0, 1, 2)]
    seen, uniq = set(), []
    for v in out:
        if v["name"] not in seen:
            seen.add(v["name"])
            uniq.append(v)
    return uniq


def seed_extensions():
    """(e) extra seeds of the single-seed train_lstm models, trained by
    training.train_lstm.train_one under their usual names (config identical to
    seed 0 apart from the seed): list of (model name, dataset, seed)."""
    out = []
    for dataset, prefix in (("matched_rho0.5", "lstm_matched_rho0.5"),
                            ("matched_sq0.26", "lstm_matched_sq0.26"),
                            ("switching", "lstm_switching")):
        out += [(f"{prefix}_s{s}", dataset, s) for s in (1, 2)]
    return out


def estimate_seconds(v):
    """Rough full-mode training cost: measured 339 s for the 128-unit LSTM on
    200k sequences, scaled linearly in n_train and in parameter count (the GPU
    is saturated at these sizes; small models are bounded below by ~0.4x)."""
    p = n_params(build(v["arch"], **v["arch_kwargs"]))
    p0 = n_params(ParityLSTM(REF_HIDDEN, LAYERS))
    arch_factor = {"lstm": 1.0, "gru": 0.8, "transformer": 1.5}[v["arch"]]
    return BASE_SECONDS * (v["n_train"] / BASE_N_TRAIN) * max(0.4, p / p0) * arch_factor


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def make_pool(spec, split, trials, seed, stream):
    """Syndromes and nearest-lattice jump labels for a noise spec."""
    rng = make_rng(split, stream, point=seed)
    eps, nu, _ = sample_noise(spec, rng, trials)
    res = Simulator(trials=trials, rounds=eps.shape[1]).run(NearestLattice(), eps, nu)
    y = (res.lattice_jumps() % 2).astype(np.int8)
    return res.syndromes.astype(np.float32), y


def elevated(spec, factor=1.3):
    out = dict(spec)
    if out.get("kind") != "fixed":
        raise ValueError("curriculum 'rare' needs a fixed-parameter dataset")
    out["sigma_q"] = spec["sigma_q"] * factor
    return out


def sizes(v, budget):
    n_train = v["n_train"] if get_mode() == "full" else max(256, v["n_train"] // 200)
    return n_train, budget.n_val, budget.batch_size


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_variant(v, budget, device):
    name = v["name"]
    path = out_dir("checkpoints") / f"{name}.pt"
    n_train, n_val, batch = sizes(v, budget)
    config = dict(name=name, arch=v["arch"], arch_kwargs=v["arch_kwargs"], dataset=v["dataset"],
                  spec=DATASETS[v["dataset"]], curriculum=v["curriculum"], seed=v["seed"],
                  n_train=n_train, n_val=n_val, batch=batch, max_epochs=budget.max_epochs,
                  patience=budget.patience, lr=LR, wd=WEIGHT_DECAY,
                  time_cap=budget.train_time_cap_s)
    if is_complete(path, config):
        log(f"{name}: checkpoint up to date, skipping")
        return path

    t0 = time.time()
    spec = DATASETS[v["dataset"]]
    s_tr, y_tr = make_pool(spec, "train", n_train, v["seed"], f"var_data_{v['dataset']}")
    s_va, y_va = make_pool(spec, "val", n_val, v["seed"], f"var_data_{v['dataset']}")
    if v["curriculum"] == "rare":
        s_hi, y_hi = make_pool(elevated(spec), "train", n_train, v["seed"],
                               f"var_data_{v['dataset']}_elevated")
        jumps = y_tr.sum(axis=1) > 0
        w_nom = np.where(jumps, 4.0, 1.0)
        w_nom /= w_nom.sum()
    t_data = time.time() - t0

    to = lambda a: torch.tensor(a, device=device)
    s_tr_t, lab_tr = to(s_tr), head_labels(to(y_tr))
    s_va_t, y_va_t = to(s_va), to(y_va)
    if v["curriculum"] == "rare":
        s_hi_t, lab_hi = to(s_hi), head_labels(to(y_hi))

    torch.manual_seed(1000 + v["seed"])
    model = build(v["arch"], **v["arch_kwargs"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = math.ceil(n_train / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=budget.max_epochs * steps)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(2000 + v["seed"])
    best, history, bad, stop = (math.inf, None, None), [], 0, "max_epochs"
    t_train = time.time()
    for epoch in range(budget.max_epochs):
        if v["curriculum"] == "rare":
            frac_hi = max(0.0, 0.5 * (1 - epoch / max(1, budget.max_epochs // 2)))
            n_hi = int(round(frac_hi * n_train))
            idx_nom = rng.choice(n_train, size=n_train - n_hi, p=w_nom)
            idx_hi = rng.integers(0, n_train, size=n_hi)
        else:
            frac_hi, idx_nom, idx_hi = 0.0, rng.permutation(n_train), np.zeros(0, dtype=int)
        order = np.concatenate([np.stack([np.zeros_like(idx_nom), idx_nom], 1),
                                np.stack([np.ones_like(idx_hi), idx_hi], 1)])
        order = order[rng.permutation(len(order))]
        running = 0.0
        for i in range(steps):
            part = order[i * batch:(i + 1) * batch]
            if len(part) == 0:
                break
            src = part[:, 0]
            ids = torch.as_tensor(part[:, 1], device=device)
            nom = torch.as_tensor(src == 0, device=device)
            s_b = s_tr_t[ids]
            lab_b = lab_tr[ids]
            if v["curriculum"] == "rare" and bool((~nom).any()):
                s_b = torch.where(nom[:, None], s_b, s_hi_t[ids])
                lab_b = torch.where(nom[:, None, None], lab_b, lab_hi[ids])
            logits, _ = model(s_b)
            loss = loss_fn(logits, lab_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        val_loss, per_head = evaluate_heads(model, s_va_t, y_va_t)
        history.append(dict(epoch=epoch, train_loss=running / steps, val_loss=val_loss,
                            heads=per_head, frac_elevated=frac_hi,
                            seconds=time.time() - t_train))
        log(f"{name} epoch {epoch}: val {val_loss:.4f} elevated {frac_hi:.2f}")
        if val_loss < best[0] - 1e-5:
            best = (val_loss, {k: t.detach().clone() for k, t in model.state_dict().items()},
                    per_head)
            bad = 0
        else:
            bad += 1
            if bad >= budget.patience:
                stop = "early_stop"
                break
        if time.time() - t_train > budget.train_time_cap_s:
            stop = "time_cap"
            break
    if best[1] is None:
        log(f"{name}: no finite validation loss; not saving")
        return None
    model.load_state_dict(best[1])
    _, per_head = evaluate_heads(model, s_va_t, y_va_t)
    chosen = min(per_head, key=lambda h: per_head[h]["mean_ler"])
    train_seconds = time.time() - t_train
    ckpt = dict(state_dict=model.state_dict(), arch=v["arch"], arch_kwargs=v["arch_kwargs"],
                hidden=v["arch_kwargs"]["hidden"], layers=v["arch_kwargs"].get("layers", LAYERS),
                n_params=n_params(model), chosen_head=chosen, val_heads=per_head,
                history=history, stop_reason=stop, train_seconds=train_seconds,
                data_seconds=t_data, config=config, params_train=None)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    write_sidecar(path, config, extra=dict(chosen_head=chosen, stop_reason=stop,
                                           train_seconds=train_seconds,
                                           n_params=n_params(model)))
    log(f"{name}: stop={stop} head={chosen} val mean LER {per_head[chosen]['mean_ler']:.5f} "
        f"({train_seconds:.0f}s, {n_params(model)} params)")
    return path


def train_all(names=None, include_seed_extensions=True):
    from experiments.runlog import RunLog
    from training.train_lstm import train_one
    budget = get_budget()
    device = torch_device()
    paths = {}
    with RunLog("train_variants", args=dict(names=names)) as rl:
        for v in variant_grid():
            if names and v["name"] not in names:
                continue
            with rl.phase("train", model=v["name"], group=v["group"]):
                paths[v["name"]] = train_variant(v, budget, device)
        if include_seed_extensions:
            for name, dataset, seed in seed_extensions():
                if names and name not in names:
                    continue
                with rl.phase("train", model=name, group="seeds"):
                    paths[name] = train_one(name, dataset, seed, budget, device)
    return paths


def cost_table():
    rows = [dict(name=v["name"], group=v["group"], params=n_params(build(v["arch"], **v["arch_kwargs"])),
                 est_seconds=round(estimate_seconds(v))) for v in variant_grid()]
    rows += [dict(name=n, group="seeds", params=n_params(ParityLSTM(REF_HIDDEN, LAYERS)),
                  est_seconds=round(BASE_SECONDS)) for n, _, _ in seed_extensions()]
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--cost", action="store_true", help="print the cost estimate and exit")
    args = parser.parse_args()
    if args.cost:
        rows = cost_table()
        for r in rows:
            print(f"{r['name']:<40} {r['group']:<11} {r['params']:>8} params  ~{r['est_seconds']:>5} s")
        print(f"total ~{sum(r['est_seconds'] for r in rows) / 3600:.1f} h")
    else:
        train_all(args.only)
