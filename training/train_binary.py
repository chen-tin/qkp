"""Train the one-bit-syndrome LSTMs (decoders/binary_lstm.py) for E14.

Data: closed-loop runs (simulator/closed_loop.py) with per-sequence noise
parameters drawn from a box around the chosen scale, measured with PhasedBits
(m bits per round), under a behaviour policy: the sBs baseline with a
per-sequence feedback size delta ~ U(0.05, 0.2) L plus N(0, 0.03 L) jitter per
round, so that the network sees a spread of its own past corrections.

Caveat (distribution shift): at test time the network's own corrections
replace the behaviour policy's, so it sees bit/correction histories it was not
trained on. Feeding the previous correction as an input and predicting the
correction-independent target x_pre_t + eps_{t+1} limits the damage but does
not remove it; a DAgger-style second round (collect data with the trained
decoder, retrain) is the natural fix and is left for later.

Run from the repo root:
    python -m training.train_binary                 # all models
    QKP_MODE=quick python -m training.train_binary
"""

import math
import time

import numpy as np
import torch
from torch import nn

from decoders.bayesian.binary_filter import PhasedBits, default_phases, unpack
from decoders.binary_lstm import BitLSTM, bin_index
from experiments.common import (
    get_budget, get_mode, is_complete, log, make_rng, out_dir, torch_device, write_sidecar,
)
from simulator import closed_loop as cl
from simulator.noise import L
from simulator.switching import sample_varied

N_BINS = 48
LAYERS = 2
LR = 2e-3
ROUNDS = 50

# per-sequence parameter boxes (absolute units), one per scale
SCALES = {
    "nominal": dict(sigma_q=(0.14 * L, 0.26 * L), sigma_m=(0.03 * L, 0.07 * L)),
    "hardware": dict(sigma_q=(0.05 * L, 0.10 * L), sigma_m=(0.08 * L, 0.13 * L)),
}
BITS = (1, 4, 8)


def model_list():
    """(name, scale, m, seed) for every binary LSTM used by E14."""
    return [(f"binary_lstm_{scale}_m{m}_s0", scale, m, 0) for scale in SCALES for m in BITS]


class Behaviour:
    """sBs feedback with per-trial delta and per-round Gaussian jitter."""

    def __init__(self, delta, phases, jitter, rng):
        self.delta, self.phases, self.jitter, self.rng = delta, phases, jitter, rng
        self.sin_bits = [j for j, ph in enumerate(phases) if j % 2 == 0]
        self.name = "behaviour"

    def reset(self, trials):
        pass

    def decide(self, bits):
        b = unpack(bits, len(self.phases))[:, self.sin_bits]
        c = self.delta * (2.0 * b - 1.0).mean(axis=1)
        return c + self.rng.normal(0.0, self.jitter, size=c.shape)


def make_data(scale, m, split, trials, seed, rounds=ROUNDS):
    box = SCALES[scale]
    rng = make_rng(split, f"binary_lstm_{scale}_m{m}", point=seed)
    rho = rng.uniform(0.0, 0.97, trials)
    sq = rng.uniform(*box["sigma_q"], trials)
    sm = rng.uniform(*box["sigma_m"], trials)
    d, n = sample_varied(rng, rho, sq, sm, rounds)
    vis = rng.choice([0.95, 1.0], size=trials)
    pfl = rng.choice([0.0, 0.01], size=trials)
    phases = default_phases(m)
    meas = PhasedBits(phases=phases, visibility=vis, p_flip=pfl, nu_sd=sm)
    beh = Behaviour(rng.uniform(0.05, 0.2, trials) * L, phases, 0.03 * L, rng)
    res = cl.run(beh, d, n, meas, rng)
    x_pre = res.residuals + res.corrections
    c_prev = np.concatenate([np.zeros((trials, 1)), res.corrections[:, :-1]], axis=1)
    ahead = np.concatenate([x_pre[:, 1:] + res.corrections[:, :-1],
                            np.full((trials, 1), np.nan)], axis=1)   # x_pre_t + eps_{t+1}
    bits = np.stack([unpack(res.observations[:, t], m) for t in range(rounds)], axis=1)
    return dict(bits=(2.0 * bits - 1.0).astype(np.float32), c_prev=c_prev.astype(np.float32),
                now=bin_index(x_pre, N_BINS), ahead=bin_index(np.nan_to_num(ahead), N_BINS),
                ahead_mask=~np.isnan(ahead))


def _loss(model, batch, ce):
    now_logits, ahead_logits, _ = model(batch["bits"], batch["c_prev"])
    nb = model.n_bins
    l_now = ce(now_logits.reshape(-1, nb), batch["now"].reshape(-1))
    mask = batch["ahead_mask"].reshape(-1)
    l_ahead = ce(ahead_logits.reshape(-1, nb)[mask], batch["ahead"].reshape(-1)[mask])
    return l_now + l_ahead


def train_one(name, scale, m, seed, budget, device):
    path = out_dir("checkpoints") / f"{name}.pt"
    config = dict(name=name, scale=scale, m=m, seed=seed, box=SCALES[scale], n_train=budget.n_train,
                  n_val=budget.n_val, hidden=budget.hidden, layers=LAYERS, n_bins=N_BINS,
                  batch=budget.batch_size, max_epochs=budget.max_epochs, patience=budget.patience,
                  lr=LR, rounds=ROUNDS, behaviour="sBs delta~U(0.05,0.2)L + N(0,0.03L) jitter")
    if is_complete(path, config):
        log(f"{name}: up to date")
        return path
    t0 = time.time()
    to = lambda a: torch.tensor(a, device=device)
    tr = {k: to(v) for k, v in make_data(scale, m, "train", budget.n_train, seed).items()}
    va = {k: to(v) for k, v in make_data(scale, m, "val", budget.n_val, seed).items()}
    torch.manual_seed(3000 + seed)
    model = BitLSTM(m, budget.hidden, LAYERS, N_BINS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    steps = math.ceil(budget.n_train / budget.batch_size)
    best, best_state, bad, history = math.inf, None, 0, []
    for epoch in range(budget.max_epochs):
        perm = torch.randperm(budget.n_train, device=device)
        model.train()
        for i in range(steps):
            idx = perm[i * budget.batch_size:(i + 1) * budget.batch_size]
            loss = _loss(model, {k: v[idx] for k, v in tr.items()}, ce)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = _loss(model, va, ce).item()
        history.append(dict(epoch=epoch, val_loss=val))
        log(f"{name} epoch {epoch}: val {val:.4f}")
        if val < best - 1e-5:
            best, best_state, bad = val, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= budget.patience:
                break
        if time.time() - t0 > budget.train_time_cap_s:
            break
    model.load_state_dict(best_state)
    ckpt = dict(state_dict=model.state_dict(), m=m, hidden=budget.hidden, layers=LAYERS,
                n_bins=N_BINS, history=history, config=config, train_seconds=time.time() - t0)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)
    write_sidecar(path, config, extra=dict(train_seconds=ckpt["train_seconds"], best_val=best))
    return path


def train_all(names=None):
    budget, device = get_budget(), torch_device()
    log(f"binary LSTMs (mode={get_mode()}, device={device})")
    return {name: train_one(name, scale, m, seed, budget, device)
            for name, scale, m, seed in model_list() if names is None or name in names}


if __name__ == "__main__":
    train_all()
