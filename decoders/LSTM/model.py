"""LSTM decoder.

The network reads the syndrome history and predicts parities of the
nearest-lattice jumps y_t (y_t = 1 when the nearest-lattice correction of
round t moves the state by an odd number of lattice spacings):

    head "k0"   : y_t                         (this round only)
    head "kK"   : W_t = y_{t-K} + ... + y_t mod 2   (window of K+1 rounds)
    head "full" : K_t = y_1 + ... + y_t mod 2       (whole history)

A head's prediction W_hat_t fixes the cumulative parity of the decoder's own
corrections, J_t = j_1 + ... + j_t mod 2, through

    J_t = W_hat_t XOR J_{t-K-1}          (J_t = K_hat_t for the full head)

and the logical state after round t is K_t XOR J_t. Wider windows let the
decoder revise more of its past decisions as new syndromes arrive. The head
used at test time is selected on validation data only.
"""

import math

import numpy as np
import torch
from torch import nn

from decoders.base import ParityDecoder
from simulator.noise import L

HEADS = (("k0", 0), ("k1", 1), ("k2", 2), ("k4", 4), ("full", None))


def features(s):
    """Per-round inputs from syndromes s (any shape): [s/L, sin, cos]."""
    a = 2 * math.pi * s / L
    return torch.stack([s / L, torch.sin(a), torch.cos(a)], dim=-1)


def head_labels(y):
    """Targets for every head from jump parities y, shape (B, T) -> (B, T, H)."""
    K = torch.cumsum(y.to(torch.int64), dim=1) % 2
    out = []
    for _, k in HEADS:
        if k is None:
            out.append(K)
        else:
            shifted = torch.zeros_like(K)
            if k + 1 < K.shape[1]:
                shifted[:, k + 1:] = K[:, : K.shape[1] - k - 1]
            out.append(K ^ shifted)
    return torch.stack(out, dim=-1).to(torch.float32)


def cumulative_from_head(w_hat, k):
    """J_t from a head's predictions w_hat (B, T) of {0,1} ints."""
    if k is None:
        return w_hat.clone()
    B, T = w_hat.shape
    J = torch.zeros_like(w_hat)
    for t in range(T):
        prev = J[:, t - k - 1] if t - k - 1 >= 0 else 0
        J[:, t] = w_hat[:, t] ^ prev
    return J


class ParityLSTM(nn.Module):
    def __init__(self, hidden=128, layers=2):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.lstm = nn.LSTM(3, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, len(HEADS))

    def forward(self, s, state=None):
        """s: (B, T) syndromes. Returns logits (B, T, H) and the LSTM state."""
        out, state = self.lstm(features(s), state)
        return self.head(out), state


class LSTMDecoder(ParityDecoder):
    """Runs a trained ParityLSTM one round at a time (closed loop)."""

    def __init__(self, model, head="full", name="LSTM", device=None, record=False):
        super().__init__(record=record)
        names = [h for h, _ in HEADS]
        self.head_index = names.index(head)
        self.k = dict(HEADS)[head]
        self.name = name
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device).eval()

    def _reset(self, trials):
        self.state = None
        self.J = []        # cumulative parity of own corrections, per round

    @torch.no_grad()
    def choose(self, syndrome):
        s = torch.as_tensor(syndrome, device=self.device, dtype=torch.float32)
        logits, self.state = self.model(s[:, None], self.state)
        logit = logits[:, 0, self.head_index]
        w_hat = (logit > 0).to(torch.int64).cpu().numpy()
        p_w = torch.sigmoid(logit).cpu().numpy()

        t = len(self.J)
        if self.k is None:
            ref = np.zeros_like(w_hat)
        else:
            ref = self.J[t - self.k - 1] if t - self.k - 1 >= 0 else np.zeros_like(w_hat)
        J_t = w_hat ^ ref
        J_prev = self.J[-1] if self.J else np.zeros_like(w_hat)
        self.J.append(J_t)
        j = J_t ^ J_prev
        # probability that j = 1 is the right choice
        flip = (ref ^ J_prev).astype(bool)
        p_one = np.where(flip, 1 - p_w, p_w)
        return j, p_one


def load_model(path, device=None):
    """Load a checkpoint written by training/train_lstm.py."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ParityLSTM(ckpt["hidden"], ckpt["layers"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
