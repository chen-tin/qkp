"""LSTM decoder for one-bit syndromes in the closed loop.

Inputs per round: the m bits of the round (as +-1) and the correction the
decoder applied in the previous round (c/L, sin and cos of 2 pi c / L), so the
network sees its own feedback. Outputs: two distributions over n_bins bins of
x mod 2L,

    head "now":   x_pre_t            (displacement when the bits were taken)
    head "ahead": x_pre_t + eps_{t+1} = x_pre_{t+1} + c_t

The "ahead" target does not depend on the correction c_t, so the decoder can
choose c_t with the same lookahead cost as the binary Bayes filter
(decoders/bayesian/binary_filter.py): minimise P(odd cell) + lam * centring
over the predicted distribution of x_pre_t + eps_{t+1} - c.

Training data come from a behaviour policy (training/train_binary.py); see
there for the distribution-shift caveat.
"""

import math

import numpy as np
import torch
from torch import nn

from decoders.bayesian.binary_filter import correction_costs, unpack
from simulator.noise import L


def features(bits_pm, c_prev):
    """bits_pm: (..., m) in {-1, +1}; c_prev: (...) correction in absolute units."""
    a = 2 * math.pi * c_prev / L
    return torch.cat([bits_pm, (c_prev / L)[..., None], torch.sin(a)[..., None],
                      torch.cos(a)[..., None]], dim=-1)


class BitLSTM(nn.Module):
    def __init__(self, m, hidden=128, layers=2, n_bins=48):
        super().__init__()
        self.m, self.hidden, self.layers, self.n_bins = m, hidden, layers, n_bins
        self.lstm = nn.LSTM(m + 3, hidden, layers, batch_first=True)
        self.head_now = nn.Linear(hidden, n_bins)
        self.head_ahead = nn.Linear(hidden, n_bins)

    def forward(self, bits_pm, c_prev, state=None):
        out, state = self.lstm(features(bits_pm, c_prev), state)
        return self.head_now(out), self.head_ahead(out), state


def bin_index(x, n_bins):
    """Bin of x mod 2L (numpy or torch)."""
    if isinstance(x, torch.Tensor):
        return (torch.remainder(x, 2 * L) / (2 * L) * n_bins).long().clamp_(0, n_bins - 1)
    return np.clip((np.remainder(x, 2 * L) / (2 * L) * n_bins).astype(np.int64), 0, n_bins - 1)


class BinaryLSTMDecoder:
    """Closed-loop decoder: one LSTM step per round, lookahead cost policy."""

    def __init__(self, model, lam=1e-2, device=None, name="LSTM (binary)"):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device).eval()
        self.name = name
        centres = (np.arange(model.n_bins) + 0.5) * 2 * L / model.n_bins
        cost, self.c_values = correction_costs(centres, lam)
        self.cost = torch.tensor(cost, device=self.device, dtype=torch.float32)

    def reset(self, trials):
        self.state = None
        self.c_prev = torch.zeros(trials, device=self.device)

    @torch.no_grad()
    def decide(self, bits):
        b = torch.tensor(unpack(bits, self.model.m), device=self.device, dtype=torch.float32)
        _, ahead, self.state = self.model((2 * b - 1)[:, None, :], self.c_prev[:, None], self.state)
        p = torch.softmax(ahead[:, 0], dim=-1)
        k = torch.argmin(p @ self.cost, dim=1)
        c = self.c_values[k.cpu().numpy()]
        self.c_prev = torch.tensor(c, device=self.device, dtype=torch.float32)
        return c


def load(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    model = BitLSTM(ck["m"], ck["hidden"], ck["layers"], ck["n_bins"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
