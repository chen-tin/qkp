"""Learned sequence decoders beyond the LSTM: GRU and causal transformer.

All models share the ParityLSTM interface of decoders/LSTM/model.py:

    forward(s, state=None) -> (logits (B, T, H), state)

with the same per-round features (features(s)) and the same parity heads
(HEADS), so the training targets, the head selection and the parity
bookkeeping are unchanged. Each model has a `.hidden` attribute (used by the
fast driver to size chunks).

  ParityGRU          nn.GRU stack + linear head
  ParityTransformer  causal (masked) self-attention over the syndrome history
                     with sinusoidal positions (no maximum length). For the
                     closed-loop reference path the state is the prefix of
                     feature vectors and the model recomputes over it (exact,
                     O(T^2)); the fast path reads the whole sequence at once,
                     which is valid because syndromes do not depend on
                     parity decisions.

Decoders:
  SeqDecoder   closed-loop reference driver (= LSTMDecoder, any model)
  FastSeq      fast driver for decoders/fast.py (`is_fast`, whole sequence)

load_model(path) reads the checkpoint's `arch` (default "lstm", so every
existing checkpoint still loads) and `arch_kwargs`.
"""

import math

import torch
from torch import nn

from decoders.fast import FastLSTM
from decoders.LSTM.model import HEADS, LSTMDecoder, ParityLSTM, features

N_FEATURES = 3


class ParityGRU(nn.Module):
    def __init__(self, hidden=128, layers=2):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.rnn = nn.GRU(N_FEATURES, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, len(HEADS))

    def forward(self, s, state=None):
        out, state = self.rnn(features(s), state)
        return self.head(out), state


def _positions(t0, T, d, device, dtype):
    """Sinusoidal position encodings for positions t0 .. t0 + T - 1, (T, d)."""
    pos = torch.arange(t0, t0 + T, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    angle = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(T, d, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle[:, : pe[:, 1::2].shape[1]])
    return pe.to(dtype)


class ParityTransformer(nn.Module):
    """Causal transformer encoder over the syndrome history."""

    def __init__(self, hidden=96, layers=2, heads=4, ff_mult=4, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.inp = nn.Linear(N_FEATURES, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, ff_mult * hidden, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, len(HEADS))

    def _encode(self, f):
        B, T, _ = f.shape
        x = self.inp(f) + _positions(0, T, self.hidden, f.device, f.dtype)[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=f.device), diagonal=1)
        return self.head(self.norm(self.encoder(x, mask=mask, is_causal=True)))

    def forward(self, s, state=None):
        """state: feature prefix (B, t, 3) from earlier calls (closed loop)."""
        f = features(s)
        if state is not None:
            full = torch.cat([state, f], dim=1)
            logits = self._encode(full)[:, state.shape[1]:]
            return logits, full
        return self._encode(f), f


ARCHS = {"lstm": ParityLSTM, "gru": ParityGRU, "transformer": ParityTransformer}


def build(arch, **kw):
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}")
    return ARCHS[arch](**kw)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def matched_kwargs(arch, target_params, layers=2, heads=4):
    """Hidden size for `arch` whose parameter count is closest to target_params."""
    best = None
    for h in range(8, 513, 4):
        if arch == "transformer" and h % heads:
            continue
        kw = dict(hidden=h, layers=layers) if arch != "transformer" else \
            dict(hidden=h, layers=layers, heads=heads)
        p = n_params(build(arch, **kw))
        if best is None or abs(p - target_params) < abs(best[1] - target_params):
            best = (kw, p)
    return best


class SeqDecoder(LSTMDecoder):
    """Closed-loop reference driver for any sequence model (same bookkeeping)."""


class FastSeq(FastLSTM):
    """Whole-sequence fast driver for any sequence model (decoders/fast.py)."""


def load_model(path, device=None):
    """Load a checkpoint from train_lstm.py or train_variants.py."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "lstm")
    kw = ckpt.get("arch_kwargs") or dict(hidden=ckpt["hidden"], layers=ckpt["layers"])
    model = build(arch, **kw)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
