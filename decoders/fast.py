"""GPU-optimised drivers for the parity decoders (same decisions, less overhead).

The reference decoders (grid_filter.py, adaptive.py, LSTM/model.py) are driven
round by round through simulator/sim.py and return numpy arrays every round,
which forces a host-device sync per round and leaves the GPU mostly idle.
Here, because syndromes do not depend on the decoder (simulator/fast.py):

  * the syndrome stream is computed up front and moved to the GPU once;
  * decisions stay on the GPU and are copied back once at the end;
  * all per-element work is float32 (the laptop RTX 5080 runs float64 at a
    small fraction of float32 speed); only the cumulative correction C and
    per-trial evidences stay float64;
  * models that share a grid are advanced with one batched matrix product;
  * the rare "belief lost all support" reset is flagged per trial without a
    host sync, and flagged trials are re-run with the reference decoder;
  * the oracle builds per-trial kernels in float32 and reuses a kernel table
    when the schedule takes few distinct values;
  * the LSTM reads the whole syndrome sequence in one cuDNN call.

Decisions agree with the reference decoders up to float32 rounding at
near-ties (tests/test_fast.py checks agreement rates).
"""

import math
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import norm

from simulator.fast import logical_errors, syndromes_and_nl
from simulator.noise import L

TWO_L = 2 * L
# fused (torch.compile) gather -> likelihood multiply -> X-marginal kernel;
# disable with QKP_FUSED=0 (e.g. where Triton is unavailable)
FUSED = __import__("os").environ.get("QKP_FUSED", "1") != "0"


def _post_eager(P, index, scale, out, dim, eps_dim=1):
    torch.gather(P, dim, index, out=out)
    out.mul_(scale)
    return out.sum(dim=eps_dim)


_compiled = {}


def _post(P, index, scale, out, dim, eps_dim=1):
    """out = gather(P, dim, index) * scale (in place); returns the eps-marginal.

    One fused GPU kernel when FUSED (reads P once, writes out once) instead of
    three passes over the belief tensor.
    """
    if not (FUSED and P.is_cuda):
        return _post_eager(P, index, scale, out, dim, eps_dim)
    fn = _compiled.get((dim, eps_dim))
    if fn is None:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)

        def post(P, index, scale, out):
            out.copy_(torch.gather(P, dim, index) * scale)
            return out.sum(dim=eps_dim)
        fn = _compiled[(dim, eps_dim)] = torch.compile(post, fullgraph=True)
    return fn(P, index, scale, out)


def _device(device):
    return device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# shared grid machinery
# ---------------------------------------------------------------------------
class Grid:
    """eps and X grids for a given sigma cap (same rule as grid_filter.py)."""

    def __init__(self, n_x, sigma_cap, device):
        self.n_x = n_x
        self.h = h = TWO_L / n_x
        R = max(5 * sigma_cap, 0.6 * L)
        self.M = int(math.ceil(R / h))
        self.n_e = 2 * self.M + 1
        self.eps_np = (np.arange(self.n_e) - self.M) * h
        self.eps = torch.tensor(self.eps_np, device=device, dtype=torch.float32)
        self.x = torch.arange(n_x, device=device, dtype=torch.float32) * h
        shift = torch.arange(self.n_e, device=device) - self.M
        xs = torch.arange(n_x, device=device)
        self.shear = (xs[None, :] - shift[:, None]) % n_x            # (n_e, n_x)
        self.device = device

    def kernel(self, sigma_q, rho):
        h, eps = self.h, self.eps_np
        sig = math.sqrt(max(sigma_q**2 * (1 - rho**2), h**2 / 12))
        hi = (eps[:, None] + h / 2 - rho * eps[None, :]) / sig
        lo = (eps[:, None] - h / 2 - rho * eps[None, :]) / sig
        K = norm.cdf(hi) - norm.cdf(lo)
        return K / K.sum(axis=0, keepdims=True)

    def prior(self, sigma_q):
        h, eps = self.h, self.eps_np
        p = norm.cdf((eps + h / 2) / sigma_q) - norm.cdf((eps - h / 2) / sigma_q)
        return p / p.sum()

    def sigma_eff(self, sigma_m):
        return math.sqrt(sigma_m**2 + self.h**2 / 12)


def _offsets(s, C):
    """(s + C) mod 2L in float64 (C accumulates), returned as float32."""
    return torch.remainder(s + C, TWO_L).to(torch.float32)


def _likelihood(x, off, sigma_eff):
    """Periodic Gaussian likelihood of the syndrome on the X grid, float32.

    Returns w (b, n_x) = sum over 5 images of exp(-(d + mL)^2 / 2 sigma^2),
    i.e. sigma sqrt(2 pi) p(s | X); the normalising constant is added to the
    evidence separately.
    """
    d = torch.remainder(off[:, None] - x[None, :] + L / 2, L) - L / 2
    inv = -0.5 / sigma_eff**2
    w = torch.exp(d * d * inv)
    for m in (1.0, 2.0):
        w = w + torch.exp((d + m * L) ** 2 * inv) + torch.exp((d - m * L) ** 2 * inv)
    return w


def _class_one(x, off):
    u = torch.remainder(x[None, :] - off[:, None], TWO_L)
    return ((u >= L / 2) & (u < 1.5 * L)).to(torch.float32)


# ---------------------------------------------------------------------------
# grid filters
# ---------------------------------------------------------------------------
class Bank:
    """M AR(1) models on one shared grid.

    joint=False: M independent filters (each normalised and scored on its own;
                 matched Bayes, Bayes-broad, Bayes-window).
    joint=True:  one filter over (model, eps, X) with model jumps (IMM).
    """

    def __init__(self, models, n_x, device, joint=False, hazard=0.0, tf32=True):
        self.models = models
        self.M = len(models)
        self.device = device
        self.joint = joint
        self.tf32 = tf32
        self.grid = g = Grid(n_x, max(m["sigma_q"] for m in models), device)
        self.K = torch.tensor(np.stack([g.kernel(m["sigma_q"], m["rho"]) for m in models]),
                              device=device, dtype=torch.float32)
        self.prior = torch.tensor(np.stack([g.prior(m["sigma_q"]) for m in models]),
                                  device=device, dtype=torch.float32)
        se = [g.sigma_eff(m["sigma_m"]) for m in models]
        self.uniq_se = sorted(set(se))
        self.se_index = torch.tensor([self.uniq_se.index(v) for v in se], device=device)
        log_norm = [math.log(v * math.sqrt(2 * math.pi)) for v in se]
        self.log_norm = torch.tensor(log_norm, device=device, dtype=torch.float64)
        # joint filter: regimes with different sigma_m need their own
        # normalisation inside the posterior; scale relative to the smallest
        self.log_norm_min = min(log_norm)
        self.joint_scale = torch.tensor([math.exp(self.log_norm_min - v) for v in log_norm],
                                        device=device, dtype=torch.float32)
        if joint and self.M > 1:
            T = np.full((self.M, self.M), hazard / (self.M - 1))
            np.fill_diagonal(T, 1 - hazard)
            self.T = torch.tensor(T.T, device=device, dtype=torch.float32)   # new = T^T old
        else:
            self.T = None

    def bytes_per_trial(self):
        return 4 * self.M * self.grid.n_e * self.grid.n_x * 4

    def reset(self, b):
        """Allocate state for b trials (buffers are reused across rounds in place,
        so the round step can be captured in a CUDA graph)."""
        g = self.grid
        self.B = torch.empty((self.M, g.n_e, b, g.n_x), device=self.device, dtype=torch.float32)
        shape = (b,) if self.joint else (self.M, b)
        self.evidence = torch.empty(shape, device=self.device, dtype=torch.float64)
        self.invZ = torch.empty(shape, device=self.device, dtype=torch.float32)
        self.bad = torch.empty(b, device=self.device, dtype=torch.bool)
        self.index = g.shear[None, :, None, :].expand(self.M, g.n_e, b, g.n_x)
        self.reinit()

    def reinit(self):
        """Back to the prior, in place."""
        scale = 1.0 / self.M if self.joint else 1.0
        self.B.zero_()
        self.B[:, :, :, 0] = self.prior[:, :, None] * scale
        self.evidence.zero_()
        # 1/Z of the previous round, folded into this round's likelihood multiply
        # (prediction is linear, so normalising late gives the same posterior)
        self.invZ.fill_(1.0)
        self.bad.zero_()

    def step(self, s, C):
        """Advance one round. Returns p_one: (b,) if joint else (M, b)."""
        g = self.grid
        B = self.B
        M, n_e, b, n_x = B.shape
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.tf32
        try:
            X = B.reshape(M, -1)
            if self.T is not None:
                X = self.T @ X
            P = torch.bmm(self.K, X.reshape(M, n_e, b * n_x)).reshape(M, n_e, b, n_x)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        off = _offsets(s, C)
        ws = torch.stack([_likelihood(g.x, off, se) for se in self.uniq_se])  # (S, b, n_x)
        w = ws[self.se_index] if len(self.uniq_se) > 1 else ws.expand(M, b, n_x)
        mask = _class_one(g.x, off)
        if self.joint:
            if len(self.uniq_se) > 1:
                w = w * self.joint_scale[:, None, None]
            px = _post(P, self.index, (w * self.invZ[None, :, None])[:, None], B, 3).sum(dim=0)
            Z = px.sum(dim=-1)                                      # (b,)
            ok = (Z > 0) & torch.isfinite(Z)
            self.evidence.add_(torch.log(Z.to(torch.float64)) - self.log_norm_min)
            bad = ~ok
        else:
            px = _post(P, self.index, (w * self.invZ[:, :, None])[:, None], B, 3)  # (M, b, n_x)
            Z = px.sum(dim=-1)                                      # (M, b)
            ok = (Z > 0) & torch.isfinite(Z)
            self.evidence.add_(torch.log(Z.to(torch.float64)) - self.log_norm[:, None])
            bad = (~ok).any(dim=0)
        self.invZ.copy_(torch.where(ok, 1.0 / Z, torch.ones_like(Z)))
        self.bad.logical_or_(bad)
        return (px * mask).sum(dim=-1) * self.invZ


class FastGridDecoder:
    is_fast = True   # experiments.common.run_blocked routes these to fast.run
    """Matched / fixed-model Bayes, Bayes-broad (full-history average) and
    Bayes-window W (sliding-window average): independent models, grouped by
    grid so each group is one batched product. IMM: joint=True."""

    def __init__(self, models, n_x=96, device=None, window=None, joint=False, hazard=0.0,
                 reference=None, name="Bayes (fast)"):
        self.device = _device(device)
        self.name = name
        self.window = window
        self.joint = joint
        self.reference = reference
        if joint:
            self.banks = [Bank(models, n_x, self.device, joint=True, hazard=hazard)]
        else:
            groups = defaultdict(list)
            for m in models:
                groups[round(m["sigma_q"], 12)].append(m)
            self.banks = [Bank(ms, n_x, self.device) for ms in groups.values()]
        self.n_models = len(models)

    def bytes_per_trial(self):
        return sum(b.bytes_per_trial() for b in self.banks)

    _BANK_STATE = ("B", "evidence", "invZ", "bad", "index")

    def _alloc(self, T, b):
        dev = self.device
        for bank in self.banks:
            bank.reset(b)
        st = dict(
            # each cached graph owns its banks' buffers (a graph replays raw
            # pointers, so they must stay alive and be restored before use)
            banks=[{k: getattr(bk, k) for k in self._BANK_STATE} for bk in self.banks],
            s=torch.zeros((T, b), device=dev, dtype=torch.float64),
            t=torch.zeros(1, device=dev, dtype=torch.long),
            C=torch.zeros(b, device=dev, dtype=torch.float64),
            J=torch.zeros((T, b), device=dev, dtype=torch.int8),
        )
        if not self.joint and self.n_models > 1:
            st["logw"] = torch.zeros((self.n_models, b), device=dev, dtype=torch.float64)
            if self.window:
                st["ring"] = torch.zeros((self.window, self.n_models, b), device=dev,
                                         dtype=torch.float64)
        return st

    def _reinit(self, st):
        for bank in self.banks:
            bank.reinit()
        for key in ("t", "C", "J", "logw", "ring"):
            if key in st:
                st[key].zero_()

    def _round(self, st):
        """One round, every state update in place (CUDA-graph safe)."""
        t = st["t"]
        s = st["s"].index_select(0, t)[0]
        C = st["C"]
        if self.joint:
            p_one = self.banks[0].step(s, C)
        elif self.n_models == 1:
            p_one = self.banks[0].step(s, C)[0]
        else:
            before = torch.cat([bk.evidence for bk in self.banks])
            p = torch.cat([bk.step(s, C) for bk in self.banks])
            inc = torch.cat([bk.evidence for bk in self.banks]) - before
            logw = st["logw"]
            if self.window:
                slot = torch.remainder(t, self.window)
                logw.add_(inc - st["ring"].index_select(0, slot)[0])
                st["ring"].index_copy_(0, slot, inc[None])
            else:
                logw.add_(inc)
            w = torch.softmax(logw, dim=0).to(torch.float32)
            p_one = (w * p).sum(dim=0)
        j = p_one > 0.5
        st["J"].index_copy_(0, t, j.to(torch.int8)[None])
        C.copy_(torch.remainder(C + s + j.to(torch.float64) * L, TWO_L))
        t.add_(1)

    def decide(self, s_all, graph=None):
        """s_all: (T, b) float64 syndromes on device. Returns j (T, b) int8 on device.

        graph=None uses a CUDA graph of one round on CUDA devices (one launch per
        round instead of dozens); graph=False runs the same code eagerly.
        """
        T, b = s_all.shape
        use_graph = (self.device.type == "cuda") if graph is None else graph
        key = (T, b, FUSED)          # a graph bakes in the kernels it was captured with
        cache = getattr(self, "_graphs", {})
        if use_graph and key in cache:
            g, st = cache[key]
            for bk, saved in zip(self.banks, st["banks"]):
                for k, v in saved.items():
                    setattr(bk, k, v)
        else:
            st = self._alloc(T, b)
            g = None
            if use_graph:
                st["s"].copy_(s_all)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):                  # warm-up (cuBLAS workspaces)
                        self._round(st)
                torch.cuda.current_stream().wait_stream(side)
                self._reinit(st)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._round(st)
                cache[key] = (g, st)
                self._graphs = cache
        self._reinit(st)
        st["s"].copy_(s_all)
        for _ in range(T):
            if g is not None:
                g.replay()
            else:
                self._round(st)
        self.bad = torch.stack([bk.bad for bk in self.banks]).any(dim=0)
        return st["J"].clone()


# ---------------------------------------------------------------------------
# oracle: per-trial, per-round kernels
# ---------------------------------------------------------------------------
class FastOracle:
    is_fast = True   # experiments.common.run_blocked routes these to fast.run
    """Oracle Bayes filter, float32 kernels, kernel table for repeated values."""

    def __init__(self, n_x=96, device=None, table_limit=1024, name="Bayes (oracle, fast)",
                 reference=None):
        self.device = _device(device)
        self.reference = reference
        self.n_x = n_x
        self.table_limit = table_limit
        self.name = name

    def bytes_per_trial(self, sigma_cap=0.5 * L):
        g = Grid(self.n_x, sigma_cap, torch.device("cpu"))
        return 4 * (g.n_e * g.n_e + 4 * g.n_e * g.n_x)

    def decide(self, s_all, schedule):
        dev = self.device
        T, b = s_all.shape
        g = Grid(self.n_x, schedule.max_sigma(), dev)
        sq = schedule.sigma_q.T                                      # (T, b) numpy
        prev = np.concatenate([sq[:1], sq[:-1]], axis=0)
        a = (schedule.rho.T * sq / prev).astype(np.float32)
        sd = np.maximum(sq * np.sqrt(1 - schedule.rho.T ** 2), g.h / math.sqrt(12)).astype(
            np.float32)
        pairs = np.stack([a, sd], axis=-1).reshape(-1, 2)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
        inv = inv.reshape(T, b)
        use_table = len(uniq) <= self.table_limit
        if use_table:
            table = self._kernels(torch.tensor(uniq[:, 0], device=dev),
                                  torch.tensor(uniq[:, 1], device=dev), g)
            inv_t = torch.tensor(inv, device=dev)
            single = [len(np.unique(inv[t])) == 1 for t in range(T)]
        else:
            a_t = torch.tensor(a, device=dev)
            sd_t = torch.tensor(sd, device=dev)
        se = np.sqrt(schedule.sigma_m.T ** 2 + g.h**2 / 12).astype(np.float32)
        se_t = torch.tensor(se, device=dev)
        kick_rows = [np.flatnonzero(schedule.kick[:, t] > 0) for t in range(T)]
        kick_t = torch.tensor(schedule.kick.T.astype(np.float32), device=dev)

        # belief layout (b, n_e, n_x); prior N(0, sigma_1^2)
        sd0 = torch.tensor(schedule.sigma_q[:, 0].astype(np.float32), device=dev)
        prior = torch.special.ndtr((g.eps[None, :] + g.h / 2) / sd0[:, None]) - \
            torch.special.ndtr((g.eps[None, :] - g.h / 2) / sd0[:, None])
        prior = prior / prior.sum(dim=1, keepdim=True)
        B = torch.zeros((b, g.n_e, g.n_x), device=dev, dtype=torch.float32)
        B[:, :, 0] = prior
        index = g.shear[None].expand(b, g.n_e, g.n_x)
        C = torch.zeros(b, device=dev, dtype=torch.float64)
        J = torch.empty((T, b), device=dev, dtype=torch.int8)
        bad = torch.zeros(b, device=dev, dtype=torch.bool)
        invZ = torch.ones(b, device=dev, dtype=torch.float32)
        offsets = torch.remainder(torch.arange(g.n_x, device=dev, dtype=torch.float32) * g.h + L,
                                  TWO_L) - L
        for t in range(T):
            if use_table and single[t]:
                K = table[int(inv[t, 0])]
                B = (K @ B.permute(1, 0, 2).reshape(g.n_e, -1)).reshape(g.n_e, b, g.n_x) \
                    .permute(1, 0, 2)
            else:
                K = table[inv_t[t]] if use_table else self._kernels(a_t[t], sd_t[t], g)
                B = torch.bmm(K, B)
            s = s_all[t]
            off = _offsets(s, C)
            d = torch.remainder(off[:, None] - g.x[None, :] + L / 2, L) - L / 2
            inv2 = (-0.5 / se_t[t] ** 2)[:, None]
            w = torch.exp(d * d * inv2)
            for m in (1.0, 2.0):
                w = w + torch.exp((d + m * L) ** 2 * inv2) + torch.exp((d - m * L) ** 2 * inv2)
            scale = (w * invZ[:, None])[:, None, :]
            rows = kick_rows[t]
            if len(rows):            # burst round: shear, kick, then multiply
                B = torch.gather(B, 2, index)
                B = self._kick(B, torch.tensor(rows, device=dev), kick_t[t], offsets, g)
                B = B * scale
                px = B.sum(dim=1)
            else:
                out = torch.empty((b, g.n_e, g.n_x), device=dev, dtype=torch.float32)
                px = _post(B.contiguous(), index, scale, out, 2)
                B = out
            Z = px.sum(dim=-1)
            ok = (Z > 0) & torch.isfinite(Z)
            bad |= ~ok
            invZ = torch.where(ok, 1.0 / Z, torch.ones_like(Z))
            p_one = (px * _class_one(g.x, off)).sum(dim=-1) * invZ
            j = p_one > 0.5
            J[t] = j.to(torch.int8)
            C = torch.remainder(C + s + j.to(torch.float64) * L, TWO_L)
        self.bad = bad
        return J

    @staticmethod
    def _kernels(a, sd, g):
        e_new = g.eps[None, :, None]
        e_old = g.eps[None, None, :]
        a = a[:, None, None]
        sd = sd[:, None, None]
        K = torch.special.ndtr((e_new + g.h / 2 - a * e_old) / sd) - \
            torch.special.ndtr((e_new - g.h / 2 - a * e_old) / sd)
        return K / K.sum(dim=1, keepdim=True).clamp_min(1e-30)

    @staticmethod
    def _kick(B, rows, kick, offsets, g):
        k = kick[rows]
        m = torch.arange(-2, 3, device=B.device, dtype=torch.float32) * TWO_L
        ker = torch.exp(-((offsets[None, :, None] + m) ** 2) / (2 * k[:, None, None] ** 2)).sum(2)
        ker = ker / ker.sum(dim=1, keepdim=True)
        sub = B[rows]
        conv = torch.fft.irfft(torch.fft.rfft(sub, dim=-1) * torch.fft.rfft(ker, dim=-1)[:, None],
                               n=g.n_x, dim=-1).clamp_min(0)
        B = B.clone()
        B[rows] = conv
        return B


# ---------------------------------------------------------------------------
# LSTM: whole sequence in one call
# ---------------------------------------------------------------------------
def lstm_decide(model, head_index, k, s_all, chunk=None):
    """j (T, b) from a trained ParityLSTM, reading all rounds at once."""
    T, b = s_all.shape
    s = s_all.T.to(torch.float32).contiguous()                       # (b, T)
    chunk = chunk or max(1, int(4e8 // max(1, T * model.hidden * 8)))
    outs = []
    with torch.no_grad():
        for start in range(0, b, chunk):
            logits, _ = model(s[start:start + chunk])
            outs.append(logits[..., head_index] > 0)
    w = torch.cat(outs).to(torch.int64)                              # (b, T)
    if k is None:
        Jc = w
    else:
        step = k + 1
        pad = (-T) % step
        wp = torch.nn.functional.pad(w, (0, pad))
        Jc = (torch.cumsum(wp.reshape(b, -1, step), dim=1) % 2).reshape(b, -1)[:, :T]
    prev = torch.nn.functional.pad(Jc[:, :-1], (1, 0))
    return (Jc ^ prev).T.to(torch.int8)


# ---------------------------------------------------------------------------
# end-to-end runners (noise -> logical-error record)
# ---------------------------------------------------------------------------
def run(decoder, displacement_noise, measurement_noise, schedule=None, block=None):
    """Logical-error record (trials, rounds) for a fast decoder or LSTM spec."""
    s, K = syndromes_and_nl(displacement_noise, measurement_noise)
    b_total = s.shape[0]
    block = block or b_total
    out = np.zeros_like(K)
    for start in range(0, b_total, block):
        stop = min(start + block, b_total)
        s_t = torch.tensor(s[start:stop].T, device=decoder.device, dtype=torch.float64)
        if isinstance(decoder, FastOracle):
            J = decoder.decide(s_t, schedule.slice(start, stop))
        else:
            J = decoder.decide(s_t)
        j = J.T.cpu().numpy()
        rec = logical_errors(K[start:stop], j)
        bad = getattr(decoder, "bad", None)
        idx = np.flatnonzero(bad.cpu().numpy()) if bad is not None else []
        decoder.resets = getattr(decoder, "resets", 0) + len(idx)
        if len(idx):
            factory = getattr(decoder, "reference", None)
            if factory is None:
                raise FloatingPointError(
                    f"{decoder.name}: {len(idx)} trials lost all belief support and no "
                    "reference decoder was given to re-run them")
            from experiments.common import run_blocked
            sub_sched = None if schedule is None else _rows(schedule.slice(start, stop), idx)
            rec[idx] = run_blocked(factory(), displacement_noise[start:stop][idx],
                                   measurement_noise[start:stop][idx], len(idx),
                                   schedule=sub_sched)
        out[start:stop] = rec
    return out


def _rows(schedule, idx):
    from simulator.nonstationary import Schedule
    return Schedule(schedule.sigma_q[idx], schedule.rho[idx], schedule.sigma_m[idx],
                    schedule.kick[idx], schedule.tau[idx], dict(schedule.info))


class FastLSTM:
    is_fast = True   # experiments.common.run_blocked routes these to fast.run
    def __init__(self, model, head, device=None, name="LSTM (fast)"):
        from decoders.LSTM.model import HEADS
        self.device = _device(device)
        self.model = model.to(self.device).eval()
        names = [h for h, _ in HEADS]
        self.head_index = names.index(head)
        self.k = dict(HEADS)[head]
        self.name = name
        self.reference = None

    def decide(self, s_all):
        return lstm_decide(self.model, self.head_index, self.k, s_all)


def fast_decoders(device):
    """name -> fn(d, n, schedule) -> logical-error record, for profiling."""
    from decoders.bayesian.adaptive import OracleBayesDecoder
    from decoders.bayesian.grid_filter import BayesGridDecoder
    from experiments import run_all
    from experiments.common import NOMINAL, out_dir
    from experiments.run_nonstationary import ADAPTIVE_GRID
    nom = [dict(NOMINAL)]
    out = {
        "Bayes (matched)": FastGridDecoder(
            nom, device=device,
            reference=lambda: BayesGridDecoder(**NOMINAL, n_x=96, device=device)),
        "Bayes-broad": FastGridDecoder(run_all.BROAD_COMPONENTS, device=device),
        "Bayes (oracle)": FastOracle(device=device),
        "Bayes-IMM": FastGridDecoder(ADAPTIVE_GRID, device=device, joint=True, hazard=0.005),
        "Bayes-window 200": FastGridDecoder(ADAPTIVE_GRID, device=device, window=200),
    }
    path = out_dir("checkpoints") / "lstm_nominal_s0.pt"
    if path.exists():
        from decoders.LSTM.model import load_model
        model, ck = load_model(path, device)
        out["LSTM-nominal"] = FastLSTM(model, ck["chosen_head"], device=device)
    for dec in out.values():
        if not hasattr(dec, "reference"):
            dec.reference = None
    return {name: (lambda d, n, s, dec=dec: run(dec, d, n, schedule=s)) for name, dec in out.items()}
