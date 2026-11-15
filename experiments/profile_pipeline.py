"""Profile simulate / decode / encode and break down where the time goes.

Every run is recorded by experiments/runlog.py. For the GPU decoders the
PyTorch profiler splits device time into kernel categories (GEMM, shear
gather, likelihood math, reductions, copies) and the script reports how much
of the wall time the GPU was busy at all (the rest is Python / launch /
host-sync overhead).

    MPLBACKEND=Agg python -m experiments.profile_pipeline --impl reference
    MPLBACKEND=Agg python -m experiments.profile_pipeline --impl fast
    MPLBACKEND=Agg python -m experiments.profile_pipeline --impl both

Writes results/benchmarks/profile_<impl>.json.
"""

import argparse
import json
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from experiments import run_all
from experiments.common import NOMINAL, get_mode, log, make_rng, out_dir, torch_device
from experiments.runlog import RunLog
from simulator import nonstationary as ns
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice, Simulator

OUT = out_dir() / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = get_mode() == "quick"

CATEGORIES = [
    ("matmul (predict eps)", ("gemm", "bmm", "matmul", "mm", "addmm", "cutlass", "sgemm",
                              "xmma", "ampere", "sm80", "sm90", "sm100", "sm120", "cublas")),
    ("lstm (cuDNN)", ("lstm", "rnn", "cudnn")),
    ("gather / shift (shear)", ("gather", "index", "scatter")),
    ("likelihood math (exp/log/remainder)", ("exp", "log", "remainder", "fmod", "pow", "ndtr",
                                             "erf", "sqrt", "logsumexp")),
    ("reductions (sums, max, softmax)", ("reduce", "sum", "max", "amax", "softmax", "norm")),
    ("host<->device copies", ("memcpy", "copy_", "to_copy", "dtoh", "htod")),
    ("fft", ("fft",)),
]


def categorize(name):
    n = name.lower()
    for cat, keys in CATEGORIES:
        if any(k in n for k in keys):
            return cat
    return "other elementwise"


def profile_gpu(fn, device):
    """Run fn under the profiler; return (wall_s, gpu_busy_s, {category: s})."""
    if device.type != "cuda":
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0, None, {}
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    cats = defaultdict(float)
    busy = 0.0
    for e in prof.events():
        dev_type = getattr(e, "device_type", None)
        if dev_type is not None and str(dev_type).endswith("CUDA") and e.device_time_total:
            # kernel / memcpy events on the device
            if getattr(e, "is_async", False) or True:
                t = e.self_device_time_total / 1e6 if hasattr(e, "self_device_time_total") \
                    else e.device_time_total / 1e6
                cats[categorize(e.name)] += t
                busy += t
    return wall, busy, dict(cats)


def timed(fn, device, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


# ---------------------------------------------------------------------------
# reference implementations (what experiments use today)
# ---------------------------------------------------------------------------
def reference_decoders(device):
    from decoders.bayesian.adaptive import OracleBayesDecoder, RegimeBankDecoder, WindowedBayesDecoder
    from decoders.bayesian.grid_filter import BayesGridDecoder, MultiModelBayesDecoder
    from decoders.LSTM.model import LSTMDecoder, load_model
    from experiments.run_nonstationary import ADAPTIVE_GRID
    out = {
        "Bayes (matched)": lambda: BayesGridDecoder(**NOMINAL, n_x=96, device=device),
        "Bayes-broad": lambda: MultiModelBayesDecoder(run_all.BROAD_COMPONENTS, n_x=96,
                                                      device=device),
        "Bayes (oracle)": lambda: OracleBayesDecoder(n_x=96, device=device),
        "Bayes-IMM": lambda: RegimeBankDecoder(ADAPTIVE_GRID, hazard=0.005, n_x=96,
                                               device=device),
        "Bayes-window 200": lambda: WindowedBayesDecoder(ADAPTIVE_GRID, window=200, n_x=96,
                                                         device=device),
    }
    path = out_dir("checkpoints") / "lstm_nominal_s0.pt"
    if path.exists():
        def lstm():
            model, ck = load_model(path, device)
            return LSTMDecoder(model, head=ck["chosen_head"], device=device)
        out["LSTM-nominal"] = lstm
    return out


def run_reference(make, d, n, sch):
    from experiments.common import run_blocked
    dec = make()
    return run_blocked(dec, d, n, d.shape[0], schedule=sch)


def reference_simulate(d, n):
    return Simulator(trials=d.shape[0], rounds=d.shape[1]).run(NearestLattice(), d, n)


def reference_encode(delta, n_cutoff, k_max, grid):
    from quantum.encoding import encode_gkp
    import qutip as qt
    psi = encode_gkp(delta, n_cutoff, k_max)
    w = qt.wigner(psi, grid, grid, method="clenshaw")
    return psi, w


# ---------------------------------------------------------------------------
def profile(impl, log_run, device):
    batch = 512 if QUICK else 4096
    heavy = 256 if QUICK else 1024
    rounds = 50
    res = dict(impl=impl, batch=batch, heavy_batch=heavy, rounds=rounds, decoders={})
    d, n, sch = ns.sample(ns.spec("stationary"), make_rng("dev", "profile_pipeline", 0),
                          batch, rounds)

    # --- simulate --------------------------------------------------------
    with log_run.phase("simulate.noise", trials=batch, rounds=rounds):
        t_noise = timed(lambda: NoiseModel(**NOMINAL).sample(
            make_rng("dev", "profile_noise", 0), batch, rounds), torch.device("cpu"))
    if impl == "reference":
        sim_fn = lambda: reference_simulate(d, n)
    else:
        from simulator.fast import simulate_nl
        sim_fn = lambda: simulate_nl(d, n, device=device)
    with log_run.phase("simulate.loop", trials=batch, rounds=rounds):
        t_sim = timed(sim_fn, device)
    res["simulate"] = dict(noise_s=t_noise, loop_s=t_sim)
    log(f"[{impl}] simulate: noise {t_noise * 1e3:.1f} ms, loop {t_sim * 1e3:.1f} ms")

    # --- decode ------------------------------------------------------------
    if impl == "reference":
        decs = {k: (lambda m=m: (lambda dd, nn, ss: run_reference(m, dd, nn, ss)))()
                for k, m in reference_decoders(device).items()}
    else:
        from decoders.fast import fast_decoders
        decs = fast_decoders(device)
    for name, fn in decs.items():
        b = heavy if name in ("Bayes-IMM", "Bayes-window 200", "Bayes-broad", "Bayes (oracle)") else batch
        dd, nn, ss = d[:b], n[:b], sch.slice(0, b)
        fn(dd[:8], nn[:8], ss.slice(0, 8))                       # warm-up / compile
        with log_run.phase("decode", decoder=name, trials=b, rounds=rounds):
            wall = timed(lambda: fn(dd, nn, ss), device, repeats=2)
        wall_p, busy, cats = profile_gpu(lambda: fn(dd, nn, ss), device)
        # single-trial latency
        lat = timed(lambda: fn(dd[:1], nn[:1], ss.slice(0, 1)), device, repeats=3) / rounds
        res["decoders"][name] = dict(
            trials=b, wall_s=wall, us_per_trial_round=wall / (b * rounds) * 1e6,
            latency_ms_per_round=lat * 1e3, profiled_wall_s=wall_p, gpu_busy_s=busy,
            gpu_busy_fraction=(busy / wall_p) if busy else None, categories_s=cats)
        log(f"[{impl}] {name:<18} {wall / (b * rounds) * 1e6:8.2f} us/tr  "
            f"latency {lat * 1e3:7.3f} ms/round  GPU busy "
            f"{(busy / wall_p * 100 if busy else float('nan')):5.1f}%")

    # --- encode ------------------------------------------------------------
    grid = np.linspace(-5 * np.sqrt(2), 5 * np.sqrt(2), 201 if QUICK else 401)
    enc = {}
    for delta, ncut in ((0.30, 100), (0.20, 200)):
        if impl == "reference":
            from quantum.encoding import encode_gkp
            import qutip as qt
            with log_run.phase("encode.state", delta=delta, n_cutoff=ncut):
                t_state = timed(lambda: encode_gkp(delta, ncut, 8), torch.device("cpu"),
                                repeats=2)
            psi = encode_gkp(delta, ncut, 8)
            with log_run.phase("encode.wigner", delta=delta, grid=len(grid)):
                t_w = timed(lambda: qt.wigner(psi, grid, grid, method="clenshaw"),
                            torch.device("cpu"), repeats=1)
        else:
            from quantum.fast_encoding import encode_gkp_fast, wigner_fft
            with log_run.phase("encode.state", delta=delta, n_cutoff=ncut):
                t_state = timed(lambda: encode_gkp_fast(delta, ncut, 8, device=device), device,
                                repeats=3)
            c = encode_gkp_fast(delta, ncut, 8, device=device)
            wigner_fft(c, grid, device=device)
            with log_run.phase("encode.wigner", delta=delta, grid=len(grid)):
                t_w = timed(lambda: wigner_fft(c, grid, device=device), device, repeats=3)
        enc[f"delta={delta},N={ncut}"] = dict(state_s=t_state, wigner_s=t_w)
        log(f"[{impl}] encode delta={delta} N={ncut}: state {t_state * 1e3:.2f} ms, "
            f"wigner {t_w * 1e3:.1f} ms")
    res["encode"] = enc
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("reference", "fast", "both", "ab"), default="reference")
    args = parser.parse_args()
    if args.impl == "ab":
        return main_ab()
    device = torch_device()
    run_all.DEVICE = device
    impls = ("reference", "fast") if args.impl == "both" else (args.impl,)
    for impl in impls:
        with RunLog("profile_pipeline", args=dict(impl=impl)) as rl:
            res = profile(impl, rl, device)
            rl.note(summary={k: v["us_per_trial_round"] for k, v in res["decoders"].items()})
        (OUT / f"profile_{impl}.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"profile_pipeline: wrote {OUT}")



# ---------------------------------------------------------------------------
# interleaved A/B timing (robust to the laptop's power-limit drift)
# ---------------------------------------------------------------------------
def ab_compare(log_run, device, reps=5):
    from decoders.fast import FastGridDecoder, fast_decoders, run as fast_run
    batch = 512 if QUICK else 4096
    heavy = 256 if QUICK else 1024
    rounds = 50
    d, n, sch = ns.sample(ns.spec("stationary"), make_rng("dev", "profile_ab", 0), batch, rounds)
    ref = reference_decoders(device)
    fast = fast_decoders(device)
    # eager (no CUDA graph) variants of the grid decoders, to isolate the graph gain
    eager = {}
    from decoders import fast as fastmod
    for name, fn in fast.items():
        dec = fn.__defaults__[0] if fn.__defaults__ else None
        if isinstance(dec, FastGridDecoder):
            eager[name] = (lambda dd, nn, ss, dec=dec: _run_eager(dec, dd, nn))
    rows = {}

    def one(fn, dd, nn, ss):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(dd, nn, ss)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    for name in fast:
        b = heavy if name in ("Bayes-IMM", "Bayes-window 200", "Bayes-broad", "Bayes (oracle)") \
            else batch
        for label, size in (("throughput", b), ("latency", 1)):
            dd, nn, ss = d[:size], n[:size], sch.slice(0, size)
            variants = {"reference": (lambda a, b_, c, m=ref[name]: run_reference(m, a, b_, c)),
                        "fast": fast[name]}
            if name in eager:
                variants["fast, no CUDA graph"] = eager[name]
            if name != "LSTM-nominal":
                variants["fast, unfused"] = (lambda a, b_, c, f=fast[name]: _unfused(f, a, b_, c))
            for fn in variants.values():
                fn(dd, nn, ss)                       # warm-up / graph capture
            times = {k: [] for k in variants}
            for _ in range(reps):
                for k, fn in variants.items():
                    times[k].append(one(fn, dd, nn, ss))
            med = {k: float(np.median(v)) for k, v in times.items()}
            unit = (size * rounds) if label == "throughput" else rounds
            row = {k: v / unit * (1e6 if label == "throughput" else 1e3) for k, v in med.items()}
            row["speedup"] = med["reference"] / med["fast"]
            rows[f"{name} | {label}"] = row
            with log_run.phase("ab", decoder=name, mode=label, trials=size, rounds=rounds,
                               **{k.replace(" ", "_").replace(",", ""): v for k, v in row.items()}):
                pass
            log(f"[ab] {name:<18} {label:<10} " + "  ".join(
                f"{k} {v:9.3f}" for k, v in row.items()))

    # simulate and encode
    from simulator.fast import simulate_nl
    t_ref, t_fast = [], []
    for _ in range(reps):
        t_ref.append(timed(lambda: reference_simulate(d, n), torch.device("cpu"), 1))
        t_fast.append(timed(lambda: simulate_nl(d, n), torch.device("cpu"), 1))
    rows["simulate (NL loop)"] = dict(reference=np.median(t_ref) * 1e3,
                                      fast=np.median(t_fast) * 1e3,
                                      speedup=np.median(t_ref) / np.median(t_fast))
    from quantum.encoding import encode_gkp
    from quantum.fast_encoding import encode_gkp_fast, wigner_fft
    import qutip as qt
    grid = np.linspace(-5 * np.sqrt(2), 5 * np.sqrt(2), 201)
    for delta, ncut in ((0.30, 100), (0.20, 200)):
        tr = [timed(lambda: encode_gkp(delta, ncut, 8), torch.device("cpu"), 1) for _ in range(3)]
        tf = [timed(lambda: encode_gkp_fast(delta, ncut, 8), torch.device("cpu"), 1)
              for _ in range(3)]
        rows[f"encode state delta={delta} N={ncut}"] = dict(
            reference=np.median(tr) * 1e3, fast=np.median(tf) * 1e3,
            speedup=np.median(tr) / np.median(tf))
        psi = encode_gkp(delta, ncut, 8)
        c = encode_gkp_fast(delta, ncut, 8)
        wigner_fft(c, grid, device=device)
        wr = [timed(lambda: qt.wigner(psi, grid, grid, method="clenshaw"), torch.device("cpu"), 1)
              for _ in range(2)]
        wf = [timed(lambda: wigner_fft(c, grid, device=device), device, 1) for _ in range(3)]
        rows[f"encode wigner 201x201 delta={delta} N={ncut}"] = dict(
            reference=np.median(wr) * 1e3, fast=np.median(wf) * 1e3,
            speedup=np.median(wr) / np.median(wf))
    for k in [k for k in rows if k.startswith(("simulate", "encode"))]:
        log(f"[ab] {k:<40} ref {rows[k]['reference']:9.2f} ms  fast {rows[k]['fast']:8.2f} ms  "
            f"x{rows[k]['speedup']:.1f}")
    return rows


def _unfused(fn, d, n, s):
    from decoders import fast as fastmod
    fastmod.FUSED = False
    try:
        return fn(d, n, s)
    finally:
        fastmod.FUSED = True


def _run_eager(dec, d, n):
    from decoders.fast import run as fast_run
    orig = dec.decide
    dec.decide = lambda s, graph=False: orig(s, graph=False)
    try:
        return fast_run(dec, d, n)
    finally:
        dec.decide = orig


def main_ab():
    device = torch_device()
    run_all.DEVICE = device
    with RunLog("profile_ab") as rl:
        rows = ab_compare(rl, device)
        rl.note(rows=rows)
    (OUT / "profile_ab.json").write_text(json.dumps(rows, indent=1, default=float))
    log(f"profile_pipeline: wrote {OUT / 'profile_ab.json'}")


if __name__ == "__main__":
    main()
