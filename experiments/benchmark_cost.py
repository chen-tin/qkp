"""Cost benchmark: latency, throughput, memory, scaling, matched-accuracy
comparison and end-to-end time-to-solution, for every decoder.

Follows the benchmarking recipe in the QUARK / JUQCS line of work: measure
the full cost and its components on fixed workloads, compare methods at the
accuracy they reach, and report end-to-end time to solution (here: LSTM
training + inference vs. filters that need no training). Run on an idle
machine; the script records other GPU processes and the CPU load average
so contaminated timings can be spotted.

Run from the repo root:
    MPLBACKEND=Agg python -m experiments.benchmark_cost
Output: results/benchmarks/cost.json, cost_*.csv, cost_pareto.png
"""

import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from decoders.bayesian.adaptive import OracleBayesDecoder, RegimeBankDecoder, WindowedBayesDecoder
from decoders.bayesian.grid_filter import BayesGridDecoder, MultiModelBayesDecoder, block_size
from decoders.LSTM.model import LSTMDecoder, load_model
from experiments import run_all
from experiments.common import (
    NOMINAL, flips_per_trial, get_budget, get_mode, log, make_rng, out_dir, run_blocked,
    torch_device,
)
from experiments.run_nonstationary import ADAPTIVE_GRID, IMM_HAZARD, POINTS, WINDOWS
from simulator import nonstationary as ns
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice

OUT = out_dir() / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)
QUICK = get_mode() == "quick"
CKPT = out_dir("checkpoints")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def factories(device):
    """name -> (make(), kind). kind: 'cpu' (numpy), 'gpu' (torch), 'oracle'."""
    nom = NOMINAL
    f = {
        "NL": (lambda: NearestLattice(), "cpu"),
        "Bayes (matched) n_x=64": (lambda: BayesGridDecoder(**nom, n_x=64, device=device), "gpu"),
        "Bayes (matched) n_x=96": (lambda: BayesGridDecoder(**nom, n_x=96, device=device), "gpu"),
        "Bayes (matched) n_x=128": (lambda: BayesGridDecoder(**nom, n_x=128, device=device), "gpu"),
        "Bayes (matched) n_x=192": (lambda: BayesGridDecoder(**nom, n_x=192, device=device), "gpu"),
        "Bayes-broad": (lambda: MultiModelBayesDecoder(run_all.BROAD_COMPONENTS, n_x=96,
                                                       device=device), "gpu"),
        "Bayes (oracle)": (lambda: OracleBayesDecoder(n_x=96, device=device), "oracle"),
        "Bayes-IMM": (lambda: RegimeBankDecoder(ADAPTIVE_GRID, hazard=IMM_HAZARD, n_x=96,
                                                device=device), "gpu"),
    }
    for w in WINDOWS:
        f[f"Bayes-window {w}"] = (lambda w=w: WindowedBayesDecoder(ADAPTIVE_GRID, window=w,
                                                                   n_x=96, device=device), "gpu")
    # LSTMs: the 50-round nominal model and, once trained, the long-sequence drift model
    for stem, label in (("lstm_nominal_s0", "LSTM-nominal"), ("lstm_drift_s0", "LSTM-drift")):
        path = CKPT / f"{stem}.pt"
        if path.exists():
            def make_lstm(path=path, label=label):
                model, ck = load_model(path, device)
                return LSTMDecoder(model, head=ck["chosen_head"], name=label, device=device)
            f[label] = (make_lstm, "gpu")
    return f


def noise(batch, rounds, seed):
    """Stationary nominal noise plus its constant schedule (for the oracle)."""
    d, n, sch = ns.sample(ns.spec("stationary"), make_rng("dev", "benchmark_cost", seed),
                          batch, rounds)
    return d, n, sch


def timed_run(make, kind, batch, rounds, device, seed=0, repeats=1):
    d, n, sch = noise(batch, rounds, seed)
    dec = make()
    # warm-up on a small slice (kernel compilation, allocator)
    run_blocked(dec, d[:min(batch, 16)], n[:min(batch, 16)], batch,
                schedule=sch.slice(0, min(batch, 16)))
    best = math.inf
    peak = None
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sync(device)
        t0 = time.perf_counter()
        rec = run_blocked(dec, d, n, batch, schedule=sch)
        sync(device)
        best = min(best, time.perf_counter() - t0)
        if device.type == "cuda" and kind != "cpu":
            peak = torch.cuda.max_memory_allocated() / 1e9
    return best, peak, rec


def scaling(device):
    """Latency (batch 1) and throughput vs batch and rounds."""
    rows = []
    batches = (1, 64, 512) if QUICK else (1, 64, 512, 4096)
    for name, (make, kind) in factories(device).items():
        for batch in batches:
            rounds = 50
            if kind != "cpu" and batch >= 4096 and name.startswith(("Bayes-window", "Bayes-IMM")):
                batch = 1024           # 12-model decoders: keep memory bounded
            try:
                dt, peak, _ = timed_run(make, kind, batch, rounds, device,
                                        repeats=3 if batch == 1 else 1)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append(dict(decoder=name, batch=batch, rounds=rounds, oom=True))
                continue
            rows.append(dict(decoder=name, device=device.type if kind != "cpu" else "cpu",
                             batch=batch, rounds=rounds, seconds=dt,
                             ms_per_round=dt / rounds * 1e3,
                             us_per_trial_round=dt / (batch * rounds) * 1e6,
                             trial_rounds_per_s=batch * rounds / dt, peak_gpu_gb=peak))
            log(f"cost: {name:<24} batch={batch:<5} {dt / rounds * 1e3:9.3f} ms/round "
                f"{batch * rounds / dt:12.0f} trial-rounds/s")
    # rounds scaling at fixed batch: is cost per trial-round constant?
    for name in ("Bayes (matched) n_x=96", "Bayes (oracle)", "Bayes-IMM", "Bayes-window 200",
                 "LSTM-nominal", "NL"):
        if name not in factories(device):
            continue
        make, kind = factories(device)[name]
        for rounds in ((50, 200) if QUICK else (50, 200, 1000)):
            batch = 256
            dt, peak, _ = timed_run(make, kind, batch, rounds, device)
            rows.append(dict(decoder=name, batch=batch, rounds=rounds, seconds=dt,
                             us_per_trial_round=dt / (batch * rounds) * 1e6,
                             trial_rounds_per_s=batch * rounds / dt, peak_gpu_gb=peak,
                             sweep="rounds"))
            log(f"cost: {name:<24} rounds={rounds:<5} {dt / (batch * rounds) * 1e6:8.2f} "
                f"us/trial-round")
    return rows


def components(device):
    """Noise generation vs simulation loop vs decoding, nominal, 4096 x 50."""
    batch, rounds = (512, 50) if QUICK else (4096, 50)
    t0 = time.perf_counter()
    d, n = NoiseModel(**NOMINAL).sample(make_rng("dev", "benchmark_cost_c", 0), batch, rounds)
    t_noise = time.perf_counter() - t0
    t0 = time.perf_counter()
    run_blocked(NearestLattice(), d, n, batch)            # simulator loop + trivial decoder
    t_loop = time.perf_counter() - t0
    dec = BayesGridDecoder(**NOMINAL, n_x=96, device=device)
    run_blocked(dec, d[:16], n[:16], batch)
    sync(device)
    t0 = time.perf_counter()
    run_blocked(dec, d, n, batch)
    sync(device)
    t_bayes = time.perf_counter() - t0
    return dict(batch=batch, rounds=rounds, noise_s=t_noise, sim_loop_with_NL_s=t_loop,
                bayes_total_s=t_bayes, bayes_decoding_share=1 - t_loop / t_bayes)


def matched_accuracy(device):
    """Accuracy and cost from the same run, nominal point (Pareto comparison)."""
    batch, rounds = (1000, 50) if QUICK else (8000, 50)
    rows = []
    for name, (make, kind) in factories(device).items():
        b = batch if not name.startswith(("Bayes-window", "Bayes-IMM")) else min(batch, 2000)
        dt, peak, rec = timed_run(make, kind, b, rounds, device, seed=11)
        per = flips_per_trial(rec) / rounds
        rows.append(dict(decoder=name, trials=b, rounds=rounds,
                         flip_rate=float(per.mean()),
                         flip_rate_se=float(per.std(ddof=1) / math.sqrt(b)),
                         mean_ler=float(rec.mean()),
                         us_per_trial_round=dt / (b * rounds) * 1e6, peak_gpu_gb=peak))
        log(f"pareto: {name:<24} flip {per.mean():.2e} at {dt / (b * rounds) * 1e6:.2f} us/tr")
    return rows


def end_to_end(scale_rows, pareto_rows):
    """Time to decode a workload, including training where a decoder needs it."""
    thr = {}
    for r in scale_rows:
        if r.get("sweep") is None and not r.get("oom"):
            thr[r["decoder"]] = max(thr.get(r["decoder"], 0), r["trial_rounds_per_s"])
    train = {}
    for p in sorted(CKPT.glob("lstm_*.pt")):
        meta = json.loads(p.with_name(p.name + ".meta.json").read_text())
        ck = torch.load(p, map_location="cpu", weights_only=False)
        train[p.stem] = dict(train_s=meta.get("train_seconds"), data_s=ck.get("data_seconds"))
    lstm_train = train.get("lstm_nominal_s0", {})
    train_s = (lstm_train.get("train_s") or 0) + (lstm_train.get("data_s") or 0)
    workloads = {"1e6 trial-rounds": 1e6, "E1-size (100k x 50)": 5e6, "1e9 trial-rounds": 1e9}
    out = []
    for name, rate in thr.items():
        row = dict(decoder=name, best_trial_rounds_per_s=rate,
                   training_s=train_s if name.startswith("LSTM") else 0.0)
        for wname, size in workloads.items():
            row[wname + " (s)"] = row["training_s"] + size / rate
        out.append(row)
    # break-even workload: when does LSTM training pay off vs matched Bayes?
    if "LSTM-nominal" in thr and "Bayes (matched) n_x=96" in thr:
        a, b = thr["LSTM-nominal"], thr["Bayes (matched) n_x=96"]
        breakeven = train_s / (1 / b - 1 / a) if a > b else None
    else:
        breakeven = None
    return dict(rows=out, lstm_training_s=train_s, training=train,
                breakeven_trial_rounds_lstm_vs_bayes=breakeven)


def e9_e11_projection(scale_rows):
    """Projected wall-clock for the full E9-E11 run from measured throughputs."""
    from experiments.common import BUDGETS
    full = BUDGETS["full"]
    tr = full.ns_max_test_trials * full.ns_rounds          # trial-rounds per decoder per point
    rate = {}
    for r in scale_rows:
        if r.get("sweep") == "rounds" or r.get("oom"):
            continue
        rate[r["decoder"]] = max(rate.get(r["decoder"], 0), r["trial_rounds_per_s"])
    per_point = {
        "Bayes (oracle)": rate.get("Bayes (oracle)"),
        "NL": rate.get("NL"),
        "Bayes (nominal)": rate.get("Bayes (matched) n_x=96"),
        "Bayes-broad": rate.get("Bayes-broad"),
        "Bayes-IMM": rate.get("Bayes-IMM"),
        **{f"Bayes-window {w}": rate.get(f"Bayes-window {w}") for w in WINDOWS},
        "LSTM x 9 (3 families x 3 seeds)": rate.get("LSTM-nominal") / 9
        if rate.get("LSTM-nominal") else None,
    }
    secs = {k: tr / v for k, v in per_point.items() if v}
    n_points = sum(len(v) for v in POINTS.values())
    total = sum(secs.values()) * n_points
    return dict(trial_rounds_per_decoder_per_point=tr, seconds_per_point=secs,
                points=n_points, total_hours=total / 3600)


def main():
    device = torch_device()
    run_all.DEVICE = device
    env = dict(mode=get_mode(), device=str(device),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               other_gpu_processes=run_all._other_gpu_processes(),
               load_avg=os.getloadavg(), torch=torch.__version__)
    log(f"benchmark_cost: {env}")
    scale = scaling(device)
    comp = components(device)
    pareto = matched_accuracy(device)
    e2e = end_to_end(scale, pareto)
    proj = e9_e11_projection(scale)
    env["load_avg_end"] = os.getloadavg()
    out = dict(env=env, scaling=scale, components=comp, matched_accuracy=pareto,
               end_to_end=e2e, e9_e11_projection=proj)
    (OUT / "cost.json").write_text(json.dumps(out, indent=1, default=float))
    for key, rows in (("scaling", scale), ("pareto", pareto)):
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "decoder", k))
        with open(OUT / f"cost_{key}.csv", "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in pareto:
        if r["flip_rate"] <= 0:
            continue
        ax.errorbar(r["us_per_trial_round"], r["flip_rate"], yerr=2 * r["flip_rate_se"], fmt="o")
        ax.annotate(r["decoder"], (r["us_per_trial_round"], r["flip_rate"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cost: microseconds per trial-round (batched, this machine)")
    ax.set_ylabel("logical flips per round (nominal point)")
    ax.set_title("Accuracy vs cost at the nominal point (lower-left is better)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "cost_pareto.png", dpi=150)
    plt.close(fig)
    log(f"benchmark_cost: wrote {OUT}; projected full E9-E11 = {proj['total_hours']:.1f} h")


if __name__ == "__main__":
    main()
