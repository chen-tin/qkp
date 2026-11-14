"""Run every experiment (E0-E8) and cache raw results.

Each (experiment, point, decoder) is cached under results/cache/ with a
sidecar, so the script can be stopped and resumed; LSTM decoders whose
checkpoints are not trained yet are deferred and picked up on the next run.
Analysis and figures are produced by experiments/analyze.py from the cache.

Run from the repo root:
    MPLBACKEND=Agg python -m experiments.run_all           # full run
    QKP_MODE=quick MPLBACKEND=Agg python -m experiments.run_all
    python -m experiments.run_all --only E1 E2
"""

import argparse
import json
import time
import traceback

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from decoders.bayesian.grid_filter import BayesGridDecoder, MultiModelBayesDecoder, block_size
from decoders.LSTM.model import LSTMDecoder, load_model
from experiments.common import (
    BROAD_BOX, NOMINAL, ROOT, ROUNDS, SEED_ROOT, _stable_int, eval_split, get_budget,
    get_mode, is_complete, log, make_rng, out_dir, run_blocked, save_json, save_npz,
    sidecar_path, torch_device,
)
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice, NoCorrection
from simulator.switching import SwitchingNoiseModel

DEVICE = None
FAILURES = []


def _code_hash():
    """Hash of the code that determines decoder outputs; part of every cache key."""
    import hashlib
    h = hashlib.sha1()
    for rel in ("simulator/sim.py", "simulator/noise.py", "simulator/switching.py",
                "decoders/base.py", "decoders/bayesian/grid_filter.py",
                "decoders/LSTM/model.py"):
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()[:12]


CODE_HASH = _code_hash()


def point_seed(exp, point_id):
    """Noise-stream index derived from the point's identity, not its position."""
    return _stable_int(f"{exp}/{point_id}") % (2**31)


def checkpoint_identity(dec):
    """Sidecar digest + write time of an LSTM checkpoint (changes on retrain)."""
    meta = json.loads(sidecar_path(dec["checkpoint"]).read_text())
    return dict(digest=meta["digest"], written=meta["written"],
                chosen_head=meta.get("chosen_head"))

# Bayes-broad: model average over a coarse grid covering the broad box
BROAD_COMPONENTS = [
    dict(sigma_q=sq * L, sigma_m=0.05 * L, rho=rho)
    for rho in (0.0, 0.6, 0.85, 0.95)
    for sq in (0.15, 0.25)
]

SWITCHING = dict(sigma_q=0.20 * L, sigma_m=0.05 * L, rho_lo=0.3, rho_hi=0.97, p_switch=0.05)


# decoder factories
def n_x_for(sigma_m):
    """Grid points over 2L: finer grid when the assumed sigma_m is small."""
    return 192 if sigma_m < 0.035 * L else 96


def bayes(sigma_q, sigma_m, rho, name, **kw):
    return dict(
        name=name, kind="bayes",
        spec=dict(sigma_q=sigma_q, sigma_m=sigma_m, rho=rho, **kw),
        make=lambda: BayesGridDecoder(
            sigma_q, sigma_m, rho, name=name, n_x=n_x_for(sigma_m), device=DEVICE, **kw
        ),
    )


def bayes_broad():
    return dict(
        name="Bayes-broad", kind="bayes", spec=dict(components=BROAD_COMPONENTS),
        make=lambda: MultiModelBayesDecoder(BROAD_COMPONENTS, n_x=96, device=DEVICE),
    )


def simple(name):
    cls = {"NL": NearestLattice, "No correction": NoCorrection}[name]
    return dict(name=name, kind="simple", spec={}, make=lambda: cls())


def lstm(model_name, label, head=None):
    path = out_dir("checkpoints") / f"{model_name}.pt"

    def make():
        model, ckpt = load_model(path, DEVICE)
        return LSTMDecoder(model, head=head or ckpt["chosen_head"], name=label, device=DEVICE)

    return dict(name=label, kind="lstm", spec=dict(model=model_name, head=head),
                checkpoint=path, make=make)


def lstm_ready(dec):
    path = dec["checkpoint"]
    meta = path.with_name(path.name + ".meta.json")
    if not (path.exists() and meta.exists()):
        return False
    return json.loads(meta.read_text()).get("mode") == get_mode()


def lstm_family(prefix, label, seeds):
    return [lstm(f"{prefix}_s{s}", f"{label} s{s}") for s in range(seeds)]


# evaluation of one point with a shared, adaptively sized test set
def point_dir(exp, point_id):
    return out_dir() / "cache" / exp / point_id


def noise_chunk_with_schedule(noise_spec, exp, point_index, chunk, trials, rounds):
    """(displacement, measurement, schedule) of one test chunk.

    schedule is None for the stationary kinds; for kind "schedule" it is the
    simulator.nonstationary.Schedule the noise was drawn from (the oracle
    decoder reads it; every other decoder sees only syndromes).
    """
    rng = make_rng(eval_split(), exp, point_index, chunk)
    kind = noise_spec["kind"]
    if kind == "ar1":
        model = NoiseModel(noise_spec["sigma_q"], noise_spec["sigma_m"], noise_spec["rho"])
        return (*model.sample(rng, trials, rounds), None)
    if kind == "switching":
        model = SwitchingNoiseModel(*(noise_spec[k] for k in
                                      ("sigma_q", "sigma_m", "rho_lo", "rho_hi", "p_switch")))
        return (*model.sample(rng, trials, rounds), None)
    if kind == "schedule":
        from simulator import nonstationary
        return nonstationary.sample(noise_spec["spec"], rng, trials, rounds)
    raise ValueError(kind)


def noise_chunk(noise_spec, exp, point_index, chunk, trials, rounds):
    d, n, _ = noise_chunk_with_schedule(noise_spec, exp, point_index, chunk, trials, rounds)
    return d, n


def run_decoder(dec, noise_spec, exp, point_index, n_chunks, chunk_trials, rounds, budget):
    """Logical-error record (trials, rounds) of one decoder on the point's test set."""
    decoder = dec["make"]()
    block = block_size(decoder, budget.filter_block_bytes) if dec["kind"] == "bayes" else 50_000
    parts = []
    t0 = time.perf_counter()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    for c in range(n_chunks):
        d, n, sched = noise_chunk_with_schedule(noise_spec, exp, point_index, c,
                                                chunk_trials, rounds)
        parts.append(run_blocked(decoder, d, n, block, schedule=sched))
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    resets = getattr(decoder, "resets", 0)
    return np.concatenate(parts), seconds, resets


def evaluate_point(exp, point_index, point_id, noise_spec, decoders, reference,
                   cap, rounds=ROUNDS, min_events=None, chunk=None):
    """Evaluate all decoders on one point.

    The number of test chunks is set once, by growing the test set until the
    reference decoder (normally matched Bayes) has min_events logical errors
    at the last round, or the cap is reached. Every decoder then sees exactly
    the same noise.
    """
    budget = get_budget()
    chunk = chunk or budget.chunk_trials
    cap = min(cap, budget.max_test_trials) if get_mode() == "quick" else cap
    if min_events is None:
        min_events = budget.min_events
    point_index = point_seed(exp, point_id)
    seed_info = dict(root=SEED_ROOT, split=eval_split(), point=point_index)
    pdir = point_dir(exp, point_id)
    pdir.mkdir(parents=True, exist_ok=True)
    n_file = pdir / "n_chunks.json"
    n_config = dict(exp=exp, point=point_id, noise=noise_spec, chunk=chunk, cap=cap,
                    rounds=rounds, reference=reference["name"], min_events=min_events,
                    ref_spec=reference["spec"], seed=seed_info, code=CODE_HASH)

    def cache_path(dec):
        safe = dec["name"].replace(" ", "_").replace("/", "-")
        return pdir / f"{safe}.npz"

    def dec_config(dec, n_chunks):
        config = dict(exp=exp, point=point_id, noise=noise_spec, chunk=chunk,
                      n_chunks=n_chunks, rounds=rounds, decoder=dec["name"],
                      spec={k: v for k, v in dec["spec"].items()},
                      seed=seed_info, code=CODE_HASH)
        if dec["kind"] == "lstm":
            config["checkpoint"] = checkpoint_identity(dec)
        return config

    # decide the test-set size with the reference decoder
    if is_complete(n_file, n_config):
        n_chunks = json.loads(n_file.read_text())["n_chunks"]
    else:
        try:
            n_chunks = _size_test_set(exp, point_id, noise_spec, reference, cap, chunk,
                                      rounds, min_events, point_index, budget, n_file,
                                      n_config, cache_path, dec_config)
        except Exception:
            FAILURES.append(f"{exp}/{point_id}/{reference['name']} (reference)")
            log(f"{exp}/{point_id}: reference FAILED, skipping point\n{traceback.format_exc()}")
            return
    _run_decoders(exp, point_id, noise_spec, decoders, n_chunks, chunk, rounds,
                  point_index, budget, cache_path, dec_config)


def _size_test_set(exp, point_id, noise_spec, reference, cap, chunk, rounds, min_events,
                   point_index, budget, n_file, n_config, cache_path, dec_config):
    """Grow the test set chunk by chunk with the reference decoder."""
    parts, seconds, resets = [], 0.0, 0
    n_chunks = 0
    decoder = reference["make"]()
    block = (block_size(decoder, budget.filter_block_bytes)
             if reference["kind"] == "bayes" else 50_000)
    while True:
        t0 = time.perf_counter()
        d, n, sched = noise_chunk_with_schedule(noise_spec, exp, point_index, n_chunks,
                                                chunk, rounds)
        parts.append(run_blocked(decoder, d, n, block, schedule=sched))
        seconds += time.perf_counter() - t0
        n_chunks += 1
        events = int(np.concatenate(parts)[:, -1].sum())
        if events >= min_events or (n_chunks + 1) * chunk > cap:
            break
    record = np.concatenate(parts)
    resets = getattr(decoder, "resets", 0)
    save_npz(cache_path(reference), dec_config(reference, n_chunks),
             extra=dict(seconds=seconds, resets=resets, trials=record.shape[0]),
             bits=np.packbits(record, axis=1), trials=record.shape[0])
    save_json(n_file, n_config, dict(n_chunks=n_chunks))
    log(f"{exp}/{point_id}: {reference['name']} sets N={n_chunks * chunk} "
        f"({events} final-round errors, {seconds:.1f}s)")
    return n_chunks


def _run_decoders(exp, point_id, noise_spec, decoders, n_chunks, chunk, rounds,
                  point_index, budget, cache_path, dec_config):
    for dec in decoders:
        if dec["kind"] == "lstm" and not lstm_ready(dec):
            log(f"{exp}/{point_id}: {dec['name']} deferred (checkpoint not ready)")
            continue
        path = cache_path(dec)
        config = dec_config(dec, n_chunks)
        if is_complete(path, config):
            continue
        try:
            record, seconds, resets = run_decoder(
                dec, noise_spec, exp, point_index, n_chunks, chunk, rounds, budget
            )
        except Exception:
            FAILURES.append(f"{exp}/{point_id}/{dec['name']}")
            log(f"{exp}/{point_id}: {dec['name']} FAILED\n{traceback.format_exc()}")
            continue
        save_npz(path, config, extra=dict(seconds=seconds, resets=resets,
                                          trials=record.shape[0]),
                 bits=np.packbits(record, axis=1), trials=record.shape[0])
        log(f"{exp}/{point_id}: {dec['name']:<22} LER50={record[:, -1].mean():.5f} "
            f"mean={record.mean():.5f} ({seconds:.1f}s, resets={resets})")


def ar1(sigma_q, sigma_m, rho):
    return dict(kind="ar1", sigma_q=float(sigma_q), sigma_m=float(sigma_m), rho=float(rho))


def standard_set(truth, budget, extra=()):
    nom = NOMINAL
    return [
        simple("NL"),
        bayes(truth["sigma_q"], truth["sigma_m"], truth["rho"], "Bayes (matched)"),
        bayes(nom["sigma_q"], nom["sigma_m"], nom["rho"], "Bayes (nominal)"),
        bayes_broad(),
        *lstm_family("lstm_nominal", "LSTM-nominal", budget.lstm_seeds),
        *lstm_family("lstm_broad", "LSTM-broad", budget.lstm_seeds),
        *extra,
    ]


# experiments
def e1():
    budget = get_budget()
    truth = NOMINAL
    decs = [
        simple("No correction"),
        simple("NL"),
        bayes(**truth, name="Bayes (matched)"),
        bayes(**truth, name="Bayes (per-round)", per_round=True),
        bayes_broad(),
        *lstm_family("lstm_nominal", "LSTM-nominal", budget.lstm_seeds),
        lstm("lstm_nominal_s0", "LSTM-nominal s0 (k0 head)", head="k0"),
        *lstm_family("lstm_broad", "LSTM-broad", budget.lstm_seeds),
    ]
    evaluate_point("E1", 0, "nominal", ar1(**truth), decs, decs[2], cap=100_000)


E2_RHOS = (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995)
E2_LEVELS = {"A": (0.20, 0.05), "B": (0.30, 0.08)}
E3_SQ = (0.08, 0.10, 0.14, 0.18, 0.20, 0.22, 0.26, 0.30, 0.34)
E4_SM = (0.01, 0.02, 0.035, 0.05, 0.07, 0.10, 0.12)


def e2():
    budget = get_budget()
    i = 0
    for level, (sq, sm) in E2_LEVELS.items():
        for rho in E2_RHOS:
            truth = dict(sigma_q=sq * L, sigma_m=sm * L, rho=rho)
            extra = []
            if level == "A" and rho == 0.5:
                extra = [lstm("lstm_matched_rho0.5_s0", "LSTM-matched")]
            decs = standard_set(truth, budget, extra)
            evaluate_point("E2", i, f"{level}_rho{rho}", ar1(**truth), decs, decs[1],
                           cap=40_000)
            i += 1


def e3():
    budget = get_budget()
    for i, sq in enumerate(E3_SQ):
        truth = dict(sigma_q=sq * L, sigma_m=0.05 * L, rho=0.9)
        extra = [lstm("lstm_matched_sq0.26_s0", "LSTM-matched")] if sq == 0.26 else []
        decs = standard_set(truth, budget, extra)
        evaluate_point("E3", i, f"sq{sq}", ar1(**truth), decs, decs[1], cap=20_000)


def e4():
    budget = get_budget()
    for i, sm in enumerate(E4_SM):
        truth = dict(sigma_q=0.20 * L, sigma_m=sm * L, rho=0.9)
        decs = standard_set(truth, budget)
        evaluate_point("E4", i, f"sm{sm}", ar1(**truth), decs, decs[1], cap=20_000)


E5_RHOS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
E5_SQ_SCALE = (0.5, 0.75, 1.0, 1.5, 2.0)
E5_SM_SCALE = (0.5, 1.0, 2.0)


def e5():
    budget = get_budget()
    truth = NOMINAL
    decs = [bayes(**truth, name="Bayes (matched)"), simple("NL")]
    for rho in E5_RHOS:
        for k in E5_SQ_SCALE:
            if rho == truth["rho"] and k == 1.0:
                continue
            decs.append(bayes(k * truth["sigma_q"], truth["sigma_m"], rho,
                              name=f"Bayes rho'={rho} sq'x{k}"))
    for k in E5_SM_SCALE:
        if k != 1.0:
            decs.append(bayes(truth["sigma_q"], k * truth["sigma_m"], truth["rho"],
                              name=f"Bayes sm'x{k}"))
    decs += lstm_family("lstm_nominal", "LSTM-nominal", budget.lstm_seeds)
    decs += lstm_family("lstm_broad", "LSTM-broad", budget.lstm_seeds)
    # fixed test-set size: same as E1's resolution would be too costly for 38 filters
    evaluate_point("E5", 0, "nominal", ar1(**truth), decs, decs[0], cap=40_000,
                   min_events=10**9)


def e6():
    budget = get_budget()
    sw = SWITCHING
    noise_spec = dict(kind="switching", **{k: float(v) for k, v in sw.items()})
    decs = [
        simple("NL"),
        bayes(sw["sigma_q"], sw["sigma_m"], sw["rho_lo"], "Bayes (regime-aware)",
              rho_hi=sw["rho_hi"], p_switch=sw["p_switch"]),
        bayes(sw["sigma_q"], sw["sigma_m"], 0.9, "Bayes (rho=0.9)"),
        bayes(sw["sigma_q"], sw["sigma_m"], sw["rho_hi"], "Bayes (rho=0.97)"),
        bayes_broad(),
        *lstm_family("lstm_nominal", "LSTM-nominal", budget.lstm_seeds),
        *lstm_family("lstm_broad", "LSTM-broad", budget.lstm_seeds),
        lstm("lstm_switching_s0", "LSTM-switching"),
    ]
    evaluate_point("E6", 0, "switching", noise_spec, decs, decs[1], cap=60_000)


def e7():
    """Timing: per-round latency (1 trial) and throughput (large batch)."""
    budget = get_budget()
    from training.train_lstm import model_list
    path = out_dir() / "E7_timing.json"
    models = [lstm(m, m) for m, _, _ in model_list(budget, include_nonstationary=False)]
    if not all(lstm_ready(m) for m in models):
        log("E7: deferred until all LSTMs are trained (timing needs an idle GPU)")
        return
    config = dict(exp="E7", budget_hidden=budget.hidden, mode=get_mode(), code=CODE_HASH,
                  checkpoints={m["name"]: checkpoint_identity(m) for m in models})
    if is_complete(path, config):
        return
    other_gpu = _other_gpu_processes()
    truth = NOMINAL
    model = NoiseModel(**truth)
    rows = []
    decs = [
        simple("NL"),
        bayes(**truth, name="Bayes (matched)"),
        bayes_broad(),
        lstm("lstm_nominal_s0", "LSTM-nominal s0"),
    ]
    global DEVICE
    for device in ([torch.device("cuda")] if torch.cuda.is_available() else []) + [
            torch.device("cpu")]:
        DEVICE = device
        for dec in decs:
            if dec["kind"] == "lstm" and not lstm_ready(dec):
                log("E7: LSTM checkpoint not ready, deferring timing")
                DEVICE = torch_device()
                return
            if dec["kind"] == "simple" and device.type == "cuda":
                continue
            for batch, rounds in ((1, 50), (4096 if device.type == "cuda" else 512, 20)):
                d, n = model.sample(make_rng("dev", "E7", batch), batch, rounds)
                decoder = dec["make"]()
                run_blocked(decoder, d[:min(batch, 64)], n[:min(batch, 64)], batch)  # warm-up
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                run_blocked(decoder, d, n, batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None
                rows.append(dict(decoder=dec["name"], device=device.type, batch=batch,
                                 rounds=rounds, seconds=dt,
                                 latency_ms_per_round=dt / rounds * 1e3,
                                 throughput_trial_rounds_per_s=batch * rounds / dt,
                                 peak_gpu_gb=peak))
                log(f"E7: {dec['name']:<18} {device.type} batch={batch:<5} "
                    f"{dt / rounds * 1e3:8.2f} ms/round  "
                    f"{batch * rounds / dt:12.0f} trial-rounds/s")
    DEVICE = torch_device()
    # training times from the checkpoints
    training = {}
    for p in sorted(out_dir("checkpoints").glob("*.pt")):
        meta = p.with_name(p.name + ".meta.json")
        if meta.exists():
            info = json.loads(meta.read_text())
            training[p.stem] = dict(seconds=info.get("train_seconds"),
                                    stop=info.get("stop_reason"),
                                    head=info.get("chosen_head"))
    save_json(path, config, dict(rows=rows, training=training, other_gpu_processes=other_gpu,
                                 gpu=torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else None))


def _other_gpu_processes():
    """Other compute processes on the GPU (timing is unreliable if any)."""
    import os
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True)
    except Exception:
        return []
    procs = [l.strip() for l in out.splitlines() if l.strip()]
    return [p for p in procs if not p.startswith(f"{os.getpid()},")]


def e8():
    """Diagnostics on recorded subsets (per-round decisions and confidences)."""
    budget = get_budget()
    n = budget.diag_trials
    points = [
        ("nominal", NOMINAL),
        ("rho0", dict(NOMINAL, rho=0.0)),
        ("rho0.5", dict(NOMINAL, rho=0.5)),
        ("rho0.99", dict(NOMINAL, rho=0.99)),
        ("sq0.10", dict(NOMINAL, sigma_q=0.10 * L)),
        ("sq0.30", dict(NOMINAL, sigma_q=0.30 * L)),
    ]
    from simulator.sim import Simulator
    for i, (pid, truth) in enumerate(points):
        path = out_dir() / "cache" / "E8" / f"{pid}.npz"
        decs = [
            bayes(**truth, name="Bayes (matched)"),
            bayes(**truth, name="Bayes (per-round)", per_round=True),
            bayes(NOMINAL["sigma_q"], NOMINAL["sigma_m"], 0.5, "Bayes rho'=0.5"),
            bayes(NOMINAL["sigma_q"], NOMINAL["sigma_m"], 0.99, "Bayes rho'=0.99"),
            lstm("lstm_nominal_s0", "LSTM-nominal s0"),
            lstm("lstm_nominal_s0", "LSTM-nominal s0 (k0 head)", head="k0"),
            lstm("lstm_broad_s0", "LSTM-broad s0"),
        ]
        if not all(lstm_ready(d) for d in decs if d["kind"] == "lstm"):
            log(f"E8/{pid}: deferred (checkpoints not ready)")
            continue
        config = dict(exp="E8", point=pid, truth=truth, n=n, index=i, split=eval_split(),
                      code=CODE_HASH, decoders=[d["name"] for d in decs],
                      checkpoints=[checkpoint_identity(d) for d in decs if d["kind"] == "lstm"])
        if is_complete(path, config):
            continue
        model = NoiseModel(**truth)
        eps, nu = model.sample(make_rng(eval_split(), "E8", i), n, ROUNDS)
        nl = Simulator(trials=n, rounds=ROUNDS).run(NearestLattice(), eps, nu)
        arrays = dict(y=(nl.lattice_jumps() % 2).astype(np.int8),
                      K=nl.logical_errors.astype(np.int8),
                      syndromes=nl.syndromes.astype(np.float32),
                      eps=eps.astype(np.float32), nu=nu.astype(np.float32))
        names = []
        for k, dec in enumerate(decs):
            decoder = dec["make"]()
            decoder.record = True
            r = Simulator(trials=n, rounds=ROUNDS).run(decoder, eps, nu)
            j, p = decoder.recorded()
            arrays[f"d{k}_j"] = j
            arrays[f"d{k}_p"] = p
            arrays[f"d{k}_err"] = r.logical_errors
            names.append(dec["name"])
        save_npz(path, config, extra=dict(decoders=names), **arrays)
        log(f"E8/{pid}: recorded {len(decs)} decoders on {n} trials")

    # (e) long horizon: 200 rounds with models trained on 50
    pid = "horizon200"
    path = out_dir() / "cache" / "E8" / f"{pid}.npz"
    decs = [simple("NL"), bayes(**NOMINAL, name="Bayes (matched)"),
            lstm("lstm_nominal_s0", "LSTM-nominal s0"), lstm("lstm_broad_s0", "LSTM-broad s0")]
    if not all(lstm_ready(d) for d in decs if d["kind"] == "lstm"):
        log("E8/horizon200: deferred")
        return
    config = dict(exp="E8", point=pid, truth=NOMINAL, n=budget.chunk_trials, rounds=200,
                  split=eval_split(), code=CODE_HASH,
                  checkpoints=[checkpoint_identity(d) for d in decs if d["kind"] == "lstm"])
    if is_complete(path, config):
        return
    model = NoiseModel(**NOMINAL)
    eps, nu = model.sample(make_rng(eval_split(), "E8", 99), budget.chunk_trials, 200)
    arrays, names = {}, []
    for k, dec in enumerate(decs):
        decoder = dec["make"]()
        block = block_size(decoder, budget.filter_block_bytes) if dec["kind"] == "bayes" else 50_000
        arrays[f"d{k}_err"] = np.packbits(run_blocked(decoder, eps, nu, block), axis=1)
        names.append(dec["name"])
    save_npz(path, config, extra=dict(decoders=names, trials=budget.chunk_trials), **arrays)
    log("E8/horizon200: done")


def e0():
    """Simulator validation (existing script) and Bayes grid convergence."""
    import runpy
    path = out_dir() / "E0_grid_convergence.json"
    config = dict(exp="E0", cases="v1")
    if not is_complete(path, config):
        rows = []
        cases = [("nominal", 0.2, 0.05, 0.9), ("sm0.02_rho0.99", 0.2, 0.02, 0.99),
                 ("sq0.30", 0.3, 0.05, 0.9), ("rho0", 0.2, 0.05, 0.0)]
        n = get_budget().chunk_trials
        for i, (name, sq, sm, rho) in enumerate(cases):
            model = NoiseModel(sq * L, sm * L, rho)
            d, nn_ = model.sample(make_rng("dev", "E0", i), n, ROUNDS)
            ref = None
            for nx in (192, 128, 96, 64):
                dec = BayesGridDecoder(model.sigma_q, model.sigma_m, rho, n_x=nx, device=DEVICE)
                t0 = time.perf_counter()
                r = run_blocked(dec, d, nn_, block_size(dec, get_budget().filter_block_bytes))
                if ref is None:
                    ref = r
                rows.append(dict(case=name, n_x=nx, seconds=time.perf_counter() - t0,
                                 ler50=float(r[:, -1].mean()), mean_ler=float(r.mean()),
                                 disagree_vs_192=float((r != ref).mean())))
                log(f"E0 grid: {name} n_x={nx} mean LER {r.mean():.5f} "
                    f"disagree {rows[-1]['disagree_vs_192']:.5f}")
        save_json(path, config, dict(rows=rows, trials=n))
    if get_mode() == "full":
        fig_dir = out_dir() / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        val = out_dir() / "E0_validate_sim.txt"
        if not val.exists():
            import os
            import subprocess
            import sys
            work = out_dir() / "E0_validate_sim"
            work.mkdir(exist_ok=True)
            env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
            out = subprocess.run([sys.executable, "-m", "experiments.validate_sim"],
                                 cwd=work, env=env, capture_output=True, text=True)
            val.write_text(out.stdout + out.stderr)
            log("E0: validate_sim reproduced (see results/E0_validate_sim.txt; "
                "figure in results/E0_validate_sim/docs/figures)")


def _nonstationary(name):
    def run():
        from experiments import run_nonstationary
        getattr(run_nonstationary, name.lower())()
    return run


EXPERIMENTS = dict(E0=e0, E1=e1, E5=e5, E2=e2, E3=e3, E4=e4, E6=e6, E8=e8, E7=e7,
                   E9=_nonstationary("E9"), E10=_nonstationary("E10"),
                   E11=_nonstationary("E11"))


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    assert matplotlib.get_backend().lower() == "agg", matplotlib.get_backend()
    DEVICE = torch_device()
    log(f"run_all: mode={get_mode()} device={DEVICE} budget={get_budget()}")
    failures = []
    for name, fn in EXPERIMENTS.items():
        if args.only and name not in args.only:
            continue
        t0 = time.perf_counter()
        log(f"=== {name} ===")
        try:
            fn()
            log(f"=== {name} done in {time.perf_counter() - t0:.0f}s ===")
        except Exception:
            failures.append(name)
            log(f"=== {name} FAILED ===\n{traceback.format_exc()}")
    log(f"run_all finished; failed experiments: {failures or 'none'}; "
        f"failed decoders/points: {FAILURES or 'none'}")


if __name__ == "__main__":
    main()
