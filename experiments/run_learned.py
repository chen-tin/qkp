"""E13: learned-decoder study (what limits the LSTM where errors are rare?).

Evaluates every training/train_variants.py variant, plus the extra seeds of the
matched and switching LSTMs, against matched Bayes on the same machinery as
E1-E8 (run_all.evaluate_point: shared test noise, cached, resumable).

Points (50 rounds, stationary unless noted):
    nominal      sigma_q = 0.2L, sigma_m = 0.05L, rho = 0.9  (rare errors: the LSTM is
                 3-4x worse than Bayes here)
    rho0.5       E2 level A, rho = 0.5                        (matched LSTM within 2%)
    sq0.26       E3, sigma_q = 0.26L                           (matched LSTM within 17%)
    switching    E6 regime-switching noise                    (seed spread of LSTM-switching)
Reference: matched Bayes (regime-aware Bayes at the switching point).

    python -m training.train_variants        # train first
    python -m experiments.run_learned        # evaluate (deferred if checkpoints missing)
    python -m experiments.run_learned --analyze
Nothing here has been run at full size.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import numpy as np

from experiments import run_all
from experiments.common import NOMINAL, get_budget, get_mode, log, out_dir, torch_device
from experiments.run_all import SWITCHING, ar1, bayes, evaluate_point, lstm_family, simple
from simulator.noise import L
from training.train_variants import seed_extensions, variant_grid

EXP = "E13"
POINTS = {
    "nominal": dict(NOMINAL),
    "rho0.5": dict(sigma_q=0.20 * L, sigma_m=0.05 * L, rho=0.5),
    "sq0.26": dict(sigma_q=0.26 * L, sigma_m=0.05 * L, rho=0.9),
}
CAP = 60_000               # test trials per point (adaptive growth stops earlier if resolved)


def variant_decoder(name, label=None):
    """run_all-style decoder dict for a train_variants checkpoint (any arch)."""
    path = out_dir("checkpoints") / f"{name}.pt"

    import os
    fast = os.environ.get("QKP_FAST", "1") != "0"

    def make():
        from decoders.learned import FastSeq, SeqDecoder, load_model
        model, ck = load_model(path, run_all.DEVICE)
        cls = FastSeq if fast else SeqDecoder
        return cls(model, ck["chosen_head"], name=label or name, device=run_all.DEVICE)

    return dict(name=label or name, kind="lstm",
                spec=dict(model=name, impl="fast" if fast else "reference"), checkpoint=path,
                make=make)


def learned_set(budget):
    decs = [variant_decoder(v["name"]) for v in variant_grid()]
    decs += lstm_family("lstm_nominal", "LSTM-nominal", budget.lstm_seeds)
    return decs


def seed_set(prefix, label):
    from experiments.run_all import lstm
    names = [f"{prefix}_s0"] + [n for n, _, _ in seed_extensions() if n.startswith(prefix)]
    return [lstm(n, f"{label} s{n.rsplit('_s', 1)[1]}") for n in names]


def e13():
    budget = get_budget()
    for i, (pid, truth) in enumerate(POINTS.items()):
        decs = [bayes(**truth, name="Bayes (matched)"), simple("NL"), *learned_set(budget)]
        if pid == "rho0.5":
            decs += seed_set("lstm_matched_rho0.5", "LSTM-matched")
        if pid == "sq0.26":
            decs += seed_set("lstm_matched_sq0.26", "LSTM-matched")
        evaluate_point(EXP, i, pid, ar1(**truth), decs, decs[0], cap=CAP)
    sw = SWITCHING
    noise_spec = dict(kind="switching", **{k: float(v) for k, v in sw.items()})
    decs = [bayes(sw["sigma_q"], sw["sigma_m"], sw["rho_lo"], "Bayes (regime-aware)",
                  rho_hi=sw["rho_hi"], p_switch=sw["p_switch"]),
            simple("NL"), *seed_set("lstm_switching", "LSTM-switching"),
            *learned_set(budget)]
    evaluate_point(EXP, len(POINTS), "switching", noise_spec, decs, decs[0], cap=CAP)


# ---------------------------------------------------------------------------
def analyze():
    """Table per point: P_L(50), time-averaged error, ratio to the reference,
    parameter count and training seconds of every learned decoder."""
    import torch
    from experiments.analyze import load_point, write_csv
    rows = []
    for pid in list(POINTS) + ["switching"]:
        data = load_point(EXP, pid)
        if not data:
            log(f"{EXP}/{pid}: no results yet")
            continue
        ref_name = "Bayes (regime-aware)" if pid == "switching" else "Bayes (matched)"
        ref = data.get(ref_name, {}).get("errors")
        for name, d in data.items():
            rec = d["errors"]
            per_trial = rec.mean(axis=1)
            row = dict(point=pid, decoder=name, trials=rec.shape[0],
                       ler50=float(rec[:, -1].mean()), mean_ler=float(per_trial.mean()),
                       mean_ler_se=float(per_trial.std(ddof=1) / np.sqrt(len(per_trial))))
            if ref is not None:
                r = ref.mean()
                row["mean_ler_vs_reference"] = row["mean_ler"] / r if r > 0 else None
                diff = per_trial - ref.mean(axis=1)
                row["diff_vs_reference"] = float(diff.mean())
                row["diff_vs_reference_se"] = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            ck = out_dir("checkpoints") / f"{name}.pt"
            if ck.exists():
                c = torch.load(ck, map_location="cpu", weights_only=False)
                row["arch"] = c.get("arch", "lstm")
                row["params"] = c.get("n_params") or sum(t.numel() for t in c["state_dict"].values())
                row["train_seconds"] = c.get("train_seconds")
                row["stop"] = c.get("stop_reason")
            rows.append(row)
    if rows:
        write_csv(out_dir() / "E13_summary.csv", rows)
        (out_dir() / "E13_summary.json").write_text(json.dumps(rows, indent=1, default=float))
        log(f"E13: wrote {out_dir() / 'E13_summary.csv'} ({len(rows)} rows)")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        analyze()
        return
    from experiments.runlog import RunLog
    run_all.DEVICE = torch_device()
    log(f"run_learned: mode={get_mode()} device={run_all.DEVICE}")
    with RunLog("run_learned") as rl:
        with rl.phase(EXP):
            e13()
        rl.note(failed_decoders=list(run_all.FAILURES))
    log(f"run_learned finished; failed decoders/points: {run_all.FAILURES or 'none'}")


if __name__ == "__main__":
    main()
