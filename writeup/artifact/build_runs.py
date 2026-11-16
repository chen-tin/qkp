"""Dashboard of all current runs: how decoder performance scales.

Reads the result CSV/JSON files and the per-trial caches (full mode) and writes
writeup/artifact/runs.html. Safe to re-run while jobs are still going: points
that are not cached yet are shown as pending.

    python -m writeup.artifact.build_runs
"""

import csv
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("QKP_MODE", "full")
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "writeup" / "artifact" / "runs.html"

FAMILIES = ["NL", "Bayes (matched)", "Bayes (nominal)", "Bayes-broad", "LSTM-nominal", "LSTM-broad"]


def rows(name):
    path = RES / f"{name}_summary.csv"
    return list(csv.DictReader(open(path))) if path.exists() else []


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sweep(exp, xkey, point_filter=lambda p: True):
    out = {fam: [] for fam in FAMILIES}
    for r in rows(exp):
        if r["decoder"] in out and point_filter(r["point"]):
            out[r["decoder"]].append([f(r[xkey]), f(r["ler50"]), f(r["mean_ler"])])
    return {k: sorted(v) for k, v in out.items() if v}


def mismatch_grid(rows_by_decoder, matched):
    """{(rho', sq'): ler50/matched} from decoder names "Bayes rho'=R sq'xK"."""
    cells = []
    for name, ler in rows_by_decoder.items():
        m = re.match(r"Bayes rho'=([\d.]+) sq'x([\d.]+)", name)
        if m and ler is not None and matched:
            cells.append([float(m.group(1)), float(m.group(2)), ler / matched, ler])
    return cells


def e5_grids():
    grids = {}
    r5 = {r["decoder"]: f(r["ler50"]) for r in rows("E5")}
    if r5:
        grids["nominal (E5)"] = dict(cells=mismatch_grid(r5, r5.get("Bayes (matched)")),
                                     matched=r5.get("Bayes (matched)"), nl=r5.get("NL"))
    try:
        from experiments.analyze import load_point
        for pid in ("rho0.5", "sq0.26", "hardware"):
            data = load_point("E5b", pid)
            if not data:
                continue
            ler = {k: float(v["errors"][:, -1].mean()) for k, v in data.items()}
            grids[f"{pid} (E5b)"] = dict(cells=mismatch_grid(ler, ler.get("Bayes (matched)")),
                                         matched=ler.get("Bayes (matched)"), nl=ler.get("NL"))
    except Exception as e:                          # cache missing: show what we have
        grids["E5b error"] = dict(error=str(e))
    return grids


def nonstationary():
    out = {}
    try:
        from experiments.analyze_nonstationary import POINTS, decoder_rows
        from experiments.analyze import load_point
        for exp, pts in POINTS.items():
            done, items = 0, []
            for pid, spec, rel in pts:
                data = load_point(exp, pid)
                if not data or "Bayes (oracle)" not in data:
                    items.append(dict(point=pid, pending=True))
                    continue
                done += 1
                rs = decoder_rows(exp, pid, data, rel)
                items.append(dict(point=pid, decoders={
                    re.sub(r" s\d+$", "", r["decoder"]): r.get("excess_vs_oracle") for r in rs
                    if r.get("excess_vs_oracle") is not None},
                    pl={re.sub(r" s\d+$", "", r["decoder"]): r.get("pl_h") for r in rs}))
            out[exp] = dict(done=done, total=len(pts), items=items)
    except Exception as e:
        out["error"] = str(e)
    return out


def stress():
    p = RES / "benchmarks" / "stress_bayes.json"
    if not p.exists():
        return []
    s = json.loads(p.read_text())
    return [dict(point=k, fixed=v.get("excess_fixed"), imm=v.get("excess_imm"),
                 eps=v["eps"].get("Bayes (oracle)")) for k, v in s.items()]


def binary():
    p = RES / "E14_summary.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def continuous():
    p = RES / "E15_summary.json"
    return json.loads(p.read_text()) if p.exists() else []


def cost():
    p = RES / "benchmarks" / "cost.json"
    if not p.exists():
        return {}
    c = json.loads(p.read_text())
    batch = {}
    for r in c["scaling"]:
        if r.get("sweep") is None and not r.get("oom"):
            batch.setdefault(r["decoder"], []).append([r["batch"], r["us_per_trial_round"]])
    return dict(batch={k: sorted(v) for k, v in batch.items()},
                pareto=[[r["decoder"], r["us_per_trial_round"], r["flip_rate"]]
                        for r in c["matched_accuracy"]])


def status():
    def tail(name, n=1):
        p = RES / "logs" / name
        if not p.exists():
            return "not started"
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        return lines[-1][:160] if lines else "started"
    ck = ROOT / "checkpoints"
    from training.train_lstm import model_list
    from experiments.common import get_budget
    budget = get_budget()
    wanted = [m for m, _, _ in model_list(budget)]
    have = [m for m in wanted if (ck / f"{m}.pt").exists()]
    var = len(list(ck.glob("var_*.pt")))
    binl = len(list(ck.glob("binary_lstm_*.pt")))
    dqn = len(list(ck.glob("dqn_*.pt")))
    return dict(
        built=time.strftime("%Y-%m-%d %H:%M"),
        jobs={"E9–E11": tail("full_E9-E11.log"), "E12": tail("full_E12.log"),
              "E5b": tail("full_E5b.log"), "E14": tail("full_E14.log"), "E15": tail("full_E15.log"),
              "train LSTMs + variants": tail("full_train_lstm_variants.log"),
              "train binary + DQN": tail("full_train_binary_dqn.log")},
        models=dict(lstm=f"{len(have)}/{len(wanted)}", variants=var, binary=binl, dqn=dqn),
    )


def build():
    data = dict(
        rho_A=sweep("E2", "rho", lambda p: p.startswith("A_")),
        rho_B=sweep("E2", "rho", lambda p: p.startswith("B_")),
        sigma_q=sweep("E3", "sigma_q"), sigma_m=sweep("E4", "sigma_m"),
        mismatch=e5_grids(), nonstat=nonstationary(), stress=stress(),
        binary=binary(), continuous=continuous(), cost=cost(), status=status(),
    )
    html = (Path(__file__).with_name("runs_template.html")).read_text()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":"), default=float))
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
