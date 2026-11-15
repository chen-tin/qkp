"""Effective-length headline metrics from existing results (no new simulations).

Analogue of the "effective fermionic length" application benchmark
(Dallaire-Demers et al., arXiv:2003.01862): one number per decoder that says
how far it can be pushed before a quality threshold is crossed.

1. Effective noise length: along the E3 sigma_q sweep (rho = 0.9) and the E4
   sigma_m sweep, the largest noise level at which P_L(50) stays below
   {0.01, 0.05, 0.10} and the time-averaged error below 0.01; along E2
   (levels A and B), the smallest rho needed. Log-linear interpolation
   between sweep points, over the contiguous "good" region that starts at
   the easy end of the sweep.
2. Effective round length: first round from which P_L(t) stays above
   {0.01, 0.05, 0.10} for the rest of the horizon, at the nominal point
   (E1, 50 rounds; E8 horizon200, 200 rounds), with a Wilson-interval range
   [earliest, latest]. ("Stays above" because round 1 has no history: every
   decoder starts at P_L(1) ~ 0.015, and the memory decoders then recover.)
3. Advantage length: the rho (E2) and sigma_q (E3) values at which the
   decoder's time-averaged error is at most half of NL's, with a 2-sigma
   test of mean_NL - 2 mean_dec > 0 (independent SEs, conservative).
4. Bare-round break-even: the sigma_q* above which the decoder's logical
   flip rate per round is below the odd-jump probability p(sigma_q) of a
   single unprotected round (simulator.sim.odd_jump_probability). Below
   sigma_q*, measurement noise makes repeated correction worse than one
   bare round; above it, correcting with memory wins.

Run from the repo root (CPU is enough):
    python -m experiments.benchmark_effective_length
Writes results/benchmarks/effective_length.{json,csv}.
"""

import csv
import json
import math

import numpy as np

from experiments.common import ROOT, wilson
from simulator.noise import L
from simulator.sim import odd_jump_probability

RES = ROOT / "results"
OUT = RES / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)
FAMILIES = ("NL", "Bayes (matched)", "Bayes (nominal)", "Bayes-broad", "LSTM-nominal",
            "LSTM-broad")
PL_THRESHOLDS = (0.01, 0.05, 0.10)
MEAN_THRESHOLD = 0.01
FLOOR = 1e-7


def rows(exp):
    with open(RES / f"{exp}_summary.csv") as f:
        return list(csv.DictReader(f))


def num(x):
    return None if x in ("", None) else float(x)


def family_rows(exp, family, point_prefix=None):
    out = [r for r in rows(exp) if r["decoder"] == family
           and (point_prefix is None or r["point"].startswith(point_prefix))]
    return out


def value(r, key):
    """Metric value; zero counts are replaced by a floor for log interpolation."""
    v = num(r[key])
    return max(v, FLOOR) if v is not None else None


# 1. effective noise length
def boundary(xs, ys, thr, easy_end):
    """Edge of the contiguous region y < thr starting from the easy end.

    xs ascending. easy_end = "low" (noise sweeps) or "high" (rho sweep).
    Returns (value, status): status "interpolated", "above range" (whole
    sweep good), "below range" (even the easiest point fails).
    """
    xs, ys = list(xs), list(ys)
    order = range(len(xs)) if easy_end == "low" else range(len(xs) - 1, -1, -1)
    order = list(order)
    if ys[order[0]] >= thr:
        return None, "fails at easiest point"
    for a, b in zip(order[:-1], order[1:]):
        if ys[b] >= thr:
            la, lb = math.log(ys[a]), math.log(ys[b])
            f = (math.log(thr) - la) / (lb - la) if lb != la else 0.0
            return float(xs[a] + f * (xs[b] - xs[a])), "interpolated"
    return None, "passes whole sweep"


def noise_length(exp, xkey, family, easy_end, prefix=None):
    rs = sorted(family_rows(exp, family, prefix), key=lambda r: float(r[xkey]))
    if len(rs) < 2:
        return None
    xs = [float(r[xkey]) for r in rs]
    out = {"range": [xs[0], xs[-1]]}
    for thr in PL_THRESHOLDS:
        out[f"ler50<{thr}"] = boundary(xs, [value(r, "ler50") for r in rs], thr, easy_end)
    out[f"mean_ler<{MEAN_THRESHOLD}"] = boundary(xs, [value(r, "mean_ler") for r in rs],
                                                 MEAN_THRESHOLD, easy_end)
    return out


# 2. effective round length
def round_length(curve_k, trials):
    """First round (1-based) from which P_L stays > thr, with a Wilson range."""
    T = len(curve_k)
    p = curve_k / trials
    lo = np.array([wilson(int(k), trials)[1] for k in curve_k])
    hi = np.array([wilson(int(k), trials)[2] for k in curve_k])
    out = {}
    for thr in PL_THRESHOLDS:
        def first(arr):
            above = arr > thr
            if not above[-1]:
                return None
            below = np.flatnonzero(~above)
            return int(below[-1]) + 2 if len(below) else 1
        out[f"P_L>{thr}"] = dict(round=first(p), earliest=first(hi), latest=first(lo),
                                 horizon=T)
    return out


def e1_curves():
    from experiments.analyze import load_point
    data = load_point("E1", "nominal")
    fams = {}
    for name, d in data.items():
        fam = name.rsplit(" s", 1)[0] if name.startswith("LSTM") and "head" not in name else name
        fams.setdefault(fam, []).append(d["errors"])
    out = {}
    for fam, recs in fams.items():
        trials = recs[0].shape[0]
        k = np.mean([r.sum(axis=0) for r in recs], axis=0)   # mean over seeds
        out[fam] = dict(trials=trials, seeds=len(recs), **round_length(np.rint(k), trials))
    return out


def horizon_curves():
    path = RES / "cache" / "E8" / "horizon200.npz"
    meta = json.loads((path.with_name(path.name + ".meta.json")).read_text())
    names, trials = meta["decoders"], int(meta["trials"])
    out = {}
    with np.load(path) as z:
        for k, name in enumerate(names):
            rec = np.unpackbits(z[f"d{k}_err"], axis=1, count=200).astype(bool)[:trials]
            out[name] = dict(trials=trials, **round_length(rec.sum(axis=0), trials))
    return out


# 3. advantage length
def advantage(exp, xkey, family, prefix=None):
    nl = {r["point"]: r for r in family_rows(exp, "NL", prefix)}
    pts = []
    for r in sorted(family_rows(exp, family, prefix), key=lambda r: float(r[xkey])):
        n = nl.get(r["point"])
        if n is None:
            continue
        md, mn = num(r["mean_ler"]), num(n["mean_ler"])
        sd, sn = num(r["mean_ler_se"]) or 0.0, num(n["mean_ler_se"]) or 0.0
        margin = mn - 2 * md
        se = math.sqrt(sn**2 + 4 * sd**2)
        pts.append(dict(x=float(r[xkey]), ratio=mn / md if md > 0 else float("inf"),
                        at_least_2x=margin > 0, significant=margin > 2 * se if se > 0
                        else margin > 0))
    good = [p["x"] for p in pts if p["at_least_2x"]]
    sig = [p["x"] for p in pts if p["significant"]]
    return dict(points=pts, x_at_least_2x=good, x_significant=sig,
                interval=[min(good), max(good)] if good else None)


# 4. bare-round break-even
def break_even(family):
    rs = sorted(family_rows("E3", family), key=lambda r: float(r["sigma_q"]))
    pts = []
    for r in rs:
        sq = float(r["sigma_q"])
        p = odd_jump_probability(sq * L)
        f, se, trials = num(r["flip_rate"]), num(r["flip_rate_se"]) or 0.0, int(float(r["trials"]))
        upper = f + 2 * se if f > 0 else 3.0 / (trials * 50)
        lower = max(f - 2 * se, 0.0)
        verdict = "beats" if upper < p else "worse" if lower > p else "undetermined"
        pts.append(dict(sigma_q=sq, flip=f, p_bare=p, verdict=verdict))
    # crossing where the log ratio flip / p changes sign (flip > 0 points only)
    cross = None
    for a, b in zip(pts[:-1], pts[1:]):
        if a["flip"] > 0 and b["flip"] > 0:
            ra = math.log(a["flip"] / a["p_bare"])
            rb = math.log(b["flip"] / b["p_bare"])
            if ra > 0 >= rb:
                cross = a["sigma_q"] + ra / (ra - rb) * (b["sigma_q"] - a["sigma_q"])
                break
    beats = [p["sigma_q"] for p in pts if p["verdict"] == "beats"]
    return dict(points=pts, sigma_q_star=cross,
                beats_from=min(beats) if beats else None,
                beats_whole_resolved_range=bool(beats) and all(
                    p["verdict"] != "worse" for p in pts if p["sigma_q"] >= min(beats)))


def fmt_boundary(b):
    if b is None:
        return "–"
    v, status = b
    if v is not None:
        return f"{v:.3f}"
    return {"passes whole sweep": "whole sweep", "fails at easiest point": "nowhere"}[status]


def main():
    out = dict(noise_length={}, round_length={}, advantage={}, break_even={})
    for fam in FAMILIES:
        out["noise_length"][fam] = dict(
            E3_sigma_q=noise_length("E3", "sigma_q", fam, "low"),
            E4_sigma_m=noise_length("E4", "sigma_m", fam, "low"),
            E2A_rho=noise_length("E2", "rho", fam, "high", "A_"),
            E2B_rho=noise_length("E2", "rho", fam, "high", "B_"),
        )
        if fam != "NL":
            out["advantage"][fam] = dict(
                E2A_rho=advantage("E2", "rho", fam, "A_"),
                E2B_rho=advantage("E2", "rho", fam, "B_"),
                E3_sigma_q=advantage("E3", "sigma_q", fam),
            )
        out["break_even"][fam] = break_even(fam)
    out["round_length"] = dict(E1_nominal_50=e1_curves(), E8_nominal_200=horizon_curves())
    (OUT / "effective_length.json").write_text(json.dumps(out, indent=1, default=str))

    # flat CSV: one row per decoder family
    table = []
    for fam in FAMILIES:
        nlen = out["noise_length"][fam]
        r200 = out["round_length"]["E8_nominal_200"]
        r50 = out["round_length"]["E1_nominal_50"].get(fam, {})
        name200 = {"LSTM-nominal": "LSTM-nominal s0", "LSTM-broad": "LSTM-broad s0"}.get(fam, fam)
        rl = r200.get(name200) or r50
        adv = out["advantage"].get(fam, {})
        be = out["break_even"][fam]
        row = dict(
            decoder=fam,
            sq_max_ler50_0p05=fmt_boundary(nlen["E3_sigma_q"]["ler50<0.05"]),
            sq_max_ler50_0p10=fmt_boundary(nlen["E3_sigma_q"]["ler50<0.1"]),
            sq_max_mean_0p01=fmt_boundary(nlen["E3_sigma_q"][f"mean_ler<{MEAN_THRESHOLD}"]),
            sm_max_ler50_0p05=fmt_boundary(nlen["E4_sigma_m"]["ler50<0.05"]),
            rho_min_ler50_0p05_A=fmt_boundary(nlen["E2A_rho"]["ler50<0.05"]),
            rho_min_ler50_0p05_B=fmt_boundary(nlen["E2B_rho"]["ler50<0.05"]),
            rounds_to_PL_0p01=_rounds(rl, 0.01),
            rounds_to_PL_0p10=_rounds(rl, 0.1),
            rho_2x_vs_NL_A=_interval(adv.get("E2A_rho")),
            sq_2x_vs_NL=_interval(adv.get("E3_sigma_q")),
            breakeven_sq_star=f"{be['sigma_q_star']:.3f}" if be["sigma_q_star"] else
            ("never" if not be["beats_from"] else f"<= {be['beats_from']:g}"),
        )
        table.append(row)
    with open(OUT / "effective_length.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for r in table:
        print(r)
    return out, table


def _rounds(rl, thr):
    if not rl:
        return "–"
    d = rl[f"P_L>{thr}"]
    if d["round"] is None:
        return f"> {d['horizon']}"
    lo, hi = d["earliest"], d["latest"]
    return f"{d['round']} [{lo}–{hi if hi else '>' + str(d['horizon'])}]"


def _interval(a):
    if not a or not a["x_at_least_2x"]:
        return "none"
    xs = a["x_at_least_2x"]
    sig = a["x_significant"]
    return f"{min(xs):g}–{max(xs):g} ({len(sig)}/{len(xs)} pts at 2σ)"


if __name__ == "__main__":
    main()
