"""Benchmark the simulator against real data and published experimental numbers.

A. Drift features: measured transmon T1 series (Zhu et al., zenodo 16754689;
   fetch with `python -m experiments.fetch_real_data`) against the drift
   generators in simulator/nonstationary.py. Photon loss maps to Gaussian
   displacement noise with variance close to the loss probability (Albert et
   al., arXiv:1708.05010, Sec. II.2), so a decay rate Gamma(t) maps to
   sigma_q(t)^2 proportional to Gamma(t). Only relative fluctuations and
   their time structure are compared, in units of samples: one T1 sample is
   minutes apart, one QEC round is microseconds, so the generators are a
   compressed version of the real drift and only the *shape* is comparable.

B. GKP calibration: Sivak et al. (Nature 616, 50 (2023), arXiv:2211.09116)
   report a cavity T1 of 610 us, a QEC cycle of 2 x 4.924 us, envelope
   Delta = 0.34 and a logical error probability p_X = p_Z = 1.81e-3 per
   cycle. Loss alone gives sigma_q = sqrt(gamma / (1 - gamma)) per cycle with
   gamma = 1 - exp(-t_c / T1). We sweep the measurement-noise level sigma_m
   at that sigma_q and find where our decoders reach the published per-cycle
   error. This is a feature-level comparison: the experiment measures one
   ancilla bit per round with finite-energy states, our simulator an analog
   syndrome of an ideal code.

Run from the repo root:
    python -m experiments.fetch_real_data
    MPLBACKEND=Agg python -m experiments.benchmark_real
Outputs: results/benchmarks/ (or results/quick/benchmarks/ in quick mode).
"""

import json
import math
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

from decoders.bayesian.grid_filter import BayesGridDecoder, block_size
from experiments.common import (
    flips_per_trial, get_budget, get_mode, log, make_rng, out_dir, run_blocked, torch_device,
)
from experiments.fetch_real_data import DATA_DIR, fetch
from simulator import nonstationary as ns
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice, odd_jump_probability

OUT = out_dir() / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)
HEADLINE_MAX_MK = 119          # above ~120 mK T1 collapses (quasiparticle regime)
OUTLIER_FACTOR = 10.0          # T1 fits > 10x from the trace median are dropped

# Sivak et al. 2023 (arXiv:2211.09116), quoted values
SIVAK = dict(T1_cavity_us=610.0, t_cycle_us=2 * 4.924, delta=0.34, p_logical=1.81e-3,
             p_logical_se=0.04e-3, T_logical_ms=2.20)


# ---------------------------------------------------------------------------
# A. drift features
# ---------------------------------------------------------------------------
def load_traces():
    """List of dict(qubit, temp_mK, t_s, T1_us, kept) for every trace."""
    import numpy as np
    traces = []
    for path in sorted(DATA_DIR.glob("*_T1_timeseries.npz")):
        with np.load(path, allow_pickle=False) as z:
            for key in z.files:
                if key.endswith("_time"):
                    continue
                m = re.match(r"T1_(\w+?)_(\d+)mK$", key)
                if not m:
                    continue
                t, T1 = z[key + "_time"].astype(float), z[key].astype(float)
                med = np.median(T1)
                kept = np.isfinite(T1) & (T1 > 0) & (T1 < OUTLIER_FACTOR * med) & \
                    (T1 > med / OUTLIER_FACTOR)
                traces.append(dict(qubit=m.group(1), temp_mK=int(m.group(2)), t_s=t[kept],
                                   T1_us=T1[kept], dropped=int((~kept).sum())))
    return traces


def acf(x, max_lag):
    x = x - x.mean()
    v = (x * x).mean()
    return np.array([1.0] + [(x[:-k] * x[k:]).mean() / v for k in range(1, max_lag + 1)])


def spectral_exponent(x, n_bins=12):
    """alpha in S(f) ~ 1/f^alpha from a log-binned periodogram (unit spacing)."""
    f, p = periodogram(x - x.mean(), detrend=False)
    f, p = f[1:], p[1:]
    edges = np.geomspace(f[0], f[-1] * 1.0001, n_bins + 1)
    fx, py = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi)
        if sel.sum() >= 1 and np.all(p[sel] > 0):
            fx.append(np.exp(np.log(f[sel]).mean()))
            py.append(np.exp(np.log(p[sel]).mean()))
    if len(fx) < 4:
        return float("nan")
    slope = np.polyfit(np.log(fx), np.log(py), 1)[0]
    return float(-slope)


def features(gamma_rel):
    """Shape features of a relative decay-rate series g = Gamma / <Gamma>.

    sigma_q is proportional to sqrt(g), so 'relstd_sigma' is the relative
    spread a simulated sigma_q(t) would need.
    """
    g = np.asarray(gamma_rel, dtype=float)
    s = np.sqrt(g)
    a = acf(g, min(len(g) // 4, 400))
    below = np.flatnonzero(a < 1 / math.e)
    d = np.diff(g)
    return dict(
        n=len(g),
        relstd_gamma=float(g.std() / g.mean()),
        relstd_sigma=float(s.std() / s.mean()),
        maxmin_gamma=float(g.max() / g.min()),
        acf1=float(a[1]),
        tau_samples=float(below[0]) if len(below) else float("nan"),
        alpha=spectral_exponent(g),
        # 1 for white noise, -> 0 for smooth drift
        roughness=float(d.std() / (math.sqrt(2) * g.std())),
        # heavy tails of the increments signal abrupt (telegraph-like) jumps
        jump_kurtosis=float(((d - d.mean()) ** 4).mean() / d.var() ** 2) if d.var() > 0
        else float("nan"),
    )


def real_features():
    rows = []
    for tr in load_traces():
        gamma = 1.0 / tr["T1_us"]
        f = features(gamma / gamma.mean())
        dt_min = float(np.median(np.diff(tr["t_s"])) / 60)
        f.update(qubit=tr["qubit"], temp_mK=tr["temp_mK"], dropped=tr["dropped"],
                 step_min=dt_min, tau_hours=f["tau_samples"] * dt_min / 60,
                 hours=float((tr["t_s"][-1] - tr["t_s"][0]) / 3600),
                 mean_T1_us=float(tr["T1_us"].mean()),
                 headline=tr["temp_mK"] <= HEADLINE_MAX_MK)
        rows.append(f)
    return rows


# generator families compared with the data (illustrative E9/E10 settings)
GENERATORS = {
    "one_over_f rel=0.25 (E9)": ns.spec("one_over_f", rel=0.25),
    "sinusoid amp=0.3 P=1000 (E9)": ns.spec("sinusoid", target="sigma_q", amp=0.3,
                                            period=1000.0),
    "sinusoid amp=0.3 P=200 (E9)": ns.spec("sinusoid", target="sigma_q", amp=0.3,
                                           period=200.0),
    "telegraph 0.20/0.26 L (E10)": ns.spec("telegraph", target="sigma_q",
                                           levels=[0.20 * L, 0.26 * L], p_switch=0.005),
    "step 0.20->0.26 L (E10)": ns.spec("step", target="sigma_q", before=0.20 * L,
                                       after=0.26 * L),
    "ramp 0.16->0.26 L (E9)": ns.spec("ramp", target="sigma_q", start=0.16 * L, end=0.26 * L),
}


def generator_features(spec, n_series, length, seed):
    sch = ns.make_schedule(spec, make_rng("dev", "benchmark_real", seed), n_series, length)
    rows = []
    for s in sch.sigma_q:
        g = (s / s.mean()) ** 2
        rows.append(features(g))
    return rows


def calibrate_one_over_f(target_relstd_sigma, length, seed=99):
    """rel for the one_over_f family whose median relstd_sigma matches the data."""
    lo, hi = 0.01, 1.5
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        rows = generator_features(ns.spec("one_over_f", rel=mid), 64, length, seed)
        if np.median([r["relstd_sigma"] for r in rows]) < target_relstd_sigma:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def summarize_rows(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                          p90=float(np.percentile(v, 90)))
    return out


FEATURE_KEYS = ("relstd_sigma", "relstd_gamma", "maxmin_gamma", "acf1", "tau_samples",
                "alpha", "roughness", "jump_kurtosis")


def part_a():
    fetch()
    real = real_features()
    head = [r for r in real if r["headline"]]
    length = int(np.median([r["n"] for r in head]))
    n_series = 64 if get_mode() == "quick" else 256
    gens = {name: generator_features(sp, n_series, length, k)
            for k, (name, sp) in enumerate(GENERATORS.items())}
    target = float(np.median([r["relstd_sigma"] for r in head]))
    rel = calibrate_one_over_f(target, length)
    cal_name = f"one_over_f rel={rel:.3f} (calibrated)"
    gens[cal_name] = generator_features(ns.spec("one_over_f", rel=rel), n_series, length, 77)

    table = {"real T1 (<=119 mK)": summarize_rows(head, FEATURE_KEYS),
             "real T1 (all)": summarize_rows(real, FEATURE_KEYS)}
    table.update({name: summarize_rows(rows, FEATURE_KEYS) for name, rows in gens.items()})
    _write_csv(OUT / "real_t1_features.csv", real)
    (OUT / "drift_feature_comparison.json").write_text(json.dumps(
        dict(series_length=length, calibrated_rel=rel, table=table), indent=1))

    # figure: feature clouds
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    pairs = (("relstd_sigma", "alpha"), ("roughness", "jump_kurtosis"), ("acf1", "tau_samples"))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, (kx, ky) in zip(axes, pairs):
        for k, (name, rows) in enumerate(gens.items()):
            ax.scatter([r[kx] for r in rows], [r[ky] for r in rows], s=6, alpha=0.35,
                       color=colors[k % 10], label=name)
        ax.scatter([r[kx] for r in head], [r[ky] for r in head], s=46, marker="*",
                   color="k", label="real T1 traces (<=119 mK)", zorder=5)
        ax.set_xlabel(kx)
        ax.set_ylabel(ky)
        ax.grid(alpha=0.3)
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=6.5, loc="best")
    fig.suptitle("Drift generators vs measured T1 drift (features in units of samples)")
    fig.tight_layout()
    fig.savefig(OUT / "drift_features.png", dpi=150)
    plt.close(fig)
    log(f"A: {len(real)} traces ({len(head)} headline); calibrated one_over_f rel={rel:.3f}")
    return dict(table=table, calibrated_rel=rel, series_length=length,
                n_traces=len(real), n_headline=len(head))


# ---------------------------------------------------------------------------
# B. GKP calibration against Sivak et al.
# ---------------------------------------------------------------------------
def loss_sigma(T1_us, t_us):
    gamma = 1 - math.exp(-t_us / T1_us)
    return math.sqrt(gamma / (1 - gamma))


def cavity_T1_for_sigma(sigma, t_us):
    gamma = sigma**2 / (1 + sigma**2)
    return -t_us / math.log(1 - gamma)


def part_b():
    device = torch_device()
    budget = get_budget()
    trials = 4000 if get_mode() == "quick" else 20000
    rounds = 100
    sq = loss_sigma(SIVAK["T1_cavity_us"], SIVAK["t_cycle_us"])
    sm_delta = SIVAK["delta"] / math.sqrt(2)
    sm_grid = sorted({0.0, 0.03, 0.06, 0.08, 0.10, 0.12, round(sm_delta / L, 4), 0.15,
                      0.18, 0.21})
    rows = []
    for k, sm_l in enumerate(sm_grid):
        sm = sm_l * L
        model = NoiseModel(sq, sm, 0.0)       # loss is memoryless: rho = 0
        d, n = model.sample(make_rng("dev", "benchmark_gkp", k), trials, rounds)
        nl = run_blocked(NearestLattice(), d, n, trials)
        dec = BayesGridDecoder(sq, max(sm, 0.005 * L), 0.0,
                               n_x=192 if sm < 0.035 * L else 96, device=device)
        by = run_blocked(dec, d, n, block_size(dec, budget.filter_block_bytes))
        row = dict(sigma_q_L=sq / L, sigma_m_L=sm_l,
                   nl_flip_exact=odd_jump_probability(math.sqrt(sq**2 + 2 * sm**2)))
        for name, rec in (("nl", nl), ("bayes", by)):
            per = flips_per_trial(rec) / rounds
            row[f"{name}_flip"] = float(per.mean())
            row[f"{name}_flip_se"] = float(per.std(ddof=1) / math.sqrt(trials))
        rows.append(row)
        log(f"B: sigma_m={sm_l:.3f}L  NL {row['nl_flip']:.2e}  Bayes {row['bayes_flip']:.2e}")

    def crossing(key):
        xs = np.array([r["sigma_m_L"] for r in rows])
        ys = np.array([r[key] for r in rows])
        target = SIVAK["p_logical"]
        for i in range(len(xs) - 1):
            if ys[i] < target <= ys[i + 1] and ys[i] > 0:
                f = (math.log(target) - math.log(ys[i])) / (math.log(ys[i + 1]) - math.log(ys[i]))
                return float(xs[i] + f * (xs[i + 1] - xs[i]))
        return None

    out = dict(
        sivak=SIVAK, sigma_q_from_loss_L=sq / L, sigma_m_from_delta_L=sm_delta / L,
        nominal_equivalent_cavity_T1_us=cavity_T1_for_sigma(0.20 * L, SIVAK["t_cycle_us"]),
        sigma_m_matching_sivak_nl_L=crossing("nl_flip"),
        sigma_m_matching_sivak_bayes_L=crossing("bayes_flip"),
        trials=trials, rounds=rounds, rows=rows,
    )
    _write_csv(OUT / "gkp_calibration.csv", rows)
    (OUT / "gkp_calibration.json").write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    xs = [r["sigma_m_L"] for r in rows]
    ax.plot(xs, [r["nl_flip_exact"] for r in rows], "k--", lw=1, label="NL, closed form")
    for name, color, label in (("nl", "0.45", "nearest lattice (sim)"),
                               ("bayes", "C0", "Bayes, matched (sim)")):
        ys = np.array([r[f"{name}_flip"] for r in rows])
        se = np.array([r[f"{name}_flip_se"] for r in rows])
        ok = ys > 0
        ax.errorbar(np.array(xs)[ok], ys[ok], yerr=2 * se[ok], fmt="o-", color=color,
                    label=label, ms=4)
    ax.axhline(SIVAK["p_logical"], color="C3", lw=2,
               label="Sivak et al. 2023: p_X = 1.81e-3 per cycle")
    ax.axvline(sm_delta / L, color="C3", ls=":", label="sigma_m = Delta/sqrt(2), Delta = 0.34")
    ax.set_yscale("log")
    ax.set_xlabel("measurement noise sigma_m / L")
    ax.set_ylabel("logical flips per round")
    ax.set_title(f"Loss-only displacement noise sigma_q = {sq / L:.3f} L, rho = 0\n"
                 f"(cavity T1 = 610 us, cycle 9.85 us; Sivak et al. 2023)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "gkp_calibration.png", dpi=150)
    plt.close(fig)
    return out


def _write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and not isinstance(r[k], (list, np.ndarray)):
                keys.append(k)
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")


def main():
    log(f"benchmark_real: mode={get_mode()}")
    summary = dict(A=part_a(), B=part_b())
    (OUT / "benchmark_real.json").write_text(json.dumps(summary, indent=1, default=float))
    log(f"benchmark_real: wrote {OUT}")


if __name__ == "__main__":
    main()
