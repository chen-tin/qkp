"""Build the paper-oriented review page (simulation basis, validation, benchmarks).

Reads result files from results/ and writes writeup/artifact/review.html. Run
from the repo root after the scripts that produce them:

    python -m experiments.validation_rerun
    python -m experiments.benchmark_real          # gkp_calibration.json
    python -m experiments.benchmark_cost          # cost.json
    python -m experiments.profile_pipeline --impl ab
    python -m writeup.artifact.build_review
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "benchmarks"
OUT = ROOT / "writeup" / "artifact" / "review.html"


def load(name):
    return json.loads((RES / name).read_text())


def data():
    v = load("validation_rerun.json")
    cal = load("gkp_calibration.json")
    cost = load("cost.json")
    ab = load("profile_ab.json")
    checks = (
        [dict(g="A · single round", l=f"σ={r['sigma']}", z=r["z"], exact=True) for r in v["A"]]
        + [dict(g="B · per-round flip", l="round 1", z=v["B"][0]["z1"], exact=True)]
        + [dict(g="B · per-round flip", l=f"ρ={r['rho']}", z=r["z_later"], exact=True)
           for r in v["B"]]
        + [dict(g="C1 · P_L(t), ρ=0", l="max |z| of 50", z=v["C1"]["zmax"], exact=True)]
        + [dict(g="C2 · approx.", l="ρ=0, σ_m>0", z=v["C2"]["z"], exact=False)]
        + [dict(g="D · ρ→1 limit", l=str(r["rho"]), z=r["z"], exact=False) for r in v["D"]]
    )
    pareto = [dict(n=r["decoder"].replace("Bayes (matched) ", "Bayes "), x=r["us_per_trial_round"],
                   y=r["flip_rate"]) for r in cost["matched_accuracy"] if r["flip_rate"] > 0]
    return dict(
        checks=checks,
        n_exact=sum(c["exact"] for c in checks),
        n_exact_pass=sum(c["exact"] and abs(c["z"]) < 2 for c in checks),
        nominal=v["nominal"]["curves"], trace=v["trace"], rho=v["B"],
        cal=[[r["sigma_m_L"], r["nl_flip"], r["bayes_flip"], r["nl_flip_exact"]] for r in cal["rows"]],
        cal_meta=dict(sq=cal["sigma_q_from_loss_L"], sm_delta=cal["sigma_m_from_delta_L"],
                      nl=cal["sigma_m_matching_sivak_nl_L"], bayes=cal["sigma_m_matching_sivak_bayes_L"],
                      T1eq=cal["nominal_equivalent_cavity_T1_us"]),
        pareto=pareto,
        ab={k: dict(ref=r["reference"], fast=r["fast"], speedup=r["speedup"]) for k, r in ab.items()},
        grid=v["grid"], grid_agree=v["grid_agree"],
    )


def ab_rows(ab):
    order = ["Bayes (matched)", "Bayes (oracle)", "Bayes-IMM", "Bayes-window 200", "Bayes-broad",
             "LSTM-nominal"]
    rows = []
    for name in order:
        t, l = ab.get(f"{name} | throughput"), ab.get(f"{name} | latency")
        if t and l:
            rows.append(f'<tr><td data-l="Decoder">{name}</td>'
                        f'<td class="n" data-l="throughput">{t["ref"]:.2f} → {t["fast"]:.2f}</td>'
                        f'<td class="n" data-l="×">{t["speedup"]:.1f}×</td>'
                        f'<td class="n" data-l="latency">{l["ref"]:.2f} → {l["fast"]:.3f}</td>'
                        f'<td class="n" data-l="×">{l["speedup"]:.0f}×</td></tr>')
    for key, label in (("encode state delta=0.2 N=200", "GKP state, Δ = 0.20, N = 200"),
                       ("encode wigner 201x201 delta=0.2 N=200", "Wigner function 201², N = 200")):
        r = ab.get(key)
        if r:
            rows.append(f'<tr><td data-l="Step">{label}</td><td class="n" data-l="ms">'
                        f'{r["ref"]:.0f} → {r["fast"]:.1f} ms</td><td class="n" data-l="×">'
                        f'{r["speedup"]:.0f}×</td><td class="n">—</td><td class="n">—</td></tr>')
    return "\n".join(rows)


def stress_rows():
    s = load("stress_bayes.json")
    fmt = lambda v: "—" if v is None else f"{v:.2f}×"
    rows = []
    for key, r in s.items():
        eo = r["eps"].get("Bayes (oracle)")
        d, se = r["fixed_minus_oracle"]
        rows.append(f'<tr><td data-l="Point">{key}</td>'
                    f'<td class="n" data-l="ε oracle">{"—" if eo is None else f"{eo:.1e}"}</td>'
                    f'<td class="n" data-l="fixed÷oracle">{fmt(r.get("excess_fixed"))}</td>'
                    f'<td class="n" data-l="IMM÷oracle">{fmt(r.get("excess_imm"))}</td>'
                    f'<td class="n" data-l="P_L gap">{d:+.3f} ± {se:.3f}</td></tr>')
    return "\n".join(rows)


def coverage_img():
    import base64
    png = (RES / "coverage_map.png").read_bytes()
    return "data:image/png;base64," + base64.b64encode(png).decode()


def build():
    d = data()
    html = TEMPLATE.replace("__DATA__", json.dumps(d, separators=(",", ":")))
    html = html.replace("__AB_ROWS__", ab_rows(d["ab"]))
    html = html.replace("__STRESS_ROWS__", stress_rows())
    html = html.replace("__COVERAGE_IMG__", coverage_img())
    html = html.replace("__N_EXACT__", f'{d["n_exact_pass"]} / {d["n_exact"]}')
    html = html.replace("__SQ_LOSS__", f'{d["cal_meta"]["sq"]:.3f}')
    html = html.replace("__T1EQ__", f'{d["cal_meta"]["T1eq"]:.0f}')
    html = html.replace("__SM_MATCH__", f'{d["cal_meta"]["nl"]:.2f}–{d["cal_meta"]["bayes"]:.2f}')
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KB)")


TEMPLATE = (Path(__file__).with_name("review_template.html")).read_text() \
    if (Path(__file__).with_name("review_template.html")).exists() else ""

if __name__ == "__main__":
    build()
