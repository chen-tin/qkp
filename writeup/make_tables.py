"""Generate LaTeX tables and number macros for the report from results/.

Every number in writeup/report.tex comes from writeup/generated/*.tex,
which this script builds from results/summary.json and the CSVs.

Run from the repo root:
    python -m writeup.make_tables
"""

import csv
import json
import math
from pathlib import Path

from experiments.common import out_dir

RES = out_dir()
GEN = Path(__file__).resolve().parent / "generated"
GEN.mkdir(exist_ok=True)


def read_csv(name):
    path = RES / name
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sci(x, digits=2):
    """Compact number for tables: 0.123, 4.5e-4 -> 4.5\\times10^{-4}."""
    x = num(x)
    if x is None or not math.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    if abs(x) >= 0.01:
        return f"{x:.{digits + 1}f}"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10**exp
    return f"{mant:.{digits - 1}f}\\!\\times\\!10^{{{exp}}}"


def ler_cell(r):
    """LER after 50 rounds; upper bound when no errors were seen."""
    if r is None:
        return "--"
    if int(float(r["ler50_events"])) == 0:
        return f"$<{sci(3.0 / float(r['trials']), 1)}$"
    s = f"${sci(r['ler50'])}$"
    if int(float(r["ler50_events"])) < 10:
        s += "$^\\dagger$"
    return s


def mean_cell(r, seeds=False):
    if r is None:
        return "--"
    s = f"${sci(r['mean_ler'])}$"
    if seeds and r.get("mean_ler_seed_min"):
        s += f" {{\\scriptsize[{sci(r['mean_ler_seed_min'], 1)},\\,{sci(r['mean_ler_seed_max'], 1)}]}}"
        s = s.replace("[$", "[").replace("$]", "]")
    return s


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def rows_by(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def one(rows, **kw):
    m = rows_by(rows, **kw)
    return m[0] if m else None


FAMILY_ORDER = ["NL", "Bayes (matched)", "Bayes (nominal)", "Bayes-broad",
                "LSTM-nominal", "LSTM-broad", "LSTM-matched"]


def table_e1(rows):
    order = ["No correction", "NL", "Bayes (matched)", "Bayes (per-round)", "Bayes-broad",
             "LSTM-nominal", "LSTM-nominal s0", "LSTM-nominal s1", "LSTM-nominal s2",
             "LSTM-nominal s0 (k0 head)", "LSTM-broad", "LSTM-broad s0", "LSTM-broad s1",
             "LSTM-broad s2"]
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Decoder & $P_L(10)$ & $P_L(50)$ & time-avg.\\ error & paired diff.\\ vs Bayes \\\\",
             "\\midrule"]
    for name in order:
        r = one(rows, decoder=name)
        if r is None:
            continue
        p10 = f"${sci(r['ler10'])}$"
        diff = "--"
        if r.get("diff_mean_ler_vs_bayes") and name != "Bayes (matched)":
            d, se = num(r["diff_mean_ler_vs_bayes"]), num(r.get("diff_mean_ler_vs_bayes_se"))
            if d is not None:
                diff = f"${'+' if d >= 0 else '-'}{sci(abs(d))}$"
                if se:
                    diff += f" {{\\scriptsize($\\pm{sci(2 * se, 1)}$)}}"
        label = name.replace("(k0 head)", "(per-round head)")
        if name in ("LSTM-nominal", "LSTM-broad"):
            label = f"\\textbf{{{name} (mean of seeds)}}"
        lines.append(f"{label} & {p10} & {ler_cell(r)} & {mean_cell(r)} & {diff} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_e1.tex").write_text("\n".join(lines))


def table_sweep(rows, exp, points, header, fname, metric="ler50"):
    present = [d for d in FAMILY_ORDER if rows_by(rows, decoder=d)]
    lines = ["\\begin{tabular}{l" + "r" * len(present) + "}", "\\toprule",
             header + " & " + " & ".join(present) + " \\\\", "\\midrule"]
    for pid, label in points:
        cells = []
        for d in present:
            r = one(rows, point=pid, decoder=d)
            if metric == "ler50":
                cell = ler_cell(r)
            else:
                cell = mean_cell(r)
            if r is not None and d.startswith(("LSTM-broad", "Bayes-broad")) \
                    and r.get("in_dist_broad") == "False":
                cell = f"\\textit{{{cell}}}"
            cells.append(cell)
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / fname).write_text("\n".join(lines))


def main():
    summary = json.loads((RES / "summary.json").read_text())
    macros = ""

    e1 = read_csv("E1_summary.csv")
    if e1:
        table_e1(e1)
        for key, name in (("NL", "NL"), ("Bayes", "Bayes (matched)"),
                          ("BayesPR", "Bayes (per-round)"), ("BayesBroad", "Bayes-broad"),
                          ("LSTMnom", "LSTM-nominal"), ("LSTMbroad", "LSTM-broad"),
                          ("LSTMkzero", "LSTM-nominal s0 (k0 head)"),
                          ("None", "No correction")):
            r = one(e1, decoder=name)
            if r:
                macros += macro(f"Eone{key}LERfifty", sci(r["ler50"]))
                macros += macro(f"Eone{key}Mean", sci(r["mean_ler"]))
                macros += macro(f"Eone{key}Trials", f"{int(float(r['trials'])):,}".replace(",", "{,}"))
                macros += macro(f"Eone{key}Events", r["ler50_events"])
        nl, by = one(e1, decoder="NL"), one(e1, decoder="Bayes (matched)")
        if nl and by and num(by["mean_ler"]):
            macros += macro("EoneGain", f"{num(nl['mean_ler']) / num(by['mean_ler']):.0f}")
        ln = one(e1, decoder="LSTM-nominal")
        if ln and by and num(by["mean_ler"]):
            macros += macro("EoneLSTMoverBayes", f"{num(ln['mean_ler']) / num(by['mean_ler']):.1f}")

    e2 = read_csv("E2_summary.csv")
    from experiments.run_all import E2_LEVELS, E2_RHOS, E3_SQ, E4_SM
    for lv in E2_LEVELS:
        pts = [(f"{lv}_rho{r}", f"{r}") for r in E2_RHOS]
        table_sweep(e2, "E2", pts, "$\\rho$", f"table_e2_{lv}.tex")
        table_sweep(e2, "E2", pts, "$\\rho$", f"table_e2_{lv}_mean.tex", metric="mean")
    e3 = read_csv("E3_summary.csv")
    table_sweep(e3, "E3", [(f"sq{s}", f"{s}") for s in E3_SQ], "$\\sigma_q/L$", "table_e3.tex")
    e4 = read_csv("E4_summary.csv")
    table_sweep(e4, "E4", [(f"sm{s}", f"{s}") for s in E4_SM], "$\\sigma_m/L$", "table_e4.tex")

    # E5 compact table: time-averaged error by assumed rho' and sigma_q'
    e5 = read_csv("E5_summary.csv")
    if e5:
        from experiments.run_all import E5_RHOS, E5_SM_SCALE, E5_SQ_SCALE
        lines = ["\\begin{tabular}{l" + "r" * len(E5_SQ_SCALE) + "}", "\\toprule",
                 "$\\rho'$ \\textbackslash{} $\\sigma_q'/\\sigma_q$ & "
                 + " & ".join(f"{k}" for k in E5_SQ_SCALE) + " \\\\", "\\midrule"]
        for rho in E5_RHOS:
            cells = []
            for k in E5_SQ_SCALE:
                name = "Bayes (matched)" if (rho == 0.9 and k == 1.0) else f"Bayes rho'={rho} sq'x{k}"
                r = one(e5, decoder=name)
                c = mean_cell(r)
                if name == "Bayes (matched)":
                    c = f"\\textbf{{{c}}}"
                cells.append(c)
            lines.append(f"{rho} & " + " & ".join(cells) + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (GEN / "table_e5.tex").write_text("\n".join(lines))
        for name, key in (("Bayes sm'x0.5", "SmHalf"), ("Bayes sm'x2.0", "SmTwo"),
                          ("NL", "NL"), ("LSTM-nominal", "LSTMnom"), ("LSTM-broad", "LSTMbroad"),
                          ("Bayes (matched)", "Bayes"), ("Bayes rho'=0.0 sq'x1.0", "RhoZero"),
                          ("Bayes rho'=0.5 sq'x1.0", "RhoHalf"),
                          ("Bayes rho'=0.99 sq'x1.0", "RhoHigh"),
                          ("Bayes rho'=0.9 sq'x2.0", "SqTwo"),
                          ("Bayes rho'=0.9 sq'x0.5", "SqHalf")):
            r = one(e5, decoder=name)
            if r:
                macros += macro(f"Efive{key}Mean", sci(r["mean_ler"]))
        if e5:
            macros += macro("EfiveTrials", f"{int(float(e5[0]['trials'])):,}".replace(",", "{,}"))

    e6 = read_csv("E6_summary.csv")
    if e6:
        order = ["NL", "Bayes (regime-aware)", "Bayes (rho=0.9)", "Bayes (rho=0.97)",
                 "Bayes-broad", "LSTM-nominal", "LSTM-broad", "LSTM-switching"]
        lines = ["\\begin{tabular}{lrrr}", "\\toprule",
                 "Decoder & $P_L(10)$ & $P_L(50)$ & time-avg.\\ error \\\\", "\\midrule"]
        for name in order:
            r = one(e6, decoder=name)
            if r:
                fam = name in ("LSTM-nominal", "LSTM-broad")
                lines.append(f"{name}{' (mean of seeds)' if fam else ''} & ${sci(r['ler10'])}$ & "
                             f"{ler_cell(r)} & {mean_cell(r, seeds=fam)} \\\\")
                macros += macro("Esix" + name.replace(" ", "").replace("(", "").replace(")", "")
                                .replace("-", "").replace("=", "").replace(".", "")
                                .replace("0", "zero").replace("9", "nine").replace("7", "seven")
                                + "Mean", sci(r["mean_ler"]))
        lines += ["\\bottomrule", "\\end{tabular}"]
        (GEN / "table_e6.tex").write_text("\n".join(lines))

    e7 = summary.get("E7")
    if e7:
        lines = ["\\begin{tabular}{llrrr}", "\\toprule",
                 "Decoder & device & batch & ms / round & trial-rounds / s \\\\", "\\midrule"]
        for r in e7["rows"]:
            lines.append(f"{r['decoder']} & {r['device']} & {r['batch']} & "
                         f"{r['latency_ms_per_round']:.3g} & {r['throughput_trial_rounds_per_s']:.3g} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (GEN / "table_e7.tex").write_text("\n".join(lines))
        if e7.get("gpu"):
            macros += macro("GPUname", e7["gpu"])

    tr = summary.get("training") or []
    if tr:
        lines = ["\\begin{tabular}{lllrr}", "\\toprule",
                 "Model & chosen head & stop & epochs & minutes \\\\", "\\midrule"]
        for r in tr:
            lines.append(f"{r['model'].replace('_', '\\_')} & {r['head']} & "
                         f"{r['stop'].replace('_', ' ')} & {r['epochs']} & {r['seconds'] / 60:.1f} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (GEN / "table_training.tex").write_text("\n".join(lines))

    e0 = summary.get("E0")
    if e0:
        lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
                 "Case & $n_x$ & mean error & $P_L(50)$ & disagreement vs $n_x{=}192$ \\\\",
                 "\\midrule"]
        for r in e0["rows"]:
            lines.append(f"{r['case'].replace('_', ' ')} & {r['n_x']} & ${sci(r['mean_ler'], 3)}$ & "
                         f"${sci(r['ler50'], 3)}$ & ${sci(r['disagree_vs_192'], 1)}$ \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        (GEN / "table_e0.tex").write_text("\n".join(lines))

    e8 = summary.get("E8") or {}
    rev = e8.get("revision") or {}
    for name, key in (("Bayes (matched)", "Bayes"), ("Bayes (per-round)", "BayesPR"),
                      ("LSTM-nominal s0", "LSTM"), ("LSTM-nominal s0 (k0 head)", "LSTMkzero")):
        if name in rev:
            macros += macro(f"Erev{key}", sci(rev[name]))
    hz = e8.get("horizon200") or {}
    for name, key in (("NL", "NL"), ("Bayes (matched)", "Bayes"),
                      ("LSTM-nominal s0", "LSTMnom"), ("LSTM-broad s0", "LSTMbroad")):
        if name in hz:
            macros += macro(f"Ehz{key}Fifty", sci(hz[name]["ler50"]))
            macros += macro(f"Ehz{key}Twohundred", sci(hz[name]["ler200"]))
    pf = e8.get("nl_pair_fraction") or {}
    for pid, key in (("rho0", "Zero"), ("nominal", "Nominal"), ("rho0.99", "High")):
        if pid in pf:
            macros += macro(f"Epair{key}", f"{pf[pid]['adjacent']:.2f}")
            macros += macro(f"Ecancel{key}", f"{pf[pid]['in_even_runs']:.2f}")
    diag = e8.get("diagnostics") or {}
    if "nominal" in diag:
        d = diag["nominal"]
        macros += macro("EdiagDisagree", sci(d["disagree"]))
        if d.get("bayes_right_when_disagree") is not None:
            macros += macro("EdiagBayesRight", f"{100 * d['bayes_right_when_disagree']:.0f}")

    (GEN / "macros.tex").write_text(macros)
    print(f"wrote {len(list(GEN.glob('*.tex')))} files to {GEN}")


if __name__ == "__main__":
    main()
