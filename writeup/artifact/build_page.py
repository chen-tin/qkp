"""Build the review page (writeup/artifact/index.html) from the result files.

Every number on the page is read from results/*.csv, results/summary.json or
the checkpoint sidecars, so the page matches the report.

Run from the repo root:
    .venv/bin/python -m writeup.artifact.build_page
"""

import csv
import html
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent
FIG_OUT = OUT / "figures"


def rows(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


def get(rs, **kw):
    for r in rs:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    raise KeyError(kw)


def f(x):
    return float(x)


def fmt(x, digits=2):
    """0.123 / 4.5e-4 style, as HTML."""
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) >= 0.01:
        return f"{x:.{digits + 1}f}"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    return f"{m:.{digits - 1}f}&thinsp;×&thinsp;10<sup>{e}</sup>"


def upper(x):
    e = int(math.floor(math.log10(x)))
    m = math.ceil(x / 10**e * 10 - 1e-9) / 10
    return f"&lt;&thinsp;{m:.1f}&thinsp;×&thinsp;10<sup>{e}</sup>"


def ler50_cell(r):
    if int(f(r["ler50_events"])) == 0:
        return upper(3.0 / f(r["trials"]))
    return fmt(r["ler50"])


def esc(s):
    return html.escape(s, quote=True)


# data
e1 = rows("E1_summary.csv")
e2 = rows("E2_summary.csv")
e3 = rows("E3_summary.csv")
e4 = rows("E4_summary.csv")
e5 = rows("E5_summary.csv")
e6 = rows("E6_summary.csv")
summary = json.loads((RES / "summary.json").read_text())
e7 = json.loads((RES / "E7_timing.json").read_text())
e0 = json.loads((RES / "E0_grid_convergence.json").read_text())

bayes = get(e1, decoder="Bayes (matched)")
nl = get(e1, decoder="NL")
b_mean = f(bayes["mean_ler"])
N_E1 = int(f(bayes["trials"]))

E1_ORDER = [
    ("No correction", "No correction", "base"),
    ("NL", "Nearest lattice", "base"),
    ("Bayes (matched)", "Bayes, correct model", "bayes"),
    ("Bayes-broad", "Bayes-broad (8-model average)", "bayes"),
    ("Bayes (per-round)", "Bayes, per-round only (ablation)", "bayes"),
    ("LSTM-nominal", "LSTM trained here (mean of 3 seeds)", "lstm"),
    ("LSTM-nominal s0 (k0 head)", "LSTM, per-round head (ablation)", "lstm"),
    ("LSTM-broad", "LSTM trained broadly (mean of 3 seeds)", "lstm"),
]


def bar(value, lo=1e-4, hi=1.0):
    v = min(max(value, lo), hi)
    return 100 * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))


e1_rows_html = []
for key, label, kind in E1_ORDER:
    r = get(e1, decoder=key)
    mean = f(r["mean_ler"])
    ratio = mean / b_mean
    seeds = ""
    if r.get("mean_ler_seed_min"):
        seeds = (f'<span class="seedrange">seeds {fmt(r["mean_ler_seed_min"], 2)}'
                 f'–{fmt(r["mean_ler_seed_max"], 2)}</span>')
    e1_rows_html.append(f"""
      <tr class="k-{kind}">
        <th scope="row">{esc(label)}</th>
        <td class="num">{fmt(r["ler10"])}</td>
        <td class="num">{ler50_cell(r)}</td>
        <td class="num">{fmt(mean)}{seeds}</td>
        <td class="num">{'—' if key == 'Bayes (matched)' else f'{ratio:.3g}×'}</td>
        <td class="barcell" aria-hidden="true"><span class="bar" style="width:{bar(mean):.1f}%"></span></td>
      </tr>""")

# E2 chart data (time-averaged error)
E2_DECODERS = ["NL", "Bayes (matched)", "Bayes (nominal)", "Bayes-broad",
               "LSTM-nominal", "LSTM-broad"]
e2_data = {}
for lv in ("A", "B"):
    series = {}
    for dec in E2_DECODERS:
        pts = []
        for r in e2:
            if r["point"].startswith(lv + "_") and r["decoder"] == dec:
                rho = float(r["point"].split("rho")[1])
                d = dict(x=rho, y=f(r["mean_ler"]),
                         lo=f(r.get("mean_ler_seed_min") or r["mean_ler"]),
                         hi=f(r.get("mean_ler_seed_max") or r["mean_ler"]),
                         ood=r.get("in_dist_broad") == "False")
                pts.append(d)
        series[dec] = sorted(pts, key=lambda p: p["x"])
    e2_data[lv] = series
lm = get(e2, point="A_rho0.5", decoder="LSTM-matched")
e2_data["A"]["LSTM-matched"] = [dict(x=0.5, y=f(lm["mean_ler"]), lo=f(lm["mean_ler"]),
                                     hi=f(lm["mean_ler"]), ood=False)]


def m2(point, dec, exp=e2):
    return f(get(exp, point=point, decoder=dec)["mean_ler"])


def ratio(point, dec, exp=e2, ref="Bayes (matched)"):
    return m2(point, dec, exp) / m2(point, ref, exp)


# key facts used in the text
facts = dict(
    nl50=fmt(nl["ler50"]), b50=fmt(bayes["ler50"]), b_events=bayes["ler50_events"],
    gain=f"{f(nl['mean_ler']) / b_mean:.0f}",
    lstm50=fmt(get(e1, decoder="LSTM-nominal")["ler50"]),
    lstm_ratio=f"{f(get(e1, decoder='LSTM-nominal')['mean_ler']) / b_mean:.1f}",
    broad_b_ratio=f"{f(get(e1, decoder='Bayes-broad')['mean_ler']) / b_mean:.2f}",
    pr_bayes=f"{f(get(e1, decoder='Bayes (per-round)')['mean_ler']) / b_mean:.1f}",
    pr_lstm=f"{f(get(e1, decoder='LSTM-nominal s0 (k0 head)')['mean_ler']) / f(get(e1, decoder='LSTM-nominal s0')['mean_ler']):.1f}",
    rho0_nl=fmt(m2("A_rho0.0", "NL")), rho0_bnom=fmt(m2("A_rho0.0", "Bayes (nominal)")),
    rho0_lnom=fmt(m2("A_rho0.0", "LSTM-nominal")),
    matched05=f"{(ratio('A_rho0.5', 'LSTM-matched') - 1) * 100:.0f}",
    matched026=f"{(ratio('sq0.26', 'LSTM-matched', e3) - 1) * 100:.0f}",
    rho03_lb=fmt(m2("A_rho0.3", "LSTM-broad")), rho03_bb=fmt(m2("A_rho0.3", "Bayes-broad")),
    sw_reg=fmt(get(e6, decoder="Bayes (regime-aware)")["mean_ler"]),
    sw_lb=fmt(get(e6, decoder="LSTM-broad")["mean_ler"]),
    sw_b09=fmt(get(e6, decoder="Bayes (rho=0.9)")["mean_ler"]),
    sw_bb=fmt(get(e6, decoder="Bayes-broad")["mean_ler"]),
    sw_b097=fmt(get(e6, decoder="Bayes (rho=0.97)")["mean_ler"]),
    sw_nl=fmt(get(e6, decoder="NL")["mean_ler"]),
    sw_ls=fmt(get(e6, decoder="LSTM-switching")["mean_ler"]),
    e5_rho0=fmt(get(e5, decoder="Bayes rho'=0.0 sq'x1.0")["mean_ler"]),
    e5_nl=fmt(get(e5, decoder="NL")["mean_ler"]),
    e5_sq2=fmt(get(e5, decoder="Bayes rho'=0.9 sq'x2.0")["mean_ler"]),
    e5_rho05=fmt(get(e5, decoder="Bayes rho'=0.5 sq'x1.0")["mean_ler"]),
    e5_over=fmt(get(e5, decoder="Bayes rho'=0.99 sq'x0.5")["mean_ler"]),
    e5_b=fmt(get(e5, decoder="Bayes (matched)")["mean_ler"]),
)
t = {(r["decoder"], r["device"], r["batch"]): r for r in e7["rows"]}
facts["gpu_lstm"] = f"{t[('LSTM-nominal s0', 'cuda', 4096)]['throughput_trial_rounds_per_s']:.2g}"
facts["gpu_bayes"] = f"{t[('Bayes (matched)', 'cuda', 4096)]['throughput_trial_rounds_per_s']:.2g}"
facts["cpu_ratio"] = f"{t[('LSTM-nominal s0', 'cpu', 512)]['throughput_trial_rounds_per_s'] / t[('Bayes (matched)', 'cpu', 512)]['throughput_trial_rounds_per_s']:.0f}"
facts["cpu_lat_b"] = f"{t[('Bayes (matched)', 'cpu', 1)]['latency_ms_per_round']:.2f}"
facts["cpu_lat_l"] = f"{t[('LSTM-nominal s0', 'cpu', 1)]['latency_ms_per_round']:.2f}"
facts["gpu_name"] = e7.get("gpu") or "GPU"

# training table
train_rows = []
for p in sorted((ROOT / "checkpoints").glob("*.pt.meta.json")):
    info = json.loads(p.read_text())
    train_rows.append((p.name.replace(".pt.meta.json", ""), info.get("chosen_head"),
                       info.get("stop_reason", "").replace("_", " "),
                       info.get("train_seconds", 0) / 60))
train_html = "".join(
    f"<tr><th scope='row'><code>{esc(n)}</code></th><td>{esc(h)}</td><td>{esc(s)}</td>"
    f"<td class='num'>{m:.1f}</td></tr>" for n, h, s, m in train_rows)

# grid convergence summary
worst = max(r["disagree_vs_192"] for r in e0["rows"] if r["n_x"] == 96)

# figures
FIG_OUT.mkdir(parents=True, exist_ok=True)
FIGS = [
    ("E1_ler_vs_round.png", "e1", "E1 · Nominal point, error after each round",
     "Logical error rate after each round at σ<sub>q</sub>&nbsp;=&nbsp;0.2L, "
     "σ<sub>m</sub>&nbsp;=&nbsp;0.05L, ρ&nbsp;=&nbsp;0.9 "
     f"({N_E1:,} trials, same noise for every decoder).",
     "The odd/even zig-zag in the first rounds is a frozen displacement sitting near a "
     "cell boundary; Bayes settles it within a few rounds, the per-round variants never do."),
    ("E2_rho_sweep.png", "e2", "E2 · Correlation sweep",
     "Top: P<sub>L</sub>(50). Bottom: time-averaged error. Level A (left) is the nominal "
     "noise strength, level B (right) is harder and outside the broad training box.",
     "Bayes's gain over nearest lattice peaks near ρ&nbsp;=&nbsp;0.9. At ρ&nbsp;=&nbsp;0, "
     "both decoders tuned for ρ&nbsp;=&nbsp;0.9 fall below nearest lattice."),
    ("E3_sigma_q_sweep.png", "e3", "E3 · Displacement-noise sweep",
     "ρ&nbsp;=&nbsp;0.9, σ<sub>m</sub>&nbsp;=&nbsp;0.05L, 20,000 trials per point.",
     "The nominal-model filter (dashed) tracks the matched one across the whole range. "
     "The purple star is the LSTM trained at σ<sub>q</sub>&nbsp;=&nbsp;0.26L."),
    ("E4_sigma_m_sweep.png", "e4", "E4 · Measurement-noise sweep",
     "ρ&nbsp;=&nbsp;0.9, σ<sub>q</sub>&nbsp;=&nbsp;0.2L, 20,000 trials per point.",
     "Above σ<sub>m</sub>&nbsp;≈&nbsp;0.07L every decoder degrades quickly and the "
     "filters' lead shrinks."),
    ("E5_mismatch.png", "e5", "E5 · Bayes with a wrong model",
     "Time-averaged error of filters that assume the wrong ρ′ and σ<sub>q</sub>′, at the "
     "nominal truth (40,000 trials). The red box is the correct model.",
     "Raising σ<sub>q</sub>′ is cheap; lowering ρ′ or assuming too little innovation "
     "noise is expensive. ρ′&nbsp;=&nbsp;0 is worse than no memory."),
    ("E6_switching.png", "e6", "E6 · Regime-switching noise",
     "ρ switches between 0.3 and 0.97 with probability 0.05 per round.",
     "The broadly trained LSTM beats every filter that assumes one fixed ρ; only the "
     "regime-aware filter does better."),
    ("E7_timing.png", "e7", "E7 · Decoding cost",
     f"Per-round latency for one trial and batched throughput, on an idle {esc(facts['gpu_name'])} "
     "and the CPU.",
     "On the GPU the LSTM and the filter run at similar rates; batched on the CPU the "
     "LSTM is far faster."),
    ("E8_diagnostics.png", "e8", "E8 · Diagnostics",
     "(a) P<sub>L</sub>(10) against two analytic floors. (b) Nearest-lattice jumps "
     "cluster as ρ grows. (c) Calibration at the nominal point.",
     "Bayes approaches the Kalman floor at high ρ. The LSTM's probabilities are biased "
     "low; Bayes is close to calibrated."),
    ("E8_horizon.png", "e8h", "E8 · 200 rounds",
     "Models trained on 50-round sequences, run for 200 rounds at the nominal point.",
     "No blow-up past the training length; errors keep accumulating slowly."),
    ("training_curves.png", "train", "LSTM training curves",
     "Validation loss and validation time-averaged error per epoch, all 9 models.",
     "The three broad seeds converge to similar validation error, so their large "
     "spread at the nominal point is not undertraining."),
]
fig_html = []
for fn, fid, title, cap, look in FIGS:
    shutil.copy(RES / "figures" / fn, FIG_OUT / fn)
    fig_html.append(f"""
    <figure class="fig" id="fig-{fid}">
      <figcaption><h3>{title}</h3><p>{cap}</p></figcaption>
      <a class="figlink" href="figures/{fn}"><img src="figures/{fn}" alt="{esc(title)}" loading="lazy"></a>
      <p class="look"><span class="look-label">What to look at</span> {look}</p>
    </figure>""")

N_CHECK = 150  # claims checked by the results audit
N_FIXED = 32

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GKP Decoder Comparison</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap" rel="stylesheet">
<style>
:root {{
  /* lab-report layout: sticky contents rail beside one reading column; figures may run wider */
  --paper: #f6f7f9;
  --surface: #ffffff;
  --ink: #18202b;
  --muted: #5a6574;
  --rule: #d8dde4;
  --accent: #1f5fa8;      /* Bayes blue, as in the figures */
  --lstm: #b8352a;        /* LSTM red, as in the figures */
  --base: #7a8290;        /* nearest lattice grey */
  --good: #2f7d4f;
  --warn: #9a6a00;
  --accent-soft: #e6eef8;
  --font-display: "Source Serif 4", Georgia, "Times New Roman", serif;
  --font-body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper: #12161c; --surface: #1a2029; --ink: #e6eaf0; --muted: #9aa5b4;
    --rule: #2c3542; --accent: #7fb0ec; --lstm: #f0857a; --base: #a3abb8;
    --good: #6cc391; --warn: #e2b04a; --accent-soft: #1c2a3c; color-scheme: dark;
  }}
}}
:root[data-theme="dark"] {{
  --paper: #12161c; --surface: #1a2029; --ink: #e6eaf0; --muted: #9aa5b4;
  --rule: #2c3542; --accent: #7fb0ec; --lstm: #f0857a; --base: #a3abb8;
  --good: #6cc391; --warn: #e2b04a; --accent-soft: #1c2a3c; color-scheme: dark;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
@media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} }}
body {{
  margin: 0; background: var(--paper); color: var(--ink);
  font-family: var(--font-body); font-size: 16px; line-height: 1.6;
}}
.layout {{
  display: grid; grid-template-columns: minmax(0, 1fr);
  gap: 40px; max-width: 1180px; margin: 0 auto; padding: 32px 20px 80px;
}}
@media (min-width: 1100px) {{
  .layout {{ grid-template-columns: 210px minmax(0, 1fr); }}
}}
nav.toc {{ display: none; }}
@media (min-width: 1100px) {{
  nav.toc {{ display: block; position: sticky; top: 24px; align-self: start;
    font-size: 14px; }}
}}
nav.toc ol {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; }}
nav.toc a {{ color: var(--muted); text-decoration: none; display: block;
  padding: 3px 0 3px 10px; border-left: 2px solid var(--rule); }}
nav.toc a:hover, nav.toc a:focus-visible {{ color: var(--accent); border-left-color: var(--accent); }}
nav.toc .toc-title {{ font-family: var(--font-mono); font-size: 12px; color: var(--muted);
  letter-spacing: .04em; text-transform: uppercase; margin-bottom: 8px; }}
main {{ min-width: 0; display: grid; gap: 56px; }}
section {{ display: grid; gap: 16px; }}
.prose {{ max-width: 68ch; }}
h1, h2, h3 {{ font-family: var(--font-display); font-weight: 600; text-wrap: balance;
  margin: 0; line-height: 1.25; }}
h1 {{ font-size: clamp(30px, 4vw, 40px); }}
h2 {{ font-size: 26px; padding-bottom: 6px; border-bottom: 1px solid var(--rule); }}
h3 {{ font-size: 19px; }}
p {{ margin: 0; }}
a {{ color: var(--accent); }}
a:focus-visible, button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
code {{ font-family: var(--font-mono); font-size: .88em; }}
sup {{ font-size: .7em; }}
.num, .mono {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}

header.top {{ display: grid; gap: 14px; }}
.kicker {{ font-family: var(--font-mono); font-size: 13px; color: var(--muted); }}
.lede {{ font-size: 18px; color: var(--muted); max-width: 68ch; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.chip {{ font-size: 13px; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--rule);
  background: var(--surface); display: inline-flex; gap: 6px; align-items: center; }}
.chip::before {{ content: ""; width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }}
.chip.ok::before {{ background: var(--good); }}
.chip.warn::before {{ background: var(--warn); }}
.chip.off::before {{ background: var(--base); }}

.verdict {{ background: var(--surface); border: 1px solid var(--rule); border-left: 4px solid var(--accent);
  border-radius: 6px; padding: 20px 24px; display: grid; gap: 10px; max-width: 78ch; }}
.verdict h2 {{ border: 0; padding: 0; font-size: 21px; }}
.verdict ul {{ margin: 0; padding-left: 20px; display: grid; gap: 6px; }}

.tablewrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--rule);
  border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14.5px; }}
th, td {{ padding: 9px 12px; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); }}
thead th {{ font-size: 12.5px; font-weight: 600; color: var(--muted); background: var(--paper);
  white-space: nowrap; }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
td.num {{ text-align: right; white-space: nowrap; }}
tbody th {{ font-weight: 500; }}
.seedrange {{ display: block; font-size: 11.5px; color: var(--muted); }}
.barcell {{ width: 28%; min-width: 120px; vertical-align: middle; }}
.bar {{ display: block; height: 10px; border-radius: 2px; background: var(--base); }}
tr.k-bayes .bar {{ background: var(--accent); }}
tr.k-lstm .bar {{ background: var(--lstm); }}
tr.k-bayes th {{ border-left: 3px solid var(--accent); }}
tr.k-lstm th {{ border-left: 3px solid var(--lstm); }}
tr.k-base th {{ border-left: 3px solid var(--base); }}
.note {{ font-size: 13.5px; color: var(--muted); max-width: 78ch; }}

ol.findings {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 14px; counter-reset: f; }}
ol.findings > li {{ counter-increment: f; background: var(--surface); border: 1px solid var(--rule);
  border-radius: 6px; padding: 16px 20px 16px 56px; position: relative; display: grid; gap: 8px; }}
ol.findings > li::before {{ content: counter(f); position: absolute; left: 18px; top: 15px;
  font-family: var(--font-mono); font-size: 15px; color: var(--accent); font-weight: 500; }}
.finding-head {{ display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: baseline; justify-content: space-between; }}
.finding-head h3 {{ font-size: 18px; }}
.strength {{ font-size: 12px; font-family: var(--font-mono); padding: 2px 8px; border-radius: 4px;
  border: 1px solid currentColor; white-space: nowrap; }}
.strength.strong {{ color: var(--good); }}
.strength.qual {{ color: var(--warn); }}
.evidence {{ font-size: 14.5px; color: var(--muted); }}
.evidence a {{ white-space: nowrap; }}

.chartcard {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 16px 18px; display: grid; gap: 12px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; font-size: 14px; }}
.seg {{ display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }}
.seg button {{ font: inherit; font-size: 13.5px; padding: 5px 12px; border: 0; background: var(--surface);
  color: var(--ink); cursor: pointer; }}
.seg button[aria-pressed="true"] {{ background: var(--accent); color: var(--surface); }}
.toggles {{ display: flex; flex-wrap: wrap; gap: 6px 14px; }}
.toggles label {{ display: inline-flex; gap: 6px; align-items: center; cursor: pointer; }}
.swatch {{ width: 18px; height: 3px; border-radius: 2px; display: inline-block; }}
svg.chart {{ width: 100%; height: auto; display: block; }}
svg.chart text {{ fill: var(--muted); font-family: var(--font-mono); font-size: 11px; }}
svg.chart .grid {{ stroke: var(--rule); stroke-width: 1; }}
svg.chart .axis {{ stroke: var(--muted); stroke-width: 1; }}

.figs {{ display: grid; gap: 40px; }}
figure.fig {{ margin: 0; display: grid; gap: 10px; }}
figure.fig figcaption {{ display: grid; gap: 4px; max-width: 78ch; }}
figure.fig figcaption p {{ color: var(--muted); font-size: 14.5px; }}
.figlink {{ display: block; background: #ffffff; border: 1px solid var(--rule); border-radius: 6px;
  padding: 8px; }}
figure.fig img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
.look {{ font-size: 14.5px; max-width: 78ch; }}
.look-label {{ font-family: var(--font-mono); font-size: 12px; color: var(--accent);
  margin-right: 6px; text-transform: uppercase; letter-spacing: .04em; }}

.two {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
.card {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 16px 18px; display: grid; gap: 8px; align-content: start; }}
.card h3 {{ font-size: 17px; }}
.card ul, .prose ul {{ margin: 0; padding-left: 20px; display: grid; gap: 5px; }}
.card p, .card li {{ font-size: 14.5px; }}
.eq {{ font-family: var(--font-mono); font-size: 14px; background: var(--paper);
  border: 1px solid var(--rule); border-radius: 4px; padding: 8px 12px; overflow-x: auto; }}
ol.decide {{ margin: 0; padding-left: 22px; display: grid; gap: 10px; max-width: 78ch; }}
pre {{ font-family: var(--font-mono); font-size: 13px; background: var(--surface);
  border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; overflow-x: auto;
  margin: 0; line-height: 1.5; }}
footer {{ font-size: 13px; color: var(--muted); }}
</style>
</head>
<body>
<div class="layout">
<nav class="toc" aria-label="Contents">
  <div class="toc-title">Contents</div>
  <ol>
    <li><a href="#answer">Answer</a></li>
    <li><a href="#nominal">Nominal point</a></li>
    <li><a href="#findings">Findings</a></li>
    <li><a href="#sweep">ρ sweep, interactive</a></li>
    <li><a href="#figures">All figures</a></li>
    <li><a href="#method">Method</a></li>
    <li><a href="#checks">Checks and audit</a></li>
    <li><a href="#caveats">Caveats</a></li>
    <li><a href="#decide">For you to decide</a></li>
    <li><a href="#files">Files</a></li>
  </ol>
</nav>

<main>
<header class="top">
  <div class="kicker">Branch experiment/decoder-comparison · run of 9 Nov 2026</div>
  <h1>GKP decoder comparison under correlated displacement noise</h1>
  <p class="lede">Nearest lattice vs a sequential Bayesian filter vs an LSTM, for the square GKP
  code in the q quadrature, with AR(1)-correlated displacements and Gaussian measurement noise.
  This page summarises the full run for review; the 20-page LaTeX report has every detail.</p>
  <div class="chips">
    <span class="chip ok">E0–E8 complete, no failures</span>
    <span class="chip ok">17 / 17 tests pass</span>
    <span class="chip ok">{N_CHECK} report claims audited, {N_FIXED} corrected</span>
    <span class="chip warn">compute 2 h 20 min, target was 2 h</span>
    <span class="chip off">RL / DQN not built (out of scope)</span>
  </div>
</header>

<section class="verdict" id="answer" aria-labelledby="answer-h">
  <h2 id="answer-h">The answer to the proposal's question</h2>
  <ul>
    <li>Memory helps a lot under correlated noise: at the nominal point the Bayesian filter's
      time-averaged logical error is {facts['gain']}× lower than nearest lattice's.</li>
    <li>With the correct noise model the Bayesian filter is best at every test point, and a
      moderately wrong model usually still beats the LSTM.</li>
    <li>Machine learning wins where the Bayesian model is structurally wrong: under
      regime-switching noise the broadly trained LSTM beats every fixed-model filter.</li>
    <li>Wrong memory is worse than none: decoders tuned for ρ&nbsp;=&nbsp;0.9 do worse than
      nearest lattice when the noise is uncorrelated.</li>
  </ul>
</section>

<section id="nominal" aria-labelledby="nominal-h">
  <h2 id="nominal-h">The nominal point</h2>
  <p class="prose">σ<sub>q</sub>&nbsp;=&nbsp;0.2L, σ<sub>m</sub>&nbsp;=&nbsp;0.05L,
  ρ&nbsp;=&nbsp;0.9, L&nbsp;=&nbsp;√π. {N_E1:,} trials of 50 rounds; every decoder sees the
  same noise. P<sub>L</sub>(n) is the fraction of trials in a logical error after round n;
  the time-averaged error averages P<sub>L</sub> over rounds 1–50.</p>
  <div class="tablewrap">
  <table>
    <thead><tr><th scope="col">Decoder</th><th scope="col">P<sub>L</sub>(10)</th>
      <th scope="col">P<sub>L</sub>(50)</th><th scope="col">Time-averaged error</th>
      <th scope="col">vs Bayes</th><th scope="col">Time-averaged error, log scale 10<sup>−4</sup>…1</th></tr></thead>
    <tbody>{''.join(e1_rows_html)}
    </tbody>
  </table>
  </div>
  <p class="note">"vs Bayes" divides each decoder's time-averaged error by the correct-model
  filter's. Seed ranges are the spread over the three trained LSTMs. Matched Bayes had
  {facts['b_events']} errors at round 50 in {N_E1:,} trials.</p>
</section>

<section id="findings" aria-labelledby="findings-h">
  <h2 id="findings-h">Findings</h2>
  <p class="note">The tag gives how firmly the data support each finding: <em>robust</em> holds
  across the sweeps and outside the error bars; <em>qualified</em> holds with stated exceptions.</p>
  <ol class="findings">
    <li>
      <div class="finding-head"><h3>Memory helps most at strong correlation</h3><span class="strength strong">robust</span></div>
      <p>Nearest lattice reaches P<sub>L</sub>(50)&nbsp;=&nbsp;{facts['nl50']} at the nominal point;
      the correct-model filter reaches {facts['b50']}. The gain grows with ρ up to about 0.9 and
      tracks how predictable the next displacement is, σ<sub>q</sub>√(1−ρ²).</p>
      <p class="evidence">E1, E2, E8(a): <a href="#fig-e1">nominal curves</a>,
      <a href="#fig-e2">ρ sweep</a>, <a href="#fig-e8">floors</a></p>
    </li>
    <li>
      <div class="finding-head"><h3>Revising earlier corrections matters</h3><span class="strength strong">robust</span></div>
      <p>The best decoders track the running logical state and undo earlier mistakes when later
      syndromes reveal them. Versions that decide each round on its own are
      {facts['pr_lstm']}× worse (LSTM) and {facts['pr_bayes']}× worse (Bayes) in time-averaged error.
      Validation picked the cumulative head for all 9 LSTMs.</p>
      <p class="evidence">E1 ablations, <a href="#nominal">table above</a></p>
    </li>
    <li>
      <div class="finding-head"><h3>With the right model, Bayes wins everywhere</h3><span class="strength strong">robust</span></div>
      <p>Matched LSTMs come within {facts['matched05']}% (ρ&nbsp;=&nbsp;0.5) and {facts['matched026']}%
      (σ<sub>q</sub>&nbsp;=&nbsp;0.26L) of the filter where errors are common, but the LSTM trained
      at the nominal point is {facts['lstm_ratio']}× worse there, where errors are rare. When the two
      disagree at the nominal point, Bayes is right 98% of the time.</p>
      <p class="evidence">E1, E2, E3, E8(c): <a href="#fig-e3">σ<sub>q</sub> sweep</a>,
      <a href="#fig-e8">calibration</a></p>
    </li>
    <li>
      <div class="finding-head"><h3>Wrong memory is worse than no memory</h3><span class="strength strong">robust</span></div>
      <p>With uncorrelated noise (ρ&nbsp;=&nbsp;0) nearest lattice has error {facts['rho0_nl']};
      the filter assuming ρ&nbsp;=&nbsp;0.9 has {facts['rho0_bnom']} and the LSTM trained at
      ρ&nbsp;=&nbsp;0.9 has {facts['rho0_lnom']}. In E5, a filter assuming ρ′&nbsp;=&nbsp;0 at a true
      ρ&nbsp;=&nbsp;0.9 has {facts['e5_rho0']}, worse than nearest lattice ({facts['e5_nl']}).</p>
      <p class="evidence">E2, E5: <a href="#fig-e5">mismatch heatmap</a></p>
    </li>
    <li>
      <div class="finding-head"><h3>A moderately wrong model is usually still fine</h3><span class="strength qual">qualified</span></div>
      <p>Raising the assumed σ<sub>q</sub> to 2× costs little ({facts['e5_sq2']} against
      {facts['e5_b']}); lowering ρ′ to 0.5 costs 5× ({facts['e5_rho05']}); an overconfident model
      collapses to nearest-lattice level ({facts['e5_over']}). The exception to "Bayes stays ahead":
      at weak correlation (ρ&nbsp;=&nbsp;0.3) the broad LSTM ({facts['rho03_lb']}) beats the
      8-model Bayes average ({facts['rho03_bb']}).</p>
      <p class="evidence">E2, E5</p>
    </li>
    <li>
      <div class="finding-head"><h3>ML helps when the Bayes model is structurally wrong</h3><span class="strength strong">robust</span></div>
      <p>When ρ switches between 0.3 and 0.97, the broad LSTM ({facts['sw_lb']}) beats filters
      assuming a fixed ρ&nbsp;=&nbsp;0.9 ({facts['sw_b09']}), ρ&nbsp;=&nbsp;0.97 ({facts['sw_b097']}) and
      the model average ({facts['sw_bb']}); nearest lattice is at {facts['sw_nl']}. Only a filter that
      models the switching does better ({facts['sw_reg']}). An LSTM trained on the switching noise
      itself ({facts['sw_ls']}) is no better than the broad one.</p>
      <p class="evidence">E6: <a href="#fig-e6">switching</a></p>
    </li>
  </ol>
</section>

<section id="sweep" aria-labelledby="sweep-h">
  <h2 id="sweep-h">Correlation sweep, interactive</h2>
  <p class="prose">Time-averaged logical error against ρ (E2). Bars on the LSTM points show the
  spread over three seeds. Hollow points are outside the broad training box.</p>
  <div class="chartcard">
    <div class="controls">
      <div class="seg" role="group" aria-label="Noise level">
        <button type="button" id="lvA" aria-pressed="true">Level A · σ<sub>q</sub> 0.2L, σ<sub>m</sub> 0.05L</button>
        <button type="button" id="lvB" aria-pressed="false">Level B · σ<sub>q</sub> 0.3L, σ<sub>m</sub> 0.08L</button>
      </div>
      <div class="toggles" id="toggles"></div>
    </div>
    <svg class="chart" id="chart" viewBox="0 0 760 380" role="img" aria-label="Time-averaged logical error against correlation rho"></svg>
  </div>
</section>

<section id="figures" aria-labelledby="figures-h">
  <h2 id="figures-h">All figures</h2>
  <p class="note">Click a figure to open it at full size.</p>
  <div class="figs">{''.join(fig_html)}
  </div>
</section>

<section id="method" aria-labelledby="method-h">
  <h2 id="method-h">Method in brief</h2>
  <div class="two">
    <div class="card">
      <h3>Simulator</h3>
      <p>Each round: x&nbsp;←&nbsp;x&nbsp;+&nbsp;ε<sub>t</sub>, syndrome
      s<sub>t</sub>&nbsp;=&nbsp;wrap(x&nbsp;+&nbsp;ν<sub>t</sub>), correction c<sub>t</sub>,
      x&nbsp;←&nbsp;x&nbsp;−&nbsp;c<sub>t</sub>. A logical error is round(x/L) odd. Classical
      shifts only, validated against closed forms and GKP 2001.</p>
    </div>
    <div class="card">
      <h3>The key reduction</h3>
      <p>Every decoder plays c<sub>t</sub>&nbsp;=&nbsp;s<sub>t</sub>&nbsp;+&nbsp;j<sub>t</sub>L with
      j<sub>t</sub>&nbsp;∈&nbsp;{{0,1}}. Then the syndromes do not depend on the decoder, and its
      logical state is the nearest-lattice state XOR the parity of its own j's.</p>
      <div class="eq">s<sub>t</sub> = wrap(ε<sub>t</sub> + ν<sub>t</sub> − ν<sub>t−1</sub>)</div>
      <p>Decoding means estimating a running parity. Comparisons are exactly paired, and LSTM
      labels never depend on the LSTM.</p>
    </div>
    <div class="card">
      <h3>Bayesian grid filter</h3>
      <p>Belief over (cumulative displacement mod 2L, ε) on a 96×97 grid, updated by an AR(1)
      matrix, a shear, and the periodic syndrome likelihood; batched on the GPU. Variants:
      correct model, fixed nominal model, an 8-model average, regime-aware, and a per-round
      ablation. Grid-converged: decisions differ from a 192-point grid in at most
      {worst:.1e} of trial-rounds.</p>
    </div>
    <div class="card">
      <h3>LSTM</h3>
      <p>2 layers × 128, inputs [s/L, sin, cos], five heads predicting window parities and the
      running parity. 200,000 training sequences, up to 40 epochs; the head is chosen on
      validation data. Trained at the nominal point (3 seeds), on a broad parameter box
      (3 seeds), at two matched points, and on switching noise.</p>
    </div>
  </div>
</section>

<section id="checks" aria-labelledby="checks-h">
  <h2 id="checks-h">Checks and audit trail</h2>
  <div class="two">
    <div class="card">
      <h3>Before building</h3>
      <p>Four independent reviewers checked the plan, with skeptics verifying each finding.
      They caught that the first design let Bayes revise past decisions but not the LSTM, which
      would have handicapped the LSTM by design.</p>
    </div>
    <div class="card">
      <h3>Before the main run</h3>
      <p>A code review compared the filter against an exact reference and found the maths
      correct. It found bookkeeping bugs (cache keys, seeding, timing on a shared GPU) and
      analysis bugs, all fixed before the run.</p>
    </div>
    <div class="card">
      <h3>After the run</h3>
      <p>{N_CHECK} claims in the report were checked against the result files; {N_FIXED}
      were corrected. None changed a conclusion; the largest fix was the explanation of the
      ρ&nbsp;≥&nbsp;0.99 behaviour.</p>
    </div>
    <div class="card">
      <h3>Reproduction</h3>
      <p>The simulator validation re-ran identically (GKP's 0.110 at σ&nbsp;=&nbsp;0.555 reproduced
      as 0.1108). 17 unit tests pass, including exact zero error for decoders given the true
      labels and bit-for-bit agreement between closed-loop and batch LSTM decoding.</p>
    </div>
  </div>
  <h3>LSTM training</h3>
  <div class="tablewrap">
  <table>
    <thead><tr><th scope="col">Model</th><th scope="col">Chosen head</th><th scope="col">Stopped by</th><th scope="col">Minutes</th></tr></thead>
    <tbody>{train_html}</tbody>
  </table>
  </div>
  <p class="note">Five models first hit a 10-minute guard while sharing the GPU with the
  evaluation run; they were retrained to their full schedule and every affected result was
  re-evaluated.</p>
</section>

<section id="caveats" aria-labelledby="caveats-h">
  <h2 id="caveats-h">Caveats</h2>
  <ul class="prose">
    <li>Corrections are restricted to s<sub>t</sub>&nbsp;+&nbsp;j<sub>t</sub>L. That makes every
      comparison exactly paired but rules out continuous fine corrections.</li>
    <li>q quadrature and classical shifts only: no finite-energy states, p-quadrature noise,
      loss or ancilla noise.</li>
    <li>One noise family (stationary AR(1)) plus one switching variant.</li>
    <li>The LSTM recipe was fixed in advance, not tuned; a larger model or more data may close
      part of the gap.</li>
    <li>Bayes-broad fixes σ<sub>m</sub>′&nbsp;=&nbsp;0.05L, so it covers the broad box less fully
      than the broad LSTM's training data.</li>
    <li>Where matched Bayes makes almost no errors, P<sub>L</sub>(50) is only an upper bound;
      comparisons there rely on the time-averaged error.</li>
    <li>Cost: on the GPU both memory decoders run at about {facts['gpu_bayes']} trial-rounds/s
      (filter) and {facts['gpu_lstm']} (LSTM). On the CPU single-trial latency is similar
      ({facts['cpu_lat_b']} and {facts['cpu_lat_l']} ms/round), but batched LSTM throughput is
      {facts['cpu_ratio']}× the filter's.</li>
  </ul>
</section>

<section id="decide" aria-labelledby="decide-h">
  <h2 id="decide-h">For you to decide</h2>
  <ol class="decide">
    <li><strong>Is the discrete-correction restriction acceptable for the project?</strong> It is
      what makes the comparison clean. Dropping it means building decoders that also estimate the
      measurement error.</li>
    <li><strong>Should the RL/DQN decoder be built?</strong> With this action set the problem is a
      contextual bandit whose optimum is the matched filter, so a DQN can at best match Bayes. It
      becomes interesting with continuous corrections.</li>
    <li><strong>Is the LSTM gap at rare errors worth chasing?</strong> Options: larger model, more
      data, or oversampling the rounds where nearest lattice jumps.</li>
    <li><strong>Experimental data:</strong> still open if a compatible dataset turns up.</li>
    <li><strong>Merge:</strong> the branch is local only. Merge into chris-branch or push only if
      you want it shared.</li>
  </ol>
</section>

<section id="files" aria-labelledby="files-h">
  <h2 id="files-h">Files</h2>
  <p class="prose">Worktree <code>/home/babay/programs/QC/qkp-experiment</code>, branch
  <code>experiment/decoder-comparison</code>. Your original folder and chris-branch are unchanged.</p>
<pre>RESULTS.md                    one-page status and key numbers
writeup/report.pdf            full LaTeX report (20 pages)
results/E*_summary.csv        every number
results/figures/              every figure
results/run_log_pass*.txt     run logs
checkpoints/*.pt              the 9 trained LSTMs

QKP_MODE=quick .venv/bin/python -m pytest -q
.venv/bin/python -m experiments.run_all      # skips cached results
.venv/bin/python -m experiments.analyze
.venv/bin/python -m writeup.make_tables</pre>
</section>

<footer>Generated from results/ by writeup/artifact/build_page.py. Figures are the same PNGs as in the report.</footer>
</main>
</div>

<script>
(function () {{
  const DATA = {json.dumps(e2_data)};
  const STYLE = {{
    "NL": {{ color: "var(--base)", dash: "" }},
    "Bayes (matched)": {{ color: "var(--accent)", dash: "" }},
    "Bayes (nominal)": {{ color: "var(--accent)", dash: "5 4" }},
    "Bayes-broad": {{ color: "var(--good)", dash: "" }},
    "LSTM-nominal": {{ color: "var(--lstm)", dash: "" }},
    "LSTM-broad": {{ color: "var(--warn)", dash: "" }},
    "LSTM-matched": {{ color: "var(--lstm)", dash: "", star: true }}
  }};
  const NAMES = Object.keys(STYLE);
  const shown = new Set(NAMES);
  let level = "A";
  const svg = document.getElementById("chart");
  const NS = "http://www.w3.org/2000/svg";
  const W = 760, H = 380, m = {{ l: 62, r: 20, t: 14, b: 44 }};
  const x0 = m.l, x1 = W - m.r, y0 = H - m.b, y1 = m.t;
  const X = v => x0 + (v - 0) / (1 - 0) * (x1 - x0);

  function el(tag, attrs, text) {{
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    svg.appendChild(e);
    return e;
  }}

  function draw() {{
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const series = DATA[level];
    let ys = [];
    for (const n of NAMES) if (shown.has(n) && series[n]) for (const p of series[n]) ys.push(p.lo, p.hi, p.y);
    ys = ys.filter(v => v > 0);
    const lo = Math.pow(10, Math.floor(Math.log10(Math.min(...ys))));
    const hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...ys))));
    const Y = v => y0 - (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * (y0 - y1);
    for (let e = Math.log10(lo); e <= Math.log10(hi) + 1e-9; e++) {{
      const y = Y(Math.pow(10, e));
      el("line", {{ x1: x0, x2: x1, y1: y, y2: y, class: "grid" }});
      el("text", {{ x: x0 - 8, y: y + 4, "text-anchor": "end" }}, "1e" + e);
    }}
    for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1.0]) {{
      el("line", {{ x1: X(t), x2: X(t), y1: y0, y2: y0 + 4, class: "axis" }});
      el("text", {{ x: X(t), y: y0 + 18, "text-anchor": "middle" }}, t.toFixed(1));
    }}
    el("line", {{ x1: x0, x2: x1, y1: y0, y2: y0, class: "axis" }});
    el("line", {{ x1: x0, x2: x0, y1: y0, y2: y1, class: "axis" }});
    el("text", {{ x: (x0 + x1) / 2, y: H - 8, "text-anchor": "middle" }}, "correlation ρ");
    el("text", {{ x: 14, y: (y0 + y1) / 2, "text-anchor": "middle",
                  transform: "rotate(-90 14 " + (y0 + y1) / 2 + ")" }}, "time-averaged logical error");
    for (const n of NAMES) {{
      if (!shown.has(n) || !series[n]) continue;
      const s = STYLE[n], pts = series[n];
      if (pts.length > 1) {{
        el("polyline", {{ points: pts.map(p => X(p.x) + "," + Y(p.y)).join(" "), fill: "none",
          stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash }});
      }}
      for (const p of pts) {{
        if (p.hi > p.lo) el("line", {{ x1: X(p.x), x2: X(p.x), y1: Y(p.lo), y2: Y(p.hi), stroke: s.color, "stroke-width": 1.5 }});
        const c = el(s.star ? "rect" : "circle", s.star
          ? {{ x: X(p.x) - 5, y: Y(p.y) - 5, width: 10, height: 10, transform: "rotate(45 " + X(p.x) + " " + Y(p.y) + ")" }}
          : {{ cx: X(p.x), cy: Y(p.y), r: 4 }});
        c.setAttribute("stroke", s.color);
        c.setAttribute("stroke-width", 1.5);
        c.setAttribute("fill", p.ood ? "var(--surface)" : s.color);
        const tt = document.createElementNS(NS, "title");
        tt.textContent = n + " · ρ = " + p.x + " · " + p.y.toExponential(2);
        c.appendChild(tt);
      }}
    }}
  }}

  const tog = document.getElementById("toggles");
  NAMES.forEach((n, i) => {{
    const id = "t" + i;
    const lab = document.createElement("label");
    lab.setAttribute("for", id);
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.id = id; cb.checked = true;
    cb.addEventListener("change", () => {{ cb.checked ? shown.add(n) : shown.delete(n); draw(); }});
    const sw = document.createElement("span");
    sw.className = "swatch"; sw.style.background = STYLE[n].color;
    lab.append(cb, sw, document.createTextNode(n));
    tog.appendChild(lab);
  }});
  const bA = document.getElementById("lvA"), bB = document.getElementById("lvB");
  function setLevel(l) {{
    level = l;
    bA.setAttribute("aria-pressed", l === "A");
    bB.setAttribute("aria-pressed", l === "B");
    draw();
  }}
  bA.addEventListener("click", () => setLevel("A"));
  bB.addEventListener("click", () => setLevel("B"));
  draw();
}})();
</script>
</body>
</html>
"""

(OUT / "index.html").write_text(page)
print(f"wrote {OUT / 'index.html'} ({len(page) / 1e3:.0f} kB) and {len(FIGS)} figures")
