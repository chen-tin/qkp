"""Reproducible teaching figures from the repository's GKP encoder and simulator.

Run: .venv/bin/python -m writeup.artifact.build_gkp_walkthrough
This builds illustrations, not a new decoder benchmark or training run.
"""
import base64
import csv
import hashlib
import io
import json
import math
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import qutip as qt
import torch
from scipy.stats import norm

from quantum.encoding import encode_gkp
from simulator.noise import L, NoiseModel
from simulator.sim import Simulator, NearestLattice, wrap, logical_flip
from simulator.fast import syndromes_and_nl
from simulator.switching import SwitchingNoiseModel
from simulator.nonstationary import make_schedule, sample_from_schedule
from simulator.closed_loop import PhaseBit
from decoders.bayesian.grid_filter import BayesGridDecoder

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("gkp-walkthrough")
INK, MUTED, GRID = "#172b42", "#617285", "#e5edf2"
TEAL, PURPLE, CORAL, BLUE, GOLD = "#087e82", "#7960bd", "#ca634b", "#467fc3", "#bc8631"
PALE, ERROR = "#e5f3ee", "#f8e5df"
SEED, TRIALS, ROUNDS = 20261117, 32, 50
CHARTS, FRAMES = [], []


def style(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0, pad=7, colors=MUTED)


def uri(path):
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def table_csv(rows):
    if not rows:
        return ""
    f = io.StringIO(newline="")
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return f.getvalue()


def save(fig, slug, title, category, label, summary, explanation, code, sources, rows=None):
    svg, png = [OUT / "figures" / f"{slug}.{ext}" for ext in ("svg", "png")]
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    fig.savefig(png, bbox_inches="tight", dpi=170)
    plt.close(fig)
    rows = rows or []
    raw = table_csv(rows)
    (OUT / "data" / f"{slug}.csv").write_text(raw)
    CHARTS.append(dict(id=slug, title=title, category=category, label=label,
                       summary=summary, explanation=explanation, code=code,
                       sources=sources, svg=uri(svg), png=uri(png), csv=raw,
                       rows=rows))


def cell_bands(ax, lo, hi, vertical=True):
    for k in range(math.floor(lo)-1, math.ceil(hi)+2):
        if vertical:
            ax.axvspan(k-.5, k+.5, color=ERROR if k % 2 else PALE, zorder=-3)
        else:
            ax.axhspan(k-.5, k+.5, color=ERROR if k % 2 else PALE, zorder=-3)


def wave_density(coeffs, q):
    n = len(coeffs)
    phi = np.empty((n, len(q)))
    phi[0] = np.pi**(-.25) * np.exp(-q*q/2)
    if n > 1:
        phi[1] = np.sqrt(2) * q * phi[0]
    for i in range(1, n-1):
        phi[i+1] = np.sqrt(2/(i+1))*q*phi[i] - np.sqrt(i/(i+1))*phi[i-1]
    return np.abs(np.asarray(coeffs) @ phi)**2


class RecordedBayes(BayesGridDecoder):
    def _reset(self, trials):
        super()._reset(trials)
        self.marginals, self.before_C = [], []

    def choose(self, syndrome):
        self.before_C.append(self.C.cpu().numpy().copy())
        ans = super().choose(syndrome)
        self.marginals.append(self.filter.B.sum(dim=(0, 1)).cpu().numpy().copy())
        return ans


def lattice():
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 6.4), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1, .75]})
    for ax, parity, col in zip(axs[:2], (0, 1), (TEAL, PURPLE)):
        xs = np.array([k for k in range(-4, 5) if k % 2 == parity])
        ax.vlines(xs, 0, 1, color=col, linewidth=3)
        ax.scatter(xs, np.ones(len(xs)), color=col, s=35)
        ax.set_ylim(0, 1.35); ax.set_yticks([])
        ax.set_title(f"Logical {parity}: ideal peaks on {'even' if parity == 0 else 'odd'} lattice sites", loc="left")
        style(ax, "x")
    cell_bands(axs[2], -4.5, 4.5)
    for k in range(-4, 5):
        axs[2].text(k, .5, "flipped" if k % 2 else "correct", ha="center", va="center", fontsize=9,
                    color=CORAL if k % 2 else TEAL)
    axs[2].set_title("Residual displacement: nearest even / odd cell sets the logical result", loc="left")
    axs[2].set_ylim(0, 1); axs[2].set_yticks([]); axs[2].set_xticks(range(-4, 5))
    axs[2].set_xlim(-4.5, 4.5); axs[2].set_xlabel("Position or residual displacement / L, with L = √π")
    style(axs[2], "x"); fig.tight_layout(h_pad=1.6)
    save(fig, "01-lattice", "A qubit stored in a repeating pattern", "encoding", "Ideal-code schematic",
         "Even sites represent logical 0; odd sites represent logical 1. A displacement of L changes logical parity.",
         "The vertical lines indicate ideal peak locations, not finite-height probability densities. The lower strip is the simulator's readout rule for a residual shift relative to the intended code state. An ideal final correction removes the within-cell offset; an odd remaining cell index is a logical error.",
         "k = round(x / L)\nlogical_error = (k % 2 == 1)",
         ["simulator/sim.py", "docs/sim_validation.md"],
         [dict(cell=k, position_L=k, logical_parity=k % 2) for k in range(-4, 5)])


def encoding():
    delta, ncut, kmax = .30, 100, 8
    psi = encode_gkp(delta, ncut, kmax)
    coeff = psi.full().ravel()
    assert abs(np.vdot(coeff, coeff).real-1) < 1e-12
    n = np.arange(ncut)
    fig, axs = plt.subplots(1, 3, figsize=(12.4, 3.9), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    k, ell = np.meshgrid(np.arange(-kmax, kmax+1), np.arange(-kmax, kmax+1))
    plus = (k*ell) % 2 == 0
    for m, col, marker, label in [(plus, TEAL, "o", "+ amplitude"), (~plus, CORAL, "x", "− amplitude")]:
        axs[0].scatter(2*k[m], ell[m], s=14, c=col, marker=marker, label=label)
    axs[0].set(xlabel="Coherent-state center q / L", ylabel="Center p / L", title="1  Sum coherent-state amplitudes")
    axs[0].legend(fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(.5, -.22), frameon=False)
    axs[1].plot(n, np.exp(-delta**2*n), color=PURPLE, lw=2.4)
    axs[1].set(xlabel="Oscillator energy level n", ylabel="Amplitude multiplier", title="2  Suppress high-energy terms", ylim=(0, 1.05))
    axs[1].text(.37, .78, "exp(−Δ² n)\nΔ = 0.30", transform=axs[1].transAxes, color=PURPLE)
    axs[2].bar(n, np.abs(coeff)**2, color=TEAL, width=.9)
    axs[2].set(xlabel="Oscillator energy level n", ylabel="Probability |cₙ|²", title="3  Normalize the state", xlim=(-1, 60))
    axs[2].text(.42, .83, f"Σ |cₙ|² = 1\nMean n = {np.sum(n*abs(coeff)**2):.2f}", transform=axs[2].transAxes, color=TEAL)
    for ax in axs: style(ax)
    fig.tight_layout(w_pad=2.5)
    save(fig, "02-encoder", "What encode_gkp actually builds", "encoding", "Actual encoder output",
         "A 17 × 17 coherent-state sum, an energy envelope, and a normalized vector of 100 amplitudes.",
         "The left plot shows centers of the 289 coherent-state terms and the signs of their amplitudes. It is not the final state's probability distribution. The center plot shows the finite-energy filter. The right plot is the normalized Fock-level probability from the actual encode_gkp(0.30, 100, 8) output (display zoomed to n ≤ 60). This function prepares approximate logical zero, rather than an arbitrary input qubit or a physical preparation circuit.",
         "coeffs += (-1)**(k * ell) * coherent_state\ncoeffs *= exp(-delta**2 * n)\ncoeffs /= norm(coeffs)",
         ["quantum/encoding.py"],
         [dict(n=int(i), amplitude_multiplier=float(np.exp(-delta**2*i)), probability=float(abs(c)**2)) for i, c in enumerate(coeff)])
    q = np.linspace(-4.5*L, 4.5*L, 2701)
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    rows = []
    for d, col in [(.25, BLUE), (.30, TEAL), (.40, CORAL)]:
        state = encode_gkp(d, 240, 12).full().ravel()
        density = wave_density(state, q) * L
        for ax in axs: ax.plot(q/L, density, color=col, lw=1.7, label=f"Δ = {d:.2f}")
        rows.extend(dict(delta=d, q_L=float(x), density_per_q_over_L=float(y)) for x, y in zip(q/L, density))
    axs[0].set(xlim=(-4.5, 4.5), xlabel="Position q / L", ylabel="Probability density in q / L", title="Finite-width peaks near even lattice sites")
    axs[1].set(xlim=(-.55, .55), xlabel="Position q / L", title="Zoom into the central peak")
    for ax in axs:
        style(ax); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(w_pad=2)
    save(fig, "03-finite-energy", "Finite energy turns ideal spikes into peaks", "encoding", "Actual encoder output",
         "Changing Δ changes the shape and energy of the prepared quantum state.",
         "These position probability densities are calculated from the encoder's complex Fock coefficients using harmonic-oscillator wavefunctions. Smaller Δ gives narrower peaks and a broader overall envelope. All three states use 240 Fock levels and k_max = 12 for this comparison. The plotted position range is cropped; the densities have not been renormalized to that crop. Δ is a state-preparation parameter and is separate from the noise engine's σq and σm.",
         "psi_q = sum(c_n * harmonic_oscillator_wavefunction(n, q))\nP_q = abs(psi_q)**2",
         ["quantum/encoding.py", "quantum/fast_encoding.py"], rows)
    grid = np.linspace(-4*L, 4*L, 181)
    W = qt.wigner(psi, grid, grid, method="clenshaw")
    densq = wave_density(coeff, grid)*L
    densp = wave_density(coeff*((-1j)**np.arange(ncut)), grid)*L
    # Fixed physical sizes keep the marginal coordinates aligned with the square image.
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_axes([.14, .20, .50, .5625])
    top = fig.add_axes([.14, .80, .50, .12])
    side = fig.add_axes([.685, .20, .18, .5625])
    top.plot(grid/L, densq, color=TEAL, lw=1.7); top.set_xlim(-4, 4); top.set_ylabel("P(q/L)"); top.tick_params(labelbottom=False); style(top)
    limit=np.max(abs(W))
    im=ax.imshow(W, origin="lower", extent=(-4, 4, -4, 4), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), interpolation="nearest", aspect="equal")
    ax.set(xlabel="Position q / L", ylabel="Momentum p / L")
    side.plot(densp, grid/L, color=PURPLE, lw=1.7); side.set_ylim(-4, 4); side.set_xlabel("P(p/L)"); side.tick_params(labelleft=False); style(side, "x")
    cb = fig.add_axes([.14, .075, .725, .025])
    fig.colorbar(im, cax=cb, orientation="horizontal", label="Wigner function W(q, p) · signed quasiprobability")
    fig.suptitle("Actual approximate logical-zero state · Δ = 0.30 · 100 Fock levels", fontsize=12, y=.97)
    save(fig, "04-wigner", "See the quantum state in phase space", "encoding", "Actual encoder output",
         "The center is a Wigner function; the side curves are ordinary position and momentum probability densities.",
         "Red and blue represent positive and negative Wigner values. Negative regions describe quantum interference, not negative probabilities of measurement outcomes. The marginals are nonnegative. This is the actual encoder state computed by QuTiP; the main decoder benchmarks do not evolve this full Wigner function each round. Axis coordinates are divided by L, while the Wigner values retain QuTiP's q,p normalization.",
         "psi = encode_gkp(0.30, 100, 8)\nW = qutip.wigner(psi, q_grid, p_grid)",
         ["quantum/encoding.py"],
         [dict(q_L=float(x), p_L=float(y), W_qp=float(W[j, i])) for j, y in enumerate(grid/L) for i, x in enumerate(grid/L)])


def noise_figures(model, d, nu, trial):
    t = np.arange(1, ROUNDS+1)
    fresh = np.full(ROUNDS, np.nan)
    fresh[1:] = (d[trial, 1:] - model.rho*d[trial, :-1])/L
    fig, axs = plt.subplots(3, 1, figsize=(11.8, 6.3), sharex=True)
    for ax, y, std, col, title in [
        (axs[0], d[trial]/L, model.sigma_q/L, TEAL, "1  Physical displacement dₜ · correlated from round to round"),
        (axs[1], fresh, model.innovation_std/L, PURPLE, "2  Fresh innovation ηₜ = dₜ − ρdₜ₋₁ · independent Gaussian input"),
        (axs[2], nu[trial]/L, model.sigma_m/L, CORAL, "3  Measurement error νₜ · independent readout noise")]:
        ax.axhspan(-std, std, color=col, alpha=.09)
        ax.axhline(0, color=MUTED, lw=.7)
        ax.plot(t, y, color=col, lw=1.7, marker="o", ms=2.5)
        ax.set_ylabel("Shift / L"); ax.set_title(title, loc="left", fontsize=10)
        ax.text(.99, .88, f"Band = ±1 SD ({std:.3f}L)", transform=ax.transAxes, ha="right", fontsize=8, color=col,
                bbox=dict(facecolor="white",edgecolor="none",alpha=.8,pad=2))
        style(ax)
    axs[-1].set_xlabel("Correction round"); axs[-1].set_xlim(1, ROUNDS)
    fig.tight_layout(h_pad=1.2)
    save(fig, "05-noise-profile", "The three signals inside the noise engine", "noise", "Seeded illustration",
         "A physical kick, its fresh random component, and a separate measurement error.",
         f"This is trial {trial} (zero-based) from {TRIALS} trials using seed {SEED}; the same path is used in the decoding walkthrough. The shaded bands are distribution standard deviations, not confidence intervals or hard bounds. The first displacement is sampled directly from the stationary distribution, so a separate innovation is not plotted for round 1. dₜ is a per-round kick, not the accumulated position x.",
         "d[0] = Normal(0, sigma_q)\nd[t] = rho*d[t-1] + Normal(0, sigma_q*sqrt(1-rho**2))\nnu[t] = Normal(0, sigma_m)",
         ["simulator/noise.py"],
         [dict(round=int(k), displacement_L=float(a), innovation_L=None if not np.isfinite(b) else float(b), measurement_noise_L=float(c)) for k,a,b,c in zip(t,d[trial]/L,fresh,nu[trial]/L)])
    fig, axs = plt.subplots(3, 1, figsize=(11.8, 5.6), sharex=True, sharey=True)
    rows=[]
    for ax, rho, col in zip(axs, (0, .5, .9), (BLUE, PURPLE, TEAL)):
        m = NoiseModel(model.sigma_q, model.sigma_m, rho)
        y = m.sample_displacement(np.random.default_rng(11), 1, 70)[0]/L
        ax.axhspan(-.2, .2, color=col, alpha=.08)
        ax.plot(np.arange(1,71), y, color=col, lw=1.8)
        ax.axhline(0, color=MUTED, lw=.6)
        ax.set(title=f"ρ = {rho:g} · stationary SD = 0.20L · fresh-innovation SD = {m.innovation_std/L:.3f}L", ylabel="dₜ / L", ylim=(-.65,.65))
        style(ax)
        rows.extend(dict(rho=rho,round=i+1,displacement_L=float(x)) for i,x in enumerate(y))
    axs[-1].set_xlabel("Round · same underlying standard-normal draws across the three cases")
    fig.tight_layout(h_pad=1)
    save(fig, "06-correlation", "Memory changes the texture of noise", "noise", "Seeded illustration",
         "Increasing ρ creates persistent runs while keeping the long-run displacement variance fixed.",
         "These three paths use common standard-normal random draws and the same σq. Higher ρ gives the next kick a stronger dependence on the previous kick. Each finite path can have a different sample mean or sample variance even though their stationary distributions have the same zero mean and SD. Correlation is not the same as a larger noise amplitude.",
         "innovation_std = sigma_q * sqrt(1 - rho**2)", ["simulator/noise.py"], rows)
    fig, axs = plt.subplots(1,2,figsize=(11.8,4))
    x=np.linspace(-.65,.65,700)
    rows=[]
    for name, sd, col in [("Physical kick",.2,TEAL),("Fresh innovation",model.innovation_std/L,PURPLE),("Measurement error",.05,CORAL)]:
        y=norm.pdf(x,scale=sd);axs[0].plot(x,y,label=f"{name} · SD {sd:.3f}L",color=col,lw=2)
        rows.extend(dict(kind="density",series=name,x=float(a),value=float(b)) for a,b in zip(x,y))
    axs[0].set(xlabel="Shift / L",ylabel="Probability density",title="Analytic distributions at the nominal settings")
    lags=np.arange(0,21)
    for rho,col in [(0,BLUE),(.5,PURPLE),(.9,TEAL)]:
        y=rho**lags;axs[1].plot(lags,y,"o-",color=col,ms=3,label=f"Displacement ρ = {rho:g}")
        rows.extend(dict(kind="autocorrelation",series=f"rho={rho}",x=int(a),value=float(b)) for a,b in zip(lags,y))
    axs[1].set(xlabel="Separation in rounds",ylabel="Expected correlation",title="Memory decays as ρᵏ",ylim=(-.03,1.08))
    for ax in axs: style(ax);ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(w_pad=2)
    save(fig,"07-noise-statistics","How much noise, and how much memory?","noise","Analytic curves",
         "The width of a distribution and its correlation over time are separate properties.",
         "The left curves are exact Gaussian densities for the default parameters. The right curves are the exact stationary AR(1) autocorrelations, not estimates from one short path. Independent measurement noise has correlation zero at every nonzero lag, the same shape as the ρ = 0 curve. Neither plot is an experimental device measurement.",
         "Var(d_t) = sigma_q**2\nCorr(d_t, d_(t+k)) = rho**k",["simulator/noise.py"],rows)


def round_figure(xprev, d, nu, c, title):
    xpre=xprev+d; y=xpre+nu; s=float(wrap(y*L)/L); xafter=xpre-c
    lo=math.floor(min(xprev,xpre,xafter))-.5; hi=math.ceil(max(xprev,xpre,xafter))+.5
    if hi-lo<2: lo-=.5;hi+=.5
    fig,axs=plt.subplots(2,2,figsize=(11.7,5.5))
    def ruler(ax, label):
        cell_bands(ax,lo,hi)
        ax.set(xlim=(lo,hi),ylim=(-.8,.8),yticks=[],xlabel="Residual displacement / L")
        ax.set_title(label,loc="left",fontsize=11)
        ax.axhline(0,color=MUTED,lw=.8);style(ax,"x")
    for ax,title_ in [(axs[0,0],"1  Add a physical displacement"),(axs[1,0],"3  Apply the nearest-lattice correction"),(axs[1,1],"4  Read the remaining logical parity")]:ruler(ax,title_)
    a=axs[0,0];a.scatter([xprev,xpre],[0,0],s=65,c=[BLUE,TEAL],zorder=5)
    a.annotate("",(xpre,.15),(xprev,.15),arrowprops=dict(arrowstyle="->",color=TEAL,lw=2))
    a.text(.03,.83,f"x before = {xprev:+.3f}L",transform=a.transAxes,color=BLUE,fontsize=9)
    a.text(.03,.10,f"d = {d:+.3f}L  →  x pre = {xpre:+.3f}L",transform=a.transAxes,color=TEAL,fontsize=9)
    a=axs[0,1];style(a,"x");a.axhspan(-.1,.1,color="#edf2f6");a.axhline(0,color=MUTED,lw=.8)
    a.scatter([s],[0],s=75,color=PURPLE,zorder=5)
    a.set(xlim=(-.55,.55),ylim=(-.8,.8),yticks=[],xlabel="Measured syndrome / L",xticks=[-.5,-.25,0,.25,.5])
    a.set_title("2  Add readout error, then wrap",loc="left",fontsize=11)
    a.text(.03,.83,f"ν = {nu:+.3f}L;  x pre + ν = {y:+.3f}L",transform=a.transAxes,fontsize=9,color=CORAL)
    a.text(.03,.10,f"wrap({y:+.3f}L) = {s:+.3f}L",transform=a.transAxes,fontsize=10,color=PURPLE)
    a=axs[1,0];a.scatter([xpre,xafter],[0,0],s=65,c=[TEAL,INK],zorder=5)
    a.annotate("",(xafter,.15),(xpre,.15),arrowprops=dict(arrowstyle="->",color=INK,lw=2))
    a.text(.03,.83,f"c = s = {c:+.3f}L",transform=a.transAxes,fontsize=10,color=PURPLE)
    a.text(.03,.10,f"x after = x pre − c = {xafter:+.3f}L",transform=a.transAxes,fontsize=9,color=INK)
    k=int(np.rint(xafter));bad=k%2==1
    a=axs[1,1];a.scatter([xafter],[0],s=90,color=CORAL if bad else TEAL,zorder=5)
    a.axvline(k,color=INK,lw=.7,ls="--")
    a.text(.03,.83,f"Nearest cell k = {k} · {'odd' if bad else 'even'}",transform=a.transAxes,fontsize=10)
    a.text(.03,.10,"LOGICAL FLIP" if bad else "LOGICAL STATE PRESERVED",transform=a.transAxes,color=CORAL if bad else TEAL,fontsize=10,fontweight="bold")
    fig.suptitle(title,fontsize=13,fontweight="bold",y=1.01)
    fig.tight_layout(h_pad=2,w_pad=2)
    return fig


def decoding_examples():
    for slug,d,title in [("08-small-shift",.15,"A small shift: the nearest-lattice correction succeeds"),("09-ambiguous-shift",.60,"A larger shift: wrapping can hide a logical error")]:
        nu=.05;s=float(wrap((d+nu)*L)/L);after=d-s
        fig=round_figure(0,d,nu,s,"One constructed correction round · all values in lattice units")
        save(fig,slug,title,"decoding","Constructed teaching example",
             f"Physical shift {d:.2f}L + readout error {nu:.2f}L → syndrome {s:+.2f}L → residual {after:+.2f}L.",
             "These values were chosen to explain the arithmetic; they are not a sampled benchmark. Pale green cells have even nearest-lattice index, and pale red cells have odd index. Readout error changes the measured value. It affects the physical state indirectly when the decoder acts on that value. The logical-error rule assumes an ideal final within-cell correction.",
             "x_pre = x_prev + d\ns = wrap(x_pre + nu)\nc = s\nx_after = x_pre - c",
             ["simulator/sim.py"], [dict(x_before_L=0,displacement_L=d,measurement_error_L=nu,syndrome_L=s,correction_L=s,residual_L=after,logical_error=bool(logical_flip(after*L)))])
    fig,axs=plt.subplots(1,2,figsize=(11.8,4.2))
    x=np.linspace(-1.8,1.8,1600);y=wrap(x*L)/L;y_plot=y.copy();y_plot[np.r_[False,np.abs(np.diff(y))>.5]]=np.nan
    axs[0].plot(x,y_plot,color=PURPLE,lw=2)
    points=np.array([-1.35,-.35,.65,1.65]);sy=-.35
    axs[0].scatter(points,np.full(4,sy),s=55,color=CORAL,zorder=5)
    axs[0].axhline(sy,color=CORAL,lw=.8,ls="--")
    axs[0].set(xlabel="Unwrapped noisy value (x + ν) / L",ylabel="Syndrome s / L",title="Different cells give the same observation",ylim=(-.58,.58))
    ax=axs[1];cell_bands(ax,-1.5,1.5);ax.axhline(0,color=MUTED,lw=.6)
    ax.scatter([.95,-.05],[.65,-.65],s=80,c=[CORAL,TEAL],zorder=5)
    ax.text(-1.43,.88,"j = 0:  c = −0.35L  →  x after = +0.95L",fontsize=9,color=CORAL)
    ax.text(-1.43,-.42,"j = 1:  c = +0.65L  →  x after = −0.05L",fontsize=9,color=TEAL)
    ax.set(xlim=(-1.5,1.5),ylim=(-1.05,1.25),yticks=[],xlabel="Resulting residual / L",title="Two parity choices for the +0.60L example")
    for ax in axs:style(ax,"x")
    fig.tight_layout(w_pad=2)
    save(fig,"10-wrap","Why decoding needs more than one measurement","decoding","Exact mapping + teaching example",
         "The syndrome reveals the offset within a cell. The decoder must infer the cell's parity.",
         "The left curve is the actual wrap function. All four marked inputs yield s = −0.35L. The right panel returns to the constructed +0.60L displacement with +0.05L readout error: corrections c = s and c = s + L lead to different logical outcomes. History changes which choice is plausible; it does not reveal the hidden noise directly.",
         "s = (x + nu + L/2) % L - L/2\nc = s + j*L  # j is 0 or 1",
         ["simulator/sim.py","decoders/base.py"],
         [dict(noisy_input_L=float(a),syndrome_L=float(b)) for a,b in zip(x,y)])


def trace_figures(model,d,nu,nl,bayes,decoder,trial,event):
    t=np.arange(1,ROUNDS+1)
    fig,axs=plt.subplots(3,1,figsize=(11.8,7),sharex=True,gridspec_kw={"height_ratios":[1,1.5,.65]})
    axs[0].plot(t,nl.syndromes[trial]/L,color=PURPLE,lw=1.7,marker="o",ms=3)
    axs[0].set(ylabel="sₜ / L",title="Same observed syndrome stream for these two parity decoders",ylim=(-.55,.55));style(axs[0])
    low=min(nl.residuals[trial].min(),bayes.residuals[trial].min())/L-.35
    high=max(nl.residuals[trial].max(),bayes.residuals[trial].max())/L+.35
    cell_bands(axs[1],low,high,vertical=False)
    for result,col,name in [(nl,CORAL,"Nearest lattice"),(bayes,TEAL,"Matched Bayesian filter")]:
        axs[1].plot(t,result.residuals[trial]/L,color=col,lw=1.8,label=name)
    axs[1].set(ylabel="Residual xₜ / L",ylim=(low,high),title="Different corrections leave different residual cell indices")
    axs[1].legend(frameon=False,fontsize=9);style(axs[1])
    errors=np.stack([nl.logical_errors[trial],bayes.logical_errors[trial]])
    axs[2].imshow(errors,aspect="auto",interpolation="nearest",cmap=ListedColormap([PALE,CORAL]),vmin=0,vmax=1,extent=(.5,ROUNDS+.5,1.5,-.5))
    axs[2].set(yticks=[0,1],yticklabels=["Nearest lattice","Bayes"],xlabel="Round · green = correct logical state · coral = logical flip")
    axs[2].tick_params(length=0)
    for ax in axs:
        ax.axvline(event+1,color=INK,lw=.9,ls="--");ax.set_xlim(.5,ROUNDS+.5)
    fig.tight_layout(h_pad=1.5)
    save(fig,"11-history","Watch memory change a correction decision","decoding","Selected seeded trajectory",
         f"At round {event+1}, nearest lattice is in an odd cell while matched Bayes is in an even cell.",
         f"This is trial {trial} from a seeded set of {TRIALS} paths. It was selected because it contains a round where nearest lattice is logically wrong and Bayes is correct. It illustrates a mechanism and does not estimate decoder performance. The true residuals shown here are diagnostics; neither decoder sees them. A residual near −2L is in an even cell and is logically correct under the ideal-code scoring rule.",
         "NL: c = s\nBayes: c = s + j*L\nerror = round(x_after/L) % 2",
         ["simulator/sim.py","decoders/bayesian/grid_filter.py","decoders/base.py"],
         [dict(round=int(i+1),displacement_L=float(d[trial,i]/L),measurement_noise_L=float(nu[trial,i]/L),syndrome_L=float(nl.syndromes[trial,i]/L),nl_correction_L=float(nl.corrections[trial,i]/L),bayes_correction_L=float(bayes.corrections[trial,i]/L),nl_residual_L=float(nl.residuals[trial,i]/L),bayes_residual_L=float(bayes.residuals[trial,i]/L),nl_error=bool(nl.logical_errors[trial,i]),bayes_error=bool(bayes.logical_errors[trial,i])) for i in range(ROUNDS)])
    px=decoder.marginals[event][trial].astype(float)
    C=decoder.before_C[event][trial];s=bayes.syndromes[trial,event]
    grid=decoder.filter.x.cpu().numpy()
    u=((grid-(C+s)+L)%(2*L)-L)/L
    order=np.argsort(u);u=u[order];px=px[order]
    _,probs=decoder.recorded();p=float(probs[trial,event])
    assert abs(px.sum()-1)<1e-5
    assert abs(px[np.abs(u)>=.5].sum()-p)<1e-5
    fig,axs=plt.subplots(1,2,figsize=(11.8,4.2),gridspec_kw={"width_ratios":[1.6,1]})
    cell_bands(axs[0],-1,1)
    axs[0].plot(u,px/(2/decoder.filter.n_x),color=TEAL,lw=2)
    axs[0].fill_between(u,px/(2/decoder.filter.n_x),color=TEAL,alpha=.12)
    axs[0].set(xlim=(-1,1),xlabel="Candidate residual after c = s, modulo 2L, divided by L",ylabel="Posterior density",title=f"Actual Bayesian posterior at round {event+1}")
    axs[1].bar([0,1],[1-p,p],color=[CORAL,TEAL],width=.55)
    def probability_label(v):
        return "≥99.9%" if v >= .999 else "<0.1%" if v < .001 else f"{v:.1%}"
    for i,v in enumerate([1-p,p]):axs[1].text(i,v+.025,probability_label(v),ha="center",color=INK,fontweight="bold")
    axs[1].set(xticks=[0,1],xticklabels=["j = 0\nc = s","j = 1\nc = s + L"],ylim=(0,1.16),ylabel="Posterior probability",title="Choose the more likely correction class")
    for ax in axs:style(ax)
    fig.tight_layout(w_pad=2)
    save(fig,"12-posterior","What the Bayesian decoder believes","decoding","Actual grid-filter posterior",
         f"The filter gives the j = 1 correction class {probability_label(p)} posterior probability at the selected round.",
         "This posterior comes from the repository's 128-point Bayesian grid filter after processing the syndrome history. We sum over hidden displacement and express position relative to the base correction c = s. The middle green region supports j = 0; the outer red regions support j = 1. The two plot edges meet because position is periodic modulo 2L. These are model beliefs, not measured error frequencies or guarantees. Grid discretization and assumed noise parameters affect the belief.",
         "predict noise → shift position belief → multiply by syndrome likelihood\nnormalize posterior\nj = 1 if posterior_mass_of_class_1 > 0.5 else 0",
         ["decoders/bayesian/grid_filter.py"],
         [dict(candidate_residual_L=float(a),posterior_mass=float(b),correction_class=int(abs(a)>=.5)) for a,b in zip(u,px)])
    for r in range(ROUNDS):
        prev=0 if r==0 else nl.residuals[trial,r-1]/L
        fig=round_figure(prev,d[trial,r]/L,nu[trial,r]/L,nl.corrections[trial,r]/L,f"Nearest-lattice path · round {r+1} of {ROUNDS}")
        path=OUT/"frames"/f"round-{r+1:02d}.png"
        fig.savefig(path,bbox_inches="tight",dpi=125);plt.close(fig)
        FRAMES.append(dict(round=r+1,png=uri(path),displacement=float(d[trial,r]/L),measurement=float(nu[trial,r]/L),syndrome=float(nl.syndromes[trial,r]/L),nl_error=bool(nl.logical_errors[trial,r]),bayes_error=bool(bayes.logical_errors[trial,r])))


def extensions(model):
    fig,axs=plt.subplots(3,1,figsize=(11.8,6.5),sharex=True)
    rng=np.random.default_rng(802);switch=SwitchingNoiseModel();d,_,reg=switch.sample_with_regimes(rng,1,80)
    t=np.arange(1,81);rows=[]
    ax=axs[0];ax.plot(t,d[0]/L,color=PURPLE,lw=1.5)
    for i,r in enumerate(reg[0]):
        if r:ax.axvspan(i+.5,i+1.5,color=PURPLE,alpha=.08,lw=0)
    ax.set_title("Hidden regime switching · shade marks ρ = 0.97; unshaded ρ = 0.30",loc="left",fontsize=10)
    rows.extend(dict(family="switching",round=int(i+1),displacement_L=float(d[0,i]/L),sigma_q_L=.2,rho=float(switch.rhos[reg[0,i]]),burst_sd_L=0) for i in range(80))
    base=dict(sigma_q=.2*L,sigma_m=.05*L,rho=.9)
    specs=[dict(**base,family="ramp",target="sigma_q",start=.1*L,end=.3*L),dict(**base,family="burst",kick=.6*L,duration=2,tau=(.48,.5))]
    for ax,spec,col in zip(axs[1:],specs,[TEAL,CORAL]):
        rng=np.random.default_rng(803);schedule=make_schedule(spec,rng,1,80);d,_=sample_from_schedule(schedule,rng)
        ax.plot(t,d[0]/L,color=col,lw=1.6)
        if spec["family"]=="ramp":
            ax.plot(t,schedule.sigma_q[0]/L,ls="--",color=MUTED,lw=1,label="± target SD")
            ax.plot(t,-schedule.sigma_q[0]/L,ls="--",color=MUTED,lw=1)
            ax.set_title("Gradual drift · displacement SD rises from 0.10L to 0.30L",loc="left",fontsize=10)
            ax.legend(frameon=False,fontsize=8)
        else:
            for i in np.flatnonzero(schedule.kick[0]>0):ax.axvspan(i+.5,i+1.5,color=CORAL,alpha=.18,lw=0)
            ax.set_title("Burst · an extra random kick during two highlighted rounds",loc="left",fontsize=10)
        rows.extend(dict(family=spec["family"],round=int(i+1),displacement_L=float(d[0,i]/L),sigma_q_L=float(schedule.sigma_q[0,i]/L),rho=float(schedule.rho[0,i]),burst_sd_L=float(schedule.kick[0,i]/L)) for i in range(80))
    for ax in axs:ax.set_ylabel("dₜ / L");ax.axhline(0,color=MUTED,lw=.6);style(ax)
    axs[-1].set_xlabel("Round");fig.tight_layout(h_pad=1.3)
    save(fig,"13-noise-extensions","The engine can change its noise conditions","noise","Seeded illustrations",
         "Regime switching, slow drift, and sudden bursts are separate stress-test mechanisms.",
         "These are illustrative samples from the repository's extension modules, not completed-study results. Switching changes correlation through a hidden two-state Markov chain. The ramp changes the target displacement variance while rescaling the AR state. A burst adds a separate kick during its active rounds; that kick is not fed into later AR noise. The displacement it causes can remain in the oscillator until corrected.",
         "d_t = AR_displacement_t + burst_sd_t * fresh_normal\n# the extra burst is not part of the next AR state",
         ["simulator/switching.py","simulator/nonstationary.py"],rows)
    x=np.linspace(-1.5,1.5,700);fig,ax=plt.subplots(figsize=(10.8,3.6));rows=[]
    for v,pf,col in [(1,0,TEAL),(.8,0,PURPLE),(1,.1,CORAL)]:
        p=PhaseBit(visibility=v,p_flip=pf).prob_one(x*L);p=pf+(1-2*pf)*p
        ax.plot(x,p,color=col,lw=2,ls="--" if pf else "-",label=f"visibility {v:g}, bit-flip probability {pf:g}")
        rows.extend(dict(displacement_L=float(a),visibility=v,p_flip=pf,p_bit_one=float(b)) for a,b in zip(x,p))
    ax.set(xlabel="Residual displacement x / L · readout offset ν = 0 for this plot",ylabel="P(ancilla bit = 1)",ylim=(-.04,1.04),title="Binary measurement returns a random bit, not the analog syndrome")
    style(ax);ax.legend(frameon=False,fontsize=9);fig.tight_layout()
    save(fig,"14-binary-readout","The binary-readout extension","decoding","Analytic measurement model",
         "A one-bit measurement is sampled from a periodic probability curve.",
         "This is the PhaseBit model used by the closed-loop extension. The graph shows the probability of observing bit 1, not a logical-error probability. Visibility below one reduces measurement contrast. A readout bit flip further mixes the outcomes; in this simple curve, visibility 0.8 and independent flip probability 0.1 produce the same effective contrast, so those two curves coincide. Full ancilla dynamics and back-action are not simulated here.",
         "p1 = (1 - visibility*cos(2*pi*(x + nu)/L)) / 2\nbit = Bernoulli(p1)\nbit ^= Bernoulli(p_flip)",
         ["simulator/closed_loop.py"],rows)


def architecture():
    fig,ax=plt.subplots(figsize=(11.8,4.8));ax.set(xlim=(0,12),ylim=(0,5));ax.axis("off")
    def box(x,y,w,h,title,detail,col):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.12,rounding_size=.08",facecolor=col,edgecolor="none"))
        ax.text(x+w/2,y+h*.69,title,ha="center",va="center",fontsize=11,fontweight="bold",color=INK)
        ax.text(x+w/2,y+h*.30,detail,ha="center",va="center",fontsize=9,color=MUTED)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,color=MUTED,lw=1.4))
    box(.25,3.25,3,1.25,"Quantum encoder","Δ, Fock cutoff, lattice terms", "#e8e4f3")
    box(4.5,3.25,3,1.25,"Finite-energy quantum state","Complex amplitudes cₙ", "#e8e4f3")
    box(8.7,3.25,3,1.25,"State diagnostics","Wigner function, marginals", "#e8e4f3")
    arrow((3.4,3.875),(4.3,3.875));arrow((7.65,3.875),(8.5,3.875))
    box(.25,.35,2.4,1.25,"Noise engine","dₜ and νₜ arrays", "#e1f1ee")
    box(3.5,.35,2.4,1.25,"Shift + measurement","x → x + dₜ; wrap(x + νₜ)", "#e1f1ee")
    box(6.8,.35,2.1,1.25,"Decoder","Choose cₜ; x ← x − cₜ", "#e1f1ee")
    box(9.8,.35,1.9,1.25,"Score","Odd / even cell", "#f8e5df")
    arrow((2.8,.975),(3.3,.975));arrow((6.05,.975),(6.6,.975));arrow((9.05,.975),(9.6,.975))
    ax.text(6,2.48,"Separate computational paths in the current repository",ha="center",fontsize=12,fontweight="bold",color=INK)
    ax.text(6,1.99,"The benchmark loop starts with x = 0; it does not propagate the encoder's quantum state.",ha="center",fontsize=10,color=MUTED)
    fig.tight_layout()
    save(fig,"15-code-map","How the current code fits together","overview","Code architecture",
         "State construction and decoder benchmarking are separate computational paths.",
         "The encoder constructs a one-oscillator quantum state. The main simulator tracks one classical residual displacement x in the q quadrature, initialized at zero, and evaluates a decoder against synthetic noise. The state-preparation parameter Δ does not automatically feed into σq or σm. The shift loop omits finite-energy state evolution, the p-quadrature error channel, photon loss, and detailed ancilla back-action. Separate finite-energy checks in the repository investigate some connections between the two representations.",
         "quantum/encoding.py → quantum state and visual diagnostics\nsimulator/noise.py → simulator/sim.py → decoder → logical-error record",
         ["quantum/encoding.py","simulator/noise.py","simulator/sim.py","simulator/closed_loop.py"],
         [dict(component="encoder",representation="complex Fock amplitudes"),dict(component="main simulator",representation="q-quadrature residual displacement")])


def build():
    for p in (OUT,OUT/"figures",OUT/"frames",OUT/"data",OUT/"sources"):p.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11,"axes.titleweight":"bold","axes.labelcolor":MUTED,"text.color":INK,"axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,"savefig.facecolor":"white","savefig.pad_inches":.18,"svg.fonttype":"path"})
    torch.set_num_threads(2)
    model=NoiseModel();sim=Simulator(model,trials=TRIALS,rounds=ROUNDS)
    d,nu=sim.sample_noise(SEED)
    decoder=RecordedBayes(model.sigma_q,model.sigma_m,model.rho,n_x=128,device=torch.device("cpu"),record=True)
    nl=sim.run(NearestLattice(),d,nu);bayes=sim.run(decoder,d,nu)
    candidates=np.argwhere(nl.logical_errors & ~bayes.logical_errors)
    if not len(candidates):raise RuntimeError("No illustrative decoder disagreement in seeded paths")
    trial,event=map(int,candidates[0])
    fast_s,fast_k=syndromes_and_nl(d,nu)
    assert np.allclose(fast_s,nl.syndromes) and np.array_equal(fast_k,nl.logical_errors)
    assert np.allclose(nl.syndromes,bayes.syndromes)
    lattice();encoding();noise_figures(model,d,nu,trial);decoding_examples()
    trace_figures(model,d,nu,nl,bayes,decoder,trial,event);extensions(model);architecture()
    chapter_order = {name:i for i,name in enumerate(("encoding","noise","decoding","overview"))}
    CHARTS.sort(key=lambda c:(chapter_order[c["category"]],c["id"]))
    sources=sorted({p for c in CHARTS for p in c["sources"]})
    manifest=[]
    for path in sources:
        raw=(ROOT/path).read_bytes();dest=OUT/"sources"/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
        manifest.append(dict(path=path,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    payload=dict(charts=CHARTS,frames=FRAMES,seed=SEED,trial=trial,event=event+1,trials=TRIALS,rounds=ROUNDS,
        source_commit=commit,generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),sources=manifest,
        parameters=dict(sigma_q_L=.2,sigma_m_L=.05,rho=.9,delta=.3,n_cutoff=100,k_max=8),
        packages=dict(matplotlib=matplotlib.__version__,numpy=np.__version__,qutip=qt.__version__,torch=torch.__version__))
    json_text=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(",",":")).replace("<","\\u003c")
    template=Path(__file__).with_name("gkp_walkthrough_template.html").read_text()
    from writeup.artifact.paper_guide import include_paper_guide
    template = include_paper_guide(template, Path(__file__).resolve().parent, OUT)
    for suffix in ("html", "css", "js"):
        content = Path(__file__).with_name(f"gkp_noise_playground.{suffix}").read_text()
        template = template.replace(f"__NOISE_{suffix.upper()}__", content)
        content = Path(__file__).with_name(f"gkp_decoder_heatmaps.{suffix}").read_text()
        template = template.replace(f"__HEATMAP_{suffix.upper()}__", content)
    heatmap_path = OUT / "data" / "decoder-heatmaps.json"
    heatmaps = json.loads(heatmap_path.read_text()) if heatmap_path.exists() else None
    template = template.replace("__HEATMAP_DATA__", json.dumps(heatmaps, ensure_ascii=False).replace("<", "\\u003c"))
    (OUT/"viewer.html").write_text(template.replace("__PAYLOAD__",json_text))
    small={**payload,"charts":[{k:v for k,v in c.items() if k not in ("png","svg","csv")} for c in CHARTS],"frames":[{k:v for k,v in f.items() if k!="png"} for f in FRAMES]}
    (OUT/"illustration_data.json").write_text(json.dumps(small,indent=2,ensure_ascii=False,allow_nan=False))
    # Save the full generated noise arrays so selection of the illustrative path is auditable.
    np.savez_compressed(OUT/"data"/"seeded-noise.npz",displacement=d,measurement=nu)
    (OUT/"README.txt").write_text("GKP illustrated walkthrough\n\nOpen viewer.html in a modern browser. It works offline with embedded Matplotlib figures.\n"
        "15 standalone PNG/SVG figures and 50 round-by-round PNG frames are included.\n"
        "The Nature paper guide is included in viewer.html and in paper.html, with eight primary references and paper-references.bib.\n"
        "The Noise playground adds live, seeded JavaScript visualizations with parameter sliders, paired reference traces, and CSV export.\n"
        "Decoder heatmaps show an exploratory paired sweep with confidence intervals; run experiments.noise_heatmaps to refresh its data.\n"
        "Figures distinguish actual encoder output, analytic curves, and illustrative trajectories.\n"
        "Illustrations are teaching examples; decoder heatmaps are a separate exploratory development sweep, not a replacement for the benchmark suite.\n"
        f"Seed: {SEED}; selected trial (zero-based): {trial}; highlighted round: {event+1}.\n"
        "The first trial/round with nearest-lattice error and Bayes success was selected from 32 trials.\n\n"
        "Regenerate in the qkp repository:\n.venv/bin/python -m writeup.artifact.build_gkp_walkthrough\n\n"
        "Decoder heatmaps: three pairwise stationary-noise sweeps with fixed checkpoints, paired trials, and 95% pointwise Wilson intervals.\n"
        "Run: .venv/bin/python -m experiments.noise_heatmaps\n"
        "Refresh the offline page without recomputing illustrations: .venv/bin/python -m writeup.artifact.refresh_gkp_viewer\n"
        "Train/resume the 36-model LSTM width-depth sweep and populate its heatmaps: .venv/bin/python -m experiments.lstm_size_sweep\n")
    archive=OUT.parent/"qkp-gkp-walkthrough.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():z.write(p,p.relative_to(OUT))
    print(json.dumps(dict(viewer=str(OUT/"viewer.html"),charts=len(CHARTS),round_frames=len(FRAMES),selected_trial=trial,highlighted_round=event+1,archive=str(archive)),indent=2))


if __name__=="__main__":build()
