#!/usr/bin/env python3
"""Figures for the ICAIF paper from the locked-clock, tuned run.
Produces TWO multi-panel figures (for an 8-page two-column paper) plus the six
single panels, as vector PDF (LaTeX) and 600-dpi PNG.

Data source (multi-GPU): results/<GPU_SLUG>/timings.json, default GPU_SLUG=rtx6000ada
(the GPU in the paper). Set GPU_SLUG=h100 etc. to plot another GPU's run. If the file
is missing, falls back to the embedded RTX 6000 Ada numbers so the paper figures
always reproduce.

Output dir: $FIGDIR (default: the paper's top-level figures/ next to this package),
so re-running keeps the paper figures in sync."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.normpath(os.path.join(HERE, ".."))
GPU_SLUG = os.environ.get("GPU_SLUG", "rtx6000ada")
FIGDIR = os.environ.get("FIGDIR") or os.path.normpath(os.path.join(PKG_ROOT, "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("default")
plt.rcParams.update({
    "figure.dpi": 600, "savefig.dpi": 600,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "axes.grid": True, "grid.alpha": 0.35,
    "savefig.bbox": "tight",
})

C = "CUDA C"; M = "Mojo (tuned)"; P = "PyTorch"
col = {C: "#1f77b4", M: "#d62728", P: "#7f7f7f"}

# embedded fallback (RTX 6000 Ada, locked 2505 MHz) -- keeps paper figures reproducible
_FALLBACK = {
    "paths": [1e6, 1e7, 1e8],
    "cuda":    {"median_ms": [0.0074, 0.0153, 0.0938], "std_ms": [0.0008, 0.0012, 0.0045]},
    "mojo":    {"median_ms": [0.0072, 0.0146, 0.0971], "std_ms": [0.0002, 0.0005, 0.0006]},
    "pytorch": {"median_ms": [0.0823, 0.2695, 7.9695], "std_ms": [0.0039, 0.0018, 0.0089]},
    "mojo_tuning_100m_ms": {"baseline": 0.1142, "unroll8": 0.1019, "block64": 0.0971},
}

_json_path = os.path.join(PKG_ROOT, "results", GPU_SLUG, "timings.json")
if os.path.exists(_json_path):
    with open(_json_path) as _f:
        D = json.load(_f)
    print(f"plotting {GPU_SLUG} from {_json_path}")
else:
    D = _FALLBACK
    print(f"no timings.json for '{GPU_SLUG}', using embedded RTX 6000 Ada numbers")

paths  = np.array(D["paths"], dtype=float)
labels = ["1M" if p == 1e6 else "10M" if p == 1e7 else "100M" if p == 1e8
          else f"{p/1e6:g}M" for p in paths]
med = {C: np.array(D["cuda"]["median_ms"], dtype=float),
       M: np.array(D["mojo"]["median_ms"], dtype=float),
       P: np.array(D["pytorch"]["median_ms"], dtype=float)}
std = {C: np.array(D["cuda"]["std_ms"], dtype=float),
       M: np.array(D["mojo"]["std_ms"], dtype=float),
       P: np.array(D["pytorch"]["std_ms"], dtype=float)}
_tune = D.get("mojo_tuning_100m_ms", _FALLBACK["mojo_tuning_100m_ms"])
x = np.arange(len(labels))


def panel_label(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def draw_scaling(ax):
    for k in (C, M, P):
        ax.errorbar(paths, med[k], yerr=std[k], marker="o", capsize=3, lw=2,
                    color=col[k], label=k)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Monte Carlo paths"); ax.set_ylabel("time per launch (ms)")
    ax.set_xticks(paths); ax.set_xticklabels(labels)
    ax.set_title("runtime scaling"); ax.legend()


def draw_speedup(ax):
    w = 0.38
    sC = med[P] / med[C]; sM = med[P] / med[M]
    for off, s, k in ((-w/2, sC, C), (w/2, sM, M)):
        b = ax.bar(x + off, s, w, color=col[k], label=k)
        for r in b:
            ax.annotate(f"{r.get_height():.0f}x",
                        (r.get_x()+r.get_width()/2, r.get_height()),
                        ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Monte Carlo paths"); ax.set_ylabel("speedup over PyTorch (x)")
    ax.set_title("speedup vs PyTorch"); ax.legend()


def draw_throughput(ax):
    thr = {k: 1e8 / (med[k][2]*1e-3) / 1e9 for k in (C, M, P)}
    ks = [C, M, P]; vals = [thr[k] for k in ks]
    bars = ax.bar(ks, vals, color=[col[k] for k in ks])
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                    (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("billion paths / s"); ax.set_title("throughput at 100M")
    ax.tick_params(axis="x", labelsize=9)


def draw_parity(ax):
    ratio = med[M] / med[C]
    ax.axhspan(0.95, 1.05, color="green", alpha=0.10, label="+/-5% band")
    ax.axhline(1.0, color="black", lw=1, ls="--")
    ax.plot(x, ratio, marker="o", lw=2, color=col[M])
    for xi, ri in zip(x, ratio):
        ax.annotate(f"{ri:.2f}x", (xi, ri), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.85, 1.15)
    ax.set_xlabel("Monte Carlo paths"); ax.set_ylabel("Mojo time / CUDA time")
    ax.set_title("Mojo vs CUDA parity"); ax.legend(loc="upper left")


def draw_stability(ax):
    w = 0.26
    for j, k in enumerate((C, M, P)):
        ax.bar(x + (j-1)*w, 100.0*std[k]/med[k], w, color=col[k], label=k)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Monte Carlo paths"); ax.set_ylabel("coeff. of variation (%)")
    ax.set_title("timing stability"); ax.legend()


def draw_tuning(ax):
    stages = ["baseline", "+unroll x8", "+block 64"]
    vals = [_tune["baseline"], _tune["unroll8"], _tune["block64"]]
    bars = ax.bar(stages, vals, color=["#f0a0a0", "#e06666", col[M]])
    ax.axhline(med[C][2], color=col[C], lw=2, ls="--",
               label=f"hand CUDA = {med[C][2]:.3f} ms")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.4f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("time at 100M (ms)"); ax.set_title("kernel tuning")
    ax.tick_params(axis="x", labelsize=9); ax.legend()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), dpi=600)
    plt.close(fig)
    print("wrote", name)


# ---- single panels (kept for reference) ----
for nm, fn in [("fig1_runtime_scaling", draw_scaling),
               ("fig2_speedup_pytorch", draw_speedup),
               ("fig3_mojo_cuda_parity", draw_parity),
               ("fig4_throughput", draw_throughput),
               ("fig5_timing_stability", draw_stability),
               ("fig6_tuning_progression", draw_tuning)]:
    f, a = plt.subplots(figsize=(6.2, 4.0)); fn(a); save(f, nm)

# ---- combined Figure A: performance (3 panels, two-column span) ----
fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.7))
for a, fn, lab in zip(axs, (draw_scaling, draw_speedup, draw_throughput),
                      ("(a)", "(b)", "(c)")):
    fn(a); panel_label(a, lab)
save(fig, "figA_performance")

# ---- combined Figure B: reproducibility and tuning (3 panels) ----
fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.7))
for a, fn, lab in zip(axs, (draw_parity, draw_stability, draw_tuning),
                      ("(a)", "(b)", "(c)")):
    fn(a); panel_label(a, lab)
save(fig, "figB_reproducibility")

print("done")
