#!/usr/bin/env python3
"""Fancy reproducibility-index (RI) figure for the paper.

RI = 100 * std/median (%) of the per-launch time = coefficient of variation; lower is
more timing-reproducible. Three panels:
  (a) mean RI per implementation, 1-GPU vs 2-GPU grouped bars (+ "more reproducible" arrow),
  (b) RI vs problem size, 1-GPU dashed / 2-GPU solid, one color per implementation,
  (c) annotated heatmap of RI over (implementation x setup) x problem size.

2-GPU numbers come from results/<slug>/timings.json; 1-GPU from the paper run.
Output dir: $FIGDIR (default: the paper's top-level figures/)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.normpath(os.path.join(HERE, ".."))
SLUG = os.environ.get("GPU_SLUG", "rtx6000ada_x2")
FIGDIR = os.environ.get("FIGDIR") or os.path.normpath(os.path.join(PKG_ROOT, "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("default")
plt.rcParams.update({"figure.dpi": 600, "savefig.dpi": 600, "font.size": 11,
                     "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
                     "axes.grid": True, "grid.alpha": 0.30, "savefig.bbox": "tight"})

with open(os.path.join(PKG_ROOT, "results", SLUG, "timings.json")) as f:
    D = json.load(f)

labels = ["1M", "10M", "100M"]
x = np.arange(len(labels))
IMPL = ["CUDA C", "Mojo", "PyTorch"]
col = {"CUDA C": "#1f77b4", "Mojo": "#d62728", "PyTorch": "#7f7f7f"}

# 1-GPU std/median (paper run, rtx6000ada)
one = {"CUDA C":  ([0.0008,0.0012,0.0045], [0.0074,0.0153,0.0938]),
       "Mojo":    ([0.0002,0.0005,0.0006], [0.0072,0.0146,0.0971]),
       "PyTorch": ([0.0039,0.0018,0.0089], [0.0823,0.2695,7.9695])}
# 2-GPU std/median (this package's run)
two = {"CUDA C":  (D["cuda"]["std_ms"],    D["cuda"]["median_ms"]),
       "Mojo":    (D["mojo"]["std_ms"],    D["mojo"]["median_ms"]),
       "PyTorch": (D["pytorch"]["std_ms"], D["pytorch"]["median_ms"])}


def ri(std, med):
    return np.array([100.0*s/m for s, m in zip(std, med)])


RI1 = {k: ri(*one[k]) for k in IMPL}
RI2 = {k: ri(*two[k]) for k in IMPL}


def panel_label(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left")


# ---- (a) mean RI per implementation, 1 vs 2 GPU ----
def draw_mean(ax):
    w = 0.36
    m1 = [RI1[k].mean() for k in IMPL]
    m2 = [RI2[k].mean() for k in IMPL]
    b1 = ax.bar(x - w/2, m1, w, color=[col[k] for k in IMPL], alpha=0.55)
    b2 = ax.bar(x + w/2, m2, w, color=[col[k] for k in IMPL])
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.1f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(IMPL)
    ax.set_ylabel("mean RI  (%)"); ax.set_title("mean reproducibility index")
    ax.annotate("more reproducible", xy=(0.03, 0.55), xytext=(0.03, 0.86),
                xycoords="axes fraction", textcoords="axes fraction", fontsize=8.5,
                ha="left", arrowprops=dict(arrowstyle="->", color="0.3"))
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.4", alpha=0.55),
               plt.Rectangle((0, 0), 1, 1, color="0.4")]
    ax.legend(handles, ["1 GPU", "2 GPU"], loc="upper right")


# ---- (b) RI vs problem size ----
def draw_lines(ax):
    for k in IMPL:
        ax.plot(x, RI1[k], marker="o", lw=2, ls="--", color=col[k], alpha=0.9,
                label=f"{k} (1 GPU)")
        ax.plot(x, RI2[k], marker="s", lw=2, ls="-", color=col[k],
                label=f"{k} (2 GPU)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Monte Carlo paths"); ax.set_ylabel("RI  (%)")
    ax.set_title("RI across problem size"); ax.legend(ncol=1, fontsize=7.5, loc="upper right")


# ---- (c) heatmap ----
def draw_heat(ax):
    rows = [(k, g) for k in IMPL for g in ("1", "2")]
    M = np.array([(RI1 if g == "1" else RI2)[k] for (k, g) in rows])
    cmap = plt.get_cmap("YlOrRd")
    im = ax.imshow(M, aspect="auto", cmap=cmap, norm=mcolors.PowerNorm(0.6))
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{k}  {g}GPU" for (k, g) in rows], fontsize=8.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="black" if v < M.max()*0.55 else "white")
    ax.set_title("RI heatmap (%)"); ax.grid(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("RI (%)", fontsize=9)


fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.8))
draw_mean(axs[0]);  panel_label(axs[0], "(a)")
draw_lines(axs[1]); panel_label(axs[1], "(b)")
draw_heat(axs[2]);  panel_label(axs[2], "(c)")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "figD_reproducibility_index.pdf"))
fig.savefig(os.path.join(FIGDIR, "figD_reproducibility_index.png"), dpi=600)
print("wrote figD_reproducibility_index to", FIGDIR)
