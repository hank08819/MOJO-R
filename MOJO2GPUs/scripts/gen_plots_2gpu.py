#!/usr/bin/env python3
"""Two-GPU scaling figure for the paper, from results/<slug>_x2/timings.json.

Panel (a): runtime at 100M paths, 1-GPU vs 2-GPU, for CUDA / Mojo / PyTorch (log y).
Panel (b): parallel speedup (1-GPU median / 2-GPU median) vs paths, with the ideal
           2x line; shows that the compiled kernels approach 2x only at large n while
           PyTorch, being far heavier, scales almost ideally.

Output dir: $FIGDIR (default: the paper's top-level figures/)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
                     "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
                     "axes.grid": True, "grid.alpha": 0.35, "savefig.bbox": "tight"})

with open(os.path.join(PKG_ROOT, "results", SLUG, "timings.json")) as f:
    D = json.load(f)

labels = ["1M", "10M", "100M"]
x = np.arange(len(labels))
C, M, P = "CUDA C", "Mojo", "PyTorch"
col = {C: "#1f77b4", M: "#d62728", P: "#7f7f7f"}
two = {C: np.array(D["cuda"]["median_ms"]), M: np.array(D["mojo"]["median_ms"]),
       P: np.array(D["pytorch"]["median_ms"])}
one = {k: np.array(v) for k, v in
       zip((C, M, P), (D["single_gpu_median_ms"]["cuda"],
                       D["single_gpu_median_ms"]["mojo"],
                       D["single_gpu_median_ms"]["pytorch"]))}


def panel_label(ax, s):
    ax.text(0.02, 0.97, s, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left")


def draw_runtime_100m(ax):
    ks = [C, M, P]
    w = 0.38
    v1 = [one[k][2] for k in ks]
    v2 = [two[k][2] for k in ks]
    b1 = ax.bar(np.arange(3) - w/2, v1, w, color=[col[k] for k in ks], alpha=0.55,
                label="1 GPU")
    b2 = ax.bar(np.arange(3) + w/2, v2, w, color=[col[k] for k in ks], label="2 GPU")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.3f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log"); ax.set_xticks(np.arange(3)); ax.set_xticklabels(ks)
    ax.set_ylabel("time at 100M paths (ms)"); ax.set_title("1 GPU vs 2 GPU at 100M")
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.4", alpha=0.55),
               plt.Rectangle((0, 0), 1, 1, color="0.4")]
    ax.legend(handles, ["1 GPU", "2 GPU"], loc="upper left")


def draw_speedup(ax):
    ax.axhline(2.0, color="black", lw=1, ls="--", label="ideal 2x")
    for k in (C, M, P):
        sp = one[k] / two[k]
        ax.plot(x, sp, marker="o", lw=2, color=col[k], label=k)
        for xi, s in zip(x, sp):
            ax.annotate(f"{s:.2f}x", (xi, s), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.6, 2.2)
    ax.set_xlabel("Monte Carlo paths"); ax.set_ylabel("parallel speedup (1 GPU / 2 GPU)")
    ax.set_title("two-GPU scaling"); ax.legend(loc="lower right")


fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.7))
draw_runtime_100m(axs[0]); panel_label(axs[0], "(a)")
draw_speedup(axs[1]); panel_label(axs[1], "(b)")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "figC_2gpu_scaling.pdf"))
fig.savefig(os.path.join(FIGDIR, "figC_2gpu_scaling.png"), dpi=600)
print("wrote figC_2gpu_scaling to", FIGDIR)
