#!/usr/bin/env python3
"""Cross-workload figure: speedup over PyTorch and Mojo/CUDA parity, per kernel.
Reads results/rtx6000ada/kernels_1gpu.json. Output -> $FIGDIR (default paper figures/)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.normpath(os.path.join(HERE, ".."))
FIGDIR = os.environ.get("FIGDIR") or os.path.normpath(os.path.join(PKG_ROOT, "..", "figures"))
os.makedirs(FIGDIR, exist_ok=True)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("default")
plt.rcParams.update({"figure.dpi": 600, "savefig.dpi": 600, "font.size": 11,
                     "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight"})

D = json.load(open(os.path.join(PKG_ROOT, "results", "rtx6000ada", "kernels_1gpu.json")))
kernels = [("european", "European\n(memory-bound)"), ("asian", "Asian\n(path-dependent)"),
           ("american", "American\n(tree)"), ("mlp", "MLP infer.\n(compute-bound)"),
           ("bermudan", "Bermudan\n(high-dim infer.)")]
colC, colM = "#1f77b4", "#d62728"

spdC, spdM, ratio = [], [], []
for key, _ in kernels:
    k = D[key]; i = len(k["cuda"]["median_ms"]) - 1
    c, m, p = k["cuda"]["median_ms"][i], k["mojo"]["median_ms"][i], k["pytorch"]["median_ms"][i]
    spdC.append(p / c); spdM.append(p / m); ratio.append(m / c)
x = np.arange(len(kernels))

fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))

# (a) speedup over PyTorch
w = 0.38
b1 = axs[0].bar(x - w/2, spdC, w, color=colC, label="CUDA C")
b2 = axs[0].bar(x + w/2, spdM, w, color=colM, label="Mojo")
for b in list(b1) + list(b2):
    axs[0].annotate(f"{b.get_height():.0f}x", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=9)
axs[0].set_yscale("log"); axs[0].set_xticks(x); axs[0].set_xticklabels([l for _, l in kernels])
axs[0].set_ylabel("speedup over PyTorch (x)")
axs[0].set_title("framework gap shrinks with arithmetic intensity")
axs[0].legend()
axs[0].text(0.02, 0.96, "(a)", transform=axs[0].transAxes, fontsize=12, fontweight="bold", va="top")

# (b) Mojo/CUDA parity (log y: the tree kernel breaks parity, the rest hold it)
axs[1].axhspan(0.95, 1.05, color="green", alpha=0.12, label="+/-5% parity band")
axs[1].axhline(1.0, color="black", lw=1, ls="--")
axs[1].plot(x, ratio, "o-", lw=2, color=colM)
for xi, ri in zip(x, ratio):
    axs[1].annotate(f"{ri:.2f}x", (xi, ri), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
axs[1].set_yscale("log")
axs[1].set_xticks(x); axs[1].set_xticklabels([l for _, l in kernels]); axs[1].set_ylim(0.9, 7)
axs[1].set_yticks([1, 2, 5]); axs[1].set_yticklabels(["1x", "2x", "5x"])
axs[1].set_ylabel("Mojo time / CUDA time"); axs[1].set_title("parity holds, except the two stateful kernels")
axs[1].legend(loc="upper left")
axs[1].text(0.02, 0.96, "(b)", transform=axs[1].transAxes, fontsize=12, fontweight="bold", va="top")

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "figE_kernels.pdf"))
fig.savefig(os.path.join(FIGDIR, "figE_kernels.png"), dpi=600)
print("wrote figE_kernels to", FIGDIR)
