# MOJO2GPUs — Two-GPU Monte Carlo Benchmark (Mojo / CUDA C / PyTorch)

Data-parallel **two-GPU** version of the Black-Scholes Monte Carlo option pricer,
the companion to the single-GPU `MOJO-GPU-reprocubility` package. It produces the
"Scaling to Two GPUs" section of the paper *Closing the Mojo GPU Gap in Financial AI*.

The n paths are split into two contiguous **global-index** ranges
(`GPU 0 -> [0, n/2)`, `GPU 1 -> [n/2, n)`); each device runs the same kernel over its
range using global indices, the two kernels run **concurrently**, and the host adds
the two device totals in a fixed order. Because every payoff is still addressed by its
global index and the combine order is fixed, the determinism and pathwise-identity
theorems of the paper carry over: CUDA and Mojo return **bit-identical prices**, equal
to the single-GPU run.

---

## Layout

```
MOJO2GPUs/
├── README.md
├── pixi.toml / pixi.lock / requirements.txt   pinned env (Mojo 0.25.6.0, torch 2.5.1)
├── src/
│   ├── mc_cuda_2gpu.cu        CUDA C, 2-GPU concurrent launch
│   ├── mc_mojo_2gpu.mojo      Mojo, two DeviceContexts (device 0 + 1)
│   └── mc_pytorch_2gpu.py     PyTorch, cuda:0 + cuda:1 data-parallel
├── scripts/
│   ├── run_benchmark_2gpu.sh  compile + run all three on 2 GPUs; writes results/<slug>_x2/
│   ├── collect_results.py     parse run.txt -> timings.json; cross-GPU table
│   ├── gen_plots_2gpu.py      1-GPU vs 2-GPU scaling figure (figC)
│   └── gen_plots_ri.py        reproducibility-index figure (figD)
├── tuning/gpu_profiles.md     per-GPU kernel knobs (shared with single-GPU package)
└── results/
    └── rtx6000ada_x2/         manifest.txt · run.txt · timings.json · binaries
```

---

## Run

From the package root (needs **2 GPUs**, `nvcc`, the pinned Mojo via `pixi`, and a
PyTorch venv at `../.venv-torch`):

```bash
pixi run --manifest-path ../MOJO-GPU-reprocubility/pixi.toml \
    env PYTORCH_PYTHON="$PWD/../.venv-torch/bin/python" \
    bash scripts/run_benchmark_2gpu.sh
```

(or copy `pixi.toml`/`pixi.lock` here and `pixi install` to make it standalone.)

Then regenerate the paper figures:

```bash
FIGDIR=../figures python3 scripts/gen_plots_2gpu.py    # figC: 1 vs 2 GPU scaling
FIGDIR=../figures python3 scripts/gen_plots_ri.py      # figD: reproducibility index
```

---

## Measured results (2x RTX 6000 Ada, FP32, clocks locked 2505 MHz)

Median per-launch time over 50 launches (ms); CUDA & Mojo prices bit-identical (~10.45).

| Paths | CUDA C | Mojo | PyTorch | Mojo/CUDA | parallel speedup (Mojo, 1->2 GPU) |
|------:|-------:|-----:|--------:|----------:|----------------------------------:|
| 1e6   | 0.0070 | 0.0083 | 0.1019 | 1.19x | 0.87x |
| 1e7   | 0.0109 | 0.0115 | 0.1561 | 1.06x | 1.27x |
| 1e8   | 0.0500 | 0.0526 | 4.0423 | 1.05x | 1.85x |

### Speedup ratios at 1e8 paths (two GPUs)
- **Mojo vs CUDA:** 1.05x (CUDA ~5% faster — parity)
- **Mojo vs PyTorch:** 76.8x faster
- **CUDA vs PyTorch:** 80.8x faster
- **Two-GPU parallel speedup at 1e8:** CUDA 1.88x, Mojo 1.85x, PyTorch 1.97x (efficiency 0.94 / 0.93 / 0.98)

### Reproducibility index  RI = 100 x std/median (%)  (lower = more reproducible)
| Impl | 1 GPU mean | 2 GPU mean | 1e8 (1 GPU) | 1e8 (2 GPU) |
|------|-----------:|-----------:|------------:|------------:|
| CUDA C  | 7.82 | 2.55 | 4.80 | 0.60 |
| Mojo    | 2.27 | 4.89 | 0.62 | 1.14 |
| PyTorch | 1.84 | 1.62 | 0.11 | 0.21 |

**Takeaways.** Two GPUs nearly halve the 1e8 runtime of every implementation
(~1.9x), and Mojo stays at parity with hand-written CUDA on two devices just as on one,
at identical numerical results. The compiled kernels' larger RI at small sizes is an
artifact of microsecond-scale absolute times; at 1e8 all are at or below ~1.2%.

> Single-GPU baselines come from the sibling `MOJO-GPU-reprocubility` package.
