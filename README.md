# Closing the Mojo GPU Gap in Financial AI

Code, run logs, and figure scripts for a precision matched, three way comparison of
**Mojo**, **hand written CUDA C**, and **idiomatic PyTorch** on five financial GPU
workloads.

Every number and figure in the paper is produced by the code in this repository from
the recorded runs under `*/results/`. The harness is multi GPU ready: results are
stored per GPU type, so the same protocol can be re run on an H100, A100, or RTX 4090
without overwriting earlier runs.

---

## Repository layout

| Package | What it covers |
| --- | --- |
| [`MOJO-GPU-reprocubility/`](MOJO-GPU-reprocubility/) | European Monte Carlo pricer on **one** GPU: the core three way comparison at 1M / 10M / 100M paths, plus the kernel tuning sweep. |
| [`MOJO-kernels/`](MOJO-kernels/) | The **four additional workloads**: Asian (path dependent MC), American (binomial tree), MLP inference, Bermudan (trained policy inference). |
| [`MOJO2GPUs/`](MOJO2GPUs/) | The **two GPU** data parallel study and the timing reproducibility index. |

Each package is self contained with the same shape:

```
<package>/
├── src/            the three implementations (.mojo / .cu / .py)
├── scripts/        run_*.sh (compile+run), collect_results.py, gen_plots*.py
├── results/<gpu>/  manifest.txt (provenance), run.txt (raw log), *.json (parsed)
├── pixi.toml       pinned Mojo 0.25.6.0 + Python 3.12
└── requirements.txt pinned PyTorch 2.5.1+cu121
```

---

## Where each paper result comes from

| Paper | Content | Produced by |
| --- | --- | --- |
| **Table 1** | Single GPU MC timing, 1M/10M/100M | `MOJO-GPU-reprocubility/results/rtx6000ada/timings.json` |
| **Table 2** | Two GPU timing and parallel speedup | `MOJO2GPUs/results/rtx6000ada_x2/timings.json` |
| **Table 3** | All five workloads | `MOJO-kernels/results/rtx6000ada/kernels_1gpu.json` |
| **Figure 1** | Mojo vs CUDA source listing | `MOJO-GPU-reprocubility/src/mc_mojo.mojo`, `src/mc_cuda.cu` |
| **Figure 2** | Runtime, speedup, throughput | `MOJO-GPU-reprocubility/scripts/gen_plots.py` |
| **Figure 3** | Kernel tuning toward CUDA parity | `MOJO-GPU-reprocubility/scripts/gen_plots.py` (tuning panel) |
| **Figure 4** | Timing reproducibility index | `MOJO2GPUs/scripts/gen_plots_ri.py` |
| **Figure 5** | Five workload comparison | `MOJO-kernels/scripts/gen_plots_kernels.py` |

Plot scripts read `results/<gpu>/*.json` and fall back to embedded RTX 6000 Ada
numbers if that file is absent, so the paper figures always reproduce even without
the hardware.

---

## Quick start

Each package runs the same way. Using the single GPU package as the example:

```bash
cd MOJO-GPU-reprocubility

# 1. Mojo toolchain (pinned by pixi.lock)
#    install pixi once: curl -fsSL https://pixi.sh/install.sh | bash
pixi install
pixi run mojo --version          # expect 0.25.6.0

# 2. PyTorch in its own venv (pinned by requirements.txt)
python3.12 -m venv .venv-torch
.venv-torch/bin/pip install -r requirements.txt

# 3. Compile and run all three implementations
pixi run env PYTORCH_PYTHON="$PWD/.venv-torch/bin/python" bash scripts/run_benchmark.sh

# 4. Regenerate the paper figures
FIGDIR=./figures python3 scripts/gen_plots.py
```

The run script detects the GPU, derives a slug from its name, records a provenance
manifest, auto detects the architecture for `nvcc`, runs all three implementations,
and parses the printed tables into `timings.json`.

No GPU? Skip step 3 and run step 4 directly: the plot scripts use the embedded
recorded numbers.

---

## The reproducibility protocol

The comparison is only meaningful if the conditions are pinned. Every run records
these to `results/<gpu>/manifest.txt`.

**Environment.** A driver or compiler upgrade silently changes the emitted elementary
functions, which is what makes the CUDA and Mojo results comparable at the bit level.

| | |
| --- | --- |
| GPU | NVIDIA RTX 6000 Ada Generation, `sm_89` (2 present) |
| Driver / CUDA driver | 535.309.01 / 12.2 |
| nvcc toolkit | 12.6.85 |
| Mojo | 0.25.6.0 |
| Python / PyTorch | 3.12.7 / 2.5.1+cu121 |
| CUDA build flags | `-O3 -use_fast_math` (math aligned with Mojo) |

**Numerics.** Randomness is *index addressed*: the draw for path `i` is a pure
function of `(i, seed)`, independent of how the GPU schedules threads. The reduction
uses a *fixed grid stride order* over `M = 2^18` resident threads. Together these make
the price bitwise reproducible run to run, and make the CUDA and Mojo prices bitwise
identical to each other. Correctness is anchored to the closed form Black Scholes
price, `c ≈ 10.45` for `S=K=100, r=0.05, σ=0.20, T=1`.

**Measurement.** Graphics clock locked at 2505 MHz (memory 10000 MHz) to remove boost
variation; 5 warm up launches discarded; `L = 50` timed launches; only the compute
kernel is timed, with device buffers allocated once outside the timed region. Reported
as median, standard deviation, and the coefficient of variation.

> The installed driver (CUDA 12.2) caps the usable Mojo release at 0.25.6.0; newer
> Mojo needs the CUDA 12.4+ symbol `cuFuncGetParamInfo`. On driver >= 550 you may
> raise the Mojo version in `pixi.toml`.

---

## Headline results (RTX 6000 Ada, GPU 0)

Median of 50 launches, milliseconds, at the largest size of each kernel.

| Workload | Size | CUDA C | Mojo | PyTorch | Mojo/CUDA | Mojo vs PyTorch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| European (MC) | 1e8 paths | 0.0938 | 0.0971 | 7.9695 | 1.04x | 82x |
| Asian (MC) | 1e7 paths | 0.7951 | 0.8250 | 34.8709 | 1.04x | 42x |
| American (binomial) | 1e6 options | 3.8932 | 20.326 | 759.187 | 5.22x | 37x |
| MLP inference | 1e8 samples | 22.185 | 23.384 | 235.255 | 1.05x | 10x |
| Bermudan (policy) | 1.6e7 paths | 0.7296 | 1.4644 | 140.366 | 2.01x | 96x |

CUDA C and Mojo run numerically identical algorithms and return **bitwise identical**
results. Mojo matches hand written CUDA on the three streaming kernels and trails on
the two stateful ones (American, Bermudan), while beating PyTorch throughout.

The PyTorch baseline is deliberately **idiomatic and eager mode**: the vectorized code
a quantitative researcher would actually write, not the fastest PyTorch attainable.
`torch.compile` and CUDA graphs were not evaluated.

---

## License

TBD before publication. Without a license file, default copyright applies and others
cannot legally reuse the code.
