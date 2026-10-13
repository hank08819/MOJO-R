# MOJO-GPU-reprocubility — Monte Carlo GPU Benchmark

Three implementations of the same Black-Scholes Monte Carlo option pricer
(Mojo, hand-written CUDA C, PyTorch), benchmarked side-by-side on an NVIDIA GPU
under a documented reproducibility protocol.

**Purpose:** Produce the GPU timing results and figures for the paper
"Closing the Productivity and Performance Gap in Financial AI: A Reproducible
Evaluation of Mojo against CUDA and PyTorch on the GPU." The harness is
**multi-GPU ready**: results are stored per GPU type, so the same protocol can be
re-run on an H100, A100, RTX 4090, etc. without overwriting earlier runs.

---

## Package layout

```
MOJO-GPU-reprocubility/
├── README.md                     this file
├── pixi.toml / pixi.lock         pinned Mojo 0.25.6.0 + Python 3.12 env
├── requirements.txt              pinned PyTorch (2.5.1+cu121)
├── src/
│   ├── mc_mojo.mojo              Mojo GPU kernel (the language being evaluated)
│   ├── mc_cuda.cu               CUDA C reference kernel (hand-tuned baseline)
│   └── mc_pytorch.py            PyTorch GPU baseline (framework-overhead baseline)
├── scripts/
│   ├── run_benchmark.sh         compile + run all three; writes results/<gpu>/
│   ├── collect_results.py       parse runs -> timings.json; cross-GPU table
│   └── gen_plots.py             paper figures from results/<gpu>/timings.json
├── tuning/
│   └── gpu_profiles.md          per-GPU kernel knobs (block size, unroll)
└── results/
    ├── SUMMARY.md               cross-GPU comparison table (generated)
    ├── rtx6000ada/              the run reported in the paper
    │   ├── manifest.txt run.txt timings.json
    ├── h100/  a100/  rtx4090/   reserved slots (PENDING.txt until run)
```

Each `results/<slug>/` holds `manifest.txt` (hardware/tool provenance), `run.txt`
(raw console output), and `timings.json` (structured numbers used by the plots and
the cross-GPU table). The `<slug>` is derived automatically from the GPU name.

---

## Using only one GPU (you have two)

All three files are already written to use **GPU 0 only**:

- `mc_cuda.cu` calls `cudaSetDevice(0)` at the top of `run()`.
- `mc_pytorch.py` sets `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` before
  importing torch.
- `mc_mojo.mojo` opens `DeviceContext(device_id=0)`.

The second GPU is never opened or allocated. You can verify with:

```bash
nvidia-smi dmon -s u    # watch GPU utilization; only GPU 0 should show activity
```

If you want to use the second GPU instead, change `0` to `1` in all three files.

---

## Setup

### 1. Check your GPUs

```bash
nvidia-smi
```

You should see two GPUs listed. Note the GPU model (e.g., H100, A100, RTX 4090)
and the CUDA compute capability (e.g., 9.0 for H100, 8.0 for A100).

### 2. Install the Mojo toolchain (via pixi)

The Mojo compiler and its Python are pinned in `pixi.toml` / `pixi.lock`:

```bash
# install pixi once if needed: curl -fsSL https://pixi.sh/install.sh | bash
pixi install                 # resolves Mojo 0.25.6.0 + Python 3.12 from the lockfile
pixi run mojo --version
```

> This repo pins **Mojo 0.25.6.0** because the installed NVIDIA driver is CUDA 12.2;
> newer Mojo needs the CUDA 12.4+ symbol `cuFuncGetParamInfo`. On a driver >= 550
> you may raise the Mojo version in `pixi.toml`.

### 3. Install PyTorch (separate venv)

PyTorch lives in its own Python 3.12 venv (pinned by `requirements.txt`):

```bash
python3.12 -m venv .venv-torch
.venv-torch/bin/pip install -r requirements.txt   # torch 2.5.1+cu121
```

### 4. Make the run script executable

```bash
chmod +x scripts/run_benchmark.sh
```

---

## Run everything

From the package root:

```bash
pixi run env PYTORCH_PYTHON="$PWD/.venv-torch/bin/python" bash scripts/run_benchmark.sh
```

This will:
1. Detect the GPU, derive a `<slug>` from its name, and create `results/<slug>/`.
2. Write a provenance manifest to `results/<slug>/manifest.txt`.
3. Auto-detect the GPU architecture and compile `src/mc_cuda.cu` (`-O3 -use_fast_math`).
4. Run the CUDA C, PyTorch, and Mojo benchmarks, teeing console output to
   `results/<slug>/run.txt`.
5. Parse the printed tables into `results/<slug>/timings.json`.

Each benchmark prints **min / median / std** of the per-launch wall time (over 50
launches) and the computed option price at 1M, 10M, and 100M paths.

---

## Run each separately

```bash
# CUDA C (compile first, then run); -use_fast_math aligns the math with Mojo
nvcc -O3 -arch=sm_89 -use_fast_math src/mc_cuda.cu -o mc_cuda   # RTX 6000 Ada / 4090
./mc_cuda

# PyTorch (from the torch venv)
.venv-torch/bin/python src/mc_pytorch.py

# Mojo (through pixi)
pixi run mojo src/mc_mojo.mojo
```

---

## Testing on other GPU types

The harness is built to be run on several GPUs and keep the results side by side.

```bash
# on each machine, just run the one-shot script -- it figures out the rest
pixi run env PYTORCH_PYTHON="$PWD/.venv-torch/bin/python" bash scripts/run_benchmark.sh
```

- The GPU **architecture** (`sm_80`/`sm_89`/`sm_90`/...) is detected from
  `nvidia-smi` and passed to `nvcc`; no edit needed for CUDA.
- Results land in a **per-GPU folder** `results/<slug>/`, so an H100 run never
  overwrites the RTX 6000 Ada run. Reserved slots `results/{h100,a100,rtx4090}/`
  ship with a `PENDING.txt` placeholder; a real run replaces it.
- The **Mojo kernel** has two per-GPU tuning knobs (block size and unroll factor).
  Defaults are tuned for `sm_89`; see [`tuning/gpu_profiles.md`](tuning/gpu_profiles.md)
  for how to sweep them on new hardware.

After collecting runs, build the cross-GPU comparison table and (optionally)
per-GPU figures:

```bash
python3 scripts/collect_results.py --table --out results/SUMMARY.md   # all GPUs
GPU_SLUG=h100 FIGDIR=../figures python3 scripts/gen_plots.py           # one GPU's figures
```

`gen_plots.py` reads `results/$GPU_SLUG/timings.json` (default `rtx6000ada`, the GPU
in the paper) and falls back to embedded RTX 6000 Ada numbers if that file is absent,
so the paper figures always reproduce.

---

## Example output (measured on RTX 6000 Ada, GPU 0)

Per-launch wall time over 50 launches, reported as min / median / std (ms).
GPU clocks locked (graphics 2505 MHz, memory 10000 MHz); CUDA built with
`-use_fast_math` so its transcendentals match Mojo's; the Mojo kernel tuned
(loop unrolled x8, 64-thread blocks) — a fair, math-aligned comparison. See
`MOJOGPU-reprocubility.pdf` for the full report.

```
GPUs detected: 2  -- using GPU 0 only
GPU 0: NVIDIA RTX 6000 Ada Generation

       Paths     min(ms)    median(ms)     std(ms)     Price       <- CUDA C
------------------------------------------------------------
     1000000      0.0071        0.0074      0.0008   10.4561
    10000000      0.0149        0.0153      0.0012   10.4517
   100000000      0.0932        0.0938      0.0045   10.4502

Device : NVIDIA RTX 6000 Ada Generation                           <- PyTorch
       Paths     min(ms)    median(ms)     std(ms)     Price
--------------------------------------------------------------
   1,000,000      0.0730        0.0823      0.0039   10.4380
  10,000,000      0.2670        0.2695      0.0018   10.4475
 100,000,000      7.9470        7.9695      0.0089   10.4504

Device: GPU 0                                                     <- Mojo (tuned)
       Paths     min(ms)    median(ms)     std(ms)     Price
------------------------------------------------------------
     1000000      0.0069        0.0072      0.0002   10.4561
    10000000      0.0143        0.0146      0.0005   10.4517
   100000000      0.0967        0.0971      0.0006   10.4502
```

All prices converge to approximately **10.45** (the analytical Black-Scholes call
price for S=100, K=100, r=5%, σ=20%, T=1yr). CUDA C and Mojo use the same
fixed-seed algorithm and produce **bit-identical prices** (even with `-use_fast_math`);
PyTorch uses its own seeded RNG (`SEED=12345`) and differs within Monte Carlo
sampling error.

With math aligned and the Mojo kernel tuned, **CUDA and Mojo are at parity** —
within a few percent, each leading at some sizes (Mojo wins 1M/10M, CUDA wins 100M
by ~3.5%). Both fused, register-resident kernels (CUDA ~0.094 ms, Mojo ~0.097 ms at
100M) are roughly **82–85x faster** than idiomatic PyTorch (~7.97 ms), which
materializes several full 400 MB intermediate tensors and is memory-bandwidth bound.
Before tuning Mojo was ~1.24x slower than CUDA; explicit unrolling + occupancy
tuning (x8 unroll, 64-thread blocks) closed the gap.

---

## Troubleshooting

**`nvcc: command not found`**
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

**`mojo: command not found`**
Run Mojo through pixi: `pixi run mojo ...` (or enter `pixi shell` first).

**Mojo `Failed to find symbol "cuFuncGetParamInfo"`**
The NVIDIA driver is too old (CUDA 12.2) for Mojo >= 0.26. This repo pins Mojo
0.25.6.0 in `pixi.toml` to avoid it; upgrade the driver to >= 550 for newer Mojo.

**Mojo GPU kernel syntax errors**
The GPU API names are version-specific. This code targets **Mojo 0.25.6.0**:
`DeviceContext(device_id=0)`, `enqueue_function[kernel](...)`,
`enqueue_create_buffer`, `enqueue_create_host_buffer`.

---

## What to put in the paper

Once you have the numbers, update the paper as follows:

1. **Table 2** (new): copy the timing rows for 1M / 10M / 100M paths.
2. **Figure 3 panel (a)** caption: change "projected" to "measured on [GPU model]."
3. **Section 5**: add one paragraph: "Table 2 reports direct GPU measurements
   on an NVIDIA [model]. Mojo GPU reached X× the throughput of PyTorch and
   was within Y% of hand-written CUDA C."
4. **Limitations**: remove the sentence "We did not benchmark Mojo's GPU
   path end-to-end against CUDA on dedicated NVIDIA hardware."

---

## Hardware used

- GPU model: NVIDIA RTX 6000 Ada Generation (2 present; GPU 0 used, sm_89)
- CUDA version: 12.2 (driver runtime) / 12.6.85 (nvcc toolkit)
- CUDA build flags: `-O3 -use_fast_math` (math aligned with Mojo)
- Driver version: 535.309.01
- Mojo version: 0.25.6.0
- PyTorch version: 2.5.1+cu121 (Python 3.12.7)
- GPU clocks: locked at graphics 2505 MHz, memory 10000 MHz
- Date: 2026-06-15

> Provenance is recorded automatically to `results/manifest.txt` on every run.
> Note: this driver (CUDA 12.2) caps the usable Mojo release at 0.25.6.0 — newer
> Mojo needs the CUDA 12.4+ symbol `cuFuncGetParamInfo`, absent from driver 535.
