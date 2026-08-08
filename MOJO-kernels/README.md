# MOJO-kernels — Generality across Financial-AI Workloads

Extends the European-option benchmark (`MOJO-GPU-reprocubility`) to two further
workloads of increasing arithmetic intensity, in all three languages
(Mojo / CUDA C / PyTorch), to show that Mojo's CUDA parity is not specific to one
kernel. Produces the "Generality across Workloads" section of the paper.

## The three workloads

| Workload | What | Profile |
|----------|------|---------|
| European call (in the sibling package) | one draw + payoff per path | memory-bound |
| **Asian call** (`asian_*`) | arithmetic average over an `m=100`-step path; **no closed form** | path-dependent, heavier |
| **American put** (`american_*`) | early exercise via CRR binomial tree (`STEPS=128`), backward induction, one option/thread | tree, large per-thread working set |
| **MLP inference** (`mlp_*`) | 2-layer perceptron `16->64->1`, tanh, over a batch | compute-bound (GEMM-like) |
| **Bermudan** (`bermudan_*`) | d=5 high-dim optimal-stopping (LSM) inference; policy **trained in PyTorch** (`scripts/export_policy.py`), inferred on GPU | stateful, branch-heavy; train->deploy |

## Layout

```
MOJO-kernels/
├── src/  asian_{cuda.cu,mojo.mojo,pytorch.py}   mlp_{cuda.cu,mojo.mojo,pytorch.py}
├── scripts/  run_kernels.sh   gen_plots_kernels.py
└── results/rtx6000ada/  run.txt   kernels_1gpu.json
```

## Run

```bash
pixi run --manifest-path ../MOJO-GPU-reprocubility/pixi.toml \
    env PYTORCH_PYTHON="$PWD/../.venv-torch/bin/python" bash scripts/run_kernels.sh
FIGDIR=../figures python3 scripts/gen_plots_kernels.py   # figE
```

## Measured results (1x RTX 6000 Ada, FP32, clocks locked 2505 MHz)

Median ms over 50 launches, at the largest size of each kernel. CUDA & Mojo are
**bit-identical** (price for the options, mean output for the MLP); PyTorch uses its
own RNG / weight init.

| Workload | Size | CUDA C | Mojo | PyTorch | Mojo/CUDA | Mojo vs PyTorch | CUDA vs PyTorch |
|----------|-----:|-------:|-----:|--------:|----------:|----------------:|----------------:|
| European | 1e8  | 0.0938 | 0.0971 | 7.9695  | 1.04x | 82.1x | 85.0x |
| Asian    | 1e7  | 0.7951 | 0.8250 | 34.8709 | 1.04x | 42.3x | 43.9x |
| American | 1e6  | 3.8932 | 20.326 | 759.187 | **5.22x** | 37.4x | 195.0x |
| MLP      | 1e8  | 22.185 | 23.384 | 235.255 | 1.05x | 10.1x | 10.6x |
| Bermudan | 1.6e7| 0.7296 | 1.4644 | 140.366 | **2.01x** | 96.0x | 192.4x |

**Three findings.**
1. **Mojo holds CUDA parity on the streaming kernels** (European/Asian/MLP, within ~5%),
   at bit-identical results.
2. **The American tree is the exception**: Mojo trails CUDA ~5.2x there. Same bit-identical
   arithmetic, but the large per-thread backward-induction buffer is not placed as well by
   Mojo 0.25.6 as by nvcc. Mojo still beats PyTorch ~37x.
3. **The PyTorch (framework) gap shrinks with arithmetic intensity**: ~82x (European,
   memory-bound) -> ~42x (Asian) -> ~10x (MLP, matmul-dominated, cuBLAS near peak).

**Statistical tests** (50 launches/cell, computed in `gen_plots_kernels` companion notes):
- *Speedup* (Welch t, Mojo vs PyTorch): t = 5.3e3..1.0e4, p < 1e-300 on all four kernels.
- *Parity* (TOST, +/-10% vs CUDA): EQUIVALENT on European/Asian/MLP (p<0.001), not on American.
- *Reproducibility* (F-test of timing variance vs CUDA): F~0.02 (European, Mojo less variance),
  F~1.1 (Asian/MLP, indistinguishable), F~1257 (American, Mojo more variable).

> Single-GPU here. The 2-GPU split + reproducibility-index treatment (`MOJO2GPUs`) applies
> unchanged; extending these kernels to two GPUs is the natural next step.
