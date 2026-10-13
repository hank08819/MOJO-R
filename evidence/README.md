# Camera-ready evidence (October 2026)

This folder backs the reproducibility statements in the ICAIF '26 camera-ready paper. It holds two checks. Both ran on an NVIDIA RTX 6000 Ada with driver 580.178.04, nvcc 12.6 (`-O3 -use_fast_math`), Mojo 0.25.6.0, and PyTorch 2.5.1. The paper's Tables 1 to 3 come from the June 2026 runs (driver 535.309.01) in the three benchmark folders of this repository.

## 1. `bitcheck/`: bit-level comparison of CUDA C and Mojo

`patch_dump.py` adds a dump of every per-thread partial sum to copies of the benchmark kernels. It leaves the kernel code unchanged. `compare_bits.py` then compares the dumps bit for bit. It rebuilds each price the way the programs do: a sequential double-precision sum of the float32 partials, divided by n.

`cuda_vs_mojo_bits.txt` gives the result for all five kernels, every problem size, and one versus two GPUs. Prices are shown in hexadecimal.

- Run to run: each implementation ran twice. The partial-sum dumps are byte-identical across the two runs (`dump_sha256_run1.txt` and `dump_sha256_run2.txt` list the same hashes).
- European kernel, 10^6 paths: CUDA C and Mojo agree on every partial sum and on the price, bit for bit. They agree despite different block sizes (256 and 64).
- European kernel at 10^7 and 10^8 paths, and the other four kernels: CUDA C and Mojo prices differ in the last bits. All relative differences are below 5e-7, and below 1e-9 for the European kernel. At 10^7 and 10^8 the cause is the reduction order: CUDA C uses one accumulator per thread, while the tuned Mojo kernel uses eight. The other four kernels already differ in the per-path arithmetic.
- Two GPUs versus one GPU: the prices differ by about 1e-9 absolute, because the split changes which paths each thread sums.
- Seeded PyTorch (`torch_bits.py`, run next to `MOJO-GPU-reprocubility/src/mc_pytorch.py`): two separate runs give bit-identical prices.

## 2. `python_compilers/`: CUDA C, Mojo, PyTorch, Numba, Triton and JAX on the European kernel

All five implementations ran in one session, with the graphics clock locked at 2505 MHz (`run_locked.sh`). `mc_numba.py` and `mc_triton.py` mirror `mc_cuda.cu`: the same index-addressed LCG and Box-Muller, the same 2^18 lanes, one accumulator per lane, and fast-math transcendentals. Neither uses atomics.

- `raw/*.json`: every per-launch wall time (50 launches after 5 warm-ups), GPU-event kernel times (20 back-to-back launches per event pair), and the price in hexadecimal. `raw_repeat/` holds a second run.
- `results_locked.log`: the full log. The CUDA C, Mojo and PyTorch per-launch times are on its `RAW` lines; the CUDA C event times are on its `EVT` lines.
- `summary_locked.json` and `summarize.py`: the medians and standard deviations reported in the paper's Table 4.
- `partials_sha256.txt`: hashes of the partial-sum arrays. Numba's partial sums match CUDA C bit for bit at every size; Triton's do not. Both are identical across the two runs.

Table 4 reports medians of the per-launch wall time. Numba and Triton use their best block size (64 and 128).

JAX (0.10.2, `mc_jax.py`, `run_jax_locked.sh`) ran in a second clock-locked session, about 50 minutes later. CUDA C and Mojo were re-run there as anchors and reproduced their Table 4 times to within 0.2%. `raw_jax/jax_vec.json` holds the idiomatic vectorized `jax.jit` version, which is the one reported, and `raw_jax/jax_loop.json` holds a `lax.fori_loop` version that mirrors CUDA's per-lane loop. XLA evaluates exp, log and cos without CUDA's fast-math approximations, so JAX prices differ from CUDA C by about 2e-7 relative. Two runs give identical partial sums (`partials_sha256_jax.txt`).

The raw partial-sum dumps (about 110 MB) and the `.npy` arrays are not stored here. Only their SHA-256 hashes are. Regenerate them with `patch_dump.py` and `run_locked.sh`, then compare the hashes.
