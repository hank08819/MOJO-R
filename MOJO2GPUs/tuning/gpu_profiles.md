# Per-GPU kernel tuning profiles

The Mojo kernel reaches parity with hand-written CUDA only after two tuning knobs
are set (see the paper, Section "Results"). The optimal values are **GPU specific**:
they depend on register file size, SM count, and the throughput of the special
function units (SFUs) that evaluate `exp` / `log` / `cos` / `sqrt`.

The knobs are marked `TUNING` at the top of `src/mc_mojo.mojo`:

```mojo
alias BLOCK_SIZE = 64     # TUNING: threads per block (occupancy knob)
# TUNING: the inner loop is manually unrolled x8 (eight independent
# accumulators a0..a7 in mc_kernel). To change the unroll factor, add/remove
# accumulators and matching payoff() calls there; x8 is best on sm_89.
```

> The CUDA reference (`src/mc_cuda.cu`, `BLOCK_SIZE 256`) is already near optimal
> under `nvcc -O3`; only the Mojo side needs hand tuning. `nvcc` auto-selects the
> SM arch from `nvidia-smi` at build time, so CUDA needs no per-GPU edit here.

## How to tune for a new GPU

1. Build and run the default profile: `bash scripts/run_benchmark.sh`.
2. If Mojo trails CUDA at 100M paths by more than a few percent, sweep:
   - `BLOCK_SIZE` over {32, 64, 128, 256} (one-line edit),
   - the manual unroll factor over {4, 8, 16} (add/remove accumulators in `mc_kernel`).
   Rebuild Mojo each time (`mojo src/mc_mojo.mojo`) and keep the best median.
3. Record the winning pair in the table below and in that GPU's
   `results/<slug>/timings.json` (`mojo_block_size`, `mojo_unroll`).

## Known-good profiles

| GPU              | arch   | BLOCK_SIZE | UNROLL | status     |
|------------------|--------|-----------:|-------:|------------|
| RTX 6000 Ada     | sm_89  |         64 |      8 | measured   |
| RTX 4090         | sm_89  |         64 |      8 | starting point (same arch) — verify |
| H100             | sm_90  |        TBD |    TBD | to measure |
| A100             | sm_80  |        TBD |    TBD | to measure |

Rows marked "to measure" are placeholders. Run the sweep on that hardware and
replace TBD with the values that minimize the 100M-path median.
