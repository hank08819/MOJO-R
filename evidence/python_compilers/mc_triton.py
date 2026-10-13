"""Monte Carlo European call -- Triton kernel (R3-1 baseline, 2026-10-12).

Mirrors mc_cuda.cu: same index-addressed LCG -> Box-Muller -> discounted payoff.
Each of the 2^18 lanes plays the role of one CUDA thread: lane j sums paths
j, j + 2^18, j + 2*2^18, ... in one register accumulator and writes one partial
(no atomics, no in-kernel reduction). Fast-math transcendentals via libdevice.
The 2^18 partials are summed on the host in double, index order.
Timing: 5 warmups, then 50 timed launches; wall time of launch + synchronize.
"""
import json, os, sys, time
import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as ld

N_THREADS = 1 << 18
BLOCK = int(os.environ.get("BLOCK", "256"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
os.makedirs(OUT, exist_ok=True)


@triton.jit
def mc_kernel(part_ptr, n_paths, n_iter, S, K, r, sigma, T,
              NT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lane = pid * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    drift = (r - 0.5 * sigma * sigma) * T
    volT = sigma * tl.sqrt(T + tl.zeros([BLOCK], dtype=tl.float32))
    disc = ld.fast_expf(-r * T + tl.zeros([BLOCK], dtype=tl.float32))
    for k in range(0, n_iter):
        i = lane.to(tl.int64) + k * NT
        m = i < n_paths
        seed = (i.to(tl.uint32) * 2654435761) ^ 0xDEADBEEF
        s1 = seed * 1664525 + 1013904223
        s2 = s1 * 1664525 + 1013904223
        u1 = ((s1 >> 9) | 0x3F800000).to(tl.float32, bitcast=True) - 1.0 + 1e-7
        u2 = ((s2 >> 9) | 0x3F800000).to(tl.float32, bitcast=True) - 1.0
        mag = tl.sqrt(-2.0 * ld.fast_logf(u1))
        z = mag * ld.fast_cosf(2.0 * 3.14159265358979 * u2)
        ST = S * ld.fast_expf(drift + volT * z)
        pay = tl.maximum(ST - K, 0.0) * disc
        acc += tl.where(m, pay, 0.0)
    tl.store(part_ptr + lane, acc)


def main():
    part = torch.empty(N_THREADS, dtype=torch.float32, device="cuda")
    grid = (N_THREADS // BLOCK,)
    rows = []
    for n in [1_000_000, 10_000_000, 100_000_000]:
        n_iter = (n + N_THREADS - 1) // N_THREADS
        launch = lambda: mc_kernel[grid](part, n, n_iter, 100.0, 100.0, 0.05, 0.2, 1.0,
                                         NT=N_THREADS, BLOCK=BLOCK, num_warps=max(1, BLOCK // 32))
        for _ in range(5):
            launch()
        torch.cuda.synchronize()
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            launch()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ev = []
        for _ in range(10):   # 20 back-to-back launches per event pair: kernel time without host dispatch
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                launch()
            e1.record(); e1.synchronize()
            ev.append(e0.elapsed_time(e1) / 20)
        h = part.cpu().numpy()
        tot = 0.0
        for v in h.astype(np.float64):
            tot += float(v)
        price = tot / n
        np.save(os.path.join(OUT, f"triton_b{BLOCK}_{n}.npy"), h.view(np.uint32))
        rows.append(dict(impl=f"triton_b{BLOCK}", n=n, times_ms=ts, median_ms=float(np.median(ts)),
                         std_ms=float(np.std(ts)), event_ms=ev, event_median_ms=float(np.median(ev)), price=price, price_hex=price.hex()))
        print(f"triton b{BLOCK} n={n:>11,} median {np.median(ts):.4f} ms  std {np.std(ts):.4f}  price {price:.10f} {price.hex()}  event-median {np.median(ev):.4f}")
    json.dump(rows, open(os.path.join(OUT, f"triton_b{BLOCK}.json"), "w"))


if __name__ == "__main__":
    main()
