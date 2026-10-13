"""Monte Carlo European call -- Numba CUDA kernel (R3-1 baseline, 2026-10-12).

Mirrors mc_cuda.cu: same index-addressed LCG -> Box-Muller -> discounted payoff,
grid-stride over 2^18 resident threads, one register accumulator per thread, one
partial sum per thread, fast-math transcendentals (libdevice fast_expf/logf/cosf,
approximate sqrt), the 2^18 partials summed on the host in double, index order.
Timing: 5 warmups, then 50 timed launches; wall time of launch + synchronize.
Writes raw per-launch times, the price (hex) and the partials (uint32 .npy).
"""
import json, math, os, sys, time
import numpy as np
from numba import cuda, float32, uint32, int32
from numba.cuda import libdevice as ld

N_THREADS = 1 << 18
BLOCK = int(os.environ.get("BLOCK", "256"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
os.makedirs(OUT, exist_ok=True)
MASK = 0xFFFFFFFF


@cuda.jit(device=True)
def lcg(s):
    return uint32((s * uint32(1664525) + uint32(1013904223)) & MASK)


@cuda.jit(device=True)
def uniformf(s):
    return ld.int_as_float(int32((s >> uint32(9)) | uint32(0x3F800000))) - float32(1.0)


@cuda.jit(device=True)
def normalf(seed):
    s1 = lcg(seed)
    s2 = lcg(s1)
    u1 = uniformf(s1) + float32(1e-7)
    u2 = uniformf(s2)
    mag = math.sqrt(float32(-2.0) * ld.fast_logf(u1))
    return mag * ld.fast_cosf(float32(2.0) * float32(3.14159265358979) * u2)


@cuda.jit(fastmath=True)
def mc_kernel(partials, n_paths, S, K, r, sigma, T):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid >= N_THREADS:
        return
    acc = float32(0.0)
    i = tid
    while i < n_paths:
        seed = uint32(((uint32(i & MASK) * uint32(2654435761)) & MASK) ^ uint32(0xDEADBEEF))
        z = normalf(seed)
        ST = S * ld.fast_expf((r - float32(0.5) * sigma * sigma) * T + sigma * math.sqrt(T) * z)
        acc += ld.fmaxf(ST - K, float32(0.0)) * ld.fast_expf(-r * T)
        i += N_THREADS
    partials[tid] = acc


def main():
    d_part = cuda.device_array(N_THREADS, dtype=np.float32)
    grid = N_THREADS // BLOCK
    args = (np.float32(100.0), np.float32(100.0), np.float32(0.05), np.float32(0.2), np.float32(1.0))
    rows = []
    for n in [1_000_000, 10_000_000, 100_000_000]:
        for _ in range(5):
            mc_kernel[grid, BLOCK](d_part, n, *args)
        cuda.synchronize()
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            mc_kernel[grid, BLOCK](d_part, n, *args)
            cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ev = []
        for _ in range(10):   # 20 back-to-back launches per event pair: kernel time without host dispatch
            e0, e1 = cuda.event(), cuda.event()
            e0.record()
            for _ in range(20):
                mc_kernel[grid, BLOCK](d_part, n, *args)
            e1.record(); e1.synchronize()
            ev.append(cuda.event_elapsed_time(e0, e1) / 20)
        h = d_part.copy_to_host()
        tot = 0.0
        for v in h.astype(np.float64):
            tot += float(v)
        price = tot / n
        np.save(os.path.join(OUT, f"numba_b{BLOCK}_{n}.npy"), h.view(np.uint32))
        rows.append(dict(impl=f"numba_b{BLOCK}", n=n, times_ms=ts, median_ms=float(np.median(ts)),
                         std_ms=float(np.std(ts)), event_ms=ev, event_median_ms=float(np.median(ev)), price=price, price_hex=price.hex()))
        print(f"numba b{BLOCK} n={n:>11,} median {np.median(ts):.4f} ms  std {np.std(ts):.4f}  price {price:.10f} {price.hex()}  event-median {np.median(ev):.4f}")
    json.dump(rows, open(os.path.join(OUT, f"numba_b{BLOCK}.json"), "w"))


if __name__ == "__main__":
    main()
