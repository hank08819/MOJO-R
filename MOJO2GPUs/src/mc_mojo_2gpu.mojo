# Monte Carlo option pricer -- Mojo GPU kernel, TWO-GPU data-parallel version.
#
# Splits the n paths into two contiguous GLOBAL-index ranges:
#   GPU 0 -> [0, n/2),  GPU 1 -> [n/2, n)
# Each device opens its own DeviceContext and runs the same tuned grid-stride kernel
# (inner loop unrolled x8, 64-thread blocks) over its range using global path indices,
# so the per-path seeding/payoff is identical to the single-GPU run. The two device
# kernels are enqueued back-to-back (concurrent) and then both synchronized; the host
# adds the two device totals in a fixed order (device 0 then device 1).
#
# Run:
#   pixi run mojo mc_mojo_2gpu.mojo

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from math import exp, sqrt, log, cos, sin
from memory import UnsafePointer, bitcast
from time import perf_counter_ns

# ── constants ────────────────────────────────────────────────────────────────
alias S     = Float32(100.0)
alias K     = Float32(100.0)
alias R     = Float32(0.05)
alias SIGMA = Float32(0.2)
alias T     = Float32(1.0)
alias PI    = Float32(3.14159265358979)

alias N_THREADS  = 1 << 18              # 262,144 resident threads PER GPU
alias BLOCK_SIZE = 64                   # TUNING (per-GPU): see tuning/gpu_profiles.md
alias GRID_SIZE  = N_THREADS // BLOCK_SIZE
alias NGPU       = 2

# ── per-thread LCG + Box-Muller normal (identical to the single-GPU kernel) ───
@always_inline
fn lcg(seed: UInt32) -> UInt32:
    return seed * UInt32(1664525) + UInt32(1013904223)

@always_inline
fn uniform(seed: UInt32) -> Float32:
    return bitcast[DType.float32, 1]((seed >> 9) | UInt32(0x3F800000)) - Float32(1.0)

@always_inline
fn normal(seed: UInt32) -> Float32:
    var s1 = lcg(seed)
    var s2 = lcg(s1)
    var u1 = uniform(s1) + Float32(1e-7)
    var u2 = uniform(s2)
    var mag = sqrt(-Float32(2.0) * log(u1))
    return mag * cos(Float32(2.0) * PI * u2)

@always_inline
fn payoff(i: Int) -> Float32:
    var seed = UInt32(i) * UInt32(2654435761) ^ UInt32(0xDEADBEEF)
    var z    = normal(seed)
    var ST   = S * exp((R - Float32(0.5) * SIGMA * SIGMA) * T
                       + SIGMA * sqrt(T) * z)
    return max(ST - K, Float32(0.0)) * exp(-R * T)

# ── GPU kernel: grid-stride over a GLOBAL index range [lo, hi), x8 unrolled ────
fn mc_kernel(partials: UnsafePointer[Float32, origin=MutableAnyOrigin], lo: Int, hi: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= N_THREADS:
        return
    var a0 = Float32(0.0)
    var a1 = Float32(0.0)
    var a2 = Float32(0.0)
    var a3 = Float32(0.0)
    var a4 = Float32(0.0)
    var a5 = Float32(0.0)
    var a6 = Float32(0.0)
    var a7 = Float32(0.0)
    var i  = lo + tid
    while i + 7 * N_THREADS < hi:                       # unrolled main body (x8)
        a0 += payoff(i)
        a1 += payoff(i + N_THREADS)
        a2 += payoff(i + 2 * N_THREADS)
        a3 += payoff(i + 3 * N_THREADS)
        a4 += payoff(i + 4 * N_THREADS)
        a5 += payoff(i + 5 * N_THREADS)
        a6 += payoff(i + 6 * N_THREADS)
        a7 += payoff(i + 7 * N_THREADS)
        i += 8 * N_THREADS
    while i < hi:                                        # tail
        a0 += payoff(i)
        i += N_THREADS
    partials[tid] = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))

# ── tiny formatting helpers ───────────────────────────────────────────────────
fn rjust(s: String, width: Int) -> String:
    var out = s
    while len(out) < width:
        out = " " + out
    return out

fn fixed(v: Float64, decimals: Int) -> String:
    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var scaled = Int(round(v * Float64(scale)))
    var neg = scaled < 0
    if neg:
        scaled = -scaled
    var frac = String(scaled % scale)
    while len(frac) < decimals:
        frac = "0" + frac
    var res = String(scaled // scale) + "." + frac
    if neg:
        res = "-" + res
    return res

fn vmin(a: List[Float64]) -> Float64:
    var m = a[0]
    for i in range(1, len(a)):
        if a[i] < m:
            m = a[i]
    return m

fn vmean(a: List[Float64]) -> Float64:
    var s = Float64(0.0)
    for i in range(len(a)):
        s += a[i]
    return s / Float64(len(a))

fn vstd(a: List[Float64]) -> Float64:
    var mu = vmean(a)
    var s = Float64(0.0)
    for i in range(len(a)):
        var d = a[i] - mu
        s += d * d
    return sqrt(s / Float64(len(a)))

fn vmedian(a: List[Float64]) -> Float64:
    var b = List[Float64]()
    for i in range(len(a)):
        b.append(a[i])
    for i in range(1, len(b)):
        var key = b[i]
        var j = i - 1
        while j >= 0 and b[j] > key:
            b[j + 1] = b[j]
            j -= 1
        b[j + 1] = key
    return b[len(b) // 2]

fn main() raises:
    # open both GPUs
    var ctx0 = DeviceContext(device_id=0)
    var ctx1 = DeviceContext(device_id=1)

    print("Device: GPU 0 + GPU 1 (2x data-parallel)")
    print("")
    print(rjust("Paths", 12) + "  " + rjust("min(ms)", 10) + "  " + rjust("median(ms)", 12)
          + "  " + rjust("std(ms)", 10) + "  " + rjust("Price", 8))
    print("-" * 60)

    var sizes = [1_000_000, 10_000_000, 100_000_000]
    var reps = 50

    for si in range(len(sizes)):
        var n   = sizes[si]
        var mid = n // 2

        # per-device buffers allocated ONCE, before timing
        var d0 = ctx0.enqueue_create_buffer[DType.float32](N_THREADS)
        var d1 = ctx1.enqueue_create_buffer[DType.float32](N_THREADS)

        # warm up both
        for _ in range(5):
            ctx0.enqueue_function[mc_kernel](d0, 0,   mid, grid_dim=GRID_SIZE, block_dim=BLOCK_SIZE)
            ctx1.enqueue_function[mc_kernel](d1, mid, n,   grid_dim=GRID_SIZE, block_dim=BLOCK_SIZE)
        ctx0.synchronize()
        ctx1.synchronize()

        # ── timed region: enqueue both devices (concurrent), then sync both ──
        var times = List[Float64]()
        for _ in range(reps):
            var t0 = perf_counter_ns()
            ctx0.enqueue_function[mc_kernel](d0, 0,   mid, grid_dim=GRID_SIZE, block_dim=BLOCK_SIZE)
            ctx1.enqueue_function[mc_kernel](d1, mid, n,   grid_dim=GRID_SIZE, block_dim=BLOCK_SIZE)
            ctx0.synchronize()
            ctx1.synchronize()
            times.append(Float64(perf_counter_ns() - t0) / Float64(1_000_000))

        # price (untimed): copy both partials back and add in fixed order
        var h0 = ctx0.enqueue_create_host_buffer[DType.float32](N_THREADS)
        var h1 = ctx1.enqueue_create_host_buffer[DType.float32](N_THREADS)
        ctx0.enqueue_copy(h0, d0)
        ctx1.enqueue_copy(h1, d1)
        ctx0.synchronize()
        ctx1.synchronize()
        var total = Float64(0.0)
        for j in range(N_THREADS):
            total += Float64(h0[j])
        for j in range(N_THREADS):
            total += Float64(h1[j])
        var price = total / Float64(n)

        print(rjust(String(n), 12) + "  " + rjust(fixed(vmin(times), 4), 10)
              + "  " + rjust(fixed(vmedian(times), 4), 12)
              + "  " + rjust(fixed(vstd(times), 4), 10)
              + "  " + rjust(fixed(price, 4), 8))
