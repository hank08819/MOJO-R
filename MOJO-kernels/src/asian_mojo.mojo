# Arithmetic-average ASIAN call -- Mojo GPU kernel.
# Same algorithm as asian_cuda.cu: per-path GBM trajectory of M_STEPS steps with a
# per-path index-addressed LCG, running average, discounted payoff. Grid-stride.
# Run: pixi run mojo asian_mojo.mojo

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from math import exp, sqrt, log, cos
from memory import UnsafePointer, bitcast
from time import perf_counter_ns

alias S0    = Float32(100.0)
alias K     = Float32(100.0)
alias R     = Float32(0.05)
alias SIGMA = Float32(0.2)
alias T     = Float32(1.0)
alias PI    = Float32(3.14159265358979)
alias M_STEPS = 100

alias N_THREADS  = 1 << 18
alias BLOCK_SIZE = 128
alias GRID_SIZE  = N_THREADS // BLOCK_SIZE

@always_inline
fn lcg(seed: UInt32) -> UInt32:
    return seed * UInt32(1664525) + UInt32(1013904223)

@always_inline
fn uniform(seed: UInt32) -> Float32:
    return bitcast[DType.float32, 1]((seed >> 9) | UInt32(0x3F800000)) - Float32(1.0)

fn asian_kernel(partials: UnsafePointer[Float32, origin=MutableAnyOrigin], n_paths: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= N_THREADS:
        return
    var dt    = T / Float32(M_STEPS)
    var drift = (R - Float32(0.5) * SIGMA * SIGMA) * dt
    var vol   = SIGMA * sqrt(dt)
    var disc  = exp(-R * T)
    var acc   = Float32(0.0)
    var i     = tid
    while i < n_paths:
        var state = UInt32(i) * UInt32(2654435761) ^ UInt32(0xDEADBEEF)
        var S    = S0
        var sumS = Float32(0.0)
        for _ in range(M_STEPS):
            var s1 = lcg(state)
            var s2 = lcg(s1)
            state = s2
            var u1 = uniform(s1) + Float32(1e-7)
            var u2 = uniform(s2)
            var z  = sqrt(-Float32(2.0) * log(u1)) * cos(Float32(2.0) * PI * u2)
            S *= exp(drift + vol * z)
            sumS += S
        var avg = sumS / Float32(M_STEPS)
        acc += max(avg - K, Float32(0.0)) * disc
        i += N_THREADS
    partials[tid] = acc

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
    var ctx = DeviceContext(device_id=0)
    print("Device: GPU 0   (Asian, M_STEPS=", M_STEPS, ")")
    print("")
    print(rjust("Paths", 12) + "  " + rjust("min(ms)", 10) + "  " + rjust("median(ms)", 12)
          + "  " + rjust("std(ms)", 10) + "  " + rjust("Price", 8))
    print("-" * 60)
    var sizes = [100_000, 1_000_000, 10_000_000]
    var reps = 50
    for si in range(len(sizes)):
        var n = sizes[si]
        var d = ctx.enqueue_create_buffer[DType.float32](N_THREADS)
        for _ in range(5):
            ctx.enqueue_function[asian_kernel](d, n, grid_dim=GRID_SIZE, block_dim=BLOCK_SIZE)
        ctx.synchronize()
        var times = List[Float64]()
        for _ in range(reps):
            var t0 = perf_counter_ns()
            ctx.enqueue_function[asian_kernel](d, n, grid_dim=GRID_SIZE, block_dim=BLOCK_SIZE)
            ctx.synchronize()
            times.append(Float64(perf_counter_ns() - t0) / Float64(1_000_000))
        var h = ctx.enqueue_create_host_buffer[DType.float32](N_THREADS)
        ctx.enqueue_copy(h, d)
        ctx.synchronize()
        var total = Float64(0.0)
        for j in range(N_THREADS):
            total += Float64(h[j])
        print(rjust(String(n), 12) + "  " + rjust(fixed(vmin(times), 4), 10)
              + "  " + rjust(fixed(vmedian(times), 4), 12)
              + "  " + rjust(fixed(vstd(times), 4), 10)
              + "  " + rjust(fixed(total / Float64(n), 4), 8))
