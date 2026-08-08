# AMERICAN put pricing -- Mojo GPU. Early-exercise via CRR binomial backward induction.
# Same arithmetic as american_cuda.cu (bit-identical mean price). One option per thread,
# grid-stride over a batch with index-addressed spot in [80,120). No RNG in the pricer.
# Run: pixi run mojo american_mojo.mojo

from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from math import exp, sqrt
from memory import UnsafePointer, bitcast
from time import perf_counter_ns
from collections import InlineArray

alias STEPS = 128
alias N_THREADS  = 1 << 17
alias BLOCK_SIZE = 32                    # best for this local-array-heavy tree kernel
alias GRID_SIZE  = N_THREADS // BLOCK_SIZE

@always_inline
fn lcg(seed: UInt32) -> UInt32:
    return seed * UInt32(1664525) + UInt32(1013904223)

@always_inline
fn uniform(seed: UInt32) -> Float32:
    return bitcast[DType.float32, 1]((seed >> 9) | UInt32(0x3F800000)) - Float32(1.0)

@always_inline
fn american_put(S0: Float32, K: Float32, r: Float32, sigma: Float32, T: Float32) -> Float32:
    var dt   = T / Float32(STEPS)
    var lnu  = sigma * sqrt(dt)
    var u    = exp(lnu)
    var dn   = Float32(1.0) / u
    var disc = exp(-r * dt)
    var pp   = (exp(r * dt) - dn) / (u - dn)
    var qq   = Float32(1.0) - pp
    var V = InlineArray[Float32, STEPS + 1](fill=Float32(0.0))
    for j in range(STEPS + 1):
        var St = S0 * exp(Float32(STEPS - 2 * j) * lnu)
        V[j] = max(K - St, Float32(0.0))
    var s = STEPS - 1
    while s >= 0:
        for j in range(s + 1):
            var St   = S0 * exp(Float32(s - 2 * j) * lnu)
            var cont = disc * (pp * V[j] + qq * V[j + 1])
            V[j] = max(cont, K - St)               # American early exercise
        s -= 1
    return V[0]

fn american_kernel(partials: UnsafePointer[Float32, origin=MutableAnyOrigin], n: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= N_THREADS:
        return
    var acc = Float32(0.0)
    var i = Int(tid)
    while i < n:
        var S0 = Float32(80.0) + Float32(40.0) * uniform(lcg(UInt32(i)))
        acc += american_put(S0, Float32(100.0), Float32(0.05), Float32(0.2), Float32(1.0))
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
    print("Device: GPU 0   (American put, CRR tree STEPS=", STEPS, ")")
    print("")
    print(rjust("Options", 12) + "  " + rjust("min(ms)", 10) + "  " + rjust("median(ms)", 12)
          + "  " + rjust("std(ms)", 10) + "  " + rjust("MeanPrice", 10))
    print("-" * 68)
    var sizes = [10_000, 100_000, 1_000_000]
    var reps = 50
    for si in range(len(sizes)):
        var n = sizes[si]
        var dP = ctx.enqueue_create_buffer[DType.float32](N_THREADS)
        for _ in range(5):
            ctx.enqueue_function[american_kernel](dP, n, grid_dim=GRID_SIZE, block_dim=BLOCK_SIZE)
        ctx.synchronize()
        var times = List[Float64]()
        for _ in range(reps):
            var t0 = perf_counter_ns()
            ctx.enqueue_function[american_kernel](dP, n, grid_dim=GRID_SIZE, block_dim=BLOCK_SIZE)
            ctx.synchronize()
            times.append(Float64(perf_counter_ns() - t0) / Float64(1_000_000))
        var h = ctx.enqueue_create_host_buffer[DType.float32](N_THREADS)
        ctx.enqueue_copy(h, dP)
        ctx.synchronize()
        var total = Float64(0.0)
        for j in range(N_THREADS):
            total += Float64(h[j])
        print(rjust(String(n), 12) + "  " + rjust(fixed(vmin(times), 4), 10)
              + "  " + rjust(fixed(vmedian(times), 4), 12)
              + "  " + rjust(fixed(vstd(times), 4), 10)
              + "  " + rjust(fixed(total / Float64(n), 5), 10))
