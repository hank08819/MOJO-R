# Neural forecasting INFERENCE kernel -- Mojo GPU.
# 2-layer MLP (D_IN -> H -> 1, tanh) over N samples. Same index-addressed weight/input
# generation as mlp_cuda.cu (bit-identical), weights filled once by an init kernel into
# global memory, then reused by the timed inference kernel. Grid-stride over samples.
# Run: pixi run mojo mlp_mojo.mojo

from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from math import sqrt, tanh
from memory import UnsafePointer, bitcast
from time import perf_counter_ns
from collections import InlineArray

alias D_IN = 16
alias H    = 64
alias NW   = H * D_IN + H + H + 1

alias N_THREADS  = 1 << 18
alias BLOCK_SIZE = 128
alias GRID_SIZE  = N_THREADS // BLOCK_SIZE

@always_inline
fn lcg(seed: UInt32) -> UInt32:
    return seed * UInt32(1664525) + UInt32(1013904223)

@always_inline
fn uniform(seed: UInt32) -> Float32:
    return bitcast[DType.float32, 1]((seed >> 9) | UInt32(0x3F800000)) - Float32(1.0)

@always_inline
fn rnd(key: UInt32) -> Float32:
    return Float32(2.0) * uniform(lcg(key ^ UInt32(0x9E3779B9))) - Float32(1.0)

fn init_weights(W: UnsafePointer[Float32, origin=MutableAnyOrigin]):
    var t = block_idx.x * block_dim.x + thread_idx.x
    var stride = grid_dim.x * block_dim.x
    var k = Int(t)
    while k < NW:
        W[k] = rnd(UInt32(k)) * Float32(0.25)
        k += Int(stride)

fn mlp_kernel(W: UnsafePointer[Float32, origin=MutableAnyOrigin],
              partials: UnsafePointer[Float32, origin=MutableAnyOrigin], n: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid >= N_THREADS:
        return
    var b2 = W[H * D_IN + 2 * H]
    var acc = Float32(0.0)
    var i = Int(tid)
    while i < n:
        var x = InlineArray[Float32, D_IN](fill=Float32(0.0))
        for d in range(D_IN):
            x[d] = rnd(UInt32(i * D_IN + d + 1))
        var out = b2
        for h in range(H):
            var a = W[H * D_IN + h]                  # b1[h]
            for d in range(D_IN):
                a += W[h * D_IN + d] * x[d]          # W1[h][d] * x[d]
            out += W[H * D_IN + H + h] * tanh(a)     # W2[h] * tanh(a)
        acc += out
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
    print("Device: GPU 0   (MLP inference,", D_IN, "->", H, "-> 1, tanh)")
    print("")

    var dW = ctx.enqueue_create_buffer[DType.float32](NW)
    ctx.enqueue_function[init_weights](dW, grid_dim=64, block_dim=128)   # untimed
    ctx.synchronize()

    print(rjust("Samples", 14) + "  " + rjust("min(ms)", 10) + "  " + rjust("median(ms)", 12)
          + "  " + rjust("std(ms)", 10) + "  " + rjust("MeanOut", 10))
    print("-" * 68)
    var sizes = [1_000_000, 10_000_000, 100_000_000]
    var reps = 50
    for si in range(len(sizes)):
        var n = sizes[si]
        var dP = ctx.enqueue_create_buffer[DType.float32](N_THREADS)
        for _ in range(5):
            ctx.enqueue_function[mlp_kernel](dW, dP, n, grid_dim=GRID_SIZE, block_dim=BLOCK_SIZE)
        ctx.synchronize()
        var times = List[Float64]()
        for _ in range(reps):
            var t0 = perf_counter_ns()
            ctx.enqueue_function[mlp_kernel](dW, dP, n, grid_dim=GRID_SIZE, block_dim=BLOCK_SIZE)
            ctx.synchronize()
            times.append(Float64(perf_counter_ns() - t0) / Float64(1_000_000))
        var h = ctx.enqueue_create_host_buffer[DType.float32](N_THREADS)
        ctx.enqueue_copy(h, dP)
        ctx.synchronize()
        var total = Float64(0.0)
        for j in range(N_THREADS):
            total += Float64(h[j])
        print(rjust(String(n), 14) + "  " + rjust(fixed(vmin(times), 4), 10)
              + "  " + rjust(fixed(vmedian(times), 4), 12)
              + "  " + rjust(fixed(vstd(times), 4), 10)
              + "  " + rjust(fixed(total / Float64(n), 6), 10))
