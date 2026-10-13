"""Monte Carlo European call -- JAX (jax.jit, XLA) baseline (R3-1, 2026-10-13).

Same index-addressed LCG -> Box-Muller -> discounted payoff as mc_cuda.cu, in float32,
with the same 2^18 lanes: lane j owns paths j, j + 2^18, j + 2*2^18, ...
Two variants, both jit-compiled:
  vec  : idiomatic JAX -- evaluate all paths as one array program, then sum the
         (K, 2^18) payoff grid over K to get one partial per lane (XLA picks the order).
  loop : lax.fori_loop over K with one float32 accumulator per lane (CUDA's order).
The 2^18 partials are summed on the host in double, index order, as in the other programs.
XLA uses its own transcendental implementations (no CUDA fast-math switch).
Timing: 5 warm-ups, then 50 timed calls; wall time of call + block_until_ready.
Also: 20 back-to-back calls then one block, divided by 20 (device time without host sync).
"""
import json, os, sys, time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

N_THREADS = 1 << 18
OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
VARIANT = os.environ.get("VARIANT", "vec")
os.makedirs(OUT, exist_ok=True)
S, K, R, SIG, T = (np.float32(v) for v in (100.0, 100.0, 0.05, 0.2, 1.0))
DRIFT = (R - np.float32(0.5) * SIG * SIG) * T
VOLT = SIG * np.float32(np.sqrt(T))
DISC = np.float32(np.exp(-R * T))
U32 = jnp.uint32


def payoff(i):
    """i: uint32 path indices -> float32 discounted payoffs (0 where masked by caller)."""
    seed = (i * U32(2654435761)) ^ U32(0xDEADBEEF)
    s1 = seed * U32(1664525) + U32(1013904223)
    s2 = s1 * U32(1664525) + U32(1013904223)
    u1 = lax.bitcast_convert_type((s1 >> U32(9)) | U32(0x3F800000), jnp.float32) - jnp.float32(1.0) + jnp.float32(1e-7)
    u2 = lax.bitcast_convert_type((s2 >> U32(9)) | U32(0x3F800000), jnp.float32) - jnp.float32(1.0)
    z = jnp.sqrt(jnp.float32(-2.0) * jnp.log(u1)) * jnp.cos(jnp.float32(2.0) * jnp.float32(3.14159265358979) * u2)
    st = S * jnp.exp(DRIFT + VOLT * z)
    return jnp.maximum(st - K, jnp.float32(0.0)) * DISC


def make(n):
    k_iter = (n + N_THREADS - 1) // N_THREADS
    lanes = jnp.arange(N_THREADS, dtype=jnp.uint32)
    if VARIANT == "vec":
        @jax.jit
        def f():
            i = jnp.arange(k_iter * N_THREADS, dtype=jnp.uint32).reshape(k_iter, N_THREADS)
            y = jnp.where(i < U32(n), payoff(i), jnp.float32(0.0))
            return jnp.sum(y, axis=0)
    else:
        @jax.jit
        def f():
            def body(k, acc):
                i = lanes + U32(k) * U32(N_THREADS)
                return acc + jnp.where(i < U32(n), payoff(i), jnp.float32(0.0))
            return lax.fori_loop(0, k_iter, body, jnp.zeros(N_THREADS, jnp.float32))
    return f


def main():
    rows = []
    for n in [1_000_000, 10_000_000, 100_000_000]:
        f = make(n)
        for _ in range(5):
            f().block_until_ready()
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            f().block_until_ready()
            ts.append((time.perf_counter() - t0) * 1e3)
        dev = []
        for _ in range(10):
            t0 = time.perf_counter()
            for _ in range(20):
                out = f()
            out.block_until_ready()
            dev.append((time.perf_counter() - t0) * 1e3 / 20)
        h = np.asarray(f()).astype(np.float32)
        tot = 0.0
        for v in h.astype(np.float64):
            tot += float(v)
        price = tot / n
        np.save(os.path.join(OUT, f"jax_{VARIANT}_{n}.npy"), h.view(np.uint32))
        rows.append(dict(impl=f"jax_{VARIANT}", n=n, times_ms=ts, median_ms=float(np.median(ts)),
                         std_ms=float(np.std(ts)), event_ms=dev, event_median_ms=float(np.median(dev)),
                         price=price, price_hex=price.hex(), jax=jax.__version__))
        print(f"jax {VARIANT} n={n:>11,} median {np.median(ts):.4f} ms  std {np.std(ts):.4f}  price {price:.10f} {price.hex()}  batched-median {np.median(dev):.4f}")
    json.dump(rows, open(os.path.join(OUT, f"jax_{VARIANT}.json"), "w"))


if __name__ == "__main__":
    main()
