"""
Monte Carlo option pricer - PyTorch, TWO-GPU data-parallel baseline.

Splits the n paths across cuda:0 and cuda:1 (half each). The two halves are issued
back-to-back on their separate device streams (so they overlap), then both devices
are synchronized and the partial sums are combined on the host. This is the idiomatic
multi-GPU style a quantitative researcher would write -- no model parallelism, just
data-parallel replication of the same vectorized pricer.
"""
import math
import time
import statistics

import torch

SEED = 12345
NGPU = 2
S, K, r, sig, T = 100.0, 100.0, 0.05, 0.2, 1.0
DISCOUNT = math.exp(-r * T)


def _half(n_paths: int, dev: str) -> torch.Tensor:
    z   = torch.randn(n_paths, device=dev, dtype=torch.float32)
    ST  = S * torch.exp((r - 0.5 * sig ** 2) * T + sig * T ** 0.5 * z)
    pay = torch.clamp(ST - K, min=0.0) * DISCOUNT
    return pay.sum()                       # async on this device's stream


def mc_price_2gpu(n_paths: int) -> float:
    n0 = n_paths // 2
    n1 = n_paths - n0
    s0 = _half(n0, "cuda:0")               # issued on GPU 0 stream ...
    s1 = _half(n1, "cuda:1")               # ... then GPU 1 stream (overlap)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    total = s0.to("cpu") + s1.to("cpu")    # fixed-order host combine
    return (total / n_paths).item()


def benchmark(n_paths: int, warmup: int = 5, reps: int = 50):
    for _ in range(warmup):
        mc_price_2gpu(n_paths)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mc_price_2gpu(n_paths)
        times.append((time.perf_counter() - t0) * 1000.0)   # ms
    return times


if __name__ == "__main__":
    assert torch.cuda.device_count() >= NGPU, f"need {NGPU} GPUs"
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"Device : {torch.cuda.get_device_name(0)} x{NGPU} (2x data-parallel)")
    print(f"{'Paths':>12}  {'min(ms)':>10}  {'median(ms)':>12}  {'std(ms)':>10}  {'Price':>8}")
    print("-" * 62)
    for n in [1_000_000, 10_000_000, 100_000_000]:
        t  = benchmark(n)
        mn = min(t)
        md = statistics.median(t)
        sd = statistics.pstdev(t)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        px = mc_price_2gpu(n)
        print(f"{n:>12,}  {mn:>10.4f}  {md:>12.4f}  {sd:>10.4f}  {px:>8.4f}")
