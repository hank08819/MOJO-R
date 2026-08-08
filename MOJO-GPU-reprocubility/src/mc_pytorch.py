"""
Monte Carlo option pricer - PyTorch GPU baseline.
Runs on GPU 0 only (second GPU disabled via CUDA_VISIBLE_DEVICES).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use only the first GPU

import time
import statistics

import torch

SEED = 12345   # fixed for reproducibility (see MOJOGPU-reprocubility)

def mc_price(n_paths: int, device: str = "cuda") -> float:
    S, K, r, sig, T = 100.0, 100.0, 0.05, 0.2, 1.0
    z      = torch.randn(n_paths, device=device, dtype=torch.float32)
    ST     = S * torch.exp((r - 0.5 * sig**2) * T + sig * T**0.5 * z)
    payoff = torch.clamp(ST - K, min=0.0) * torch.exp(torch.tensor(-r * T, device=device))
    return payoff.mean().item()

def benchmark(n_paths: int, warmup: int = 5, reps: int = 50):
    # warm up
    for _ in range(warmup):
        mc_price(n_paths)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mc_price(n_paths)          # .item() forces a device sync each call
        times.append((time.perf_counter() - t0) * 1000.0)   # ms
    return times

if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"{'Paths':>12}  {'min(ms)':>10}  {'median(ms)':>12}  {'std(ms)':>10}  {'Price':>8}")
    print("-" * 62)
    for n in [1_000_000, 10_000_000, 100_000_000]:
        t  = benchmark(n)
        mn = min(t)
        md = statistics.median(t)
        sd = statistics.pstdev(t)
        torch.manual_seed(SEED)        # reseed -> reproducible reported price
        px = mc_price(n)
        print(f"{n:>12,}  {mn:>10.4f}  {md:>12.4f}  {sd:>10.4f}  {px:>8.4f}")
