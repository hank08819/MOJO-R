"""
Arithmetic-average ASIAN call -- PyTorch GPU baseline (idiomatic, vectorized).

Simulates all paths in lock-step: at each of M_STEPS time steps it draws a full
vector of normals, advances every path, and accumulates the running average. This
materializes several full-size tensors per step, the framework-overhead baseline.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math, time, statistics
import torch

SEED = 12345
S0, K, r, sig, T = 100.0, 100.0, 0.05, 0.2, 1.0
M_STEPS = 100


def asian_price(n_paths: int) -> float:
    dt    = T / M_STEPS
    drift = (r - 0.5 * sig ** 2) * dt
    vol   = sig * math.sqrt(dt)
    disc  = math.exp(-r * T)
    S   = torch.full((n_paths,), S0, device="cuda", dtype=torch.float32)
    acc = torch.zeros(n_paths, device="cuda", dtype=torch.float32)
    for _ in range(M_STEPS):
        z = torch.randn(n_paths, device="cuda", dtype=torch.float32)
        S = S * torch.exp(drift + vol * z)
        acc = acc + S
    avg = acc / M_STEPS
    return (torch.clamp(avg - K, min=0.0) * disc).mean().item()


def benchmark(n, warmup=5, reps=50):
    for _ in range(warmup): asian_price(n)
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); asian_price(n)
        t.append((time.perf_counter() - t0) * 1000.0)
    return t


if __name__ == "__main__":
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    print(f"Device : {torch.cuda.get_device_name(0)}   (Asian, M_STEPS={M_STEPS})")
    print(f"{'Paths':>12}  {'min(ms)':>10}  {'median(ms)':>12}  {'std(ms)':>10}  {'Price':>8}")
    print("-" * 62)
    for n in [100_000, 1_000_000, 10_000_000]:
        t = benchmark(n)
        torch.manual_seed(SEED)
        px = asian_price(n)
        print(f"{n:>12,}  {min(t):>10.4f}  {statistics.median(t):>12.4f}  "
              f"{statistics.pstdev(t):>10.4f}  {px:>8.4f}")
