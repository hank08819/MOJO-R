"""
AMERICAN put pricing -- PyTorch GPU baseline (idiomatic, vectorized binomial).

Prices a batch of N American puts with a Cox-Ross-Rubinstein tree, vectorized across
options: the backward induction is a Python loop of STEPS steps, each a vectorized
max(continuation, intrinsic) over the whole batch and node dimension. This is how a
quant would write it in PyTorch; the per-step framework dispatch is the overhead under
test. Spots are sampled in [80,120] (its own RNG), so the mean price matches the
CUDA/Mojo value only within sampling, not bitwise.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math, time, statistics
import torch

SEED = 12345
K, r, sigma, T = 100.0, 0.05, 0.2, 1.0
STEPS = 128
dt   = T / STEPS
lnu  = sigma * math.sqrt(dt)
u, d = math.exp(lnu), math.exp(-lnu)
disc = math.exp(-r * dt)
p    = (math.exp(r * dt) - d) / (u - d)
q    = 1.0 - p


@torch.no_grad()
def american_meanprice(n: int) -> float:
    S0 = torch.rand(n, device="cuda", dtype=torch.float32) * 40.0 + 80.0   # [80,120)
    j  = torch.arange(STEPS + 1, device="cuda", dtype=torch.float32)
    ST = S0[:, None] * torch.exp((STEPS - 2.0 * j)[None, :] * lnu)
    V  = torch.clamp(K - ST, min=0.0)                                       # (n, STEPS+1)
    for s in range(STEPS - 1, -1, -1):
        jj   = torch.arange(s + 1, device="cuda", dtype=torch.float32)
        St   = S0[:, None] * torch.exp((s - 2.0 * jj)[None, :] * lnu)
        cont = disc * (p * V[:, :s + 1] + q * V[:, 1:s + 2])
        V[:, :s + 1] = torch.maximum(cont, K - St)
    return V[:, 0].mean().item()


def benchmark(n, warmup=3, reps=30):
    for _ in range(warmup): american_meanprice(n)
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); american_meanprice(n)
        t.append((time.perf_counter() - t0) * 1000.0)
    return t


if __name__ == "__main__":
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    print(f"Device : {torch.cuda.get_device_name(0)}   (American put, CRR tree STEPS={STEPS})")
    print(f"{'Options':>12}  {'min(ms)':>10}  {'median(ms)':>12}  {'std(ms)':>10}  {'MeanPrice':>10}")
    print("-" * 68)
    for n in [10_000, 100_000, 1_000_000]:
        t = benchmark(n)
        torch.manual_seed(SEED)
        mp = american_meanprice(n)
        print(f"{n:>12,}  {min(t):>10.4f}  {statistics.median(t):>12.4f}  "
              f"{statistics.pstdev(t):>10.4f}  {mp:>10.5f}")
