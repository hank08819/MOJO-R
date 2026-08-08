"""
Neural forecasting INFERENCE -- PyTorch GPU baseline (idiomatic, vectorized).

A 2-layer MLP (D_IN -> H -> 1, tanh) over N feature vectors, evaluated with dense
matmuls and a tanh, the way a quant would write inference in PyTorch. Processed in
chunks so the N x H hidden activations do not exhaust GPU memory at N = 1e8; this
chunked-batch style is itself idiomatic for large-batch inference.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time, statistics
import torch

SEED = 12345
D_IN, H = 16, 64
CHUNK = 4_000_000


@torch.no_grad()
def mlp_meanout(n: int, W1, b1, W2, b2) -> float:
    total = 0.0
    done = 0
    while done < n:
        m = min(CHUNK, n - done)
        x   = torch.rand(m, D_IN, device="cuda", dtype=torch.float32) * 2.0 - 1.0
        hid = torch.tanh(x @ W1.t() + b1)
        out = hid @ W2 + b2                      # (m,)
        total += out.sum().item()
        done += m
    return total / n


def benchmark(n, W, warmup=3, reps=30):
    W1, b1, W2, b2 = W
    for _ in range(warmup): mlp_meanout(n, W1, b1, W2, b2)
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); mlp_meanout(n, W1, b1, W2, b2)
        t.append((time.perf_counter() - t0) * 1000.0)
    return t


if __name__ == "__main__":
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    W1 = (torch.rand(H, D_IN, device="cuda") * 2 - 1) * 0.25
    b1 = (torch.rand(H, device="cuda") * 2 - 1) * 0.25
    W2 = (torch.rand(H, device="cuda") * 2 - 1) * 0.25
    b2 = (torch.rand(1, device="cuda") * 2 - 1) * 0.25
    W = (W1, b1, W2, b2)
    print(f"Device : {torch.cuda.get_device_name(0)}   (MLP inference, {D_IN}->{H}->1, tanh)")
    print(f"{'Samples':>14}  {'min(ms)':>10}  {'median(ms)':>12}  {'std(ms)':>10}  {'MeanOut':>10}")
    print("-" * 68)
    for n in [1_000_000, 10_000_000, 100_000_000]:
        t = benchmark(n, W)
        mo = mlp_meanout(n, *W)
        print(f"{n:>14,}  {min(t):>10.4f}  {statistics.median(t):>12.4f}  "
              f"{statistics.pstdev(t):>10.4f}  {mo:>10.6f}")
