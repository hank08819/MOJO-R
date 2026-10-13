import sys, torch
sys.path.insert(0, ".")
from mc_pytorch import mc_price
out = []
for n in [1_000_000, 10_000_000, 100_000_000]:
    torch.manual_seed(12345); torch.cuda.manual_seed_all(12345)
    out.append(f"{n} {float(mc_price(n)).hex()}")
print("TORCH", torch.__version__, " | ".join(out))
