# Compare the CUDA and Mojo partial-sum dumps bit for bit (bitcheck_20261012).
# The price is rebuilt exactly as both programs do it: a sequential double sum of the
# float32 partials (device 0 first, then device 1), divided by n.
import glob
import sys
import os
import struct
import numpy as np

os.chdir(sys.argv[1] if len(sys.argv) > 1 else "dumps")


def load(tag, n):
    fs = sorted(glob.glob(f"{tag}_{n}_d*.txt"))
    return [np.loadtxt(f, dtype=np.uint32) for f in fs] if fs else None


def price(parts, n):
    tot = 0.0
    for p in parts:
        for v in p.view(np.float32).astype(np.float64):
            tot += float(v)
    return tot / n


def ulp(a, b):
    # distance in units of the last place between two float32 arrays (as signed ints)
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    return np.abs(ia - ib)


rows = []
sizes = {}
for f in glob.glob("*.txt"):
    impl, kern, n, _ = f[:-4].split("_")
    sizes.setdefault(kern, set()).add(int(n))

prices = {}
for kern in sorted(sizes):
    for n in sorted(sizes[kern]):
        c, m = load(f"cuda_{kern}", n), load(f"mojo_{kern}", n)
        if c is None or m is None:
            print(f"{kern:16s} n={n:<10d} missing dump (cuda={c is not None}, mojo={m is not None})")
            continue
        cc, mm = np.concatenate(c), np.concatenate(m)
        d = ulp(cc, mm)
        pc, pm = price(c, n), price(m, n)
        prices[(kern, n)] = (pc, pm)
        print(f"{kern:16s} n={n:<10d} partials differ {int((d > 0).sum()):6d}/{d.size} "
              f"(max ulp {int(d.max())})  price CUDA {pc.hex()} ({pc:.15f})  "
              f"Mojo {pm.hex()} ({pm:.15f})  identical={pc == pm}")

print()
for n in sorted(sizes.get("european2gpu", [])):
    if ("european", n) in prices and ("european2gpu", n) in prices:
        (c1, m1), (c2, m2) = prices[("european", n)], prices[("european2gpu", n)]
        print(f"2-GPU vs 1-GPU n={n:<10d} CUDA same={c1 == c2} (diff {c2 - c1:+.3e})  "
              f"Mojo same={m1 == m2} (diff {m2 - m1:+.3e})")
