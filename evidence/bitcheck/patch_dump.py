# Insert partial-sum dumps into copies of the ICAIF kernels (bitcheck_20261012).
# Each dump file: one uint32 bit pattern per line, for every per-thread partial sum.
import re

HELPER_C = r'''
#include <string.h>
static void dumpf(const char* tag, long long n, int d, const float* h, int cnt) {
    char fn[256]; snprintf(fn, sizeof fn, "dumps/%s_%lld_d%d.txt", tag, n, d);
    FILE* fp = fopen(fn, "w");
    for (int j = 0; j < cnt; j++) { unsigned u; memcpy(&u, &h[j], 4); fprintf(fp, "%u\n", u); }
    fclose(fp);
}
'''


def patch_c(fname, tag, host, nvar, cnt, dexpr="0"):
    L = open(fname).read().split("\n")
    last_inc = max(i for i, l in enumerate(L) if l.startswith("#include"))
    L.insert(last_inc + 1, HELPER_C)
    i = next(i for i, l in enumerate(L) if "DeviceToHost" in l)
    L.insert(i + 1, f'        dumpf("{tag}", (long long){nvar}, {dexpr}, {host}, {cnt});')
    open(fname, "w").write("\n".join(L))


def mojo_block(indent, tag, host, nvar, cnt, d):
    p = " " * indent
    return [
        f"{p}var _s{d} = String()",
        f"{p}for _j in range({cnt}):",
        f"{p}    _s{d} += String(bitcast[DType.uint32, 1]({host}[_j])) + \"\\n\"",
        f"{p}with open(String(\"dumps/{tag}_\") + String({nvar}) + \"_d{d}.txt\", \"w\") as _f{d}:",
        f"{p}    _f{d}.write(_s{d})",
    ]


def patch_mojo(fname, tag, host, nvar, cnt, after_pat, extra=None):
    L = open(fname).read().split("\n")
    i = next(i for i, l in enumerate(L) if re.search(after_pat, l))
    if "synchronize" not in L[i] and "synchronize" in L[i + 1]:
        i += 1
    indent = len(L[i]) - len(L[i].lstrip())
    blk = mojo_block(indent, tag, host, nvar, cnt, 0)
    if extra:
        blk += mojo_block(indent, tag, extra, nvar, cnt, 1)
    L[i + 1:i + 1] = blk
    open(fname, "w").write("\n".join(L))


for k in ["american", "asian", "mlp"]:
    patch_c(f"{k}_cuda.cu", f"cuda_{k}", "hP", "n", "N_THREADS")
    patch_mojo(f"{k}_mojo.mojo", f"mojo_{k}", "h", "n", "N_THREADS", r"enqueue_copy\(h, ")
patch_c("mc_cuda.cu", "cuda_european", "h_partials", "n", "N_THREADS")
patch_c("bermudan_cuda.cu", "cuda_bermudan", "hP", "M", "NT")
patch_c("mc_cuda_2gpu.cu", "cuda_european2gpu", "h_partials[d]", "n", "N_THREADS", "d")

# Mojo single-GPU European: the synchronize right after the host copy
L = open("mc_mojo.mojo").read().split("\n")
i = next(i for i, l in enumerate(L) if "enqueue_copy(h_partials" in l)
assert "synchronize" in L[i + 1]
L[i + 2:i + 2] = mojo_block(8, "mojo_european", "h_partials", "n", "N_THREADS", 0)
open("mc_mojo.mojo", "w").write("\n".join(L))

patch_mojo("bermudan_mojo.mojo", "mojo_bermudan", "h", "M", "NT", r"enqueue_copy\(h,dP\)")

L = open("mc_mojo_2gpu.mojo").read().split("\n")
i = next(i for i, l in enumerate(L) if "ctx1.synchronize()" in l and "enqueue_copy(h1" in L[i - 2])
L[i + 1:i + 1] = mojo_block(8, "mojo_european2gpu", "h0", "n", "N_THREADS", 0) + \
                 mojo_block(8, "mojo_european2gpu", "h1", "n", "N_THREADS", 1)
open("mc_mojo_2gpu.mojo", "w").write("\n".join(L))
print("patched")
