import numpy as np, json, re, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
log=open("results_locked.log").read()
sec=re.split(r"^### ",log,flags=re.M)
def raws(block,tag):
    return {int(m.group(1)):[float(x) for x in m.group(2).split()] for m in re.finditer(r"^%s (\d+) (.*)$"%tag,block,re.M)}
out={}
for s in sec:
    name=s.split("\n",1)[0].strip()
    if name in ("CUDA C","Mojo","PyTorch"):
        r=raws(s,"RAW"); e=raws(s,"EVT")
        out[name]={n:dict(med=float(np.median(v)),std=float(np.std(v)),L=len(v),evt=(float(np.median(e[n])) if n in e else None)) for n,v in r.items()}
for impl in ["numba_b64","numba_b128","numba_b256","triton_b64","triton_b128","triton_b256"]:
    out[impl]={d["n"]:dict(med=d["median_ms"],std=d["std_ms"],L=len(d["times_ms"]),evt=d["event_median_ms"]) for d in json.load(open("raw/%s.json"%impl))}
for impl,v in out.items():
    print(impl.ljust(12)," | ".join("n=%.0e med %.4f+-%.4f L=%d evt %s"%(n,d["med"],d["std"],d["L"],"-" if d["evt"] is None else "%.4f"%d["evt"]) for n,d in sorted(v.items())))
json.dump({k:{str(n):d for n,d in v.items()} for k,v in out.items()},open("summary_locked.json","w"),indent=1)
for n in [1000000,10000000,100000000]:
    c=np.loadtxt("../bitcheck/dumps/cuda_european_%d_d0.txt"%n,dtype=np.uint32)
    for impl in ["numba_b128","triton_b128"]:
        a=np.load("raw/%s_%d.npy"%(impl,n)); b=np.load("raw_repeat/%s_%d.npy"%(impl,n))
        print(n,impl,"run1==run2:",bool((a==b).all()),"| partials != CUDA:",int((a!=c).sum()))
