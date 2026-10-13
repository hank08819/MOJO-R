# Bermudan d=5 inference -- PyTorch baseline (vectorized policy evaluation).
import torch, numpy as np, time
dev="cuda"; K,r,div,sig,T,N,d=100.,0.05,0.10,0.2,3.0,9,5
dt=T/N; DISC=float(np.exp(-r*dt)); drift=(r-div-0.5*sig*sig)*dt; vol=sig*dt**0.5
import os
POLICY=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","rtx6000ada","policy_beta.npy")
B=torch.tensor(np.load(POLICY),device=dev,dtype=torch.float32) #(7,13)
def basis(S):
    m=S/K; mx=m.max(-1,keepdim=True).values
    return torch.cat([torch.ones(S.shape[0],1,device=dev),m,m*m,mx,mx*mx],1)
@torch.no_grad()
def price(M):
    g=torch.Generator(device=dev).manual_seed(7)
    S=torch.full((M,d),K,device=dev); stop=torch.zeros(M,dtype=torch.bool,device=dev)
    cash=torch.zeros(M,device=dev); 
    for k in range(N):
        S=S*torch.exp(drift+vol*torch.randn(M,d,device=dev,generator=g))
        pay=torch.clamp(S.max(-1).values-K,min=0.)
        if 1<=k<=7:
            itm=(pay>0)&(~stop); C=(basis(S[itm])@B[k-1]); idx=itm.nonzero().squeeze(1)
            ex=idx[pay[itm]>C]; cash[ex]=DISC**k*pay[ex]; stop[ex]=True
        elif k==N-1:
            fin=~stop; cash[fin]=DISC**k*pay[fin]; stop[fin]=True
    return cash.mean().item()
print(f"Bermudan d=5 INFERENCE (PyTorch)\n{'Paths':>12} {'median(ms)':>10} {'Price':>10}")
for M in [1000000,4000000,16000000]:
    for _ in range(5): price(M)
    torch.cuda.synchronize(); ts=[]
    for _ in range(20):
        t0=time.perf_counter(); price(M); ts.append((time.perf_counter()-t0)*1000)
    print(f"{M:>12} {sorted(ts)[len(ts)//2]:>10.4f} {price(M):>10.4f}")
