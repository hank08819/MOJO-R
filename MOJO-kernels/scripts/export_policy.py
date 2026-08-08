# Train poly-LSM policy for d=5 Bermudan max-call; export per-date coefficients
# as a flat array for the Mojo/CUDA inference kernels.
import torch, numpy as np
torch.manual_seed(1); dev="cuda" if torch.cuda.is_available() else "cpu"
K,r,div,sig,T,N,d,S0=100.,0.05,0.10,0.2,3.0,9,5,100.0
dt=T/N; DISC=float(np.exp(-r*dt))
def sim(M,seed):
    g=torch.Generator(device=dev).manual_seed(seed); drift=(r-div-0.5*sig*sig)*dt; vol=sig*dt**0.5
    S=torch.full((M,d),S0,device=dev); out=torch.empty(M,N,d,device=dev)
    for n in range(N): S=S*torch.exp(drift+vol*torch.randn(M,d,device=dev,generator=g)); out[:,n,:]=S
    return out
def pay(S): return torch.clamp(S.max(-1).values-K,min=0.0)
def basis(S):  # 13 feats for d=5: 1, m_i(5), m_i^2(5), maxm, maxm^2
    m=S/K; mx=m.max(-1,keepdim=True).values
    return torch.cat([torch.ones(S.shape[0],1,device=dev),m,m*m,mx,mx*mx],1)
P=sim(200000,10); g=torch.stack([pay(P[:,n,:]) for n in range(N)],1)
cf=g[:,N-1].clone(); tau=torch.full((P.shape[0],),N-1,device=dev); betas={}
for n in range(N-2,0,-1):
    Y=cf*DISC**(tau-n).float(); itm=g[:,n]>1e-8
    X=basis(P[itm,n,:]); b=torch.linalg.lstsq(X,Y[itm,None]).solution; betas[n]=b.squeeze(1)
    ex=torch.zeros_like(itm); idx=itm.nonzero().squeeze(1); do=g[itm,n]>(X@b).squeeze(1)
    ex[idx[do]]=True; tau[ex]=n; cf[ex]=g[ex,n]
# out-of-sample price as target
Pt=sim(200000,77); gt=torch.stack([pay(Pt[:,n,:]) for n in range(N)],1)
stp=torch.zeros(Pt.shape[0],dtype=torch.bool,device=dev); C=torch.zeros(Pt.shape[0],device=dev); td=torch.zeros(Pt.shape[0],device=dev)
for n in range(1,N-1):
    itm=(gt[:,n]>1e-8)&(~stp)
    Ch=(basis(Pt[itm,n,:])@betas[n]); idx=itm.nonzero().squeeze(1); do=gt[itm,n]>Ch
    e=idx[do]; C[e]=gt[e,n]; td[e]=n; stp[e]=True
fin=~stp; C[fin]=gt[fin,N-1]; td[fin]=N-1
price=(C*DISC**td).mean().item()
print(f"d={d} S0={S0}: OOS LSM price = {price:.4f}  (ref 26.156)")
B=np.stack([betas[n].cpu().numpy() for n in range(1,N-1)])  # (7,13)
print("DATES_USED = 1..7 ; NB=13 ; DISC=%.8f ; drift=%.8f ; vol=%.8f"%(DISC,(r-div-0.5*sig*sig)*dt,sig*dt**0.5))
np.save("MojoGPU2026_V3/results/policy_beta.npy",B)
# print as Mojo/C flat literal
flat=B.flatten()
print("BETA = [")
for i in range(0,len(flat),7): print("  "+", ".join(f"{x:.6f}" for x in flat[i:i+7])+",")
print("]  # 7 dates x 13 features, row-major")
