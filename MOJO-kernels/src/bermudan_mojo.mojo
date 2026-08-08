# Bermudan max-call (d=5) optimal-stopping INFERENCE -- Mojo GPU.
# One path per thread: simulate 5 GBM assets over 9 dates, apply the PyTorch-trained
# Longstaff-Schwartz policy (BETA), accumulate discounted cashflow. Same index-addressed
# RNG and arithmetic as bermudan_cuda.cu. Train in PyTorch (productivity), infer in Mojo.
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from math import exp, sqrt, log, cos
from memory import UnsafePointer, bitcast
from time import perf_counter_ns
from collections import InlineArray

alias D=5
alias N=9
alias NT = 1<<18
alias BLOCK=128
alias GRID = NT//BLOCK
alias K = Float32(100.0)
alias DRIFT=Float32(-0.02333333)
alias VOL=Float32(0.11547005)
alias DISC=Float32(0.98347145)

@always_inline
fn lcg(s:UInt32)->UInt32: return s*UInt32(1664525)+UInt32(1013904223)
@always_inline
fn uf(s:UInt32)->Float32: return bitcast[DType.float32,1]((s>>9)|UInt32(0x3F800000))-Float32(1.0)

fn berm(beta: UnsafePointer[Float32,origin=MutableAnyOrigin], part: UnsafePointer[Float32,origin=MutableAnyOrigin], M: Int):
    var tid = block_idx.x*block_dim.x+thread_idx.x
    if tid>=NT: return
    var acc=Float32(0.0)
    var o = Int(tid)
    while o<M:
        var st = UInt32(o)*UInt32(2654435761) ^ UInt32(0xBEEF1234)
        var S = InlineArray[Float32,D](fill=K)
        var cash=Float32(0.0)
        var stop=False
        var dpow=Float32(1.0)
        for k in range(N):
            var mx=Float32(-1e30)
            for i in range(D):
                var s1=lcg(st); var s2=lcg(s1); st=s2
                var u1=uf(s1)+Float32(1e-7); var u2=uf(s2)
                var z=sqrt(-Float32(2.0)*log(u1))*cos(Float32(6.2831853)*u2)
                S[i]=S[i]*exp(DRIFT+VOL*z)
                if S[i]>mx: mx=S[i]
            var pay=max(mx-K,Float32(0.0))
            if not stop:
                if k>=1 and k<=7 and pay>Float32(0.0):
                    var base=(k-1)*13
                    var mk=mx/K
                    var C=beta[base]+beta[base+11]*mk+beta[base+12]*mk*mk
                    for i in range(D):
                        var m=S[i]/K
                        C+=beta[base+1+i]*m+beta[base+6+i]*m*m
                    if pay>C:
                        cash=dpow*pay; stop=True
                elif k==N-1:
                    cash=dpow*pay; stop=True
            dpow*=DISC
        acc+=cash
        o+=NT
    part[tid]=acc

fn fixed(v: Float64, dec: Int) -> String:
    var sc=1
    for _ in range(dec): sc*=10
    var s=Int(round(v*Float64(sc))); var neg=s<0
    if neg: s=-s
    var fr=String(s%sc)
    while len(fr)<dec: fr="0"+fr
    var res=String(s//sc)+"."+fr
    if neg: res="-"+res
    return res

fn main() raises:
    var ctx = DeviceContext(device_id=0)
    var vals = List[Float32](Float32(39.334827), Float32(-24.232960), Float32(-30.312733), Float32(-30.686251), Float32(-28.654774), Float32(-25.382729), Float32(19.704042), Float32(22.673143), Float32(22.793228), Float32(21.636700), Float32(20.137524), Float32(5.654584), Float32(9.446648), Float32(21.960005), Float32(-19.810450), Float32(-24.112822), Float32(-22.714302), Float32(-23.395575), Float32(-20.043325), Float32(16.534216), Float32(18.733135), Float32(17.921757), Float32(18.126127), Float32(16.544779), Float32(8.421432), Float32(11.905353), Float32(2.858793), Float32(-19.511061), Float32(-17.618572), Float32(-17.208374), Float32(-18.153769), Float32(-16.728020), Float32(15.433704), Float32(14.452720), Float32(14.205855), Float32(14.673250), Float32(14.017172), Float32(20.548666), Float32(10.617363), Float32(-20.735924), Float32(-12.321731), Float32(-14.007333), Float32(-13.265647), Float32(-14.728768), Float32(-14.322659), Float32(10.971556), Float32(11.890816), Float32(11.330043), Float32(12.071499), Float32(11.908772), Float32(39.661861), Float32(6.429322), Float32(-33.445992), Float32(-11.130936), Float32(-11.520391), Float32(-11.558201), Float32(-11.732652), Float32(-11.121455), Float32(9.594664), Float32(9.788746), Float32(9.746010), Float32(9.898325), Float32(9.614931), Float32(48.805805), Float32(5.455533), Float32(-47.144600), Float32(-10.724863), Float32(-10.596046), Float32(-9.988170), Float32(-10.143642), Float32(-10.488173), Float32(8.689857), Float32(8.582464), Float32(8.232291), Float32(8.188211), Float32(8.609746), Float32(63.780602), Float32(2.327287), Float32(-63.072121), Float32(-9.514313), Float32(-6.557866), Float32(-8.470887), Float32(-8.393569), Float32(-7.818264), Float32(7.161981), Float32(5.770020), Float32(6.639585), Float32(6.590458), Float32(6.357351), Float32(77.070389), Float32(0.158136))
    var hbeta = ctx.enqueue_create_host_buffer[DType.float32](91)
    for i in range(91): hbeta[i]=vals[i]
    var dbeta = ctx.enqueue_create_buffer[DType.float32](91)
    ctx.enqueue_copy(dbeta, hbeta); ctx.synchronize()
    print("Bermudan max-call d=5 INFERENCE (Mojo)")
    print("       Paths  median(ms)      Price")
    var sizes=[1_000_000,4_000_000,16_000_000]
    for si in range(len(sizes)):
        var M=sizes[si]
        var dP=ctx.enqueue_create_buffer[DType.float32](NT)
        for _ in range(5): ctx.enqueue_function[berm](dbeta,dP,M,grid_dim=GRID,block_dim=BLOCK)
        ctx.synchronize()
        var best=Float64(1e9)
        for _ in range(30):
            var t0=perf_counter_ns()
            ctx.enqueue_function[berm](dbeta,dP,M,grid_dim=GRID,block_dim=BLOCK)
            ctx.synchronize()
            var e=Float64(perf_counter_ns()-t0)/Float64(1_000_000)
            if e<best: best=e
        var h=ctx.enqueue_create_host_buffer[DType.float32](NT)
        ctx.enqueue_copy(h,dP); ctx.synchronize()
        var tot=Float64(0.0)
        for j in range(NT): tot+=Float64(h[j])
        print("  "+String(M)+"   "+fixed(best,4)+"   "+fixed(tot/Float64(M),4))
