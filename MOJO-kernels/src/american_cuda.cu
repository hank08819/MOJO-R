/*
 * AMERICAN put pricing -- CUDA C reference.  Early-exercise: the hard case.
 *
 * Each thread prices one American put with a Cox-Ross-Rubinstein binomial tree of
 * STEPS steps and backward induction, taking max(continuation, intrinsic) at every
 * node (the early-exercise decision American options add over European/Asian). A
 * batch of N options with index-addressed spot S0 in [80,120] is priced; we report
 * the mean price. No RNG: the arithmetic is identical to the Mojo version, so the
 * mean price is bit-identical.
 *
 * Compile: nvcc -O3 -arch=sm_89 -use_fast_math american_cuda.cu -o american_cuda
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 128
#define N_THREADS  (1 << 17)            /* 131,072 threads (tree is heavy per option) */
#define GRID_SIZE  (N_THREADS / BLOCK_SIZE)
#define STEPS 128

#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__device__ __forceinline__ unsigned lcg(unsigned s){ return s*1664525u+1013904223u; }
__device__ __forceinline__ float uf(unsigned s){ return __uint_as_float((s>>9)|0x3F800000u)-1.0f; }

__device__ float american_put(float S0, float K, float r, float sigma, float T){
    float dt   = T / STEPS;
    float lnu  = sigma * sqrtf(dt);
    float u    = expf(lnu), d = 1.0f/u;
    float disc = expf(-r*dt);
    float p    = (expf(r*dt) - d) / (u - d);
    float q    = 1.0f - p;
    float V[STEPS+1];
    for (int j = 0; j <= STEPS; j++) {
        float St = S0 * expf((float)(STEPS - 2*j) * lnu);
        V[j] = fmaxf(K - St, 0.0f);
    }
    for (int s = STEPS-1; s >= 0; s--)
        for (int j = 0; j <= s; j++) {
            float St   = S0 * expf((float)(s - 2*j) * lnu);
            float cont = disc * (p*V[j] + q*V[j+1]);
            V[j] = fmaxf(cont, K - St);          /* American early exercise */
        }
    return V[0];
}

__global__ void american_kernel(float* partials, long long n){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= N_THREADS) return;
    float acc = 0.0f;
    for (long long i = tid; i < n; i += N_THREADS) {
        float S0 = 80.0f + 40.0f * uf(lcg((unsigned)i));   /* spot in [80,120) */
        acc += american_put(S0, 100.0f, 0.05f, 0.2f, 1.0f);
    }
    partials[tid] = acc;
}

double wall_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec*1e-6; }
static int dcmp(const void*a,const void*b){ double d=*(const double*)a-*(const double*)b; return (d>0)-(d<0);}
static double vmin(const double*a,int n){ double m=a[0]; for(int i=1;i<n;i++) if(a[i]<m)m=a[i]; return m;}
static double vstd(const double*a,int n){ double mu=0; for(int i=0;i<n;i++)mu+=a[i]; mu/=n; double s=0;
  for(int i=0;i<n;i++){double d=a[i]-mu; s+=d*d;} return sqrt(s/n);}
static double vmed(const double*a,int n){ double*b=(double*)malloc(n*sizeof(double));
  for(int i=0;i<n;i++)b[i]=a[i]; qsort(b,n,sizeof(double),dcmp); double m=b[n/2]; free(b); return m;}

int main(void){
    CHECK(cudaSetDevice(0));
    cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p,0));
    printf("GPU 0: %s   (American put, CRR tree STEPS=%d)\n\n", p.name, STEPS);
    float* dP=NULL; CHECK(cudaMalloc(&dP,(size_t)N_THREADS*sizeof(float)));
    float* hP=(float*)malloc((size_t)N_THREADS*sizeof(float));

    long long sizes[]={10000LL,100000LL,1000000LL};
    const int reps=50; double times[64];
    printf("%12s  %10s  %12s  %10s  %10s\n","Options","min(ms)","median(ms)","std(ms)","MeanPrice");
    printf("--------------------------------------------------------------------\n");
    for(int s=0;s<3;s++){
        long long n=sizes[s];
        for(int w=0;w<5;w++) american_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(dP,n);
        CHECK(cudaDeviceSynchronize());
        for(int r=0;r<reps;r++){ double t0=wall_ms();
            american_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(dP,n);
            CHECK(cudaDeviceSynchronize()); times[r]=wall_ms()-t0; }
        CHECK(cudaMemcpy(hP,dP,(size_t)N_THREADS*sizeof(float),cudaMemcpyDeviceToHost));
        double tot=0; for(int j=0;j<N_THREADS;j++) tot+=(double)hP[j];
        printf("%12lld  %10.4f  %12.4f  %10.4f  %10.5f\n",
               n,vmin(times,reps),vmed(times,reps),vstd(times,reps),tot/(double)n);
    }
    CHECK(cudaFree(dP)); free(hP); return 0;
}
