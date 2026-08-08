/*
 * Path-dependent Monte Carlo: arithmetic-average ASIAN call -- CUDA C reference.
 *
 * Each path simulates a full GBM trajectory of M_STEPS steps, averages the price
 * along the path, and discounts the payoff max(avg - K, 0). This is far more
 * arithmetic-intensive than the European kernel (M_STEPS normal draws per path) and
 * exercises a per-path sequential RNG, a harder test of language efficiency.
 *
 * Index-addressed RNG: path i seeds a per-path LCG state = hash(i); the state is
 * advanced through the trajectory, so payoffs are a deterministic function of i.
 *
 * Compile: nvcc -O3 -arch=sm_89 -use_fast_math asian_cuda.cu -o asian_cuda
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 128
#define N_THREADS  (1 << 18)
#define GRID_SIZE  (N_THREADS / BLOCK_SIZE)
#define M_STEPS    100

#define CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__device__ __forceinline__ unsigned lcg(unsigned s){ return s*1664525u+1013904223u; }
__device__ __forceinline__ float uf(unsigned s){ return __uint_as_float((s>>9)|0x3F800000u)-1.0f; }
__device__ __forceinline__ float normal_step(unsigned &state){
    unsigned s1 = lcg(state);
    unsigned s2 = lcg(s1);
    state = s2;
    float u1 = uf(s1)+1e-7f, u2 = uf(s2);
    return sqrtf(-2.0f*logf(u1))*cosf(2.0f*3.14159265358979f*u2);
}

__global__ void asian_kernel(float* partials, long long n_paths,
                             float S0, float K, float r, float sigma, float T)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if (tid >= N_THREADS) return;
    const float dt   = T / M_STEPS;
    const float drift= (r - 0.5f*sigma*sigma)*dt;
    const float vol  = sigma*sqrtf(dt);
    const float disc = expf(-r*T);
    float acc = 0.0f;
    for (long long i = tid; i < n_paths; i += N_THREADS) {
        unsigned state = ((unsigned)i*2654435761u) ^ 0xDEADBEEFu;
        float S = S0, sumS = 0.0f;
        for (int k = 0; k < M_STEPS; k++) {
            S *= expf(drift + vol*normal_step(state));
            sumS += S;
        }
        float avg = sumS / (float)M_STEPS;
        acc += fmaxf(avg - K, 0.0f) * disc;
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
    printf("GPU 0: %s   (Asian, M_STEPS=%d)\n\n", p.name, M_STEPS);

    long long sizes[]={100000LL,1000000LL,10000000LL};
    const int reps=50; double times[64];
    float* dP=NULL; CHECK(cudaMalloc(&dP,(size_t)N_THREADS*sizeof(float)));
    float* hP=(float*)malloc((size_t)N_THREADS*sizeof(float));

    printf("%12s  %10s  %12s  %10s  %8s\n","Paths","min(ms)","median(ms)","std(ms)","Price");
    printf("------------------------------------------------------------\n");
    for(int s=0;s<3;s++){
        long long n=sizes[s];
        for(int w=0;w<5;w++) asian_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(dP,n,100.f,100.f,0.05f,0.2f,1.f);
        CHECK(cudaDeviceSynchronize());
        for(int r=0;r<reps;r++){ double t0=wall_ms();
            asian_kernel<<<GRID_SIZE,BLOCK_SIZE>>>(dP,n,100.f,100.f,0.05f,0.2f,1.f);
            CHECK(cudaDeviceSynchronize()); times[r]=wall_ms()-t0; }
        CHECK(cudaMemcpy(hP,dP,(size_t)N_THREADS*sizeof(float),cudaMemcpyDeviceToHost));
        double tot=0; for(int j=0;j<N_THREADS;j++) tot+=(double)hP[j];
        printf("%12lld  %10.4f  %12.4f  %10.4f  %8.4f\n",
               n,vmin(times,reps),vmed(times,reps),vstd(times,reps),tot/(double)n);
    }
    CHECK(cudaFree(dP)); free(hP); return 0;
}
