/*
 * Monte Carlo option pricer -- CUDA C reference kernel.
 * Targets GPU 0 only; the second GPU is never touched.
 *
 * Normalized benchmark (matches mc_mojo.mojo / mc_pytorch.py):
 *   * identical algorithm to mc_mojo.mojo: LCG -> Box-Muller -> discounted payoff
 *   * grid-stride: 262,144 threads, each accumulates its paths in a register and
 *     writes one partial sum (the bulk of the reduction happens in-register)
 *   * device buffers allocated once, before the timed region
 *   * timed region = the MC compute kernel; the tiny 262,144-element finish that
 *     produces the price is done once, outside timing
 *   * 5 warmup launches, then the average of 20 timed launches
 *
 * Compile:
 *   nvcc -O3 -arch=sm_89 mc_cuda.cu -o mc_cuda    # RTX 6000 Ada / 4090 (sm_89)
 *   nvcc -O3 -arch=sm_90 mc_cuda.cu -o mc_cuda    # H100 (sm_90)
 *   nvcc -O3 -arch=sm_80 mc_cuda.cu -o mc_cuda    # A100 (sm_80)
 *
 * Run:
 *   ./mc_cuda
 *
 * Note: no -lcurand needed -- this version uses the same hand-rolled LCG as the
 * Mojo kernel so the two measure identical arithmetic.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 256
#define N_THREADS  (1 << 18)            /* 262,144 resident threads */
#define GRID_SIZE  (N_THREADS / BLOCK_SIZE)

#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t e = (call);                                        \
        if (e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(e));        \
            exit(1);                                                   \
        }                                                              \
    } while (0)

/* ── per-thread LCG + Box-Muller normal (identical to mc_mojo.mojo) ─────────── */
__device__ __forceinline__ unsigned lcg(unsigned seed) {
    return seed * 1664525u + 1013904223u;
}

__device__ __forceinline__ float uniformf(unsigned seed) {
    /* reinterpret top 23 bits as a float in [1, 2), then shift into (0, 1) */
    return __uint_as_float((seed >> 9) | 0x3F800000u) - 1.0f;
}

__device__ __forceinline__ float normalf(unsigned seed) {
    unsigned s1 = lcg(seed);
    unsigned s2 = lcg(s1);
    float u1 = uniformf(s1) + 1e-7f;     /* avoid log(0) */
    float u2 = uniformf(s2);
    float mag = sqrtf(-2.0f * logf(u1));
    return mag * cosf(2.0f * 3.14159265358979f * u2);
}

/* ── GPU kernel: grid-stride accumulation, one partial sum per thread ───────── */
__global__ void mc_kernel(float* partials, long long n_paths,
                          float S, float K, float r, float sigma, float T)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_THREADS) return;

    float acc = 0.0f;
    for (long long i = tid; i < n_paths; i += N_THREADS) {
        unsigned seed = ((unsigned)i * 2654435761u) ^ 0xDEADBEEFu;
        float z  = normalf(seed);
        float ST = S * expf((r - 0.5f * sigma * sigma) * T
                            + sigma * sqrtf(T) * z);
        acc += fmaxf(ST - K, 0.0f) * expf(-r * T);
    }
    partials[tid] = acc;
}

double wall_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

/* ── summary statistics over the per-launch timing samples ─────────────────── */
static int dcmp(const void* a, const void* b) {
    double d = *(const double*)a - *(const double*)b;
    return (d > 0) - (d < 0);
}
static double vmin(const double* a, int n) {
    double m = a[0];
    for (int i = 1; i < n; i++) if (a[i] < m) m = a[i];
    return m;
}
static double vstd(const double* a, int n) {
    double mu = 0.0;
    for (int i = 0; i < n; i++) mu += a[i];
    mu /= n;
    double s = 0.0;
    for (int i = 0; i < n; i++) { double d = a[i] - mu; s += d * d; }
    return sqrt(s / n);                 /* population standard deviation */
}
static double vmedian(const double* a, int n) {
    double* b = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) b[i] = a[i];
    qsort(b, n, sizeof(double), dcmp);
    double m = b[n / 2];
    free(b);
    return m;
}

int main(void) {
    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count));
    printf("GPUs detected: %d  -- using GPU 0 only\n", device_count);
    CHECK(cudaSetDevice(0));   /* always use GPU 0 */

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU 0: %s\n\n", prop.name);

    long long sizes[] = {1000000LL, 10000000LL, 100000000LL};
    const int reps = 50;
    double times[64];

    /* buffers allocated ONCE, before any timing */
    float* d_partials = NULL;
    CHECK(cudaMalloc(&d_partials, (size_t)N_THREADS * sizeof(float)));
    float* h_partials = (float*)malloc((size_t)N_THREADS * sizeof(float));

    printf("%12s  %10s  %12s  %10s  %8s\n",
           "Paths", "min(ms)", "median(ms)", "std(ms)", "Price");
    printf("%s\n", "------------------------------------------------------------");

    for (int s = 0; s < 3; s++) {
        long long n = sizes[s];

        /* warm up */
        for (int w = 0; w < 5; w++)
            mc_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_partials, n,
                                                 100.f, 100.f, 0.05f, 0.2f, 1.0f);
        CHECK(cudaDeviceSynchronize());

        /* ── timed region: per-launch wall time of the MC compute kernel ── */
        for (int r = 0; r < reps; r++) {
            double t0 = wall_ms();
            mc_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_partials, n,
                                                 100.f, 100.f, 0.05f, 0.2f, 1.0f);
            CHECK(cudaDeviceSynchronize());
            times[r] = wall_ms() - t0;
        }

        /* price (untimed): copy the small partials array back and finish on host */
        CHECK(cudaMemcpy(h_partials, d_partials,
                         (size_t)N_THREADS * sizeof(float), cudaMemcpyDeviceToHost));
        double total = 0.0;
        for (int j = 0; j < N_THREADS; j++) total += (double)h_partials[j];
        double price = total / (double)n;

        printf("%12lld  %10.4f  %12.4f  %10.4f  %8.4f\n",
               n, vmin(times, reps), vmedian(times, reps), vstd(times, reps), price);
    }

    CHECK(cudaFree(d_partials));
    free(h_partials);
    return 0;
}
