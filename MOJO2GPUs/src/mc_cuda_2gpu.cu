/*
 * Monte Carlo option pricer -- CUDA C, TWO-GPU data-parallel version.
 *
 * Splits the n paths into two contiguous global-index ranges:
 *     GPU 0 -> [0, n/2),   GPU 1 -> [n/2, n).
 * Each device runs the same grid-stride kernel over its range using GLOBAL path
 * indices, so the per-path seeding and payoff are byte-for-byte the same paths the
 * single-GPU version computes; the two device kernels run CONCURRENTLY (launched
 * back-to-back, then both synchronized), and the host adds the two device totals in
 * a fixed order (device 0 then device 1). Timed region = the concurrent two-device
 * compute; device buffers are allocated once, outside timing.
 *
 * Compile:
 *   nvcc -O3 -arch=sm_89 -use_fast_math mc_cuda_2gpu.cu -o mc_cuda_2gpu
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 256
#define N_THREADS  (1 << 18)            /* 262,144 resident threads PER GPU */
#define GRID_SIZE  (N_THREADS / BLOCK_SIZE)
#define NGPU 2

#define CHECK(call)                                                    \
    do {                                                               \
        cudaError_t e = (call);                                        \
        if (e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(e));        \
            exit(1);                                                   \
        }                                                              \
    } while (0)

/* ── per-thread LCG + Box-Muller (identical to the single-GPU kernel) ───────── */
__device__ __forceinline__ unsigned lcg(unsigned seed) {
    return seed * 1664525u + 1013904223u;
}
__device__ __forceinline__ float uniformf(unsigned seed) {
    return __uint_as_float((seed >> 9) | 0x3F800000u) - 1.0f;
}
__device__ __forceinline__ float normalf(unsigned seed) {
    unsigned s1 = lcg(seed);
    unsigned s2 = lcg(s1);
    float u1 = uniformf(s1) + 1e-7f;
    float u2 = uniformf(s2);
    float mag = sqrtf(-2.0f * logf(u1));
    return mag * cosf(2.0f * 3.14159265358979f * u2);
}

/* ── grid-stride over a GLOBAL index range [lo, hi) ─────────────────────────── */
__global__ void mc_kernel(float* partials, long long lo, long long hi,
                          float S, float K, float r, float sigma, float T)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N_THREADS) return;
    float acc = 0.0f;
    for (long long i = lo + tid; i < hi; i += N_THREADS) {
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
    return sqrt(s / n);
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
    printf("GPUs detected: %d  -- using %d GPU(s)\n", device_count, NGPU);
    if (device_count < NGPU) {
        fprintf(stderr, "need %d GPUs, found %d\n", NGPU, device_count);
        return 1;
    }
    for (int d = 0; d < NGPU; d++) {
        cudaDeviceProp prop;
        CHECK(cudaSetDevice(d));
        CHECK(cudaGetDeviceProperties(&prop, d));
        printf("GPU %d: %s\n", d, prop.name);
    }
    printf("\n");

    long long sizes[] = {1000000LL, 10000000LL, 100000000LL};
    const int reps = 50;
    double times[64];

    /* per-device partial buffers, allocated ONCE before timing */
    float* d_partials[NGPU];
    float* h_partials[NGPU];
    for (int d = 0; d < NGPU; d++) {
        CHECK(cudaSetDevice(d));
        CHECK(cudaMalloc(&d_partials[d], (size_t)N_THREADS * sizeof(float)));
        h_partials[d] = (float*)malloc((size_t)N_THREADS * sizeof(float));
    }

    printf("%12s  %10s  %12s  %10s  %8s\n",
           "Paths", "min(ms)", "median(ms)", "std(ms)", "Price");
    printf("%s\n", "------------------------------------------------------------");

    for (int s = 0; s < 3; s++) {
        long long n   = sizes[s];
        long long mid = n / 2;
        long long lo[NGPU] = {0,   mid};
        long long hi[NGPU] = {mid, n};

        /* warm up both devices */
        for (int w = 0; w < 5; w++)
            for (int d = 0; d < NGPU; d++) {
                CHECK(cudaSetDevice(d));
                mc_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_partials[d], lo[d], hi[d],
                                                     100.f, 100.f, 0.05f, 0.2f, 1.0f);
            }
        for (int d = 0; d < NGPU; d++) { CHECK(cudaSetDevice(d)); CHECK(cudaDeviceSynchronize()); }

        /* ── timed region: launch both devices concurrently, then sync both ── */
        for (int r = 0; r < reps; r++) {
            double t0 = wall_ms();
            for (int d = 0; d < NGPU; d++) {            /* async launches, no sync between */
                CHECK(cudaSetDevice(d));
                mc_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_partials[d], lo[d], hi[d],
                                                     100.f, 100.f, 0.05f, 0.2f, 1.0f);
            }
            for (int d = 0; d < NGPU; d++) {            /* wait for both to finish */
                CHECK(cudaSetDevice(d));
                CHECK(cudaDeviceSynchronize());
            }
            times[r] = wall_ms() - t0;
        }

        /* price (untimed): copy both partials back and add in fixed order */
        double total = 0.0;
        for (int d = 0; d < NGPU; d++) {
            CHECK(cudaSetDevice(d));
            CHECK(cudaMemcpy(h_partials[d], d_partials[d],
                             (size_t)N_THREADS * sizeof(float), cudaMemcpyDeviceToHost));
            for (int j = 0; j < N_THREADS; j++) total += (double)h_partials[d][j];
        }
        double price = total / (double)n;

        printf("%12lld  %10.4f  %12.4f  %10.4f  %8.4f\n",
               n, vmin(times, reps), vmedian(times, reps), vstd(times, reps), price);
    }

    for (int d = 0; d < NGPU; d++) {
        CHECK(cudaSetDevice(d));
        CHECK(cudaFree(d_partials[d]));
        free(h_partials[d]);
    }
    return 0;
}
