// BF16 GEMM (cuBLAS) — MLP shape: (M x K) @ (K x N), K=4096, N=12288
// Sweeps M = {Batch * Sequence} values. Output dtype BF16, accumulation FP32.
//
// Build : ./build.sh         (or  nvcc -O3 -std=c++17 -arch=sm_120 -lcublas mm_test.cu -o mm_test)
// Run   : ./run.sh           (binds GPU 3 + power log + run log + segments csv)
//         ./mm_test 2048 4096 8192        # custom M sweep
//
// Env vars
//   MM_GPU      = device index (default 0; with CUDA_VISIBLE_DEVICES, target = 0)
//   MM_WARMUP   = warmup iters per M  (default 5)
//   MM_ITERS    = floor for measured iters per M  (default 50)
//   MM_MIN_MS   = min wall-clock duration per M; iters auto-scale up (default 500)
//                 — ensures every M produces enough nvidia-smi samples
//   MM_GAP_MS   = idle gap (host sleep) between M values (default 1000)
//                 — leaves a visible dip in the power trace at M boundaries
//   MM_SEGMENTS = path to write per-M segment CSV (default none).
//                 columns: M,K,N,iters,t_start,t_end,ms_avg,tflops
//                 timestamps match nvidia-smi format: "YYYY/MM/DD HH:MM:SS.fff"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t _e = (call);                                             \
    if (_e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(_e));                                 \
        std::exit(1);                                                    \
    }                                                                    \
} while (0)

#define CUBLAS_CHECK(call) do {                                          \
    cublasStatus_t _s = (call);                                          \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuBLAS error %s:%d code=%d\n",                  \
                __FILE__, __LINE__, (int)_s);                            \
        std::exit(1);                                                    \
    }                                                                    \
} while (0)

__global__ void init_bf16(__nv_bfloat16* p, size_t n, uint64_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t x = i * 6364136223846793005ULL + seed;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    float v = ((int32_t)(x & 0xFFFF) - 32768) / 32768.0f * 0.1f;
    p[i] = __float2bfloat16(v);
}

static void launch_init(__nv_bfloat16* p, size_t n, uint64_t seed) {
    int tpb = 256;
    size_t blocks = (n + tpb - 1) / tpb;
    init_bf16<<<(unsigned int)blocks, tpb>>>(p, n, seed);
}

// Wall-clock timestamp in nvidia-smi "YYYY/MM/DD HH:MM:SS.fff" format (local time)
static void now_string(char* buf, size_t n) {
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    std::tm tmv{};
    localtime_r(&ts.tv_sec, &tmv);
    int ms = (int)(ts.tv_nsec / 1000000);
    std::snprintf(buf, n, "%04d/%02d/%02d %02d:%02d:%02d.%03d",
                  tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
                  tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ms);
}

int main(int argc, char** argv) {
    int device = 0;
    if (const char* env = std::getenv("MM_GPU")) device = std::atoi(env);
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    fprintf(stderr,
            "[mm_test] device=%d  name=%s  SM=%d.%d  multiProcessors=%d\n",
            device, prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    const int K = 4096;
    const int N = 12288;

    std::vector<int> Ms = {1, 32, 128, 512, 1024, 2048, 4096, 8192, 16384};
    if (argc > 1) {
        Ms.clear();
        for (int i = 1; i < argc; ++i) Ms.push_back(std::atoi(argv[i]));
    }

    int warmup       = 5;
    int iters_floor  = 50;
    double min_ms    = 500.0;
    int gap_ms       = 1000;
    const char* segments_path = std::getenv("MM_SEGMENTS");
    if (const char* env = std::getenv("MM_WARMUP")) warmup = std::atoi(env);
    if (const char* env = std::getenv("MM_ITERS"))  iters_floor = std::atoi(env);
    if (const char* env = std::getenv("MM_MIN_MS")) min_ms = std::atof(env);
    if (const char* env = std::getenv("MM_GAP_MS")) gap_ms = std::atoi(env);

    fprintf(stderr,
            "[mm_test] warmup=%d  iters_floor=%d  min_ms=%.1f  gap_ms=%d  segments=%s\n",
            warmup, iters_floor, min_ms, gap_ms,
            segments_path ? segments_path : "(none)");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int Mmax = 0;
    for (int m : Ms) if (m > Mmax) Mmax = m;

    size_t elemsA = (size_t)Mmax * K;
    size_t elemsB = (size_t)K    * N;
    size_t elemsC = (size_t)Mmax * N;

    __nv_bfloat16 *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, elemsA * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dB, elemsB * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dC, elemsC * sizeof(__nv_bfloat16)));

    launch_init(dA, elemsA, 1);
    launch_init(dB, elemsB, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const float alpha = 1.0f, beta = 0.0f;

    // cuBLAS is column-major. We have row-major A(MxK), B(KxN), C(MxN).
    // Compute C^T = B^T @ A^T in column-major  ==>  pass B first, A second,
    // with m=N, n=M, k=K, leading dims (N, K, N).
    auto run_gemm = [&](int M) {
        return cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, CUDA_R_16BF, N,
            dA, CUDA_R_16BF, K,
            &beta,
            dC, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
    };

    FILE* segf = nullptr;
    if (segments_path) {
        segf = std::fopen(segments_path, "w");
        if (!segf) {
            fprintf(stderr, "[mm_test] failed to open segments file: %s\n", segments_path);
            std::exit(1);
        }
        std::fprintf(segf, "M,K,N,iters,t_start,t_end,ms_avg,tflops\n");
        std::fflush(segf);
    }

    std::printf("%-8s %-6s %-7s %-8s %-10s %-10s %-24s %-24s\n",
                "M", "K", "N", "iters", "ms_avg", "TFLOPS", "t_start", "t_end");
    std::printf("------------------------------------------------------------------------------------------\n");
    std::fflush(stdout);

    for (size_t i = 0; i < Ms.size(); ++i) {
        int M = Ms[i];

        // ---- warmup + auto-scale iters ----
        for (int w = 0; w < warmup; ++w) CUBLAS_CHECK(run_gemm(M));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Probe: 10 iters to estimate per-iter cost
        const int probe = 10;
        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < probe; ++it) CUBLAS_CHECK(run_gemm(M));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float probe_ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&probe_ms, start, stop));
        double ms_per_iter_est = probe_ms / probe;

        int iters = iters_floor;
        if (ms_per_iter_est > 0.0) {
            int needed = (int)((min_ms / ms_per_iter_est) + 0.999);
            if (needed > iters) iters = needed;
        }

        // ---- measured run with wall-clock bracketing ----
        char ts_start[32], ts_end[32];
        now_string(ts_start, sizeof ts_start);

        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < iters; ++it) CUBLAS_CHECK(run_gemm(M));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        now_string(ts_end, sizeof ts_end);

        float ms_total = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
        double ms_avg = ms_total / iters;
        double flops  = 2.0 * (double)M * K * N;
        double tflops = flops / (ms_avg * 1e-3) / 1e12;

        std::printf("%-8d %-6d %-7d %-8d %-10.4f %-10.2f %-24s %-24s\n",
                    M, K, N, iters, ms_avg, tflops, ts_start, ts_end);
        std::fflush(stdout);

        if (segf) {
            std::fprintf(segf, "%d,%d,%d,%d,%s,%s,%.6f,%.4f\n",
                         M, K, N, iters, ts_start, ts_end, ms_avg, tflops);
            std::fflush(segf);
        }

        // ---- idle gap so power trace dips between M values ----
        if (gap_ms > 0 && i + 1 < Ms.size()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
        }
    }

    if (segf) std::fclose(segf);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cublasDestroy(handle);
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
