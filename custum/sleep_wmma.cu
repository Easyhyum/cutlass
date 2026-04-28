/*
 * WMMA GEMM with configurable __nanosleep between Tensor Core operations
 *
 * Purpose: Study power consumption patterns during MLP matrix multiplication
 *          by controlling the duty cycle of Tensor Core (WMMA) operations.
 *
 * Kernels:
 *   1. wmma_gemm_sleep_kernel          - Simple global-memory WMMA GEMM
 *   2. wmma_gemm_sleep_smem_kernel     - Shared-memory tiled 64×64
 *   3. wmma_stress_kernel              - Pure WMMA loop for power profiling
 *   4. wmma_gemm_opt_kernel            - HIGH-PERF: 128×128 block, 2×4 warp
 *                                        tiles, double-buffered smem, vec loads
 *   5. wmma_gemm_skinny_kernel         - Optimized for small M (≤16)
 *
 * GPU target : NVIDIA RTX PRO 6000 Blackwell  (sm_120, cc 12.0)
 * CUDA       : 13.0
 */

 #include <torch/extension.h>
 #include <cuda_runtime.h>
 #include <mma.h>
 #include <cuda_fp16.h>
 
 using namespace nvcuda;
 
 // ======================== Constants ========================
 static constexpr int WMMA_M = 16;
 static constexpr int WMMA_N = 16;
 static constexpr int WMMA_K = 16;
 static constexpr int WARP_SIZE = 32;
 
 // Optimized kernel constants
 static constexpr int OPT_BLOCK_M = 128;
 static constexpr int OPT_BLOCK_N = 128;
 static constexpr int OPT_BLOCK_K = 32;
 static constexpr int OPT_WARP_M  = 32;   // each warp: 2 WMMA tiles in M
 static constexpr int OPT_WARP_N  = 64;   // each warp: 4 WMMA tiles in N
 
 // Skinny kernel constants
 static constexpr int SKINNY_BLOCK_K = 64;
 
 __device__ __forceinline__ void sleep_cycles(unsigned long long cycles) {
     unsigned long long start = clock64();
     while ((clock64() - start) < cycles) {
         // busy wait
     }
 }
 
 // ======================== Kernel 1: Simple WMMA GEMM ========================
 __global__ void wmma_gemm_sleep_kernel(
     const half* __restrict__ A,
     const half* __restrict__ B,
     float*      __restrict__ C,
     int M, int N, int K,
     int sleep_ns, int sleep_freq)
 {
     // Each warp handles one 16×16 output tile
     const int warpId   = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
     const int tilesM   = M / WMMA_M;
     const int tilesN   = N / WMMA_N;
     const int totalTiles = tilesM * tilesN;
     if (warpId >= totalTiles) return;
 
     const int tileRow = warpId / tilesN;
     const int tileCol = warpId % tilesN;
 
     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
     wmma::fill_fragment(acc, 0.0f);
 
     const int aRow = tileRow * WMMA_M;
     const int bCol = tileCol * WMMA_N;
     int mma_count = 0;
 
     for (int k = 0; k < K; k += WMMA_K) {
         wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
         wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
 
         wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
         wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);
         wmma::mma_sync(acc, a_frag, b_frag, acc);
         if (sleep_ns > 0) {
             mma_count++;
             if (mma_count % sleep_freq == 0) {
                 __nanosleep(static_cast<unsigned>(sleep_ns));
                 // sleep_cycles((unsigned long long)(sleep_ns));
             }
         }
     }
    //  if (sleep_ns > 0) {
    //     __nanosleep(static_cast<unsigned>(sleep_ns));
    //     // __syncthreads();
    //     // cg::tiled_partition<32>(cg::this_thread_block()).sync();
    //  }
     wmma::store_matrix_sync(C + aRow * N + bCol, acc, N, wmma::mem_row_major);
     // if (sleep_ns > 0) {
     //     // mma_count++;
     //     // if (mma_count % sleep_freq == 0) {
     //         __nanosleep(static_cast<unsigned>(sleep_ns));
     //         // sleep_cycles((unsigned long long)(sleep_ns));
     //     // }
     // }
 }
 
 // ======================== Kernel 1b: Time-check WMMA GEMM (warp별 start/end 시각 기록) ========================
// wmma_gemm_sleep_kernel 과 동일한 연산을 수행하되,
// GPU 전역 타이머(globaltimer, 1 ns 해상도, SM 간 동기화) 로
// 각 warp 의 절대 시작·종료 시각(ns)을 start_buf / end_buf 에 각각 기록한다.
//
// globaltimer vs clock64:
//   clock64()    – SM별 사이클 카운터, 다른 SM 간 비교 불가, 클럭 주파수 변환 필요
//   globaltimer  – GPU 전역 ns 타이머, SM 간 절대 시각 비교 가능, 변환 불필요
__global__ void wmma_gemm_sleep_kernel_time_check(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ C,
    int M, int N, int K,
    int sleep_ns, int sleep_freq,
    unsigned long long* __restrict__ start_buf,
    unsigned long long* __restrict__ end_buf)
{
    const int warpId     = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int laneId     = threadIdx.x % WARP_SIZE;
    const int tilesM     = M / WMMA_M;
    const int tilesN     = N / WMMA_N;
    const int totalTiles = tilesM * tilesN;
    if (warpId >= totalTiles) return;

    const int tileRow = warpId / tilesN;
    const int tileCol = warpId % tilesN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    const int aRow = tileRow * WMMA_M;
    const int bCol = tileCol * WMMA_N;

    // warp 시작 절대 시각 기록 (GPU 전역 ns 타이머)
    unsigned long long t_start;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

        wmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + bCol, N);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
    if (sleep_ns > 0) {
        __nanosleep(static_cast<unsigned>(sleep_ns));
    }
    wmma::store_matrix_sync(C + aRow * N + bCol, acc, N, wmma::mem_row_major);

    // warp 종료 절대 시각 기록 (lane 0 만 쓰기)
    unsigned long long t_end;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
    if (laneId == 0) {
        start_buf[warpId] = t_start;
        end_buf[warpId]   = t_end;
    }
}

// ======================== Kernel 2: Shared-memory WMMA GEMM (64×64) ========================
 __global__ void wmma_gemm_sleep_smem_kernel(
     const half* __restrict__ A,
     const half* __restrict__ B,
     float*      __restrict__ C,
     int M, int N, int K,
     int sleep_ns, int sleep_freq)
 {
     constexpr int BLOCK_M = 64;
     constexpr int BLOCK_N = 64;
     constexpr int BLOCK_K = 16;
 
     __shared__ half smemA[BLOCK_M * BLOCK_K];
     __shared__ half smemB[BLOCK_K * BLOCK_N];
 
     const int blockRow = blockIdx.y * BLOCK_M;
     const int blockCol = blockIdx.x * BLOCK_N;
 
     const int warpId    = threadIdx.x / WARP_SIZE;
     const int warpRow   = (warpId / 4) * WMMA_M;   // 4 warps per row
     const int warpCol   = (warpId % 4) * WMMA_N;
 
     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
     wmma::fill_fragment(acc, 0.0f);
 
     int mma_count = 0;
     const int tid = threadIdx.x;
     const int numThreads = blockDim.x;  // 512
 
     for (int kStart = 0; kStart < K; kStart += BLOCK_K) {
         // Cooperative load: A tile [BLOCK_M × BLOCK_K]
         for (int i = tid; i < BLOCK_M * BLOCK_K; i += numThreads) {
             int r = i / BLOCK_K;
             int c = i % BLOCK_K;
             int gr = blockRow + r;
             int gc = kStart + c;
             smemA[i] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
         }
         // Cooperative load: B tile [BLOCK_K × BLOCK_N]
         for (int i = tid; i < BLOCK_K * BLOCK_N; i += numThreads) {
             int r = i / BLOCK_N;
             int c = i % BLOCK_N;
             int gr = kStart + r;
             int gc = blockCol + c;
             smemB[i] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
         }
         __syncthreads();
 
         wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
         wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
 
         wmma::load_matrix_sync(a_frag, smemA + warpRow * BLOCK_K, BLOCK_K);
         wmma::load_matrix_sync(b_frag, smemB + warpCol, BLOCK_N);
         wmma::mma_sync(acc, a_frag, b_frag, acc);
 
         // if (sleep_ns > 0) {
         //     mma_count++;
         //     if (mma_count % sleep_freq == 0) {
         //         // __nanosleep(static_cast<unsigned>(sleep_ns));
         //         sleep_cycles((unsigned long long)(sleep_ns));
         //     }
         // }
         __syncthreads();
     }
     if (sleep_ns > 0) {
         // mma_count++;
         // if (mma_count % sleep_freq == 0) {
             __nanosleep(static_cast<unsigned>(sleep_ns));
             // sleep_cycles((unsigned long long)(sleep_ns));
         // }
     }
     const int outRow = blockRow + warpRow;
     const int outCol = blockCol + warpCol;
     if (outRow < M && outCol < N) {
         wmma::store_matrix_sync(C + outRow * N + outCol, acc, N, wmma::mem_row_major);
     }
 }
 
 // ======================== Kernel 3: Stress test (pure WMMA loop) ========================
 __global__ void wmma_stress_kernel(int num_iters, int sleep_ns, int sleep_freq)
 {
     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a;
     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b;
     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c;
 
     wmma::fill_fragment(a, __float2half(1.0f));
     wmma::fill_fragment(b, __float2half(1.0f));
     wmma::fill_fragment(c, 0.0f);
 
     for (int i = 0; i < num_iters; ++i) {
         wmma::mma_sync(c, a, b, c);
         if (sleep_ns > 0 && (i + 1) % sleep_freq == 0) {
                 __nanosleep(static_cast<unsigned>(sleep_ns));
                 // sleep_cycles((unsigned long long)(sleep_ns));
         }
     }
 
     // Prevent dead-code elimination
     if (threadIdx.x == 0 && c.x[0] < -1e30f) {
         asm volatile("" :: "f"(c.x[0]));
     }
 }
 
 // ======================== Kernel 4: Optimized 128×128 WMMA GEMM ========================
 __global__ void wmma_gemm_opt_kernel(
     const half* __restrict__ A,
     const half* __restrict__ B,
     float*      __restrict__ C,
     int M, int N, int K,
     int sleep_ns, int sleep_freq)
 {
     // Block tile: 128×128, 8 warps (256 threads)
     // Each warp: 32×64 (2×4 WMMA fragments)
     constexpr int WARPS = 8;
     constexpr int WARPS_M = 4;  // 4 warps along M
     constexpr int WARPS_N = 2;  // 2 warps along N
 
     const int blockRow = blockIdx.y * OPT_BLOCK_M;
     const int blockCol = blockIdx.x * OPT_BLOCK_N;
     const int warpId   = threadIdx.x / WARP_SIZE;
     const int warpM    = warpId / WARPS_N;  // 0..3
     const int warpN    = warpId % WARPS_N;  // 0..1
 
     // Double-buffered shared memory
     // A: transposed for col-major load [BLOCK_K][BLOCK_M + 8 padding]
     // B: natural row-major [BLOCK_K][BLOCK_N + 8 padding]
     __shared__ half smemA[2][OPT_BLOCK_K][OPT_BLOCK_M + 8];
     __shared__ half smemB[2][OPT_BLOCK_K][OPT_BLOCK_N + 8];
 
     // Accumulator fragments: 2×4 per warp
     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][4];
     #pragma unroll
     for (int i = 0; i < 2; i++)
         #pragma unroll
         for (int j = 0; j < 4; j++)
             wmma::fill_fragment(acc[i][j], 0.0f);
 
     const int tid = threadIdx.x;
     const int numThreads = WARPS * WARP_SIZE;  // 256
     int mma_count = 0;
 
     // Load first K-tile into buffer 0
     auto load_tile = [&](int kStart, int buf) {
         // Load A: [BLOCK_M × BLOCK_K] → store transposed as [BLOCK_K][BLOCK_M]
         for (int i = tid; i < OPT_BLOCK_M * OPT_BLOCK_K; i += numThreads) {
             int r = i / OPT_BLOCK_K;   // row in A block
             int c = i % OPT_BLOCK_K;   // col in A block
             int gr = blockRow + r;
             int gc = kStart + c;
             smemA[buf][c][r] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
         }
         // Load B: [BLOCK_K × BLOCK_N]
         for (int i = tid; i < OPT_BLOCK_K * OPT_BLOCK_N; i += numThreads) {
             int r = i / OPT_BLOCK_N;
             int c = i % OPT_BLOCK_N;
             int gr = kStart + r;
             int gc = blockCol + c;
             smemB[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
         }
     };
 
     load_tile(0, 0);
     __syncthreads();
 
     for (int kStart = 0; kStart < K; kStart += OPT_BLOCK_K) {
         int curBuf = (kStart / OPT_BLOCK_K) & 1;
         int nxtBuf = curBuf ^ 1;
 
         // Prefetch next K-tile
         int nextK = kStart + OPT_BLOCK_K;
         if (nextK < K) {
             load_tile(nextK, nxtBuf);
         }
 
         // Compute: each warp does 2×4 WMMA tiles over 2 K-steps of 16
         #pragma unroll
         for (int kk = 0; kk < OPT_BLOCK_K; kk += WMMA_K) {
             #pragma unroll
             for (int mi = 0; mi < 2; mi++) {
                 wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
                 int localM = warpM * OPT_WARP_M + mi * WMMA_M;
                 // A is stored transposed: smemA[k][m], col-major load
                 wmma::load_matrix_sync(a_frag, &smemA[curBuf][kk][localM], OPT_BLOCK_M + 8);
 
                 #pragma unroll
                 for (int ni = 0; ni < 4; ni++) {
                     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
                     int localN = warpN * OPT_WARP_N + ni * WMMA_N;
                     wmma::load_matrix_sync(b_frag, &smemB[curBuf][kk][localN], OPT_BLOCK_N + 8);
                     wmma::mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                 }
             }
 
            //  if (sleep_ns > 0) {
            //      mma_count++;
            //      if (mma_count % sleep_freq == 0) {
            //      __nanosleep(static_cast<unsigned>(sleep_ns));
            //      // sleep_cycles((unsigned long long)(sleep_ns));
            //      }
            //  }
         }
         __syncthreads();
     }

     // Store accumulator fragments to global C
     #pragma unroll
     for (int mi = 0; mi < 2; mi++) {
         #pragma unroll
         for (int ni = 0; ni < 4; ni++) {
             int outRow = blockRow + warpM * OPT_WARP_M + mi * WMMA_M;
             int outCol = blockCol + warpN * OPT_WARP_N + ni * WMMA_N;
             if (outRow < M && outCol < N) {
                 wmma::store_matrix_sync(C + outRow * N + outCol, acc[mi][ni],
                                         N, wmma::mem_row_major);
             }
         }
     }
 }
 
 // ======================== Kernel 5: Skinny-M WMMA GEMM ========================
 __global__ void wmma_gemm_skinny_kernel(
     const half* __restrict__ A,
     const half* __restrict__ B,
     float*      __restrict__ C,
     int M, int N, int K,
     int sleep_ns, int sleep_freq)
 {
     // For small M: 8 warps share A tile, each warp handles separate N-tile
     // Block handles: [16 × (8*16)] output = [16 × 128]
     constexpr int WARPS = 8;
     constexpr int BLOCK_N = WARPS * WMMA_N;  // 128
 
     __shared__ half smemA[2][SKINNY_BLOCK_K][WMMA_M + 8];
     __shared__ half smemB[2][SKINNY_BLOCK_K][BLOCK_N + 8];
 
     const int blockRow = blockIdx.y * WMMA_M;
     const int blockCol = blockIdx.x * BLOCK_N;
     const int warpId   = threadIdx.x / WARP_SIZE;
     const int tid      = threadIdx.x;
     const int numThreads = WARPS * WARP_SIZE;  // 256
 
     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
     wmma::fill_fragment(acc, 0.0f);
 
     int mma_count = 0;
 
     auto load_skinny = [&](int kStart, int buf) {
         // Load A [WMMA_M × SKINNY_BLOCK_K] → transposed [K][M]
         for (int i = tid; i < WMMA_M * SKINNY_BLOCK_K; i += numThreads) {
             int r = i / SKINNY_BLOCK_K;
             int c = i % SKINNY_BLOCK_K;
             int gr = blockRow + r;
             int gc = kStart + c;
             smemA[buf][c][r] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
         }
         // Load B [SKINNY_BLOCK_K × BLOCK_N]
         for (int i = tid; i < SKINNY_BLOCK_K * BLOCK_N; i += numThreads) {
             int r = i / BLOCK_N;
             int c = i % BLOCK_N;
             int gr = kStart + r;
             int gc = blockCol + c;
             smemB[buf][r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
         }
     };
 
     load_skinny(0, 0);
     __syncthreads();
 
     for (int kStart = 0; kStart < K; kStart += SKINNY_BLOCK_K) {
         int curBuf = (kStart / SKINNY_BLOCK_K) & 1;
         int nxtBuf = curBuf ^ 1;
 
         int nextK = kStart + SKINNY_BLOCK_K;
         if (nextK < K) {
             load_skinny(nextK, nxtBuf);
         }
 
         #pragma unroll
         for (int kk = 0; kk < SKINNY_BLOCK_K; kk += WMMA_K) {
             wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
             wmma::load_matrix_sync(a_frag, &smemA[curBuf][kk][0], WMMA_M + 8);
 
             wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
             int localN = warpId * WMMA_N;
             wmma::load_matrix_sync(b_frag, &smemB[curBuf][kk][localN], BLOCK_N + 8);
 
             wmma::mma_sync(acc, a_frag, b_frag, acc);
 
             if (sleep_ns > 0) {
                 mma_count++;
                 if (mma_count % sleep_freq == 0) {
                 __nanosleep(static_cast<unsigned>(sleep_ns));
                 // sleep_cycles((unsigned long long)(sleep_ns));
                 }
             }
         }
         __syncthreads();
     }
 
     int outRow = blockRow;
     int outCol = blockCol + warpId * WMMA_N;
     if (outRow < M && outCol < N) {
         wmma::store_matrix_sync(C + outRow * N + outCol, acc, N, wmma::mem_row_major);
     }
 }
 
 // ======================== Kernel 6: CUTLASS-style High-Perf WMMA GEMM ========================
//
// CUTLASS 최고 성능 커널(s16816gemm_f16 계열)의 핵심 기법 적용:
//
//  vs wmma_gemm_opt_kernel 비교:
//  ┌──────────────────────────┬───────────────┬───────────────┐
//  │ 항목                     │  opt (K4)     │  hiperf (K6)  │
//  ├──────────────────────────┼───────────────┼───────────────┤
//  │ BLOCK_K                  │ 32            │ 64   (+2×)    │
//  │ Global→Shared 로드        │ scalar half   │ float4(128b)  │
//  │ Smem 레이아웃             │ 전치[K][M]    │ row-major[M][K]│
//  │ Smem 패딩                 │ +8            │ +16(16B 정렬) │
//  │ 스레드 수                 │ 256           │ 256           │
//  │ Warp 배치 (M×N)          │ 4×2           │ 4×2           │
//  │ 워프당 WMMA 타일          │ 2×4           │ 2×4           │
//  └──────────────────────────┴───────────────┴───────────────┘
//
//  개선 포인트:
//  1. BLOCK_K = 64: syncthreads 호출 횟수 절반 → 동기화 오버헤드 감소
//  2. float4 로드: 128-bit 메모리 명령 → 메모리 버스 활용률 4×
//  3. row-major smemA: 전치 없이 직접 저장 → 로드 로직 단순화
//  4. pad=16: 각 행이 16B 정렬 → float4 공유 메모리 저장 안전

// smem 크기 계산 (컴파일 타임 상수):
//   smemA[2][128][80]  × 2B = 40,960 B ≈ 40 KB
//   smemB[2][64][144]  × 2B = 36,864 B ≈ 36 KB
//   합계 ≈ 76 KB  → 96 KB carveout 필요 (SM80 이상 지원)

static constexpr int HP_BLOCK_M  = 128;
static constexpr int HP_BLOCK_N  = 128;
static constexpr int HP_BLOCK_K  = 64;
static constexpr int HP_WARPS_M  = 4;
static constexpr int HP_WARPS_N  = 2;
static constexpr int HP_WARPS    = HP_WARPS_M * HP_WARPS_N;  // 8 warps, 256 threads
static constexpr int HP_WARP_M   = HP_BLOCK_M / HP_WARPS_M;  // 32  → 2 WMMA tiles
static constexpr int HP_WARP_N   = HP_BLOCK_N / HP_WARPS_N;  // 64  → 4 WMMA tiles
static constexpr int HP_MMA_M    = HP_WARP_M / WMMA_M;       // 2
static constexpr int HP_MMA_N    = HP_WARP_N / WMMA_N;       // 4
// smem row stride: (BLOCK_K/N + 16) halfs → 16B-aligned for float4 stores
static constexpr int HP_PAD      = 16;

// ── Shared memory 레이아웃 상수 ─────────────────────────────────────
// 동적 smem 사용 이유:
//   정적 __shared__ 는 ptxas 가 컴파일 타임에 48 KB 한계를 적용.
//   extern __shared__ (동적) 는 컴파일 타임 체크를 건너뛰고
//   런타임에 cudaFuncAttributeMaxDynamicSharedMemorySize 로 96 KB 까지 허용.
//
// 레이아웃 (half 단위 stride):
//   smemA_stride = HP_BLOCK_K + HP_PAD = 64+16 = 80   (160 B → 16B 정렬 ✓)
//   smemB_stride = HP_BLOCK_N + HP_PAD = 128+16 = 144  (288 B → 16B 정렬 ✓)
//   smemA_buf_halfs = HP_BLOCK_M × smemA_stride = 128×80 = 10240
//   smemB_buf_halfs = HP_BLOCK_K × smemB_stride = 64×144 = 9216
//   total = 2×(10240+9216)×2 B = 77,824 B ≈ 76 KB
static constexpr int HP_SMEM_A_STRIDE    = HP_BLOCK_K + HP_PAD;          // 80
static constexpr int HP_SMEM_B_STRIDE    = HP_BLOCK_N + HP_PAD;          // 144
static constexpr int HP_SMEM_A_BUF_HALF  = HP_BLOCK_M * HP_SMEM_A_STRIDE; // 10240
static constexpr int HP_SMEM_B_BUF_HALF  = HP_BLOCK_K * HP_SMEM_B_STRIDE; // 9216
static constexpr size_t HP_SMEM_BYTES    =
    2 * (HP_SMEM_A_BUF_HALF + HP_SMEM_B_BUF_HALF) * sizeof(half);        // 77824

__global__ void __launch_bounds__(HP_WARPS * WARP_SIZE)
wmma_gemm_hiperf_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float*      __restrict__ C,
    int M, int N, int K,
    int sleep_ns, int sleep_freq)
{
    // 동적 shared memory: 컴파일 타임 48KB 제한 우회
    extern __shared__ char _smem_raw[];
    half* smemA = reinterpret_cast<half*>(_smem_raw);
    half* smemB = smemA + 2 * HP_SMEM_A_BUF_HALF;

    // smemA[buf][row][col] = smemA + buf*HP_SMEM_A_BUF_HALF + row*HP_SMEM_A_STRIDE + col
    // smemB[buf][row][col] = smemB + buf*HP_SMEM_B_BUF_HALF + row*HP_SMEM_B_STRIDE + col

    const int tid      = threadIdx.x;
    const int warpId   = tid / WARP_SIZE;
    const int warpM    = warpId / HP_WARPS_N;  // 0..3
    const int warpN    = warpId % HP_WARPS_N;  // 0..1

    const int blockRow = blockIdx.y * HP_BLOCK_M;
    const int blockCol = blockIdx.x * HP_BLOCK_N;
    const int numThreads = HP_WARPS * WARP_SIZE;  // 256

    // 워프당 2×4 accumulator
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[HP_MMA_M][HP_MMA_N];
    #pragma unroll
    for (int mi = 0; mi < HP_MMA_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < HP_MMA_N; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    // ── float4 (8 halfs) 로드 헬퍼 ──────────────────────────────
    // smemA: [HP_BLOCK_M × HP_BLOCK_K] = 128×64 = 8192 halfs = 1024 float4
    //   256 threads → 4 float4/thread
    // smemB: [HP_BLOCK_K × HP_BLOCK_N] = 64×128 = 8192 halfs = 1024 float4
    //   256 threads → 4 float4/thread

    auto load_A = [&](int kStart, int buf) {
        constexpr int TOTAL_F4 = HP_BLOCK_M * HP_BLOCK_K / 8;  // 1024
        half* dst = smemA + buf * HP_SMEM_A_BUF_HALF;
        #pragma unroll 4
        for (int i = tid; i < TOTAL_F4; i += numThreads) {
            const int row = i / (HP_BLOCK_K / 8);
            const int col = (i % (HP_BLOCK_K / 8)) * 8;  // 항상 float4 정렬
            const int gr  = blockRow + row;
            const int gc  = kStart   + col;
            float4 val = {};
            if (gr < M && gc + 7 < K) {
                val = *reinterpret_cast<const float4*>(A + gr * K + gc);
            } else if (gr < M) {
                const half* src = A + gr * K + gc;
                half* tmp = reinterpret_cast<half*>(&val);
                #pragma unroll
                for (int j = 0; j < 8 && gc + j < K; j++) tmp[j] = src[j];
            }
            // row stride = HP_SMEM_A_STRIDE = 80 halfs = 160 B (16B 정렬 ✓)
            *reinterpret_cast<float4*>(dst + row * HP_SMEM_A_STRIDE + col) = val;
        }
    };

    auto load_B = [&](int kStart, int buf) {
        constexpr int TOTAL_F4 = HP_BLOCK_K * HP_BLOCK_N / 8;  // 1024
        half* dst = smemB + buf * HP_SMEM_B_BUF_HALF;
        #pragma unroll 4
        for (int i = tid; i < TOTAL_F4; i += numThreads) {
            const int row = i / (HP_BLOCK_N / 8);
            const int col = (i % (HP_BLOCK_N / 8)) * 8;
            const int gr  = kStart   + row;
            const int gc  = blockCol + col;
            float4 val = {};
            if (gr < K && gc + 7 < N) {
                val = *reinterpret_cast<const float4*>(B + gr * N + gc);
            } else if (gr < K) {
                const half* src = B + gr * N + gc;
                half* tmp = reinterpret_cast<half*>(&val);
                #pragma unroll
                for (int j = 0; j < 8 && gc + j < N; j++) tmp[j] = src[j];
            }
            // row stride = HP_SMEM_B_STRIDE = 144 halfs = 288 B (16B 정렬 ✓)
            *reinterpret_cast<float4*>(dst + row * HP_SMEM_B_STRIDE + col) = val;
        }
    };

    // ── 첫 번째 타일 프리페치 ─────────────────────────────────────
    load_A(0, 0);
    load_B(0, 0);
    __syncthreads();

    int mma_count = 0;

    for (int kStart = 0; kStart < K; kStart += HP_BLOCK_K) {
        const int curBuf = (kStart / HP_BLOCK_K) & 1;
        const int nxtBuf = curBuf ^ 1;

        // 다음 K 타일을 현재 타일 계산과 겹쳐서 로드 (소프트웨어 파이프라인)
        const int nextK = kStart + HP_BLOCK_K;
        if (nextK < K) {
            load_A(nextK, nxtBuf);
            load_B(nextK, nxtBuf);
        }

        // ── 현재 버퍼에서 WMMA 연산 ─────────────────────────────
        // K-축: HP_BLOCK_K / WMMA_K = 64/16 = 4 단계
        const half* curA = smemA + curBuf * HP_SMEM_A_BUF_HALF;
        const half* curB = smemB + curBuf * HP_SMEM_B_BUF_HALF;

        #pragma unroll
        for (int kk = 0; kk < HP_BLOCK_K; kk += WMMA_K) {
            #pragma unroll
            for (int mi = 0; mi < HP_MMA_M; mi++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                                wmma::row_major> a_frag;
                wmma::load_matrix_sync(
                    a_frag,
                    curA + (warpM * HP_WARP_M + mi * WMMA_M) * HP_SMEM_A_STRIDE + kk,
                    HP_SMEM_A_STRIDE);

                #pragma unroll
                for (int ni = 0; ni < HP_MMA_N; ni++) {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                                    wmma::row_major> b_frag;
                    wmma::load_matrix_sync(
                        b_frag,
                        curB + kk * HP_SMEM_B_STRIDE + warpN * HP_WARP_N + ni * WMMA_N,
                        HP_SMEM_B_STRIDE);

                    wmma::mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }

            // if (sleep_ns > 0) {
            //     mma_count++;
            //     if (mma_count % sleep_freq == 0)
            //         __nanosleep(static_cast<unsigned>(sleep_ns));
            // }
        }

        __syncthreads();
    }
    if (sleep_ns > 0) {
        __nanosleep(static_cast<unsigned>(sleep_ns));
    }
    // ── 결과 저장 ────────────────────────────────────────────────
    #pragma unroll
    for (int mi = 0; mi < HP_MMA_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < HP_MMA_N; ni++) {
            const int outRow = blockRow + warpM * HP_WARP_M + mi * WMMA_M;
            const int outCol = blockCol + warpN * HP_WARP_N + ni * WMMA_N;
            if (outRow < M && outCol < N) {
                wmma::store_matrix_sync(
                    C + outRow * N + outCol,
                    acc[mi][ni], N, wmma::mem_row_major);
            }
        }
    }
}

// ======================== Host wrappers ========================
 
 // Helper: check tensor requirements
 #define CHECK_CUDA(x)  TORCH_CHECK(x.device().is_cuda(), #x " must be on CUDA")
 #define CHECK_CONT(x)  TORCH_CHECK(x.is_contiguous(),    #x " must be contiguous")
 #define CHECK_FP16(x)  TORCH_CHECK(x.scalar_type() == torch::kHalf, #x " must be FP16")
 
// ── gemm_time_check: warp별 절대 start/end 시각을 반환하는 simple kernel 래퍼 ──
// 반환값: (C [M,N] float32, start_ns [num_warps] int64, end_ns [num_warps] int64)
//   start_ns[i] = warp i 의 globaltimer 시작 시각 (절대 ns)
//   end_ns[i]   = warp i 의 globaltimer 종료 시각 (절대 ns)
//   elapsed[i]  = end_ns[i] - start_ns[i]  (Python 에서 계산)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> wmma_gemm_sleep_time_check(
    torch::Tensor A,
    torch::Tensor B,
    int  sleep_ns,
    int  sleep_freq)
{
    CHECK_CUDA(A); CHECK_CUDA(B);
    CHECK_CONT(A); CHECK_CONT(B);
    CHECK_FP16(A); CHECK_FP16(B);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    TORCH_CHECK(K == B.size(0), "A cols must match B rows");
    TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0,
                "M, N, K must be multiples of 16");

    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

    const int totalTiles = (M / WMMA_M) * (N / WMMA_N);
    auto start_buf = torch::zeros({totalTiles}, torch::dtype(torch::kInt64).device(A.device()));
    auto end_buf   = torch::zeros({totalTiles}, torch::dtype(torch::kInt64).device(A.device()));

    const half* aPtr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half* bPtr = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
    auto* sPtr = reinterpret_cast<unsigned long long*>(start_buf.data_ptr<int64_t>());
    auto* ePtr = reinterpret_cast<unsigned long long*>(end_buf.data_ptr<int64_t>());

    const int warpsPerBlk    = (totalTiles >= 1024) ? 8 : 4;
    const int threadsPerBlk  = warpsPerBlk * WARP_SIZE;
    const int numBlocks      = (totalTiles + warpsPerBlk - 1) / warpsPerBlk;

    wmma_gemm_sleep_kernel_time_check<<<numBlocks, threadsPerBlk>>>(
        aPtr, bPtr, C.data_ptr<float>(), M, N, K, sleep_ns, sleep_freq, sPtr, ePtr);

    return {C, start_buf, end_buf};
}

// ── gemm: simple or smem kernel ──
torch::Tensor wmma_gemm_sleep(
     torch::Tensor A,
     torch::Tensor B,
     int  sleep_ns,
     int  sleep_freq,
     bool use_smem)
 {
     CHECK_CUDA(A); CHECK_CUDA(B);
     CHECK_CONT(A); CHECK_CONT(B);
     CHECK_FP16(A); CHECK_FP16(B);
 
     const int M = A.size(0);
     const int K = A.size(1);
     const int N = B.size(1);
     TORCH_CHECK(K == B.size(0), "A cols must match B rows");
     TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0,
                 "M, N, K must be multiples of 16");
 
     auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
 
     const half* aPtr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
     const half* bPtr = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
 
     if (use_smem) {
         constexpr int BLOCK_M = 64, BLOCK_N = 64;
         dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
         dim3 block(512);  // 16 warps
         wmma_gemm_sleep_smem_kernel<<<grid, block>>>(
             aPtr, bPtr, C.data_ptr<float>(), M, N, K, sleep_ns, sleep_freq);
     } else {
         const int tilesM = M / WMMA_M;
         const int tilesN = N / WMMA_N;
         const int totalTiles = tilesM * tilesN;
         const int warpsPerBlk = (totalTiles >= 1024) ? 8 : 4;
         const int threadsPerBlk = warpsPerBlk * WARP_SIZE;
         const int numBlocks = (totalTiles + warpsPerBlk - 1) / warpsPerBlk;
         wmma_gemm_sleep_kernel<<<numBlocks, threadsPerBlk>>>(
             aPtr, bPtr, C.data_ptr<float>(), M, N, K, sleep_ns, sleep_freq);
     }
     return C;
 }
 
 torch::Tensor wmma_gemm_sleep_half(
     torch::Tensor A, torch::Tensor B,
     int sleep_ns, int sleep_freq, bool use_smem)
 {
     return wmma_gemm_sleep(A, B, sleep_ns, sleep_freq, use_smem).to(torch::kHalf);
 }
 
 // ── gemm_opt: 128×128 high-perf kernel ──
 torch::Tensor wmma_gemm_opt(
     torch::Tensor A,
     torch::Tensor B,
     int  sleep_ns,
     int  sleep_freq)
 {
     CHECK_CUDA(A); CHECK_CUDA(B);
     CHECK_CONT(A); CHECK_CONT(B);
     CHECK_FP16(A); CHECK_FP16(B);
 
     const int M = A.size(0);
     const int K = A.size(1);
     const int N = B.size(1);
     TORCH_CHECK(K == B.size(0), "A cols must match B rows");
     TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0,
                 "M, N, K must be multiples of 16");
 
     auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
 
     const half* aPtr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
     const half* bPtr = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
 
     dim3 grid((N + OPT_BLOCK_N - 1) / OPT_BLOCK_N,
               (M + OPT_BLOCK_M - 1) / OPT_BLOCK_M);
     dim3 block(256);  // 8 warps
 
     wmma_gemm_opt_kernel<<<grid, block>>>(
         aPtr, bPtr, C.data_ptr<float>(), M, N, K, sleep_ns, sleep_freq);
     return C;
 }
 
 torch::Tensor wmma_gemm_opt_half(
     torch::Tensor A, torch::Tensor B,
     int sleep_ns, int sleep_freq)
 {
     return wmma_gemm_opt(A, B, sleep_ns, sleep_freq).to(torch::kHalf);
 }
 
 // ── gemm_skinny: small-M optimized kernel ──
 torch::Tensor wmma_gemm_skinny(
     torch::Tensor A,
     torch::Tensor B,
     int  sleep_ns,
     int  sleep_freq)
 {
     CHECK_CUDA(A); CHECK_CUDA(B);
     CHECK_CONT(A); CHECK_CONT(B);
     CHECK_FP16(A); CHECK_FP16(B);
 
     const int M = A.size(0);
     const int K = A.size(1);
     const int N = B.size(1);
     TORCH_CHECK(K == B.size(0), "A cols must match B rows");
     TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0,
                 "M, N, K must be multiples of 16");
 
     auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));
 
     const half* aPtr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
     const half* bPtr = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
 
     constexpr int BLOCK_N = 8 * WMMA_N;  // 128
     dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + WMMA_M - 1) / WMMA_M);
     dim3 block(256);  // 8 warps
 
     wmma_gemm_skinny_kernel<<<grid, block>>>(
         aPtr, bPtr, C.data_ptr<float>(), M, N, K, sleep_ns, sleep_freq);
     return C;
 }
 
 torch::Tensor wmma_gemm_skinny_half(
     torch::Tensor A, torch::Tensor B,
     int sleep_ns, int sleep_freq)
 {
     return wmma_gemm_skinny(A, B, sleep_ns, sleep_freq).to(torch::kHalf);
 }
 
 // ── gemm_hiperf: Kernel 6 (CUTLASS-style: BLOCK_K=64 + float4) ──
torch::Tensor wmma_gemm_hiperf(
    torch::Tensor A,
    torch::Tensor B,
    int  sleep_ns,
    int  sleep_freq)
{
    CHECK_CUDA(A); CHECK_CUDA(B);
    CHECK_CONT(A); CHECK_CONT(B);
    CHECK_FP16(A); CHECK_FP16(B);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    TORCH_CHECK(K == B.size(0), "A cols must match B rows");
    TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0,
                "M, N, K must be multiples of 16 (got M=%d N=%d K=%d)", M, N, K);

    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

    // 동적 shared memory: 76 KB → 런타임에 96 KB carveout 요청
    // cudaFuncAttributeMaxDynamicSharedMemorySize 를 먼저 설정해야
    // 커널 런치 시 smemSize > 48KB 를 허용함
    TORCH_CHECK(
        cudaFuncSetAttribute(wmma_gemm_hiperf_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(HP_SMEM_BYTES)) == cudaSuccess,
        "cudaFuncSetAttribute MaxDynamicSharedMemorySize failed – "
        "GPU may not support ", HP_SMEM_BYTES, " bytes of shared memory");
    cudaFuncSetAttribute(
        wmma_gemm_hiperf_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);

    dim3 grid((N + HP_BLOCK_N - 1) / HP_BLOCK_N,
              (M + HP_BLOCK_M - 1) / HP_BLOCK_M);
    dim3 block(HP_WARPS * WARP_SIZE);  // 256 threads

    wmma_gemm_hiperf_kernel<<<grid, block, HP_SMEM_BYTES>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(), M, N, K, sleep_ns, sleep_freq);

    return C;
}

torch::Tensor wmma_gemm_hiperf_half(
    torch::Tensor A, torch::Tensor B,
    int sleep_ns, int sleep_freq)
{
    return wmma_gemm_hiperf(A, B, sleep_ns, sleep_freq).to(torch::kHalf);
}

// ── Auto-dispatch: pick best kernel based on matrix shape ──
 torch::Tensor wmma_gemm_fast(
     torch::Tensor A,
     torch::Tensor B,
     int  sleep_ns,
     int  sleep_freq)
 {
     const int M = A.size(0);
     const int N = B.size(1);
     const int m_tiles = (M + 127) / 128;
     const int n_tiles = (N + 127) / 128;
     const int opt_blocks = m_tiles * n_tiles;
     if (m_tiles >= 2 && n_tiles >= 2 && opt_blocks >= 128) {
         return wmma_gemm_opt(A, B, sleep_ns, sleep_freq);
     } else {
         return wmma_gemm_sleep(A, B, sleep_ns, sleep_freq, /*use_smem=*/false);
     }
 }
 
 torch::Tensor wmma_gemm_fast_half(
     torch::Tensor A, torch::Tensor B,
     int sleep_ns, int sleep_freq)
 {
     return wmma_gemm_fast(A, B, sleep_ns, sleep_freq).to(torch::kHalf);
 }
 
 // ── stress: pure MMA loop ──
 torch::Tensor wmma_stress(
     int num_blocks, int warps_per_block, int num_iters,
     int sleep_ns, int sleep_freq)
 {
     wmma_stress_kernel<<<num_blocks, warps_per_block * WARP_SIZE>>>(
         num_iters, sleep_ns, sleep_freq);
     return torch::zeros({1});
 }
 
 // ======================== Python bindings ========================
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     m.def("gemm_time_check", &wmma_gemm_sleep_time_check,
           "WMMA GEMM + warp-level start/end timing via globaltimer  (FP16 → FP32)\n"
           "\n  Returns (C [M,N] float32, start_ns [num_warps] int64, end_ns [num_warps] int64)\n"
           "  start_ns[i] = absolute globaltimer at warp i start (ns)\n"
           "  end_ns[i]   = absolute globaltimer at warp i end   (ns)\n"
           "  elapsed[i]  = end_ns[i] - start_ns[i]  (compute in Python)\n"
           "  GPU-wide synchronized 1 ns timer – no clock conversion needed",
           py::arg("A"), py::arg("B"),
           py::arg("sleep_ns") = 0,
           py::arg("sleep_freq") = 1);

     m.def("gemm", &wmma_gemm_sleep,
           "WMMA GEMM with nanosleep  (FP16 input → FP32 output)\n"
           "\n  A: [M, K] FP16\n  B: [K, N] FP16\n  → C: [M, N] FP32",
           py::arg("A"), py::arg("B"),
           py::arg("sleep_ns") = 0,
           py::arg("sleep_freq") = 1,
           py::arg("use_smem") = true);
 
     m.def("gemm_half", &wmma_gemm_sleep_half,
           "WMMA GEMM with nanosleep  (FP16 input → FP16 output)",
           py::arg("A"), py::arg("B"),
           py::arg("sleep_ns") = 0,
           py::arg("sleep_freq") = 1,
           py::arg("use_smem") = true);
 
     m.def("gemm_opt", &wmma_gemm_opt,
           "High-performance WMMA GEMM with nanosleep  (FP16 → FP32)\n"
           "\n  128×128 block tiles, 2×4 warp tiles, double-buffered smem",
           py::arg("A"), py::arg("B"),
           py::arg("sleep_ns") = 0,
           py::arg("sleep_freq") = 1);
 
     m.def("gemm_opt_half", &wmma_gemm_opt_half,
           "High-performance WMMA GEMM with nanosleep  (FP16 → FP16)\n",
           py::arg("A"), py::arg("B"),
           py::arg("sleep_ns") = 0,
           py::arg("sleep_freq") = 1);
 
     m.def("gemm_skinny", &wmma_gemm_skinny,
           "Skinny-M optimized WMMA GEMM (FP16 → FP32)\n"
           "\n  Optimized for small M: 8 warps share A tile",
           py::arg("A"), py::arg("B"),
           py::arg("sleep_ns") = 0,
           py::arg("sleep_freq") = 1);
 
     m.def("gemm_skinny_half", &wmma_gemm_skinny_half,
           "Skinny-M optimized WMMA GEMM (FP16 → FP16)\n",
           py::arg("A"), py::arg("B"),
           py::arg("sleep_ns") = 0,
           py::arg("sleep_freq") = 1);
 
     m.def("gemm_fast", &wmma_gemm_fast,
           "Auto-dispatch WMMA GEMM: picks best kernel based on M (FP16 → FP32)\n",
           py::arg("A"), py::arg("B"),
           py::arg("sleep_ns") = 0,
           py::arg("sleep_freq") = 1);
 
     m.def("gemm_fast_half", &wmma_gemm_fast_half,
           "Auto-dispatch WMMA GEMM: picks best kernel based on M (FP16 → FP16)\n",
           py::arg("A"), py::arg("B"),
           py::arg("sleep_ns") = 0,
           py::arg("sleep_freq") = 1);
 
     m.def("gemm_hiperf", &wmma_gemm_hiperf,
          "CUTLASS-style WMMA GEMM  (FP16 → FP32)\n"
          "\n  BLOCK_K=64 + float4 global loads + double-buffered smem\n"
          "  Improvements over gemm_opt:\n"
          "    - BLOCK_K doubled (32→64): half the syncthreads overhead\n"
          "    - 128-bit (float4) global memory loads: 4x fewer load instructions\n"
          "    - row-major smemA: no transpose, cleaner addressing\n"
          "    - 16-byte smem padding: float4-aligned stores",
          py::arg("A"), py::arg("B"),
          py::arg("sleep_ns") = 0,
          py::arg("sleep_freq") = 1);

    m.def("gemm_hiperf_half", &wmma_gemm_hiperf_half,
          "CUTLASS-style WMMA GEMM  (FP16 → FP16)",
          py::arg("A"), py::arg("B"),
          py::arg("sleep_ns") = 0,
          py::arg("sleep_freq") = 1);

    m.def("stress", &wmma_stress,
           "Pure WMMA stress test (no data movement)\n"
           "\n  Useful for measuring peak Tensor Core power draw",
           py::arg("num_blocks") = 128,
           py::arg("warps_per_block") = 8,
           py::arg("num_iters") = 10000,
           py::arg("sleep_ns") = 0,
           py::arg("sleep_freq") = 1);
 }
 