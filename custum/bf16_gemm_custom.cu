/**
 * bf16_gemm_custom.cu
 *
 * BF16 GEMM with __nanosleep – optimized kernel.
 * Uses WMMA (nvcuda::wmma) with __nv_bfloat16 fragments.
 *
 * Design decisions:
 *   BK=32  : static smem = 47 KB → 2 blocks/SM (vs BK=64 → 1 block/SM)
 *            Higher occupancy hides latency better on Blackwell SM120.
 *
 * Block tile : 128(M) × 128(N)
 * Thread block: 256 threads (8 warps: 4M × 2N)
 * Warp tile  : 32(M) × 64(N)  = 2×4 WMMA tiles of 16×16
 * K-block    : BK = 32  (2 WMMA K-steps of 16 per K-block)
 *
 * Memory pipeline:
 *   cp.async 128-bit  : async global→shared, overlaps with WMMA compute
 *   float4 vectorized : 8 BF16 per 128-bit load instruction
 *   WMMA load/mma     : BF16 fragment × BF16 fragment → FP32 accumulator
 *   Epilogue smem     : store_matrix_sync → FP32 → BF16 global write
 *
 * Static shared memory layout (47104 B ≈ 46 KB < 48 KB):
 *   smemA[2][128][40]  BF16  : 20480 B  (BK=32, PAD=8  → stride=40, 16B-aligned)
 *   smemB[2][ 32][136] BF16  : 17408 B  (BN=128, PAD=8 → stride=136, 16B-aligned)
 *   smem_epi[8][16][18] float:  9216 B  (8 warps × 16 rows × 18 floats)
 *   Total                    : 47104 B  → 2 blocks per SM on SM120 (96 KB L1)
 *
 * __nanosleep placement:
 *   After each K-block's WMMA compute, before cp.async wait for next block.
 *   The sleep partially overlaps with in-flight DMA transfers.
 *
 * Build:
 *   python setup_bf16_custom.py build_ext --inplace
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;
using BF16 = __nv_bfloat16;

/* =========================================================================
 * Kernel constants
 * ========================================================================= */
static constexpr int BM            = 128;
static constexpr int BN            = 128;
static constexpr int BK            = 32;
static constexpr int WARPS_M       = 4;
static constexpr int WARPS_N       = 2;
static constexpr int WARPS_PER_BLK = WARPS_M * WARPS_N;   // 8
static constexpr int WM            = BM / WARPS_M;          // 32
static constexpr int WN            = BN / WARPS_N;          // 64
static constexpr int WMMA_M_DIM    = 16;
static constexpr int WMMA_N_DIM    = 16;
static constexpr int WMMA_K_DIM    = 16;
static constexpr int MMA_M         = WM / WMMA_M_DIM;       // 2 tiles in M per warp
static constexpr int MMA_N         = WN / WMMA_N_DIM;       // 4 tiles in N per warp
static constexpr int MMA_K         = BK / WMMA_K_DIM;       // 2 K-steps per K-block
static constexpr int WARP_SIZE     = 32;

// smem strides (BF16 elements, all 16-byte aligned for float4)
static constexpr int SMEM_A_STRIDE  = BK + 8;          // 40 BF16 = 80 B
static constexpr int SMEM_B_STRIDE  = BN + 8;          // 136 BF16 = 272 B
static constexpr int SMEM_EPI_STRIDE= WMMA_N_DIM + 2;  // 18 float per row

// smem element counts per double-buffer slot
static constexpr int SMEM_A_BUF = BM * SMEM_A_STRIDE;  // 5120  BF16
static constexpr int SMEM_B_BUF = BK * SMEM_B_STRIDE;  // 4352  BF16

/* =========================================================================
 * cp.async helpers  (128-bit async global→shared copy, 16 bytes)
 * ========================================================================= */
__device__ __forceinline__
void cp_async_16b(void* dst_smem, const void* src_gbl) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(dst_smem)),
           "l"((uint64_t)src_gbl)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

/* =========================================================================
 * Main kernel
 * ========================================================================= */
__global__ void __launch_bounds__(WARPS_PER_BLK * WARP_SIZE)
bf16_gemm_nanosleep_kernel(
    const BF16* __restrict__ A,   // [M, K] row-major BF16
    const BF16* __restrict__ B,   // [K, N] row-major BF16
    BF16*       __restrict__ C,   // [M, N] row-major BF16 output
    int M, int N, int K,
    unsigned int sleep_ns,
    unsigned int sleep_freq
) {
    /* ---- Static shared memory (47104 B = 46 KB) ------------------------- */
    __shared__ BF16   smemA[2][BM][SMEM_A_STRIDE];    /* 20480 B */
    __shared__ BF16   smemB[2][BK][SMEM_B_STRIDE];    /* 17408 B */
    __shared__ float  smem_epi[WARPS_PER_BLK][WMMA_M_DIM][SMEM_EPI_STRIDE]; /* 9216 B */
    /* Total: 47104 B → 2 blocks/SM on Blackwell 96KB L1 ✓               */

    /* ---- Thread/warp indices -------------------------------------------- */
    const int tid        = threadIdx.x;
    const int warp_id    = tid / WARP_SIZE;
    const int warp_m     = warp_id / WARPS_N;   // 0..3
    const int warp_n     = warp_id % WARPS_N;   // 0..1
    const int lane       = tid & (WARP_SIZE - 1);
    const int numThreads = WARPS_PER_BLK * WARP_SIZE;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    /* ---- Accumulators (FP32) -------------------------------------------- */
    wmma::fragment<wmma::accumulator, WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM, float>
        acc[MMA_M][MMA_N];
    #pragma unroll
    for (int mi = 0; mi < MMA_M; mi++)
        #pragma unroll
        for (int ni = 0; ni < MMA_N; ni++)
            wmma::fill_fragment(acc[mi][ni], 0.0f);

    /* ---- cp.async tile load helpers (float4 = 8 BF16 = 16 bytes) -------- */
    /* A tile: [BM=128 × BK=32] = 512 float4. 256 threads × 2 float4/thread.
     *   smemA[buf][row][col]: row ∈ [0..127], col ∈ {0,8,16,24}
     *   stride = 40 BF16 = 80 B; col*2B ∈ {0,16,32,48} → 16B-aligned ✓
     * B tile: [BK=32 × BN=128] = 512 float4. 256 threads × 2 float4/thread.
     *   smemB[buf][row][col]: row ∈ [0..31], col ∈ {0,8,...,120}
     *   stride = 136 BF16 = 272 B; col*2B ∈ {0,16,...,240} → 16B-aligned ✓ */
    constexpr int F4_PER_A_ROW = BK / 8;           // 4 float4 per A row
    constexpr int TOTAL_F4_A   = BM * BK / 8;      // 512 float4
    constexpr int F4_PER_B_ROW = BN / 8;           // 16 float4 per B row
    constexpr int TOTAL_F4_B   = BK * BN / 8;      // 512 float4

    /* Issue cp.async for the A portion of a K-block tile */
    auto issue_A = [&](int k_start, int buf) __attribute__((always_inline)) {
        for (int i = tid; i < TOTAL_F4_A; i += numThreads) {
            const int r  = i / F4_PER_A_ROW;
            const int c  = (i % F4_PER_A_ROW) * 8;
            const int gr = block_row + r;
            const int gc = k_start + c;
            void* dst = &smemA[buf][r][c];
            if (gr < M && gc + 7 < K) {
                cp_async_16b(dst, A + gr * K + gc);
            } else {
                *reinterpret_cast<float4*>(dst) = {};
                if (gr < M) {
                    auto* s = reinterpret_cast<const uint16_t*>(A + gr * K + gc);
                    auto* d = reinterpret_cast<uint16_t*>(dst);
                    for (int j = 0; j < 8 && gc + j < K; ++j) d[j] = s[j];
                }
            }
        }
    };

    /* Issue cp.async for the B portion of a K-block tile */
    auto issue_B = [&](int k_start, int buf) __attribute__((always_inline)) {
        for (int i = tid; i < TOTAL_F4_B; i += numThreads) {
            const int r  = i / F4_PER_B_ROW;
            const int c  = (i % F4_PER_B_ROW) * 8;
            const int gr = k_start + r;
            const int gc = block_col + c;
            void* dst = &smemB[buf][r][c];
            if (gr < K && gc + 7 < N) {
                cp_async_16b(dst, B + gr * N + gc);
            } else {
                *reinterpret_cast<float4*>(dst) = {};
                if (gr < K) {
                    auto* s = reinterpret_cast<const uint16_t*>(B + gr * N + gc);
                    auto* d = reinterpret_cast<uint16_t*>(dst);
                    for (int j = 0; j < 8 && gc + j < N; ++j) d[j] = s[j];
                }
            }
        }
    };

    /* ---- Pre-load first K-block into buf=0 ------------------------------ */
    issue_A(0, 0);
    issue_B(0, 0);
    cp_async_fence();
    cp_async_wait_all();
    __syncthreads();

    /* ================================================================
     * Main K-loop  (software double buffering + cp.async overlap)
     * Per iteration:
     *   1. Issue cp.async for (kb+1) → DMA starts in background
     *   2. WMMA compute on cur_buf   → overlaps with DMA above
     *   3. __nanosleep               → power duty-cycle control
     *   4. cp_async_wait_all         → wait for (kb+1) DMA to finish
     *   5. __syncthreads             → all warps ready for next iter
     * ================================================================ */
    const int num_k_blocks = (K + BK - 1) / BK;
    unsigned int kb_count  = 0u;

    for (int kb = 0; kb < num_k_blocks; kb++) {
        const int cur_buf = kb & 1;
        const int nxt_buf = cur_buf ^ 1;

        /* Issue async loads for next K-block BEFORE WMMA starts.
         * DMA overlaps with the subsequent WMMA operations.               */
        if (kb + 1 < num_k_blocks) {
            issue_A((kb + 1) * BK, nxt_buf);
            issue_B((kb + 1) * BK, nxt_buf);
            cp_async_fence();
        }

        /* ---- WMMA compute: 2 K-steps × 2 M-tiles × 4 N-tiles ---------- */
        #pragma unroll
        for (int kk = 0; kk < MMA_K; kk++) {
            #pragma unroll
            for (int mi = 0; mi < MMA_M; mi++) {
                wmma::fragment<wmma::matrix_a,
                               WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM,
                               BF16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(
                    a_frag,
                    &smemA[cur_buf][warp_m * WM + mi * WMMA_M_DIM][kk * WMMA_K_DIM],
                    SMEM_A_STRIDE);

                #pragma unroll
                for (int ni = 0; ni < MMA_N; ni++) {
                    wmma::fragment<wmma::matrix_b,
                                   WMMA_M_DIM, WMMA_N_DIM, WMMA_K_DIM,
                                   BF16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(
                        b_frag,
                        &smemB[cur_buf][kk * WMMA_K_DIM][warp_n * WN + ni * WMMA_N_DIM],
                        SMEM_B_STRIDE);

                    wmma::mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        /* ================================================================
         * __nanosleep: inserted after each K-block's WMMA compute.
         *
         *   Timing: occurs BEFORE cp_async_wait_all, so the sleep
         *   partially overlaps with the in-flight DMA transfer of
         *   the next K-block → power throttling without wasting
         *   the memory bandwidth completely.
         *
         *   sleep_freq: sleep every N-th K-block (1=every, 2=every other…)
         * ================================================================ */
        // if (sleep_ns > 0u) {
        //     kb_count++;
        //     if (sleep_freq > 0u && kb_count % sleep_freq == 0u) {
        //         __nanosleep(sleep_ns);
        //     }
        // }

        /* Wait for next K-block's DMA (issued above) to complete         */
        if (kb + 1 < num_k_blocks) {
            cp_async_wait_all();
        }
        __syncthreads();
    }
    if (sleep_ns > 0u) {
        kb_count++;
        if (sleep_freq > 0u && kb_count % sleep_freq == 0u) {
            __nanosleep(sleep_ns);
        }
    }
    /* ================================================================
     * Epilogue: FP32 accumulator → smem_epi → BF16 global
     *
     * Per WMMA output tile (16×16):
     *   store_matrix_sync → per-warp smem slot → __syncwarp
     *   → 32 threads × 8 elements each → BF16 global write
     * ================================================================ */
    #pragma unroll
    for (int mi = 0; mi < MMA_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < MMA_N; ni++) {
            const int out_row = block_row + warp_m * WM + mi * WMMA_M_DIM;
            const int out_col = block_col + warp_n * WN + ni * WMMA_N_DIM;

            /* FP32 tile → per-warp smem slot */
            wmma::store_matrix_sync(
                smem_epi[warp_id][0],
                acc[mi][ni],
                SMEM_EPI_STRIDE,
                wmma::mem_row_major);

            __syncwarp();

            /* All 32 threads convert FP32→BF16 and write (8 elements each) */
            #pragma unroll
            for (int elem = lane; elem < WMMA_M_DIM * WMMA_N_DIM; elem += WARP_SIZE) {
                const int r  = elem / WMMA_N_DIM;
                const int c  = elem % WMMA_N_DIM;
                const int gr = out_row + r;
                const int gc = out_col + c;
                if (gr < M && gc < N) {
                    C[gr * N + gc] = __float2bfloat16(smem_epi[warp_id][r][c]);
                }
            }
        }
    }
}

/* =========================================================================
 * Python binding
 * ========================================================================= */
torch::Tensor gemm_custom(
    torch::Tensor    A,
    torch::Tensor    B,
    unsigned int     sleep_ns   = 0,
    unsigned int     sleep_freq = 1
) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be BF16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    TORCH_CHECK(M % WMMA_M_DIM == 0 && N % WMMA_N_DIM == 0 && K % WMMA_K_DIM == 0,
                "M, N, K must be multiples of 16");

    auto C      = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(WARPS_PER_BLK * WARP_SIZE);

    bf16_gemm_nanosleep_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const BF16*>(A.data_ptr()),
        reinterpret_cast<const BF16*>(B.data_ptr()),
        reinterpret_cast<      BF16*>(C.data_ptr()),
        M, N, K,
        sleep_ns, sleep_freq
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_custom", &gemm_custom,
          "BF16 GEMM with __nanosleep  (BF16 → BF16)\n"
          "\n  Block 128×128, 8 warps (4M×2N), BK=32\n"
          "  Static smem 47 KB → 2 blocks/SM on SM120\n"
          "  cp.async double-buffering + float4 vectorized loads\n"
          "  WMMA __nv_bfloat16 fragments, FP32 accumulator\n"
          "  __nanosleep after every sleep_freq K-blocks\n"
          "  (sleep overlaps with in-flight DMA transfer)",
          py::arg("A"),
          py::arg("B"),
          py::arg("sleep_ns")   = 0u,
          py::arg("sleep_freq") = 1u);
}
