/**
 * bf16_gemm_sm80_kernel.cu  (v2 – optimized)
 * ============================================
 * bf16_gemm_sm80.cu 가 CUTLASS headers 내부에서 실행하는 것과
 * 동일한 패턴을 PTX inline assembly 로 직접 구현한 파일.
 *
 * PTX instruction 사용:
 *   mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  (HMMA Tensor Core)
 *   ldmatrix.sync.aligned.m8n8.x4 / x2.trans              (smem → register 분배)
 *   cp.async.cg.shared.global                              (global → smem 비동기 로드)
 *
 * ── v1 → v2 변경사항 ──────────────────────────────────────────────────────
 *  [1] BK  64 → 32   : static smem 71 KB → 37 KB → SM당 블록 수 ×2
 *  [2] WARPS 4 → 8   : 256 threads → latency 은폐 강화
 *  [3] WN  64 → 32   : MMA_TN 8 → 4, 누산기 128→64 f32 → register spill 제거
 *  [4] MMA_TK 4 → 2  : inner K-loop 단축, 스케줄링 여유
 *  [5] Dynamic smem → Static smem : cudaFuncSetAttribute 불필요
 *
 * 타일 구조:
 *   Block tile : 128 × 128 × 32   (BM × BN × BK)
 *   Warp grid  : 2M × 4N = 8 warps = 256 threads/block
 *   Warp tile  :  64 ×  32 × 32   (WM × WN × BK)
 *   MMA tile   :  16 ×   8 × 16   (mma.m16n8k16)
 *   MMA 수/warp:   4M ×  4N × 2K  = 32 ops per K-block
 *   누산기     :   4 × 4 × 4 = 64 f32/warp   (v1: 128 → spill 제거)
 *
 * Shared memory (static):
 *   smemA[2][128][40] BF16  = 20,480 B   (BK=32, pad=8 → stride=40)
 *   smemB[2][ 32][136] BF16 = 17,408 B   (BN=128, pad=8 → stride=136)
 *   Total : 37,888 B ≈ 37 KB < 48 KB → cudaFuncSetAttribute 불필요,
 *           SM120(256 KB L1) 기준 6+ blocks/SM 가능
 *
 * 요구 사항: sm_80+, CUDA 11.0+
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// 컴파일 타임 상수
// ─────────────────────────────────────────────────────────────────────────────

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;       // ← v1: 64

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;   // ← v1: 2
constexpr int WARPS    = WARPS_M * WARPS_N;  // 8   ← v1: 4
constexpr int THREADS  = WARPS * 32;          // 256 ← v1: 128

constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32  ← v1: 64

constexpr int STAGES = 2;  // double buffering

// MMA shape (mma.m16n8k16)
constexpr int MMA_M = 16;
constexpr int MMA_N =  8;
constexpr int MMA_K = 16;

// MMA 타일 수 (warp당)
constexpr int MMA_TM = WM / MMA_M;  //  4
constexpr int MMA_TN = WN / MMA_N;  //  4  ← v1: 8
constexpr int MMA_TK = BK / MMA_K;  //  2  ← v1: 4
// 누산기: 4×4×4 = 64 f32/warp  ← v1: 128 (spill 위험 제거)

// smem stride: 패딩으로 bank conflict 최소화 (16B-aligned)
constexpr int LDA = BK + 8;   // 40 BF16/row  (80 B)
constexpr int LDB = BN + 8;   // 136 BF16/row (272 B)

// static smem 원소 수
constexpr int SMEM_A_HALFS = STAGES * BM * LDA;   // 2×128×40  = 10,240
constexpr int SMEM_B_HALFS = STAGES * BK * LDB;   // 2×32×136  =  8,704
// 총 bytes: 37,888 ≈ 37 KB < 48 KB

// ─────────────────────────────────────────────────────────────────────────────
// Shared memory 포인터 변환 헬퍼
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__
uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// ─────────────────────────────────────────────────────────────────────────────
// cp.async — 전역 메모리 → shared memory 비동기 복사
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__
void cp_async16(uint32_t smem_addr, const void* global_addr) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(smem_addr), "l"((uint64_t)global_addr), "n"(16)
    );
}

// pred=false → 해당 smem 위치를 0으로 채움 (범위 초과 시 패딩)
__device__ __forceinline__
void cp_async16_pred(uint32_t smem_addr, const void* global_addr, bool pred) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n"
        :: "r"(smem_addr),
           "l"((uint64_t)global_addr),
           "n"(16),
           "r"(pred ? 16 : 0)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :: );
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// ─────────────────────────────────────────────────────────────────────────────
// ldmatrix — shared memory → 레지스터 fragment 분배
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A fragment 로드: 16×16 BF16 tile → 4 b32 registers
 * ldmatrix.sync.aligned.m8n8.x4 (NO transpose)
 *
 * Thread 주소 매핑 (laneId 0..31):
 *   mat = lane >> 3        (0..3)
 *   row = (lane & 7) + ((mat & 1) << 3)   (0..15)
 *   col = (mat >> 1) << 3                 (0 or 8)
 */
__device__ __forceinline__
void ldmatrix_A(uint32_t fA[4],
                const __nv_bfloat16* __restrict__ smem_tile,
                int laneId)
{
    int mat = laneId >> 3;
    int row = (laneId & 7) + ((mat & 1) << 3);
    int col = (mat >> 1) << 3;

    uint32_t addr = smem_u32(smem_tile + row * LDA + col);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(fA[0]), "=r"(fA[1]), "=r"(fA[2]), "=r"(fA[3])
        : "r"(addr)
    );
}

/**
 * B fragment 로드: 16×8 BF16 tile → 2 b32 registers
 * ldmatrix.sync.aligned.m8n8.x2.trans (전치)
 * row-major K×N smem → col-major N×K MMA fragment 자동 변환
 *
 * Thread 주소 매핑 (laneId 0..31):
 *   k_row    = lane & 15         (0..15)
 *   n_offset = (lane >> 4) << 2  (0 or 4)
 */
__device__ __forceinline__
void ldmatrix_B(uint32_t fB[2],
                const __nv_bfloat16* __restrict__ smem_tile,
                int laneId)
{
    int k_row    = laneId & 15;
    int n_offset = (laneId >> 4) << 2;

    uint32_t addr = smem_u32(smem_tile + k_row * LDB + n_offset);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(fB[0]), "=r"(fB[1])
        : "r"(addr)
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Thread t 가 acc[4] 에서 담당하는 출력 위치 (16×8 tile 내):
//   row0 = lane/4,     row1 = row0+8
//   col0 = (lane%4)*2, col1 = col0+1
//   acc[0]=D[row0][col0], acc[1]=D[row1][col0]
//   acc[2]=D[row0][col1], acc[3]=D[row1][col1]
// ─────────────────────────────────────────────────────────────────────────────
__device__ __forceinline__
void mma_m16n8k16(float acc[4],
                  const uint32_t fA[4],
                  const uint32_t fB[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(fA[0]), "r"(fA[1]), "r"(fA[2]), "r"(fA[3]),
          "r"(fB[0]), "r"(fB[1])
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main kernel  (v2)
// ─────────────────────────────────────────────────────────────────────────────

__global__ __launch_bounds__(THREADS)
void bf16_gemm_sm80_kernel(
    const __nv_bfloat16* __restrict__ A,   // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,   // [K, N] row-major
          __nv_bfloat16* __restrict__ C,   // [M, N] row-major (output)
    int M, int N, int K)
{
    // ── Static shared memory ──────────────────────────────────────────────
    // 37,888 B ≈ 37 KB < 48 KB  → cudaFuncSetAttribute 불필요
    __shared__ __nv_bfloat16 smemA_data[SMEM_A_HALFS];  // [STAGES][BM][LDA]
    __shared__ __nv_bfloat16 smemB_data[SMEM_B_HALFS];  // [STAGES][BK][LDB]

    // ── 기본 인덱스 ──────────────────────────────────────────────────────
    const int tid    = threadIdx.x;
    const int laneId = tid & 31;
    const int warpId = tid >> 5;          // 0..7

    const int warpM  = warpId / WARPS_N;  // 0 or 1
    const int warpN  = warpId % WARPS_N;  // 0,1,2,3

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpRowOff = warpM * WM;    // 0 or 64
    const int warpColOff = warpN * WN;    // 0, 32, 64, 96

    // ── 누산기 초기화 (64 f32/warp, spill 없음) ──────────────────────────
    float acc[MMA_TM][MMA_TN][4] = {};    // 4×4×4 = 64 f32

    // ── cp.async 로드 헬퍼 ────────────────────────────────────────────────
    // A tile: [BM=128 × BK=32] = 4,096 BF16 = 512 float4
    //   256 threads × 2 loads/thread = 512 ✓
    //   BK/8 = 4 groups per row → flat/4 = row, (flat%4)*8 = col
    // B tile: [BK=32 × BN=128] = 4,096 BF16 = 512 float4
    //   BN/8 = 16 groups per row → flat/16 = row, (flat%16)*8 = col
    constexpr int A_LOADS = (BM * BK / 8) / THREADS;   // 2
    constexpr int B_LOADS = (BK * BN / 8) / THREADS;   // 2

    auto load_A_async = [&](int stage, int kStart) {
        __nv_bfloat16* dst = smemA_data + stage * BM * LDA;
        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            int flat = tid * A_LOADS + i;
            int row  = flat / (BK / 8);
            int col  = (flat % (BK / 8)) * 8;
            const __nv_bfloat16* src = A + (blockRow + row) * K + (kStart + col);
            uint32_t dstAddr = smem_u32(dst + row * LDA + col);
            bool valid = (blockRow + row < M) && (kStart + col < K);
            cp_async16_pred(dstAddr, src, valid);
        }
    };

    auto load_B_async = [&](int stage, int kStart) {
        __nv_bfloat16* dst = smemB_data + stage * BK * LDB;
        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            int flat = tid * B_LOADS + i;
            int row  = flat / (BN / 8);
            int col  = (flat % (BN / 8)) * 8;
            const __nv_bfloat16* src = B + (kStart + row) * N + (blockCol + col);
            uint32_t dstAddr = smem_u32(dst + row * LDB + col);
            bool valid = (kStart + row < K) && (blockCol + col < N);
            cp_async16_pred(dstAddr, src, valid);
        }
    };

    // ── 프롤로그: stage 0 로드 ────────────────────────────────────────────
    const int numKTiles = (K + BK - 1) / BK;

    load_A_async(0, 0);
    load_B_async(0, 0);
    cp_async_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    // ── 메인 K 루프 ──────────────────────────────────────────────────────
    // 구조 (software double buffering):
    //   1) 다음 타일(k+1) cp.async 발행  → DMA와 compute 중첩
    //   2) 현재 타일(k) WMMA 계산
    //   3) 다음 타일 완료 대기 + syncthreads
    for (int k = 0; k < numKTiles; ++k) {
        const int cur = k & 1;
        const int nxt = cur ^ 1;

        // 1) 다음 타일 prefetch
        if (k + 1 < numKTiles) {
            load_A_async(nxt, (k + 1) * BK);
            load_B_async(nxt, (k + 1) * BK);
            cp_async_commit();
        }

        // 현재 stage smem 기준 포인터
        const __nv_bfloat16* curA = smemA_data + cur * BM * LDA + warpRowOff * LDA;
        const __nv_bfloat16* curB = smemB_data + cur * BK * LDB + warpColOff;

        // 2) WMMA 계산: MMA_TK × MMA_TM × MMA_TN
        //    MMA_TK=2, MMA_TM=4, MMA_TN=4 → 32 mma ops/K-block
        #pragma unroll
        for (int ki = 0; ki < MMA_TK; ++ki) {
            // A fragments for all M-tiles at this ki step
            uint32_t fA[MMA_TM][4];
            #pragma unroll
            for (int mi = 0; mi < MMA_TM; ++mi) {
                ldmatrix_A(fA[mi],
                           curA + (mi * MMA_M) * LDA + ki * MMA_K,
                           laneId);
            }

            // B fragment 로드 후 모든 M-tile 과 MMA
            #pragma unroll
            for (int ni = 0; ni < MMA_TN; ++ni) {
                uint32_t fB[2];
                ldmatrix_B(fB,
                           curB + ki * MMA_K * LDB + ni * MMA_N,
                           laneId);

                #pragma unroll
                for (int mi = 0; mi < MMA_TM; ++mi) {
                    mma_m16n8k16(acc[mi][ni], fA[mi], fB);
                }
            }
        }

        // 3) 다음 타일 완료 대기
        if (k + 1 < numKTiles) {
            cp_async_wait_group<0>();
            __syncthreads();
        }
    }

    // ── 에필로그: FP32 누산기 → BF16 → 전역 C ────────────────────────────
    // mma.m16n8k16 출력 fragment (thread t):
    //   row0 = laneId/4,     row1 = row0+8
    //   col0 = (laneId%4)*2, col1 = col0+1
    const int out_row0 = laneId >> 2;          // 0..7
    const int out_row1 = out_row0 + 8;         // 8..15
    const int out_col0 = (laneId & 3) << 1;   // 0,2,4,6
    const int out_col1 = out_col0 + 1;         // 1,3,5,7

    #pragma unroll
    for (int mi = 0; mi < MMA_TM; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < MMA_TN; ++ni) {
            int r0 = blockRow + warpRowOff + mi * MMA_M + out_row0;
            int r1 = blockRow + warpRowOff + mi * MMA_M + out_row1;
            int c0 = blockCol + warpColOff + ni * MMA_N + out_col0;
            int c1 = blockCol + warpColOff + ni * MMA_N + out_col1;

            if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[mi][ni][0]);
            if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[mi][ni][1]);
            if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[mi][ni][2]);
            if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[mi][ni][3]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 커널 런처 (v2: static smem → cudaFuncSetAttribute 불필요)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * bf16_gemm_sm80_launch
 *
 * Grid : (N/BN, M/BM)  — 각 block이 128×128 C tile 담당
 * Block: 256 threads (8 warps)
 * Smem : 37,888 B static
 *
 * 권장 크기: M, N 는 128 배수, K 는 32 배수
 */
void bf16_gemm_sm80_launch(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
          __nv_bfloat16* C,
    int M, int N, int K,
    cudaStream_t stream = nullptr)
{
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS);

    bf16_gemm_sm80_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

// ─────────────────────────────────────────────────────────────────────────────
// PyTorch 확장 바인딩 (TORCH_EXTENSION_H 정의 시 활성화)
// ─────────────────────────────────────────────────────────────────────────────
#ifdef TORCH_EXTENSION_H

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

torch::Tensor gemm_sm80_ptx(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16,
                "A, B must be BF16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A, B must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    bf16_gemm_sm80_launch(
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
        M, N, K, stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_sm80_ptx", &gemm_sm80_ptx,
          "BF16 GEMM via PTX mma.m16n8k16 + ldmatrix + cp.async\n"
          "  v2: BK=32, 8 warps(256 th), MMA_TN=4, static smem 37KB\n"
          "  Block 128×128, Warp 64×32, double buffering");
}

#endif  // TORCH_EXTENSION_H
