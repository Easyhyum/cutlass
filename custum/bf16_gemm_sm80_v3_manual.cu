/**
 * bf16_gemm_sm80_v3_manual.cu
 *
 * Standalone BF16 GEMM kernel modeled after bf16_gemm_sm80.cu::gemm_sm80_v3:
 *   - CTA tile:  128 x 128 x 64
 *   - Warp tile:  64 x  64 x 32
 *   - WMMA tile:  16 x  16 x 16 (compiled to tensor-core MMA instructions)
 *   - 3-stage shared-memory pipeline
 *
 * This file intentionally does not instantiate CUTLASS device::Gemm.  It uses
 * inline PTX for cp.async and CUDA WMMA for tensor-core MMA so the whole kernel
 * is visible in one CUDA extension source.
 *
 * Build:
 *   python setup_bf16_sm80_v3_manual.py build_ext --inplace
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>
#include <limits>

namespace {

using BF16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WARPS = WARPS_M * WARPS_N;
constexpr int WARP_SIZE = 32;
constexpr int THREADS = WARPS * WARP_SIZE;

constexpr int WM = BM / WARPS_M;
constexpr int WN = BN / WARPS_N;

constexpr int STAGES = 3;

constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;

constexpr int MMA_TM = WM / MMA_M;      // 4
constexpr int MMA_TN = WN / MMA_N;      // 4
constexpr int MMA_TK = BK / MMA_K;      // 4 per CTA K tile
constexpr int WARP_MMA_K = 32 / MMA_K;  // CUTLASS v3 warp tile K = 32

// BF16 element strides.  v3's 128x128x64, 3-stage tile already needs 96 KiB
// without padding, so keep the strides tight to fit the dynamic smem limit.
constexpr int LDA = BK;   // 64 BF16
constexpr int LDB = BN;   // 128 BF16

constexpr int SMEM_A_ELEMS = STAGES * BM * LDA;
constexpr int SMEM_B_ELEMS = STAGES * BK * LDB;
constexpr int SMEM_A_BYTES = SMEM_A_ELEMS * static_cast<int>(sizeof(BF16));
constexpr int SMEM_B_BYTES = SMEM_B_ELEMS * static_cast<int>(sizeof(BF16));
constexpr int SMEM_BYTES = SMEM_A_BYTES + SMEM_B_BYTES;

__device__ __forceinline__ uint32_t smem_u32(void const* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__
void cp_async16_zfill(uint32_t smem_addr, void const* global_addr, int src_bytes) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
        :: "r"(smem_addr),
           "l"(reinterpret_cast<std::uint64_t>(global_addr)),
           "r"(src_bytes)
        : "memory"
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__
void clock64_wait(unsigned int cycles) {
    if (cycles == 0u) {
        return;
    }

    unsigned long long const start = clock64();
    while (clock64() - start < static_cast<unsigned long long>(cycles)) {
        asm volatile("");
    }
}

__global__ __launch_bounds__(THREADS, 1)
void bf16_gemm_sm80_v3_manual_kernel(
    BF16 const* __restrict__ A,
    BF16 const* __restrict__ B,
    BF16* __restrict__ C,
    int M,
    int N,
    int K,
    unsigned int sleep_cycles,
    unsigned int sleep_freq) {

    extern __shared__ __align__(16) unsigned char smem_raw[];
    BF16* const smem_a = reinterpret_cast<BF16*>(smem_raw);
    BF16* const smem_b = reinterpret_cast<BF16*>(smem_raw + SMEM_A_BYTES);

    int const tid = threadIdx.x;
    int const lane_id = tid & (WARP_SIZE - 1);
    int const warp_id = tid >> 5;
    int const warp_m = warp_id / WARPS_N;
    int const warp_n = warp_id % WARPS_N;

    int const block_row = blockIdx.y * BM;
    int const block_col = blockIdx.x * BN;
    int const warp_row = warp_m * WM;
    int const warp_col = warp_n * WN;

    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> acc[MMA_TM][MMA_TN];
    #pragma unroll
    for (int mi = 0; mi < MMA_TM; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < MMA_TN; ++ni) {
            wmma::fill_fragment(acc[mi][ni], 0.0f);
        }
    }

    constexpr int A_LOADS = (BM * BK / 8) / THREADS;  // 8 vector loads/thread
    constexpr int B_LOADS = (BK * BN / 8) / THREADS;  // 8 vector loads/thread

    auto load_A_async = [&](int stage, int k_start) {
        BF16* const dst = smem_a + stage * BM * LDA;

        #pragma unroll
        for (int i = 0; i < A_LOADS; ++i) {
            int const flat = tid * A_LOADS + i;
            int const row = flat / (BK / 8);
            int const col = (flat % (BK / 8)) * 8;
            int const global_row = block_row + row;
            int const global_col = k_start + col;

            BF16 const* const src = A + global_row * K + global_col;
            uint32_t const dst_addr = smem_u32(dst + row * LDA + col);
            int const a_remaining = K - global_col;
            int const valid_elems =
                (global_row < M && global_col < K) ? (a_remaining < 8 ? a_remaining : 8) : 0;
            cp_async16_zfill(dst_addr, src, valid_elems * static_cast<int>(sizeof(BF16)));
        }
    };

    auto load_B_async = [&](int stage, int k_start) {
        BF16* const dst = smem_b + stage * BK * LDB;

        #pragma unroll
        for (int i = 0; i < B_LOADS; ++i) {
            int const flat = tid * B_LOADS + i;
            int const row = flat / (BN / 8);
            int const col = (flat % (BN / 8)) * 8;
            int const global_row = k_start + row;
            int const global_col = block_col + col;

            BF16 const* const src = B + global_row * N + global_col;
            uint32_t const dst_addr = smem_u32(dst + row * LDB + col);
            int const b_remaining = N - global_col;
            int const valid_elems =
                (global_row < K && global_col < N) ? (b_remaining < 8 ? b_remaining : 8) : 0;
            cp_async16_zfill(dst_addr, src, valid_elems * static_cast<int>(sizeof(BF16)));
        }
    };

    int const num_k_tiles = (K + BK - 1) / BK;

    if (num_k_tiles > 0) {
        load_A_async(0, 0);
        load_B_async(0, 0);
        cp_async_commit();
    }

    if (num_k_tiles > 1) {
        load_A_async(1, BK);
        load_B_async(1, BK);
        cp_async_commit();
    }

    cp_async_wait_group<0>();
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        int const cur_stage = kt % STAGES;
        int const prefetch_tile = kt + (STAGES - 1);

        if (prefetch_tile < num_k_tiles) {
            int const prefetch_stage = prefetch_tile % STAGES;
            load_A_async(prefetch_stage, prefetch_tile * BK);
            load_B_async(prefetch_stage, prefetch_tile * BK);
            cp_async_commit();
        }

        BF16 const* const tile_a = smem_a + cur_stage * BM * LDA + warp_row * LDA;
        BF16 const* const tile_b = smem_b + cur_stage * BK * LDB + warp_col;

        #pragma unroll
        for (int ki = 0; ki < MMA_TK; ++ki) {
            #pragma unroll
            for (int mi = 0; mi < MMA_TM; ++mi) {
                wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, BF16, wmma::row_major> frag_a;
                wmma::load_matrix_sync(
                    frag_a,
                    tile_a + mi * MMA_M * LDA + ki * MMA_K,
                    LDA);

                #pragma unroll
                for (int ni = 0; ni < MMA_TN; ++ni) {
                    wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, BF16, wmma::row_major> frag_b;
                    wmma::load_matrix_sync(
                        frag_b,
                        tile_b + ki * MMA_K * LDB + ni * MMA_N,
                        LDB);

                    wmma::mma_sync(acc[mi][ni], frag_a, frag_b, acc[mi][ni]);
                }
            }

            bool const sleep_this_kgroup =
                sleep_cycles > 0u &&
                sleep_freq > 0u &&
                (ki % WARP_MMA_K) == 0 &&
                ((static_cast<unsigned int>(kt * MMA_TK + ki + 1) % sleep_freq) == 0u) &&
                (warp_id == (static_cast<unsigned int>(num_k_tiles - kt) % WARPS));

            if (sleep_this_kgroup) {
                clock64_wait(sleep_cycles);
            }
        }

        if (kt + 1 < num_k_tiles) {
            cp_async_wait_group<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    float* const smem_epi = reinterpret_cast<float*>(smem_raw);
    constexpr int EPI_STRIDE = MMA_N + 2;

    #pragma unroll
    for (int mi = 0; mi < MMA_TM; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < MMA_TN; ++ni) {
            float* const tile = smem_epi + warp_id * MMA_M * EPI_STRIDE;
            int const out_row = block_row + warp_row + mi * MMA_M;
            int const out_col = block_col + warp_col + ni * MMA_N;

            wmma::store_matrix_sync(tile, acc[mi][ni], EPI_STRIDE, wmma::mem_row_major);
            __syncwarp();

            #pragma unroll
            for (int elem = lane_id; elem < MMA_M * MMA_N; elem += WARP_SIZE) {
                int const r = elem / MMA_N;
                int const c = elem % MMA_N;
                int const global_row = out_row + r;
                int const global_col = out_col + c;

                if (global_row < M && global_col < N) {
                    C[global_row * N + global_col] = __float2bfloat16(tile[r * EPI_STRIDE + c]);
                }
            }
        }
    }
}

void launch_bf16_gemm_sm80_v3_manual(
    BF16 const* A,
    BF16 const* B,
    BF16* C,
    int M,
    int N,
    int K,
    unsigned int sleep_cycles,
    unsigned int sleep_freq,
    cudaStream_t stream) {

    dim3 const grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 const block(THREADS);

    cudaError_t err = cudaFuncSetAttribute(
        bf16_gemm_sm80_v3_manual_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_BYTES);
    TORCH_CHECK(err == cudaSuccess,
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: ",
                cudaGetErrorString(err));

    err = cudaFuncSetAttribute(
        bf16_gemm_sm80_v3_manual_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    TORCH_CHECK(err == cudaSuccess,
                "cudaFuncSetAttribute(PreferredSharedMemoryCarveout) failed: ",
                cudaGetErrorString(err));

    bf16_gemm_sm80_v3_manual_kernel<<<grid, block, SMEM_BYTES, stream>>>(
        A, B, C, M, N, K, sleep_cycles, sleep_freq);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "bf16_gemm_sm80_v3_manual_kernel launch failed: ",
                cudaGetErrorString(err));
}

}  // namespace

torch::Tensor gemm_sm80_v3_manual(
    torch::Tensor A,
    torch::Tensor B,
    unsigned int sleep_cycles = 0u,
    unsigned int sleep_freq = 1u) {

    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be BF16");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous row-major");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous row-major");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");
    TORCH_CHECK(A.size(0) <= std::numeric_limits<int>::max() &&
                A.size(1) <= std::numeric_limits<int>::max() &&
                B.size(1) <= std::numeric_limits<int>::max(),
                "matrix dimensions must fit in int");

    int const M = static_cast<int>(A.size(0));
    int const K = static_cast<int>(A.size(1));
    int const N = static_cast<int>(B.size(1));
    TORCH_CHECK((K % 8) == 0 && (N % 8) == 0,
                "K and N must be multiples of 8 for 128-bit BF16 vectorized cp.async loads");

    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    launch_bf16_gemm_sm80_v3_manual(
        reinterpret_cast<BF16 const*>(A.data_ptr()),
        reinterpret_cast<BF16 const*>(B.data_ptr()),
        reinterpret_cast<BF16*>(C.data_ptr()),
        M,
        N,
        K,
        sleep_cycles,
        sleep_freq,
        stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemm_sm80_v3_manual",
        &gemm_sm80_v3_manual,
        "Standalone BF16 GEMM: CTA 128x128x64, warp 64x64x32, mma.m16n8k16, 3-stage pipeline",
        py::arg("A"),
        py::arg("B"),
        py::arg("sleep_cycles") = 0u,
        py::arg("sleep_freq") = 1u);
}
