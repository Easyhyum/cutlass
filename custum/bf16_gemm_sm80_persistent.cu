/**
 * bf16_gemm_sm80_persistent.cu
 *
 * Persistent-CTA BF16 GEMM for power-control experiments.
 *
 * Design:
 *   - CTA tile:  128 x 128 x 32
 *   - Warp tile:  32 x  64 x 32
 *   - WMMA tile: 16 x 16 x 16
 *   - 2-stage cp.async shared-memory pipeline
 *   - Persistent tile queue: each resident CTA repeatedly consumes block tiles
 *   - Throttling is applied only to full dense tiles; tail tiles run unthrottled
 *
 * The micro-kernel mirrors bf16_gemm_custom.cu, which is much faster than the
 * first small-tile persistent prototype. Persistent scheduling is layered on top
 * of that compute path.
 *
 * Build:
 *   python setup_bf16_sm80_persistent.py build_ext --inplace
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
constexpr int BK = 32;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARPS = WARPS_M * WARPS_N;
constexpr int WARP_SIZE = 32;
constexpr int THREADS = WARPS * WARP_SIZE;

constexpr int WM = BM / WARPS_M;
constexpr int WN = BN / WARPS_N;

constexpr int STAGES = 2;

constexpr int MMA_M = 16;
constexpr int MMA_N = 16;
constexpr int MMA_K = 16;

constexpr int MMA_TM = WM / MMA_M;  // 4
constexpr int MMA_TN = WN / MMA_N;  // 4
constexpr int MMA_TK = BK / MMA_K;  // 4

constexpr int LDA = BK + 8;
constexpr int LDB = BN + 8;

constexpr int SMEM_A_ELEMS = STAGES * BM * LDA;
constexpr int SMEM_B_ELEMS = STAGES * BK * LDB;
constexpr int SMEM_A_BYTES = SMEM_A_ELEMS * static_cast<int>(sizeof(BF16));
constexpr int SMEM_B_BYTES = SMEM_B_ELEMS * static_cast<int>(sizeof(BF16));
constexpr int SMEM_BYTES = SMEM_A_BYTES + SMEM_B_BYTES;

enum class ThrottleMode : int {
    kClockWait = 0,
    kNopLoop = 1,
    kNanoSleep = 2,
};

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
void throttle_wait(unsigned int amount, int mode) {
    if (amount == 0u) {
        return;
    }

    if (mode == static_cast<int>(ThrottleMode::kNanoSleep)) {
        __nanosleep(amount);
        return;
    }

    if (mode == static_cast<int>(ThrottleMode::kNopLoop)) {
        unsigned int sink = 0u;
        for (unsigned int i = 0; i < amount; ++i) {
            asm volatile("add.u32 %0, %0, 0;" : "+r"(sink));
        }
        return;
    }

    unsigned long long const start = clock64();
    while (clock64() - start < static_cast<unsigned long long>(amount)) {
        asm volatile("");
    }
}

__device__ __forceinline__
void compute_one_tile(
    BF16 const* __restrict__ A,
    BF16 const* __restrict__ B,
    BF16* __restrict__ C,
    int M,
    int N,
    int K,
    int tile_id,
    int tiles_m,
    int full_tiles_m,
    int full_tiles_n,
    unsigned int throttle_amount,
    unsigned int throttle_freq,
    int throttle_mode,
    unsigned int throttle_phase,
    unsigned char* smem_raw) {

    BF16* const smem_a = reinterpret_cast<BF16*>(smem_raw);
    BF16* const smem_b = reinterpret_cast<BF16*>(smem_raw + SMEM_A_BYTES);

    int const tid = threadIdx.x;
    int const lane_id = tid & (WARP_SIZE - 1);
    int const warp_id = tid >> 5;
    int const warp_m = warp_id / WARPS_N;
    int const warp_n = warp_id % WARPS_N;

    int const tile_m = tile_id % tiles_m;
    int const tile_n = tile_id / tiles_m;
    int const block_row = tile_m * BM;
    int const block_col = tile_n * BN;

    bool const dense_tile = tile_m < full_tiles_m && tile_n < full_tiles_n;
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

    constexpr int A_LOADS = (BM * BK / 8) / THREADS;  // 2 vector loads/thread
    constexpr int B_LOADS = (BK * BN / 8) / THREADS;  // 2 vector loads/thread

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
            int const remaining = K - global_col;
            int const valid_elems =
                (global_row < M && global_col < K) ? (remaining < 8 ? remaining : 8) : 0;
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
            int const remaining = N - global_col;
            int const valid_elems =
                (global_row < K && global_col < N) ? (remaining < 8 ? remaining : 8) : 0;
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

            unsigned int const throttle_event =
                static_cast<unsigned int>((tile_id * num_k_tiles + kt) * MMA_TK + ki + 1);
            bool const apply_throttle =
                dense_tile &&
                throttle_amount > 0u &&
                throttle_freq > 0u &&
                ((throttle_event + throttle_phase) % throttle_freq == 0u) &&
                (warp_id == ((throttle_event + throttle_phase) & (WARPS - 1)));

            if (apply_throttle) {
                throttle_wait(throttle_amount, throttle_mode);
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

__global__ __launch_bounds__(THREADS)
void bf16_gemm_sm80_persistent_kernel(
    BF16 const* __restrict__ A,
    BF16 const* __restrict__ B,
    BF16* __restrict__ C,
    int M,
    int N,
    int K,
    int tiles_m,
    int tiles_n,
    int full_tiles_m,
    int full_tiles_n,
    int total_tiles,
    int chunk_tiles,
    int* __restrict__ tile_counter,
    unsigned int throttle_amount,
    unsigned int throttle_freq,
    int throttle_mode,
    unsigned int throttle_phase) {

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int cta_start_tile;

    while (true) {
        if (threadIdx.x == 0) {
            cta_start_tile = atomicAdd(tile_counter, chunk_tiles);
        }
        __syncthreads();

        int const start_tile = cta_start_tile;
        if (start_tile >= total_tiles) {
            break;
        }

        int const end_tile = min(start_tile + chunk_tiles, total_tiles);
        for (int tile_id = start_tile; tile_id < end_tile; ++tile_id) {
            compute_one_tile(
                A,
                B,
                C,
                M,
                N,
                K,
                tile_id,
                tiles_m,
                full_tiles_m,
                full_tiles_n,
                throttle_amount,
                throttle_freq,
                throttle_mode,
                throttle_phase,
                smem_raw);
            __syncthreads();
        }
    }
}

void launch_persistent(
    BF16 const* A,
    BF16 const* B,
    BF16* C,
    int M,
    int N,
    int K,
    int* counter,
    unsigned int throttle_amount,
    unsigned int throttle_freq,
    int throttle_mode,
    int ctas_per_sm,
    int chunk_tiles,
    cudaStream_t stream) {

    int sm_count = 0;
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    TORCH_CHECK(err == cudaSuccess, "cudaDeviceGetAttribute(MultiProcessorCount) failed: ",
                cudaGetErrorString(err));

    int const tiles_m = (M + BM - 1) / BM;
    int const tiles_n = (N + BN - 1) / BN;
    int const full_tiles_m = M / BM;
    int const full_tiles_n = N / BN;
    int const total_tiles = tiles_m * tiles_n;
    int const persistent_ctas = sm_count * ctas_per_sm;
    int const grid_ctas = total_tiles < persistent_ctas ? total_tiles : persistent_ctas;

    err = cudaMemsetAsync(counter, 0, sizeof(int), stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemsetAsync(tile_counter) failed: ",
                cudaGetErrorString(err));

    dim3 const grid(grid_ctas);
    dim3 const block(THREADS);

    err = cudaFuncSetAttribute(
        bf16_gemm_sm80_persistent_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_BYTES);
    TORCH_CHECK(err == cudaSuccess,
                "cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: ",
                cudaGetErrorString(err));

    err = cudaFuncSetAttribute(
        bf16_gemm_sm80_persistent_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    TORCH_CHECK(err == cudaSuccess,
                "cudaFuncSetAttribute(PreferredSharedMemoryCarveout) failed: ",
                cudaGetErrorString(err));

    bf16_gemm_sm80_persistent_kernel<<<grid, block, SMEM_BYTES, stream>>>(
        A,
        B,
        C,
        M,
        N,
        K,
        tiles_m,
        tiles_n,
        full_tiles_m,
        full_tiles_n,
        total_tiles,
        chunk_tiles,
        counter,
        throttle_amount,
        throttle_freq,
        throttle_mode,
        0u);

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "bf16_gemm_sm80_persistent_kernel launch failed: ",
                cudaGetErrorString(err));
}

}  // namespace

torch::Tensor gemm_sm80_persistent(
    torch::Tensor A,
    torch::Tensor B,
    unsigned int throttle_amount = 0u,
    unsigned int throttle_freq = 1u,
    int throttle_mode = 0,
    int ctas_per_sm = 2,
    int chunk_tiles = 1) {

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
    TORCH_CHECK(ctas_per_sm >= 1 && ctas_per_sm <= 8,
                "ctas_per_sm must be in [1, 8]");
    TORCH_CHECK(chunk_tiles >= 1 && chunk_tiles <= 64,
                "chunk_tiles must be in [1, 64]");
    TORCH_CHECK(throttle_mode >= 0 && throttle_mode <= 2,
                "throttle_mode must be 0(clock64 wait), 1(nop loop), or 2(__nanosleep)");

    int const M = static_cast<int>(A.size(0));
    int const K = static_cast<int>(A.size(1));
    int const N = static_cast<int>(B.size(1));
    TORCH_CHECK((K % 8) == 0 && (N % 8) == 0,
                "K and N must be multiples of 8 for 128-bit BF16 vectorized cp.async loads");

    auto C = torch::empty({M, N}, A.options());
    if (M == 0 || N == 0) {
        return C;
    }

    auto counter = torch::empty({1}, torch::TensorOptions()
                                      .dtype(torch::kInt32)
                                      .device(A.device()));

    auto stream = at::cuda::getCurrentCUDAStream();
    launch_persistent(
        reinterpret_cast<BF16 const*>(A.data_ptr()),
        reinterpret_cast<BF16 const*>(B.data_ptr()),
        reinterpret_cast<BF16*>(C.data_ptr()),
        M,
        N,
        K,
        counter.data_ptr<int>(),
        throttle_amount,
        throttle_freq,
        throttle_mode,
        ctas_per_sm,
        chunk_tiles,
        stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "gemm_sm80_persistent",
        &gemm_sm80_persistent,
        "Persistent-CTA BF16 GEMM with dense-tile-only throttle control",
        py::arg("A"),
        py::arg("B"),
        py::arg("throttle_amount") = 0u,
        py::arg("throttle_freq") = 1u,
        py::arg("throttle_mode") = 0,
        py::arg("ctas_per_sm") = 2,
        py::arg("chunk_tiles") = 1);
}
