/**
 * bf16_gemm_sm80_streamk.cu
 *
 * CUTLASS Stream-K BF16 GEMM PyTorch 확장 (SM120 Blackwell, HMMA legacy path)
 *
 * 원본 알고리즘 출처:
 *   /workspace/examples/47_ampere_gemm_universal_streamk/ampere_gemm_universal_streamk.cu
 *   "Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix
 *    Multiplication on the GPU" (https://arxiv.org/abs/2301.03598)
 *
 * 차이점 (vs gemm_sm80_v3):
 *   - GemmIdentityThreadblockSwizzle → ThreadblockSwizzleStreamK
 *   - Stream-K decomposition: SM 간 work-stealing 으로 wave quantization 손실 제거
 *   - 비-타일-정렬 크기에서 cuBLAS 와 거의 동등하거나 더 빠름
 *
 * Build:
 *   python setup_bf16_sm80_streamk.py build_ext --inplace
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle_streamk.h"

// Method A: SM-staggered nanosleep params (constants live in cutlass_sleep_globals.cuh,
// included by mma_multistage.h). We re-include here to get host-side symbol refs for
// cudaMemcpyToSymbol.
#ifdef CUTLASS_SLEEP_ENABLED
#include "cutlass/cutlass_sleep_globals.cuh"
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  BF16 × BF16 → BF16, FP32 accumulator, Stream-K decomposition
// ─────────────────────────────────────────────────────────────────────────────
using ElementA      = cutlass::bfloat16_t;
using ElementB      = cutlass::bfloat16_t;
using ElementC      = cutlass::bfloat16_t;
using ElementAccum  = float;
using LayoutA       = cutlass::layout::RowMajor;
using LayoutB       = cutlass::layout::RowMajor;
using LayoutC       = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 8
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 8
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, AlignmentC, ElementAccum, float>;

// Tile shapes selected for SM80 HMMA BF16:
//   BlockTile  128×128×32
//   WarpTile   64×64×32
//   MMA tile   16×8×16   (HMMA m16n8k16)
//   Stages     4         (deeper pipeline)
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int NumStages = 4;

// Stream-K GEMM (key difference from classic data-parallel = swizzle type)
using GemmStreamK = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,   // ← Stream-K
    NumStages,
    AlignmentA,
    AlignmentB>;

// Classic data-parallel GEMM (성능 비교용)
using GemmBasicDP = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB>;

static GemmStreamK gemm_op_streamk;
static GemmBasicDP gemm_op_basicdp;

static void check_cutlass(cutlass::Status s, const char* msg) {
    if (s != cutlass::Status::kSuccess) {
        AT_ERROR(std::string(msg) + ": " + cutlassGetStatusString(s));
    }
}

// Method A: host-side setter for SM-staggered sleep params.
// __constant__ 변수는 cutlass_sleep_globals.cuh 에서 정의됨.
static void set_sleep_params(unsigned int sleep_ns, unsigned int sleep_freq,
                             unsigned int stagger_ns = 0u,
                             unsigned int stagger_mod = 1u) {
#ifdef CUTLASS_SLEEP_ENABLED
    cudaMemcpyToSymbol(kCutlassSleepNs,         &sleep_ns,    sizeof(unsigned int));
    cudaMemcpyToSymbol(kCutlassSleepFreq,       &sleep_freq,  sizeof(unsigned int));
    cudaMemcpyToSymbol(kCutlassSleepStaggerNs,  &stagger_ns,  sizeof(unsigned int));
    cudaMemcpyToSymbol(kCutlassSleepStaggerMod, &stagger_mod, sizeof(unsigned int));
#else
    (void)sleep_ns; (void)sleep_freq; (void)stagger_ns; (void)stagger_mod;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
//  gemm_streamk: Stream-K BF16 (work-stealing 로 load-balance)
//
//  split_k_factor:
//    1   : Stream-K 기본 (data-parallel 부분 + Stream-K 잔여)
//    > 1 : Stream-K 가 Split-K 를 emulating
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemm_streamk(torch::Tensor A, torch::Tensor B,
                           int split_k_factor = 1, int avail_sms = -1,
                           unsigned int sleep_ns = 0u, unsigned int sleep_freq = 1u,
                           unsigned int stagger_ns = 0u, unsigned int stagger_mod = 1u) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be BF16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A,B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols must equal B.rows");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous (row-major)");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    float alpha = 1.0f, beta = 0.0f;

    // Method A: upload sleep params before kernel launch
    set_sleep_params(sleep_ns, sleep_freq, stagger_ns, stagger_mod);

    GemmStreamK::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        split_k_factor < 1 ? 1 : split_k_factor,
        {alpha, beta},
        (ElementA*)A.data_ptr(),
        (ElementB*)B.data_ptr(),
        (ElementC*)C.data_ptr(),
        (ElementC*)C.data_ptr(),
        (int64_t)M * K, (int64_t)K * N, (int64_t)M * N, (int64_t)M * N,
        K, N, N, N,
        avail_sms
    );

    size_t workspace_bytes = GemmStreamK::get_workspace_size(args);
    auto workspace = torch::empty(
        {(int64_t)workspace_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

    auto s = gemm_op_streamk.can_implement(args);
    check_cutlass(s, "streamk can_implement");
    s = gemm_op_streamk.initialize(args, workspace.data_ptr(), stream);
    check_cutlass(s, "streamk initialize");
    s = gemm_op_streamk.run(stream);
    check_cutlass(s, "streamk run");
    return C;
}

// 비교용: 동일 tile config / 다른 swizzle (classic data-parallel)
torch::Tensor gemm_basicdp(torch::Tensor A, torch::Tensor B,
                           unsigned int sleep_ns = 0u, unsigned int sleep_freq = 1u,
                           unsigned int stagger_ns = 0u, unsigned int stagger_mod = 1u) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be BF16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "must be contiguous");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    float alpha = 1.0f, beta = 0.0f;

    set_sleep_params(sleep_ns, sleep_freq, stagger_ns, stagger_mod);

    GemmBasicDP::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,
        {alpha, beta},
        (ElementA*)A.data_ptr(),
        (ElementB*)B.data_ptr(),
        (ElementC*)C.data_ptr(),
        (ElementC*)C.data_ptr(),
        (int64_t)M * K, (int64_t)K * N, (int64_t)M * N, (int64_t)M * N,
        K, N, N, N
    );

    size_t workspace_bytes = GemmBasicDP::get_workspace_size(args);
    auto workspace = torch::empty(
        {(int64_t)workspace_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

    auto s = gemm_op_basicdp.can_implement(args);
    check_cutlass(s, "basicdp can_implement");
    s = gemm_op_basicdp.initialize(args, workspace.data_ptr(), stream);
    check_cutlass(s, "basicdp initialize");
    s = gemm_op_basicdp.run(stream);
    check_cutlass(s, "basicdp run");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_streamk", &gemm_streamk,
          "BF16 Stream-K GEMM (128x128x32, 4-stage) + Method A SM-stagger",
          py::arg("A"), py::arg("B"),
          py::arg("split_k_factor") = 1,
          py::arg("avail_sms")      = -1,
          py::arg("sleep_ns")       = 0u,
          py::arg("sleep_freq")     = 1u,
          py::arg("stagger_ns")     = 0u,
          py::arg("stagger_mod")    = 1u);
    m.def("gemm_basicdp", &gemm_basicdp,
          "BF16 classic data-parallel GEMM (비교용) + Method A SM-stagger",
          py::arg("A"), py::arg("B"),
          py::arg("sleep_ns")       = 0u,
          py::arg("sleep_freq")     = 1u,
          py::arg("stagger_ns")     = 0u,
          py::arg("stagger_mod")    = 1u);
}
