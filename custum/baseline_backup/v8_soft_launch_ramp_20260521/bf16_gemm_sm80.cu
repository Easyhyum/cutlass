/**
 * bf16_gemm_sm80.cu
 *
 * CUTLASS 2.x SM80 집합체 기반 BF16 GEMM PyTorch 확장
 * - cp.async + mma.sync.aligned.m16n8k16 (HMMA)
 * - 3-stage pipeline (double buffering+)
 * - 128×256×32 block tile, 64×64×32 warp tile
 * - SM120에서 sm_120 으로 컴파일 → SM80 HMMA 경로 사용
 *
 * Build:
 *   python setup_bf16_sm80.py build_ext --inplace
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// ── CUTLASS 2.x headers ──────────────────────────────────────────────────────
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// ─────────────────────────────────────────────────────────────────────────────
//  CUTLASS 2.x GemmUniversal: SM80, BF16×BF16→BF16, FP32 accumulator
//  BlockTile 128×256×32, WarpTile 64×64×32, Instruction 16×8×16, Stages=3
// ─────────────────────────────────────────────────────────────────────────────
using ElementA      = cutlass::bfloat16_t;
using ElementB      = cutlass::bfloat16_t;
using ElementC      = cutlass::bfloat16_t;
using ElementAccum  = float;
using LayoutA       = cutlass::layout::RowMajor;
using LayoutB       = cutlass::layout::RowMajor;   // cuBLAS nn_align8 = NN layout
using LayoutC       = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,                   // output type
    128 / cutlass::sizeof_bits<ElementC>::value,  // elements_per_access
    ElementAccum,               // accumulator type
    float                       // compute type for alpha/beta
>;

// Tile shapes: BlockTile 128x256x32, WarpTile 64x64x32, MMA 16x8x16
using GemmKernel = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,   // block tile
    cutlass::gemm::GemmShape<64,  64,  32>,   // warp tile
    cutlass::gemm::GemmShape<16,   8,  16>,   // mma tile (HMMA m16n8k16)
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,    // pipeline stages (triple buffering)
    8,    // AlignmentA (128-bit)
    8,    // AlignmentB (128-bit)
    false,
    cutlass::arch::OpMultiplyAdd
>;

// Variant 2: 256×128 block tile
using GemmKernel256x128 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,  // block tile
    cutlass::gemm::GemmShape<64,  64,  32>,  // warp tile
    cutlass::gemm::GemmShape<16,   8,  16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd
>;

// Variant 3: 128×128×64 with K=64 blocking
using GemmKernel128x128k64 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,  // block tile
    cutlass::gemm::GemmShape<64,  64,  32>,  // warp tile
    cutlass::gemm::GemmShape<16,   8,  16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd
>;

// Variant 4: 128x256 with 5 stages (hide memory latency better)
using GemmKernel128x256s5 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,  // block tile
    cutlass::gemm::GemmShape<64,  64,  32>,  // warp tile
    cutlass::gemm::GemmShape<16,   8,  16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,   // 5 stages
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd
>;

// ─────────────────────────────────────────────────────────────────────────────
static GemmKernel          gemm_op_128x256;
static GemmKernel256x128   gemm_op_256x128;
static GemmKernel128x128k64 gemm_op_128x128k64;
static GemmKernel128x256s5  gemm_op_128x256s5;

static void check_cutlass(cutlass::Status s, const char* msg) {
    if (s != cutlass::Status::kSuccess) {
        AT_ERROR(std::string(msg) + ": " + cutlassGetStatusString(s));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  __nanosleep 파라미터 전달용 __constant__ 메모리
//
//  사용 방법:
//    1. Python에서 gemm_sm80_v3(A, B, sleep_ns=1000, sleep_freq=4) 호출
//    2. 이 파일에서 cudaMemcpyToSymbol 로 값을 GPU에 업로드
//    3. mma_multistage.h 의 #ifdef CUTLASS_SLEEP_ENABLED 블록에서
//       kCutlassSleepNs / kCutlassSleepFreq 를 읽어 __nanosleep 호출
//
//  CUTLASS_SLEEP_ENABLED : setup_bf16_sm80.py 의 -DCUTLASS_SLEEP_ENABLED 로 활성화
// ─────────────────────────────────────────────────────────────────────────────
#ifdef CUTLASS_SLEEP_ENABLED
// __constant__ 정의는 cutlass_sleep_globals.cuh 에 있음.
// mma_multistage.h 가 이미 include 하므로 여기서 재정의 불필요.
// (mma_multistage.h 가 먼저 include 되지 않는 경우를 위해 명시적으로 include)
#include "cutlass/cutlass_sleep_globals.cuh"
#endif

// host 측 setter (커널 런치 직전에 호출) — v4 (pipeline-aware spin only)
//
//   stagger_cycles : SM phase 당 추가 clock64 busy-spin (CYCLES, not ns)
//                    spin_cycles = (smid % stagger_mod) * stagger_cycles
//   stagger_mod    : SM phase 개수 (2 이상이어야 동작; 1이면 no-op)
//
//   stagger_cycles=0 또는 stagger_mod<=1  →  baseline (no spin)
//
//   sleep_ns/sleep_freq 인자는 API 호환성 위해 시그니처에 남아 있으나 무시됨.
static void set_sleep_params(unsigned int sleep_ns, unsigned int sleep_freq,
                             unsigned int stagger_ns  = 0u,
                             unsigned int stagger_mod = 1u) {
    (void)sleep_ns; (void)sleep_freq;  // v4: 미사용 (host API 하위 호환)
#ifdef CUTLASS_SLEEP_ENABLED
    cudaMemcpyToSymbol(kCutlassSleepStaggerNs,  &stagger_ns,  sizeof(unsigned int));
    cudaMemcpyToSymbol(kCutlassSleepStaggerMod, &stagger_mod, sizeof(unsigned int));
#else
    (void)stagger_ns; (void)stagger_mod;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
//  gemm_sm80: BF16×BF16→BF16 via CUTLASS SM80 HMMA + cp.async pipeline
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemm_sm80(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be BF16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols must equal B.rows");

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    int M = A.size(0), K = A.size(1), N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    float alpha = 1.0f, beta = 0.0f;

    // NN layout: A(M,K) stride K, B(K,N) stride N
    GemmKernel::Arguments args(
        {M, N, K},
        {(ElementA*)A.data_ptr(),  K},
        {(ElementB*)B.data_ptr(),  N},
        {(ElementC*)C.data_ptr(),  N},
        {(ElementC*)C.data_ptr(),  N},
        {alpha, beta}
    );

    cutlass::Status s = gemm_op_128x256.initialize(args, nullptr, stream);
    check_cutlass(s, "initialize");
    s = gemm_op_128x256.run(stream);
    check_cutlass(s, "run");
    return C;
}

// ─────────────────────────────────────────────────────────────────────────────
//  공통 실행 헬퍼
//  sleep_ns  : __nanosleep 지속 시간 (ns). 0 이면 sleep 없음.
//  sleep_freq: warp_mma_k 카운터가 sleep_freq 의 배수일 때만 sleep.
//              1 = 매 warp_mma_k 마다, 4 = 4번에 1번, ...
// ─────────────────────────────────────────────────────────────────────────────
template <typename GemmOp>
torch::Tensor run_gemm(GemmOp& op, torch::Tensor A, torch::Tensor B,
                       const char* name,
                       unsigned int sleep_ns    = 0u,
                       unsigned int sleep_freq  = 1u,
                       unsigned int stagger_ns  = 0u,
                       unsigned int stagger_mod = 1u) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be BF16");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous (row-major)");
    int M = A.size(0), K = A.size(1), N = B.size(1);

    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    float alpha = 1.0f, beta = 0.0f;

    // __constant__ 메모리에 sleep 파라미터 업로드 (커널 런치 전)
    set_sleep_params(sleep_ns, sleep_freq, stagger_ns, stagger_mod);

    // NN layout: A(M,K) row-major stride K,  B(K,N) row-major stride N
    typename GemmOp::Arguments args(
        {M, N, K},
        {(ElementA*)A.data_ptr(),     K},   // lda = K
        {(ElementB*)B.data_ptr(),     N},   // ldb = N  (NN layout)
        {(ElementC*)C.data_ptr(),     N},
        {(ElementC*)C.data_ptr(),     N},
        {alpha, beta}
    );

    auto s = op.initialize(args, nullptr, stream);
    check_cutlass(s, name);
    s = op.run(stream);
    check_cutlass(s, name);
    return C;
}

torch::Tensor gemm_sm80_v2(torch::Tensor A, torch::Tensor B,
                            unsigned int sleep_ns = 0u, unsigned int sleep_freq = 1u,
                            unsigned int stagger_ns = 0u, unsigned int stagger_mod = 1u) {
    return run_gemm(gemm_op_256x128, A, B, "256x128",
                    sleep_ns, sleep_freq, stagger_ns, stagger_mod);
}
torch::Tensor gemm_sm80_v3(torch::Tensor A, torch::Tensor B,
                            unsigned int sleep_ns = 0u, unsigned int sleep_freq = 1u,
                            unsigned int stagger_ns = 0u, unsigned int stagger_mod = 1u) {
    return run_gemm(gemm_op_128x128k64, A, B, "128x128k64",
                    sleep_ns, sleep_freq, stagger_ns, stagger_mod);
}
torch::Tensor gemm_sm80_v4(torch::Tensor A, torch::Tensor B,
                            unsigned int sleep_ns = 0u, unsigned int sleep_freq = 1u,
                            unsigned int stagger_ns = 0u, unsigned int stagger_mod = 1u) {
    return run_gemm(gemm_op_128x256s5, A, B, "128x256s5",
                    sleep_ns, sleep_freq, stagger_ns, stagger_mod);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_sm80",    &gemm_sm80,    "BF16 128×256×32 3stage");
    m.def("gemm_sm80_v2", &gemm_sm80_v2, "BF16 256×128×32 3stage (Method A: SM-stagger)",
          py::arg("A"), py::arg("B"),
          py::arg("sleep_ns")=0u, py::arg("sleep_freq")=1u,
          py::arg("stagger_ns")=0u, py::arg("stagger_mod")=1u);
    m.def("gemm_sm80_v3", &gemm_sm80_v3, "BF16 128×128×64 3stage (Method A: SM-stagger)",
          py::arg("A"), py::arg("B"),
          py::arg("sleep_ns")=0u, py::arg("sleep_freq")=1u,
          py::arg("stagger_ns")=0u, py::arg("stagger_mod")=1u);
    m.def("gemm_sm80_v4", &gemm_sm80_v4, "BF16 128×256×32 5stage (Method A: SM-stagger)",
          py::arg("A"), py::arg("B"),
          py::arg("sleep_ns")=0u, py::arg("sleep_freq")=1u,
          py::arg("stagger_ns")=0u, py::arg("stagger_mod")=1u);
}
