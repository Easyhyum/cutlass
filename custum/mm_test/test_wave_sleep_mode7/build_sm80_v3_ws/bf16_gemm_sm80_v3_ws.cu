/**
 * bf16_gemm_sm80_v3_ws.cu  (test_wave_sleep_mode7)
 *
 * Pure CUTLASS device::Gemm (128×128×64 tile, 3-stage,
 * GemmIdentityThreadblockSwizzle<8>) — same template as build_sm80_v3 — but
 * compiled with wave-sleep machinery on.  Adds the host setter
 *   prime_wave_sleep(num_waves, n_sm, first_smid_thr,
 *                    first_step_ns, mid_pct, mid_ns,
 *                    mode, shape, seed)
 * mirroring the streamk binding (`bf16_gemm_sm80_streamk_ws`) so the same
 * Python sweep code can drive both kernels.
 *
 * Mode 7 (SM gating) is the one we want for the first sweep:
 *   first_smid_thr = n_sm * active_pct / 100
 *   → SMs with smid >= thr return at operator() entry; SMs 0..thr-1 run.
 *
 * Build:
 *   python setup_bf16_sm80_v3_ws.py build_ext --inplace
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

// Wave-sleep `__constant__` symbol scaffolding.  Must be in the SAME TU as
// the gemm_op instantiation so cudaMemcpyToSymbol resolves to this binary's
// constants (not the production .so's).
#ifdef CUTLASS_SLEEP_ENABLED
#include "cutlass/cutlass_sleep_globals.cuh"
#endif
#ifdef CUTLASS_WAVE_SLEEP_ENABLED
#include "cutlass/cta_wave_sleep_globals.cuh"
#endif

using ElementA      = cutlass::bfloat16_t;
using ElementB      = cutlass::bfloat16_t;
using ElementC      = cutlass::bfloat16_t;
using ElementAccum  = float;
using LayoutA       = cutlass::layout::RowMajor;
using LayoutB       = cutlass::layout::RowMajor;
using LayoutC       = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccum,
    float
>;

// Same template as build_sm80_v3 (128×128×64, 3-stage, GemmIdentity<8>).
using GemmSm80V3 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64,  64,  32>,
    cutlass::gemm::GemmShape<16,   8,  16>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    8,
    8,
    false,
    cutlass::arch::OpMultiplyAdd
>;

static GemmSm80V3 gemm_op_sm80_v3;

static void check_cutlass(cutlass::Status s, const char* msg) {
    if (s != cutlass::Status::kSuccess) {
        AT_ERROR(std::string(msg) + ": " + cutlassGetStatusString(s));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  One-shot wave-aware sleep priming — same semantics / signature as the
//  production streamk binding so the Python sweep code is shared.
// ─────────────────────────────────────────────────────────────────────────────
#ifdef CUTLASS_WAVE_SLEEP_ENABLED
static int          g_one_wsleep_num_waves       = 0;
static int          g_one_wsleep_n_sm            = 0;
static int          g_one_wsleep_first_smid_thr  = 0;
static unsigned int g_one_wsleep_first_step_ns   = 0;
static unsigned int g_one_wsleep_mid_pct         = 0;
static unsigned int g_one_wsleep_mid_ns          = 0;
static unsigned int g_one_wsleep_seed            = 0xC0FFEE11u;
static int          g_one_wsleep_mode            = 0;
static int          g_one_wsleep_shape           = 0;
static bool         g_one_wsleep_armed           = false;

static void prime_wave_sleep(int num_waves, int n_sm,
                             int first_smid_thr, unsigned int first_step_ns,
                             unsigned int mid_pct, unsigned int mid_ns,
                             int mode = 0, int shape = 0,
                             unsigned int seed = 0xC0FFEE11u) {
    g_one_wsleep_num_waves      = num_waves;
    g_one_wsleep_n_sm           = n_sm;
    g_one_wsleep_first_smid_thr = first_smid_thr;
    g_one_wsleep_first_step_ns  = first_step_ns;
    g_one_wsleep_mid_pct        = mid_pct;
    g_one_wsleep_mid_ns         = mid_ns;
    g_one_wsleep_seed           = seed;
    g_one_wsleep_mode           = mode;
    g_one_wsleep_shape          = shape;
    g_one_wsleep_armed          = (num_waves >= 3);
}
#else
static void prime_wave_sleep(int, int, int, unsigned int,
                             unsigned int, unsigned int,
                             int = 0, int = 0, unsigned int = 0xC0FFEE11u) {}
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  gemm_sm80_v3_ws — runs the real device::Gemm with wave-sleep one-shot
//  applied (if primed).  Mainloop unchanged; the only effect of wave-sleep
//  is the `#ifdef CUTLASS_WAVE_SLEEP_ENABLED` block inside
//  MmaMultistage::operator() that reads the `__constant__` params we set here.
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemm_sm80_v3_ws(torch::Tensor A, torch::Tensor B) {
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

#ifdef CUTLASS_WAVE_SLEEP_ENABLED
    if (g_one_wsleep_armed) {
        cudaMemcpyToSymbol(kWaveSleepNumWaves,
                           &g_one_wsleep_num_waves,      sizeof(int));
        cudaMemcpyToSymbol(kWaveSleepNSm,
                           &g_one_wsleep_n_sm,           sizeof(int));
        cudaMemcpyToSymbol(kWaveSleepFirstWaveSmidThr,
                           &g_one_wsleep_first_smid_thr, sizeof(int));
        cudaMemcpyToSymbol(kWaveSleepFirstWaveStepNs,
                           &g_one_wsleep_first_step_ns,  sizeof(unsigned int));
        cudaMemcpyToSymbol(kWaveSleepMidWavePct,
                           &g_one_wsleep_mid_pct,        sizeof(unsigned int));
        cudaMemcpyToSymbol(kWaveSleepMidWaveNs,
                           &g_one_wsleep_mid_ns,         sizeof(unsigned int));
        cudaMemcpyToSymbol(kWaveSleepHashSeed,
                           &g_one_wsleep_seed,           sizeof(unsigned int));
        cudaMemcpyToSymbol(kWaveSleepMode,
                           &g_one_wsleep_mode,           sizeof(int));
        cudaMemcpyToSymbol(kWaveSleepShape,
                           &g_one_wsleep_shape,          sizeof(int));
        g_one_wsleep_armed = false;  // one-shot
    } else {
        int zero = 0;
        cudaMemcpyToSymbol(kWaveSleepNumWaves, &zero, sizeof(int));
    }
#endif

    GemmSm80V3::Arguments args(
        {M, N, K},
        {(ElementA*)A.data_ptr(),  K},
        {(ElementB*)B.data_ptr(),  N},
        {(ElementC*)C.data_ptr(),  N},
        {(ElementC*)C.data_ptr(),  N},
        {alpha, beta}
    );

    auto s = gemm_op_sm80_v3.initialize(args, nullptr, stream);
    check_cutlass(s, "sm80_v3_ws initialize");
    s = gemm_op_sm80_v3.run(stream);
    check_cutlass(s, "sm80_v3_ws run");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_sm80_v3_ws", &gemm_sm80_v3_ws,
          "CUTLASS device::Gemm 128x128x64 3-stage with one-shot wave-sleep.",
          py::arg("A"), py::arg("B"));
    m.def("prime_wave_sleep", &prime_wave_sleep,
          "Prime one-shot wave-aware sleep for the NEXT gemm_sm80_v3_ws call.",
          py::arg("num_waves"),
          py::arg("n_sm"),
          py::arg("first_smid_thr"),
          py::arg("first_step_ns"),
          py::arg("mid_pct"),
          py::arg("mid_ns"),
          py::arg("mode")  = 0,
          py::arg("shape") = 0,
          py::arg("seed")  = 0xC0FFEE11u);
}
