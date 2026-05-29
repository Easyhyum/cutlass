/**
 * probe_streamk.cu  (test_cta_probe)
 *
 * Stream-K BF16 GEMM PyTorch extension built with -DCUTLASS_CTA_PROBE_ENABLED.
 *
 * Each CTA records (smid, globaltimer_start, globaltimer_end, blockIdx) via
 * the hook in cutlass/gemm/threadblock/mma_multistage.h. The mainloop still
 * runs (output C is correct), so this binary is suitable for "real" timing
 * + dispatch analysis in one shot.
 *
 * Tile / stages match the production streamk (build_streamk).
 *
 * Build:
 *   python setup_probe_streamk.py build_ext --inplace
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

// Host-visible probe-output __constant__ symbols.
#include "cutlass/cta_probe_globals.cuh"

using ElementA      = cutlass::bfloat16_t;
using ElementB      = cutlass::bfloat16_t;
using ElementC      = cutlass::bfloat16_t;
using ElementAccum  = float;
using LayoutA       = cutlass::layout::RowMajor;
using LayoutB       = cutlass::layout::RowMajor;
using LayoutC       = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, AlignmentC, ElementAccum, float>;

// MUST match build_streamk/bf16_gemm_sm80_streamk.cu so the grid the
// dispatcher sees is identical to the production path.
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
constexpr int NumStages = 4;

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
    cutlass::gemm::threadblock::ThreadblockSwizzleStreamK,
    NumStages,
    AlignmentA,
    AlignmentB>;

static GemmStreamK gemm_op_streamk;

static void check_cutlass(cutlass::Status s, const char* msg) {
    if (s != cutlass::Status::kSuccess) {
        AT_ERROR(std::string(msg) + ": " + cutlassGetStatusString(s));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  set_probe_buffers — install device pointers + per-call CTA bound.
//  Caller must keep the tensors alive until after the following gemm call.
// ─────────────────────────────────────────────────────────────────────────────
static void set_probe_buffers(torch::Tensor smid,
                              torch::Tensor start_t,
                              torch::Tensor end_t,
                              torch::Tensor bx,
                              torch::Tensor by,
                              torch::Tensor bz,
                              int max_ctas) {
    TORCH_CHECK(smid.is_cuda()    && smid.dtype()    == torch::kInt32, "smid must be cuda int32");
    TORCH_CHECK(start_t.is_cuda() && start_t.dtype() == torch::kInt64, "start_t must be cuda int64");
    TORCH_CHECK(end_t.is_cuda()   && end_t.dtype()   == torch::kInt64, "end_t must be cuda int64");
    TORCH_CHECK(bx.is_cuda()      && bx.dtype()      == torch::kInt32, "bx must be cuda int32");
    TORCH_CHECK(by.is_cuda()      && by.dtype()      == torch::kInt32, "by must be cuda int32");
    TORCH_CHECK(bz.is_cuda()      && bz.dtype()      == torch::kInt32, "bz must be cuda int32");

    int*                 sp = smid.data_ptr<int>();
    unsigned long long*  tp = reinterpret_cast<unsigned long long*>(start_t.data_ptr<int64_t>());
    unsigned long long*  ep = reinterpret_cast<unsigned long long*>(end_t.data_ptr<int64_t>());
    int*                 xp = bx.data_ptr<int>();
    int*                 yp = by.data_ptr<int>();
    int*                 zp = bz.data_ptr<int>();

    cudaMemcpyToSymbol(g_cta_probe_smid_out,  &sp, sizeof(int*));
    cudaMemcpyToSymbol(g_cta_probe_start_out, &tp, sizeof(unsigned long long*));
    cudaMemcpyToSymbol(g_cta_probe_end_out,   &ep, sizeof(unsigned long long*));
    cudaMemcpyToSymbol(g_cta_probe_bx_out,    &xp, sizeof(int*));
    cudaMemcpyToSymbol(g_cta_probe_by_out,    &yp, sizeof(int*));
    cudaMemcpyToSymbol(g_cta_probe_bz_out,    &zp, sizeof(int*));
    cudaMemcpyToSymbol(kCtaProbeMaxCtas,      &max_ctas, sizeof(int));
}

static void clear_probe_buffers() {
    int* null_ptr_i  = nullptr;
    unsigned long long* null_ptr_t = nullptr;
    int zero = 0;
    cudaMemcpyToSymbol(g_cta_probe_smid_out,  &null_ptr_i, sizeof(int*));
    cudaMemcpyToSymbol(g_cta_probe_start_out, &null_ptr_t, sizeof(unsigned long long*));
    cudaMemcpyToSymbol(g_cta_probe_end_out,   &null_ptr_t, sizeof(unsigned long long*));
    cudaMemcpyToSymbol(g_cta_probe_bx_out,    &null_ptr_i, sizeof(int*));
    cudaMemcpyToSymbol(g_cta_probe_by_out,    &null_ptr_i, sizeof(int*));
    cudaMemcpyToSymbol(g_cta_probe_bz_out,    &null_ptr_i, sizeof(int*));
    cudaMemcpyToSymbol(kCtaProbeMaxCtas,      &zero,       sizeof(int));
}

// ─────────────────────────────────────────────────────────────────────────────
//  gemm_streamk_probe — runs the real streamk GEMM. Probe records happen
//  inside MmaMultistage::operator() (per-CTA single write, no atomics).
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemm_streamk_probe(torch::Tensor A, torch::Tensor B,
                                 int split_k_factor = 1,
                                 int avail_sms      = -1) {
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
        avail_sms);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("set_probe_buffers", &set_probe_buffers,
          "Install device probe-output pointers + per-call CTA bound.",
          py::arg("smid"), py::arg("start_t"), py::arg("end_t"),
          py::arg("bx"), py::arg("by"), py::arg("bz"),
          py::arg("max_ctas"));
    m.def("clear_probe_buffers", &clear_probe_buffers,
          "Clear probe pointers (turns probe writes into no-ops).");
    m.def("gemm_streamk_probe", &gemm_streamk_probe,
          "Stream-K GEMM with per-CTA dispatch probe.",
          py::arg("A"), py::arg("B"),
          py::arg("split_k_factor") = 1, py::arg("avail_sms") = -1);
}
