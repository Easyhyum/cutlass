/**
 * probe_streamk.cu
 *
 * Wrapper around the same CUTLASS Stream-K BF16 GEMM as bf16_gemm_sm80_streamk,
 * but built with -DCUTLASS_CTA_PROBE_ENABLED so MmaMultistage::operator()
 * records (smid, globaltimer, blockIdx) per CTA and skips the mainloop.
 *
 * The grid that CUTLASS launches is identical to the real GEMM — only the
 * inner work is replaced by a single recorded write per CTA. The intent is
 * to observe the hardware CTA→SM dispatch (and the streamk swizzle's effect
 * on it) without paying for real GEMM math.
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

// Provides g_cta_probe_smid_out / ..._start_out / ..._bx_out / ..._by_out /
// ..._bz_out / kCtaProbeMaxCtas as host-visible symbols (must be the SAME TU
// as the one whose mma_multistage.h instantiation will use them).
#include "cutlass/cta_probe_globals.cuh"
#include "cutlass/cta_wave_sleep_globals.cuh"

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

// MUST match bf16_gemm_sm80_streamk.cu so the grid the dispatcher sees is the
// same as the production path.
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

// Classic data-parallel (basic GemmIdentityThreadblockSwizzle) for comparison.
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

// ─────────────────────────────────────────────────────────────────────────────
//  set_probe_buffers(...) — install device pointers + per-call CTA bound.
//  Buffers must remain alive for the duration of the following gemm call.
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

// ─────────────────────────────────────────────────────────────────────────────
//  configure_wave_sleep(...) — install wave-aware sleep parameters.
//
//    num_waves     : total #hardware waves the grid will span
//                    (= ceil(CTAs_launched / n_sm)). MUST be ≥ 3 to activate.
//    n_sm          : multi_processor_count.
//    first_wave_smid_thr : SMs with smid < this take NO delay in wave 0.
//                          Remaining SMs sleep (smid - thr) * first_wave_step_ns.
//    first_wave_step_ns  : ns per smid step beyond the threshold.
//    mid_wave_pct        : 0..100 — percent of mid-wave CTAs that sleep.
//    mid_wave_ns         : ns sleep applied to selected mid-wave CTAs.
//    hash_seed           : changes which mid-wave CTAs are selected.
//
//  Calling configure_wave_sleep(0, ...) disables the feature.
// ─────────────────────────────────────────────────────────────────────────────
static void configure_wave_sleep(int num_waves,
                                 int n_sm,
                                 int first_wave_smid_thr,
                                 unsigned int first_wave_step_ns,
                                 unsigned int mid_wave_pct,
                                 unsigned int mid_wave_ns,
                                 unsigned int hash_seed) {
    cudaMemcpyToSymbol(kWaveSleepNumWaves,         &num_waves,           sizeof(int));
    cudaMemcpyToSymbol(kWaveSleepNSm,              &n_sm,                sizeof(int));
    cudaMemcpyToSymbol(kWaveSleepFirstWaveSmidThr, &first_wave_smid_thr, sizeof(int));
    cudaMemcpyToSymbol(kWaveSleepFirstWaveStepNs,  &first_wave_step_ns,  sizeof(unsigned int));
    cudaMemcpyToSymbol(kWaveSleepMidWavePct,       &mid_wave_pct,        sizeof(unsigned int));
    cudaMemcpyToSymbol(kWaveSleepMidWaveNs,        &mid_wave_ns,         sizeof(unsigned int));
    cudaMemcpyToSymbol(kWaveSleepHashSeed,         &hash_seed,           sizeof(unsigned int));
}

static void clear_wave_sleep() {
    int zero_i = 0;
    unsigned int zero_u = 0u;
    unsigned int default_seed = 0xC0FFEE11u;
    cudaMemcpyToSymbol(kWaveSleepNumWaves,         &zero_i,        sizeof(int));
    cudaMemcpyToSymbol(kWaveSleepNSm,              &zero_i,        sizeof(int));
    cudaMemcpyToSymbol(kWaveSleepFirstWaveSmidThr, &zero_i,        sizeof(int));
    cudaMemcpyToSymbol(kWaveSleepFirstWaveStepNs,  &zero_u,        sizeof(unsigned int));
    cudaMemcpyToSymbol(kWaveSleepMidWavePct,       &zero_u,        sizeof(unsigned int));
    cudaMemcpyToSymbol(kWaveSleepMidWaveNs,        &zero_u,        sizeof(unsigned int));
    cudaMemcpyToSymbol(kWaveSleepHashSeed,         &default_seed,  sizeof(unsigned int));
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

// gemm_streamk: same signature as production streamk — but mma_multistage
// recorded probe + returned, so C output is garbage. We still need a real
// GEMM problem so CUTLASS computes the correct grid + workspace.
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

torch::Tensor gemm_basicdp_probe(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16, "BF16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "must be contiguous");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    float alpha = 1.0f, beta = 0.0f;

    GemmBasicDP::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K}, 1, {alpha, beta},
        (ElementA*)A.data_ptr(), (ElementB*)B.data_ptr(),
        (ElementC*)C.data_ptr(), (ElementC*)C.data_ptr(),
        (int64_t)M * K, (int64_t)K * N, (int64_t)M * N, (int64_t)M * N,
        K, N, N, N);

    size_t workspace_bytes = GemmBasicDP::get_workspace_size(args);
    auto workspace = torch::empty({(int64_t)workspace_bytes},
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
    m.def("set_probe_buffers", &set_probe_buffers,
          "Install device probe-output pointers + per-call CTA bound.",
          py::arg("smid"), py::arg("start_t"), py::arg("end_t"),
          py::arg("bx"), py::arg("by"), py::arg("bz"),
          py::arg("max_ctas"));
    m.def("clear_probe_buffers", &clear_probe_buffers,
          "Clear probe pointers (turns probe writes into no-ops).");
    m.def("configure_wave_sleep", &configure_wave_sleep,
          "Install wave-aware per-CTA sleep parameters. num_waves ≥ 3 activates.",
          py::arg("num_waves"),
          py::arg("n_sm"),
          py::arg("first_wave_smid_thr"),
          py::arg("first_wave_step_ns"),
          py::arg("mid_wave_pct"),
          py::arg("mid_wave_ns"),
          py::arg("hash_seed") = 0xC0FFEE11u);
    m.def("clear_wave_sleep", &clear_wave_sleep,
          "Disable wave-aware sleep (next gemm call sees no delay).");
    m.def("gemm_streamk_probe", &gemm_streamk_probe,
          "Stream-K GEMM launched in PROBE mode (no real mainloop).",
          py::arg("A"), py::arg("B"),
          py::arg("split_k_factor") = 1, py::arg("avail_sms") = -1);
    m.def("gemm_basicdp_probe", &gemm_basicdp_probe,
          "Classic data-parallel GEMM in PROBE mode (no real mainloop).",
          py::arg("A"), py::arg("B"));
}
