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

#include <chrono>
#include <thread>

// Method A: SM-staggered nanosleep params (constants live in cutlass_sleep_globals.cuh,
// included by mma_multistage.h). We re-include here to get host-side symbol refs for
// cudaMemcpyToSymbol.
#ifdef CUTLASS_SLEEP_ENABLED
#include "cutlass/cutlass_sleep_globals.cuh"
#endif
#ifdef CUTLASS_WAVE_SLEEP_ENABLED
#include "cutlass/cta_wave_sleep_globals.cuh"

// One-shot wave-aware sleep — primed by host, consumed by the next
// gemm_streamk call, automatically disarmed afterwards.
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

static void set_mem_pressure_buffer(torch::Tensor buf) {
    TORCH_CHECK(buf.is_cuda() && buf.dtype() == torch::kInt32,
                "buf must be cuda int32");
    unsigned int* p = reinterpret_cast<unsigned int*>(buf.data_ptr<int>());
    int n_words = static_cast<int>(buf.numel());
    cudaMemcpyToSymbol(g_mem_pressure_buf, &p, sizeof(unsigned int*));
    cudaMemcpyToSymbol(kMemPressureWords, &n_words, sizeof(int));
}

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

// v8/v9 (A) ramp params + legacy stagger.
// v8: time-dilution per outer iter (linear activity model)
// v9: spatial SM ramp (graduated CTA entry delay by smid)
static void set_sleep_params(unsigned int sleep_ns, unsigned int sleep_freq,
                             unsigned int stagger_ns  = 0u,
                             unsigned int stagger_mod = 1u,
                             // v8 params:
                             unsigned int ramp_start_pct    = 100u,
                             unsigned int ramp_step_pct     = 100u,
                             unsigned int ramp_iter_time_ns = 0u,
                             // v9 params:
                             unsigned int v9_smid_threshold = 0u,
                             unsigned int v9_step_ns        = 0u) {
    (void)sleep_ns; (void)sleep_freq;
#ifdef CUTLASS_SLEEP_ENABLED
    cudaMemcpyToSymbol(kCutlassSleepStaggerNs,        &stagger_ns,         sizeof(unsigned int));
    cudaMemcpyToSymbol(kCutlassSleepStaggerMod,       &stagger_mod,        sizeof(unsigned int));
    cudaMemcpyToSymbol(kCutlassRampStartPct,          &ramp_start_pct,     sizeof(unsigned int));
    cudaMemcpyToSymbol(kCutlassRampStepPct,           &ramp_step_pct,      sizeof(unsigned int));
    cudaMemcpyToSymbol(kCutlassRampIterTimeNs,        &ramp_iter_time_ns,  sizeof(unsigned int));
    cudaMemcpyToSymbol(kCutlassRampV9SmidThreshold,   &v9_smid_threshold,  sizeof(unsigned int));
    cudaMemcpyToSymbol(kCutlassRampV9StepNs,          &v9_step_ns,         sizeof(unsigned int));
#else
    (void)stagger_ns; (void)stagger_mod;
    (void)ramp_start_pct; (void)ramp_step_pct; (void)ramp_iter_time_ns;
    (void)v9_smid_threshold; (void)v9_step_ns;
#endif
}

// ── v8 (A) one-shot ramp priming ─────────────────────────────────────────
// prime_ramp(start_pct, step_pct, iter_time_ns) → 다음 gemm_streamk 1회만
// 적용 → 자동 reset.
static unsigned int g_oneshot_ramp_start_pct    = 100u;
static unsigned int g_oneshot_ramp_step_pct     = 100u;
static unsigned int g_oneshot_ramp_iter_time_ns = 0u;
static bool         g_oneshot_ramp_armed        = false;

static void prime_ramp(unsigned int start_pct, unsigned int step_pct,
                       unsigned int iter_time_ns) {
    if (start_pct >= 100u || step_pct == 0u || iter_time_ns == 0u) {
        g_oneshot_ramp_armed = false;
        return;
    }
    g_oneshot_ramp_start_pct    = start_pct;
    g_oneshot_ramp_step_pct     = step_pct;
    g_oneshot_ramp_iter_time_ns = iter_time_ns;
    g_oneshot_ramp_armed        = true;
}

// v9 one-shot
static unsigned int g_oneshot_v9_smid_threshold = 0u;
static unsigned int g_oneshot_v9_step_ns        = 0u;
static bool         g_oneshot_v9_armed          = false;

static void prime_ramp_v9(unsigned int smid_threshold, unsigned int step_ns) {
    if (step_ns == 0u) {
        g_oneshot_v9_armed = false; return;
    }
    g_oneshot_v9_smid_threshold = smid_threshold;
    g_oneshot_v9_step_ns        = step_ns;
    g_oneshot_v9_armed          = true;
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
                           unsigned int stagger_ns = 0u, unsigned int stagger_mod = 1u,
                           unsigned int ramp_start_pct = 100u,
                           unsigned int ramp_step_pct  = 100u,
                           unsigned int ramp_iter_time_ns = 0u,
                           unsigned int v9_smid_threshold = 0u,
                           unsigned int v9_step_ns        = 0u) {
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

    // v8 (A): one-shot ramp 가 armed 되어 있으면 그 값을 이번 한 번만 사용.
    unsigned int eff_start = ramp_start_pct;
    unsigned int eff_step  = ramp_step_pct;
    unsigned int eff_itns  = ramp_iter_time_ns;
    if (g_oneshot_ramp_armed) {
        eff_start = g_oneshot_ramp_start_pct;
        eff_step  = g_oneshot_ramp_step_pct;
        eff_itns  = g_oneshot_ramp_iter_time_ns;
        g_oneshot_ramp_armed = false;
    }
    // v9 one-shot
    unsigned int eff_v9_thr  = v9_smid_threshold;
    unsigned int eff_v9_step = v9_step_ns;
    if (g_oneshot_v9_armed) {
        eff_v9_thr  = g_oneshot_v9_smid_threshold;
        eff_v9_step = g_oneshot_v9_step_ns;
        g_oneshot_v9_armed = false;
    }
    set_sleep_params(sleep_ns, sleep_freq, stagger_ns, stagger_mod,
                     eff_start, eff_step, eff_itns,
                     eff_v9_thr, eff_v9_step);

#ifdef CUTLASS_WAVE_SLEEP_ENABLED
    // Wire one-shot wave-sleep params (or disable if not primed).
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

// ─────────────────────────────────────────────────────────────────────────────
//  gemm_streamk_chunked — Kernel-level chunking.
//
//  Splits A by rows (M) into n_chunks of `chunk_m` rows each, then issues a
//  separate streamk launch per chunk on the same CUDA stream. Eliminates the
//  Python loop overhead that torch-level chunking incurs while preserving the
//  per-chunk power profile (each chunk is a small-M GEMM staying around the
//  600 W sweet spot).
//
//  Workspace is reused across chunks when the chunk shape doesn't change
//  (i.e. all but possibly the last chunk).
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemm_streamk_chunked(torch::Tensor A, torch::Tensor B,
                                   int chunk_m,
                                   int split_k_factor = 1,
                                   int avail_sms      = -1,
                                   int chunk_idle_us  = 0) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be BF16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A,B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols must equal B.rows");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous (row-major)");
    TORCH_CHECK(chunk_m > 0, "chunk_m must be positive");

    int const M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    float alpha = 1.0f, beta = 0.0f;

    // Disable any one-shot wave-sleep priming for chunked calls — chunked is
    // the alternative; don't mix the two unintentionally.
#ifdef CUTLASS_WAVE_SLEEP_ENABLED
    int zero = 0;
    cudaMemcpyToSymbol(kWaveSleepNumWaves, &zero, sizeof(int));
#endif

    int n_chunks = (M + chunk_m - 1) / chunk_m;
    // pre-allocate workspace for the full-size chunk (reused n_chunks-1 times).
    int last_rows = M - (n_chunks - 1) * chunk_m;
    int full_rows = (n_chunks >= 2 ? chunk_m : last_rows);

    // Reusable workspace tensor sized for the full-chunk arguments.
    torch::Tensor workspace_full;
    if (n_chunks >= 2) {
        GemmStreamK::Arguments probe_args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            {full_rows, N, K},
            split_k_factor < 1 ? 1 : split_k_factor,
            {alpha, beta},
            (ElementA*)A.data_ptr(), (ElementB*)B.data_ptr(),
            (ElementC*)C.data_ptr(), (ElementC*)C.data_ptr(),
            (int64_t)full_rows * K, (int64_t)K * N,
            (int64_t)full_rows * N, (int64_t)full_rows * N,
            K, N, N, N,
            avail_sms);
        size_t bytes = GemmStreamK::get_workspace_size(probe_args);
        workspace_full = torch::empty({(int64_t)bytes},
            torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));
    }

    for (int c = 0; c < n_chunks; ++c) {
        int const start = c * chunk_m;
        int const rows  = std::min(chunk_m, M - start);

        ElementA* A_ptr = ((ElementA*)A.data_ptr()) + (int64_t)start * K;
        ElementC* C_ptr = ((ElementC*)C.data_ptr()) + (int64_t)start * N;

        GemmStreamK::Arguments args(
            cutlass::gemm::GemmUniversalMode::kGemm,
            {rows, N, K},
            split_k_factor < 1 ? 1 : split_k_factor,
            {alpha, beta},
            A_ptr, (ElementB*)B.data_ptr(),
            C_ptr, C_ptr,
            (int64_t)rows * K, (int64_t)K * N,
            (int64_t)rows * N, (int64_t)rows * N,
            K, N, N, N,
            avail_sms);

        void* ws_ptr = nullptr;
        torch::Tensor ws_last;
        if (rows == full_rows && workspace_full.defined()) {
            ws_ptr = workspace_full.data_ptr();
        } else {
            size_t bytes = GemmStreamK::get_workspace_size(args);
            ws_last = torch::empty({(int64_t)bytes},
                torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));
            ws_ptr = ws_last.data_ptr();
        }

        auto s = gemm_op_streamk.can_implement(args);
        check_cutlass(s, "streamk_chunked can_implement");
        s = gemm_op_streamk.initialize(args, ws_ptr, stream);
        check_cutlass(s, "streamk_chunked initialize");
        s = gemm_op_streamk.run(stream);
        check_cutlass(s, "streamk_chunked run");

        // Optional inter-chunk idle gap: stream-sync + CPU usleep so the
        // GPU power rail can recover before the next chunk's MMA starts.
        // Mimics the natural Python-loop overhead that made torch-chunked
        // settle to ~618 W.
        if (chunk_idle_us > 0 && c < n_chunks - 1) {
            cudaStreamSynchronize(stream);
            std::this_thread::sleep_for(std::chrono::microseconds(chunk_idle_us));
        }
    }
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
          "BF16 Stream-K GEMM (128x128x32, 4-stage) + Method A v7/v8",
          py::arg("A"), py::arg("B"),
          py::arg("split_k_factor") = 1,
          py::arg("avail_sms")      = -1,
          py::arg("sleep_ns")       = 0u,
          py::arg("sleep_freq")     = 1u,
          py::arg("stagger_ns")     = 0u,
          py::arg("stagger_mod")    = 1u,
          py::arg("ramp_start_pct")    = 100u,
          py::arg("ramp_step_pct")     = 100u,
          py::arg("ramp_iter_time_ns") = 0u,
          py::arg("v9_smid_threshold") = 0u,
          py::arg("v9_step_ns")        = 0u);
    m.def("gemm_streamk_chunked", &gemm_streamk_chunked,
          "Stream-K GEMM with row-wise (M) chunking. Each chunk of `chunk_m` "
          "rows is dispatched as a separate streamk launch on the same CUDA "
          "stream. chunk_idle_us > 0 inserts cudaStreamSynchronize+usleep "
          "between chunks so the GPU power rail can recover.",
          py::arg("A"), py::arg("B"),
          py::arg("chunk_m"),
          py::arg("split_k_factor") = 1,
          py::arg("avail_sms")      = -1,
          py::arg("chunk_idle_us")  = 0);
    m.def("gemm_basicdp", &gemm_basicdp,
          "BF16 classic data-parallel GEMM (비교용)",
          py::arg("A"), py::arg("B"),
          py::arg("sleep_ns")       = 0u,
          py::arg("sleep_freq")     = 1u,
          py::arg("stagger_ns")     = 0u,
          py::arg("stagger_mod")    = 1u);
    // v8 (A): one-shot ramp priming. 다음 gemm_streamk 1회 호출에만 적용.
    //   prime_ramp(start_pct, step_pct, iter_time_ns)
    //     start_pct=70, step_pct=10 → activity = 70,80,90,100 (3 iters)
    //     start_pct=70, step_pct=5  → activity = 70,75,80,85,90,95,100 (6 iters)
    //     start_pct>=100 → no-op (ramp disabled for next call)
    m.def("prime_ramp", &prime_ramp,
          "Prime a one-shot linear-activity ramp (v8) for the NEXT gemm_streamk call.",
          py::arg("start_pct"), py::arg("step_pct"),
          py::arg("iter_time_ns"));
    // v9 spatial SM ramp:
    //   prime_ramp_v9(smid_threshold, step_ns) for the NEXT call.
    //   smid < threshold     → no delay
    //   smid >= threshold    → __nanosleep((smid - threshold) * step_ns) at op() entry
    m.def("prime_ramp_v9", &prime_ramp_v9,
          "Prime a one-shot SPATIAL SM ramp (v9) for the NEXT gemm_streamk call. "
          "SMs with smid < threshold start immediately; others get graduated "
          "(smid - threshold) * step_ns nanosleep at kernel entry.",
          py::arg("smid_threshold"), py::arg("step_ns"));
#ifdef CUTLASS_WAVE_SLEEP_ENABLED
    m.def("set_mem_pressure_buffer", &set_mem_pressure_buffer,
          "Install pointer + size of dummy buffer used by mode 9.",
          py::arg("buf"));
    m.def("prime_wave_sleep", &prime_wave_sleep,
          "Prime one-shot WAVE-AWARE sleep. Device-side gate: num_waves>=3.\n"
          "  mode=0 : wave-0-only staircase (post-prologue) + optional mid bubble.\n"
          "  mode=1 : ALL-wave staircase (applied at every mac_loop_iter start).\n"
          "  shape=0: linear  delay = slot * step_ns.\n"
          "  shape=1: quartile staircase (4 levels).",
          py::arg("num_waves"), py::arg("n_sm"),
          py::arg("first_smid_thr"), py::arg("first_step_ns"),
          py::arg("mid_pct"), py::arg("mid_ns"),
          py::arg("mode")  = 0,
          py::arg("shape") = 0,
          py::arg("seed")  = 0xC0FFEE11u);
#endif
}
