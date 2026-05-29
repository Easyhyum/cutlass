/**
 * bf16_gemm_sm80_v3_chunked.cu  (test_kernel_chunked)
 *
 * Pure CUTLASS device::Gemm (128×128×64 tile, 3-stage,
 * GemmIdentityThreadblockSwizzle<8>) — same template as build_sm80_v3 — but
 * exposes an extra `gemm_sm80_v3_chunked(A, B, chunk_m, chunk_idle_us)`
 * function that splits M into row-wise chunks and issues N sequential GEMM
 * launches on the same CUDA stream.  Optional `chunk_idle_us` inserts a
 * host-side stream-sync + usleep between chunks to let the GPU power rail
 * recover (mimics natural Python-loop overhead).
 *
 * No `__nanosleep` / wave-sleep / probe code is compiled in — pure baseline
 * SASS for the GEMM.  The chunking is purely host-side scheduling.
 *
 * Build:
 *   python setup_bf16_sm80_v3_chunked.py build_ext --inplace
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <chrono>
#include <thread>
#include <cstdint>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

using ElementA      = cutlass::bfloat16_t;
using ElementB      = cutlass::bfloat16_t;
using ElementC      = cutlass::bfloat16_t;
using ElementAccum  = float;
using LayoutA       = cutlass::layout::RowMajor;
using LayoutB       = cutlass::layout::RowMajor;
using LayoutC       = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccum, float>;

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
    3, 8, 8, false,
    cutlass::arch::OpMultiplyAdd>;

static GemmSm80V3 gemm_op;

static void check_cutlass(cutlass::Status s, const char* msg) {
    if (s != cutlass::Status::kSuccess) {
        AT_ERROR(std::string(msg) + ": " + cutlassGetStatusString(s));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  gemm_sm80_v3  — single-launch baseline (no chunking).
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemm_sm80_v3(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A bf16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B bf16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "contig");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    GemmSm80V3::Arguments args(
        {M, N, K},
        {(ElementA*)A.data_ptr(), K},
        {(ElementB*)B.data_ptr(), N},
        {(ElementC*)C.data_ptr(), N},
        {(ElementC*)C.data_ptr(), N},
        {1.0f, 0.0f}
    );
    auto s = gemm_op.initialize(args, nullptr, stream);
    check_cutlass(s, "sm80_v3 initialize");
    s = gemm_op.run(stream);
    check_cutlass(s, "sm80_v3 run");
    return C;
}

// ─────────────────────────────────────────────────────────────────────────────
//  gemm_sm80_v3_chunked — Kernel-level row-wise (M) chunking.
//
//    chunk_m       : rows per chunk.  M is split into ceil(M / chunk_m) chunks
//                    issued sequentially on the same CUDA stream.
//    chunk_idle_us : if > 0, cudaStreamSynchronize + sleep_for(us) between
//                    chunks so the GPU can drop out of the power-saturation
//                    region before the next chunk's MMA wave starts.  0 keeps
//                    chunks back-to-back (still sequential, but no host
//                    intervention → most chunks overlap-pipeline through the
//                    HW launch queue).
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemm_sm80_v3_chunked(torch::Tensor A, torch::Tensor B,
                                   int chunk_m,
                                   int chunk_idle_us = 0) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A bf16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B bf16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols==B.rows");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "contig");
    TORCH_CHECK(chunk_m > 0, "chunk_m > 0");

    int const M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    float alpha = 1.0f, beta = 0.0f;

    int n_chunks = (M + chunk_m - 1) / chunk_m;

    for (int c = 0; c < n_chunks; ++c) {
        int const start = c * chunk_m;
        int const rows  = std::min(chunk_m, M - start);

        ElementA* A_ptr = ((ElementA*)A.data_ptr()) + (int64_t)start * K;
        ElementC* C_ptr = ((ElementC*)C.data_ptr()) + (int64_t)start * N;

        GemmSm80V3::Arguments args(
            {rows, N, K},
            {A_ptr, K},
            {(ElementB*)B.data_ptr(), N},
            {C_ptr, N},
            {C_ptr, N},
            {alpha, beta}
        );

        auto s = gemm_op.initialize(args, nullptr, stream);
        check_cutlass(s, "sm80_v3_chunked initialize");
        s = gemm_op.run(stream);
        check_cutlass(s, "sm80_v3_chunked run");

        if (chunk_idle_us > 0 && c < n_chunks - 1) {
            cudaStreamSynchronize(stream);
            std::this_thread::sleep_for(std::chrono::microseconds(chunk_idle_us));
        }
    }
    return C;
}


// ─────────────────────────────────────────────────────────────────────────────
//  gemm_sm80_v3_chunked_ol — overlap variant
//
//  Same M-chunking as gemm_sm80_v3_chunked, but issues chunks on `n_streams`
//  alternating CUDA streams (default 2).  CUDA scheduler can overlap chunk
//  N+1's startup with chunk N's TAIL (last partial wave's idle SMs) → small
//  controlled tail-overlap recovering tail-latency without going to full
//  SM utilisation everywhere.
//
//  For chunks large enough to fill all SMs in their first wave (typical for
//  chunk_m ≥ 1024 on RTX PRO 6000), the only window where stream B's chunk
//  can run is the tail of stream A's chunk.  For very small chunks, both
//  streams' chunks run concurrently throughout → high power.
//
//  Synchronisation: chunks on the same stream are serialised by CUDA; the
//  caller does NOT need to synchronise between chunks.  We sync ONLY at the
//  end on the user's current stream so the returned C is valid downstream.
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemm_sm80_v3_chunked_ol(torch::Tensor A, torch::Tensor B,
                                      int chunk_m,
                                      int n_streams = 2) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A bf16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B bf16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.cols==B.rows");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "contig");
    TORCH_CHECK(chunk_m > 0, "chunk_m > 0");
    TORCH_CHECK(n_streams >= 1 && n_streams <= 4, "n_streams in [1,4]");

    int const M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    auto user_stream = at::cuda::getCurrentCUDAStream();
    float alpha = 1.0f, beta = 0.0f;

    // Allocate worker streams (high-priority not required; default suffices).
    cudaStream_t streams[4];
    for (int i = 0; i < n_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int const n_chunks = (M + chunk_m - 1) / chunk_m;

    // We can't share a single static GemmOp across streams because
    // initialize() writes per-launch state.  Use a small per-stream pool.
    GemmSm80V3 ops[4];

    for (int c = 0; c < n_chunks; ++c) {
        int const start = c * chunk_m;
        int const rows  = std::min(chunk_m, M - start);
        int const slot  = c % n_streams;

        ElementA* A_ptr = ((ElementA*)A.data_ptr()) + (int64_t)start * K;
        ElementC* C_ptr = ((ElementC*)C.data_ptr()) + (int64_t)start * N;

        GemmSm80V3::Arguments args(
            {rows, N, K},
            {A_ptr, K},
            {(ElementB*)B.data_ptr(), N},
            {C_ptr, N},
            {C_ptr, N},
            {alpha, beta}
        );

        auto s = ops[slot].initialize(args, nullptr, streams[slot]);
        check_cutlass(s, "sm80_v3_chunked_ol initialize");
        s = ops[slot].run(streams[slot]);
        check_cutlass(s, "sm80_v3_chunked_ol run");
    }

    // Make user's current stream wait for ALL worker streams.
    for (int i = 0; i < n_streams; ++i) {
        cudaEvent_t e; cudaEventCreate(&e);
        cudaEventRecord(e, streams[i]);
        cudaStreamWaitEvent(user_stream, e, 0);
        cudaEventDestroy(e);
    }
    for (int i = 0; i < n_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_sm80_v3", &gemm_sm80_v3,
          "CUTLASS device::Gemm 128x128x64 3-stage (single-launch baseline).",
          py::arg("A"), py::arg("B"));
    m.def("gemm_sm80_v3_chunked", &gemm_sm80_v3_chunked,
          "sm80_v3 GEMM with row-wise (M) chunking on the same CUDA stream.\n"
          "chunk_idle_us > 0 inserts cudaStreamSynchronize + usleep between chunks.",
          py::arg("A"), py::arg("B"),
          py::arg("chunk_m"),
          py::arg("chunk_idle_us") = 0);
    m.def("gemm_sm80_v3_chunked_ol", &gemm_sm80_v3_chunked_ol,
          "sm80_v3 row-wise M-chunking on N alternating CUDA streams.\n"
          "Chunks on the same stream are serialized; chunks on different\n"
          "streams may overlap when SMs are free (tail-overlap).",
          py::arg("A"), py::arg("B"),
          py::arg("chunk_m"),
          py::arg("n_streams") = 2);
}
