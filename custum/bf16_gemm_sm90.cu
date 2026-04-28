/**
 * bf16_gemm_sm90.cu
 *
 * CUTLASS 3.x SM90(Hopper) CollectiveBuilder 기반 BF16 GEMM PyTorch 확장
 * SM120(Blackwell)에서도 동작 (SM90 backward-compatible: wgmma + TMA)
 *
 * SM80 vs SM90 차이:
 *   SM80: cp.async + mma.sync (threadblock-level MMA)
 *   SM90: TMA    + wgmma     (warpgroup-level MMA, 2× IPC 향상)
 *         → cuBLAS가 절반 클럭에서 더 빠른 이유
 *
 * Build:
 *   python setup_bf16_sm90.py build_ext --inplace
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

// ── CuTe / CUTLASS 3.x headers ───────────────────────────────────────────────
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ─────────────────────────────────────────────────────────────────────────────
//  타입 / 레이아웃 정의
//  bench_mark.py 에서 넘기는 A[M,K] B[K,N] 모두 RowMajor
// ─────────────────────────────────────────────────────────────────────────────
using ElementA          = cutlass::bfloat16_t;
using ElementB          = cutlass::bfloat16_t;
using ElementC          = cutlass::bfloat16_t;
using ElementD          = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute    = float;

using LayoutA = cutlass::layout::RowMajor;   // A[M,K] stride (K,1)
using LayoutB = cutlass::layout::RowMajor;   // B[K,N] stride (N,1)
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// 16B 정렬 → TMA 사용 가능 (BF16 = 2B → 8개 = 16B)
static constexpr int AlignmentA = 8;
static constexpr int AlignmentB = 8;
static constexpr int AlignmentC = 8;
static constexpr int AlignmentD = 8;

// ─────────────────────────────────────────────────────────────────────────────
//  Tile 형상: 128×128×64
//  SM90 wgmma 는 warpgroup(4 warps = 128 threads) 단위로 MMA 수행
// ─────────────────────────────────────────────────────────────────────────────
using TileShape   = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;   // 1×1 cluster (단일 SM 기본)

// ─────────────────────────────────────────────────────────────────────────────
//  CollectiveBuilder (CUTLASS 3.x Auto 선택)
//  KernelScheduleAuto → SM90 TmaWarpSpecialized 선택
//  EpilogueScheduleAuto → TMA epilogue
// ─────────────────────────────────────────────────────────────────────────────
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,   // (M, N, K, BatchCount)
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// ─────────────────────────────────────────────────────────────────────────────
//  stride 헬퍼
// ─────────────────────────────────────────────────────────────────────────────
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// ─────────────────────────────────────────────────────────────────────────────
//  PyTorch 확장 함수
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor gemm_sm90_bf16(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kBFloat16, "A must be BF16");
    TORCH_CHECK(B.dtype() == torch::kBFloat16, "B must be BF16");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous (row-major)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "K dimension mismatch");

    const int M = A.size(0), K = A.size(1), N = B.size(1);

    auto D = torch::empty({M, N}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    // ── stride 계산 (packed_stride: RowMajor stride = (N,1) 형식) ──────────
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {K, N, 1});
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    float alpha = 1.0f, beta = 0.0f;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},                               // problem size + batch
        {                                            // mainloop args
            reinterpret_cast<const ElementA*>(A.data_ptr()),
            stride_A,
            reinterpret_cast<const ElementB*>(B.data_ptr()),
            stride_B
        },
        {                                            // epilogue args
            {alpha, beta},
            reinterpret_cast<const ElementC*>(D.data_ptr()),  // C (source, beta=0 → unused)
            stride_C,
            reinterpret_cast<ElementD*>(D.data_ptr()),
            stride_D
        }
    };

    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS SM90 BF16 GEMM: can_implement failed: ",
                cutlassGetStatusString(status));

    // workspace
    size_t workspace_bytes = gemm_op.get_workspace_size(args);
    auto workspace = torch::empty(
        {static_cast<int64_t>(workspace_bytes)},
        torch::TensorOptions().dtype(torch::kByte).device(A.device()));

    status = gemm_op.initialize(args, workspace.data_ptr(), stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS SM90 BF16 GEMM: initialize failed: ",
                cutlassGetStatusString(status));

    status = gemm_op.run(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS SM90 BF16 GEMM: run failed: ",
                cutlassGetStatusString(status));

    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_sm90_bf16", &gemm_sm90_bf16,
          "BF16 GEMM via CUTLASS 3.x SM90 CollectiveBuilder\n"
          "  wgmma (warpgroup MMA) + TMA (Tensor Memory Access)\n"
          "  SM120(Blackwell) 에서도 동작 (SM90 backward-compatible)\n"
          "  Tile 128×128×64, Auto pipeline stages",
          py::arg("A"), py::arg("B"));
}
