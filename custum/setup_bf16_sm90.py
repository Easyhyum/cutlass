"""
Build: python setup_bf16_sm90.py build_ext --inplace

bf16_gemm_sm90.cu:
  CUTLASS 3.x SM90 CollectiveBuilder 기반 BF16 GEMM
    - wgmma (warpgroup-level MMA) : SM80 mma.sync 대비 ~2× IPC
    - TMA (Tensor Memory Access)  : cp.async 대비 효율적인 메모리 로드
    - KernelScheduleAuto → TmaWarpSpecialized 자동 선택
  SM120(Blackwell) 에서도 동작 (SM90 instruction 집합 backward-compatible)
  Tile 128×128×64, Stage 수 자동 최적화
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="bf16_gemm_sm90",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm90",
            sources=["bf16_gemm_sm90.cu"],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_120",         # SM120 하드웨어, SM90 ISA 포함
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                    "-std=c++17",
                    # CUTLASS 3.x SM90 MMA 활성화
                    "-DCUTLASS_ARCH_MMA_SM90_SUPPORTED",
                    "-DCUTLASS_ARCH_MMA_SM90A_SUPPORTED",
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
