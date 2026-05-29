"""
Build:
  python setup_bf16_sm80_streamk.py build_ext --inplace
Optional env var:
  METHOD_A=1   include Method A v7 inner-loop sleep code
               (baseline regresses; required only for Method A configs)
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

CUTLASS_INC = "/workspace/include"

nvcc_args = [
    "-arch=sm_120",
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    # host setter / __constant__ symbol scaffolding (always on)
    "-DCUTLASS_SLEEP_ENABLED",
]
if os.environ.get("METHOD_A", "0") == "1":
    # v7 ring-rotating warp mis-aligned bubble sync inside warp_mma_k loop.
    # Adds inner-loop conditional → baseline drops ~15%. Only enable when
    # you actually want to test BM-* configs.
    nvcc_args.append("-DCUTLASS_METHOD_A_BM_ENABLED")

if os.environ.get("METHOD_A_RAMP", "0") == "1":
    # v8 soft-launch ramp at outer mainloop entry. outer-loop scope so
    # baseline regression should be minimal (does not perturb warp_mma_k
    # inner unrolled loop). Required only for RAMP-* configs.
    nvcc_args.append("-DCUTLASS_METHOD_A_RAMP_ENABLED")

setup(
    name="bf16_gemm_sm80_streamk",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80_streamk",
            sources=["bf16_gemm_sm80_streamk.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
