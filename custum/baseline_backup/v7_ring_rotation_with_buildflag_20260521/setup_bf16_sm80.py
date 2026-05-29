"""
Build:
  python setup_bf16_sm80.py build_ext --inplace
Env var:
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
    "-DCUTLASS_SLEEP_ENABLED",
]
if os.environ.get("METHOD_A", "0") == "1":
    nvcc_args.append("-DCUTLASS_METHOD_A_BM_ENABLED")

setup(
    name="bf16_gemm_sm80",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80",
            sources=["bf16_gemm_sm80.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
