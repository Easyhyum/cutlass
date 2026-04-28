"""
Build: python setup_bf16_sm80.py build_ext --inplace
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

CUTLASS_INC = "/workspace/include"

setup(
    name="bf16_gemm_sm80",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80",
            sources=["bf16_gemm_sm80.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_120",
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                    # mma_multistage.h 의 #ifdef CUTLASS_SLEEP_ENABLED 블록 활성화
                    # → __constant__ kCutlassSleepNs / kCutlassSleepFreq 사용 가능
                    "-DCUTLASS_SLEEP_ENABLED",
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
