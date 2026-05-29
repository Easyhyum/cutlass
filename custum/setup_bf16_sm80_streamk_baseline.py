"""
Pristine baseline build of bf16_gemm_sm80_streamk.

Same .cu source as the production extension, but **CUTLASS_WAVE_SLEEP_ENABLED
is NOT defined**, so the wave-aware sleep block inside MmaMultistage::operator()
is completely #if-out — the compiled SASS has no trace of the wave-sleep
logic. This is the apples-to-apples baseline to compare against the
wave-sleep build (which adds an unconditional `if (kWaveSleepNumWaves >= 3)`
runtime check at operator() entry).

Build:
  python setup_bf16_sm80_streamk_baseline.py build_ext --inplace
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
    # NOTE: NO -DCUTLASS_WAVE_SLEEP_ENABLED — wave-sleep block compiled out.
    # NOTE: NO -DCUTLASS_CTA_PROBE_ENABLED       either.
]

setup(
    name="bf16_gemm_sm80_streamk_baseline",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80_streamk_baseline",
            sources=["bf16_gemm_sm80_streamk.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
