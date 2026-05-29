"""
Build streamk with `gemm_streamk_chunked` exposed — same .cu as production
streamk, but compiled WITHOUT wave-sleep / probe macros.  Provides a clean
baseline so chunking measurements are not contaminated by sleep-block
overhead.

Build:
  python setup_bf16_sm80_streamk_chunked.py build_ext --inplace
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

CUTLASS_INC = "/workspace/include"

nvcc_args = [
    "-arch=sm_120",
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    # Pristine baseline — no sleep / probe / wave-sleep / ramp.
]

setup(
    name="bf16_gemm_sm80_streamk_chunked",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80_streamk_chunked",
            sources=["bf16_gemm_sm80_streamk.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
