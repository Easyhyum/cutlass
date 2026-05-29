"""
Build streamk extension for the OVERLAP variant test (test_kernel_chunked_ol).
Same source as the production streamk + a new `gemm_streamk_chunked_ol` host
function that issues chunks on N alternating CUDA streams for tail-overlap.

Pristine baseline — no sleep / probe / wave-sleep / ramp macros.

Build:
  python setup_bf16_sm80_streamk_chunked_ol.py build_ext --inplace
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

CUTLASS_INC = "/workspace/include"
nvcc_args = [
    "-arch=sm_120", "-O3", "--use_fast_math",
    "-lineinfo", "--expt-relaxed-constexpr",
]
setup(
    name="bf16_gemm_sm80_streamk_chunked_ol",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80_streamk_chunked_ol",
            sources=["bf16_gemm_sm80_streamk.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
