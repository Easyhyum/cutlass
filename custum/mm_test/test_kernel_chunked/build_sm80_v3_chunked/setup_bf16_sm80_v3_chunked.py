"""
Build the kernel-chunked sm80_v3 extension.

Pure CUTLASS device::Gemm (128×128×64, 3-stage, GemmIdentityThreadblockSwizzle<8>)
exposed via:
  gemm_sm80_v3(A, B)
  gemm_sm80_v3_chunked(A, B, chunk_m, chunk_idle_us=0)

No sleep / probe / ramp macros — pure baseline SASS for the GEMM mainloop.
Chunking is host-side scheduling only.

Extension name `bf16_gemm_sm80_v3_chunked` to coexist with the pristine
`cutlass_sm80_v3` .so at /workspace/custum.

Build:
  python setup_bf16_sm80_v3_chunked.py build_ext --inplace
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
    # Intentionally NO sleep / probe / wave-sleep / ramp macros.
]

setup(
    name="bf16_gemm_sm80_v3_chunked",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80_v3_chunked",
            sources=["bf16_gemm_sm80_v3_chunked.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
