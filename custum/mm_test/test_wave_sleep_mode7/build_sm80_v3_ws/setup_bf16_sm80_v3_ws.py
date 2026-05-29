"""
Wave-sleep build of CUTLASS sm80_v3 (device::Gemm 128×128×64, 3-stage,
GemmIdentityThreadblockSwizzle<8>).  Same template as the production
cutlass_sm80_v3 build, but with `-DCUTLASS_SLEEP_ENABLED` and
`-DCUTLASS_WAVE_SLEEP_ENABLED` so the wave-sleep block inside
MmaMultistage::operator() is compiled in.

Extension name `bf16_gemm_sm80_v3_ws` so it coexists with the pristine
`cutlass_sm80_v3` .so at /workspace/custum.

Build:
  python setup_bf16_sm80_v3_ws.py build_ext --inplace
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
    "-DCUTLASS_SLEEP_ENABLED",
    "-DCUTLASS_WAVE_SLEEP_ENABLED",
]

setup(
    name="bf16_gemm_sm80_v3_ws",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80_v3_ws",
            sources=["bf16_gemm_sm80_v3_ws.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
