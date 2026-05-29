"""
Wave-sleep build of streamk for test_wave_sleep_mode7.

Compiles the same bf16_gemm_sm80_streamk.cu as the production streamk, with
WAVE_SLEEP and SLEEP scaffolding macros on so `prime_wave_sleep(...)` reaches
the device-side `kWaveSleep*` __constant__ symbols.

Extension is named `bf16_gemm_sm80_streamk_ws` (suffix `_ws`) so it can
co-exist with the production `bf16_gemm_sm80_streamk` .so at /workspace/custum.

Build:
  python setup_bf16_sm80_streamk_ws.py build_ext --inplace
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
    # Host-setter scaffolding + wave-aware per-CTA sleep block in
    # mma_multistage.h.
    "-DCUTLASS_SLEEP_ENABLED",
    "-DCUTLASS_WAVE_SLEEP_ENABLED",
]

setup(
    name="bf16_gemm_sm80_streamk_ws",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80_streamk_ws",
            sources=["bf16_gemm_sm80_streamk.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
