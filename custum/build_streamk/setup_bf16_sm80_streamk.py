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
    # v8 soft-launch ramp at outer mainloop entry.
    nvcc_args.append("-DCUTLASS_METHOD_A_RAMP_ENABLED")

if os.environ.get("METHOD_A_RAMP_V9", "0") == "1":
    # v9 spatial SM ramp at operator() entry (one nanosleep per kernel launch).
    # Code is at operator() scope (outside mainloop) → near-zero baseline cost.
    nvcc_args.append("-DCUTLASS_METHOD_A_RAMP_V9_ENABLED")

# Wave-aware per-CTA sleep (wave 0 staircase + mid-wave random + last wave none).
# Compiled-in by default; runtime cost is 0 unless host calls prime_wave_sleep().
if os.environ.get("WAVE_SLEEP", "1") == "1":
    nvcc_args.append("-DCUTLASS_WAVE_SLEEP_ENABLED")

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
