"""
Pure CUTLASS sm80_v3 build — bf16_gemm_sm80.cu compiled with ALL sleep / probe
machinery DISABLED.  mma_multistage.h's wave-sleep, CTA-probe, V7/V8/V9 ramps
are all #ifdef-guarded; without those macros the SASS has zero overhead from
that code.

Build:
  python setup_cutlass_sm80_v3.py build_ext --inplace
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

CUTLASS_INC = "/workspace/include"

# NO sleep / probe / ramp macros — pure baseline build.
nvcc_args = [
    "-arch=sm_120",
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "--expt-relaxed-constexpr",
    # Intentionally NOT setting:
    #   -DCUTLASS_SLEEP_ENABLED
    #   -DCUTLASS_WAVE_SLEEP_ENABLED
    #   -DCUTLASS_CTA_PROBE_ENABLED
    #   -DCUTLASS_METHOD_A_BM_ENABLED
    #   -DCUTLASS_METHOD_A_RAMP_ENABLED
    #   -DCUTLASS_METHOD_A_RAMP_V9_ENABLED
]

setup(
    name="cutlass_sm80_v3",
    ext_modules=[
        CUDAExtension(
            name="cutlass_sm80_v3",
            sources=["bf16_gemm_sm80.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
