"""
Build:
  python setup_probe_streamk.py build_ext --inplace
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
    # Enable the CTA-probe hook inside mma_multistage::operator().
    "-DCUTLASS_CTA_PROBE_ENABLED",
    # Enable the wave-aware entry-delay hook inside mma_multistage::operator().
    "-DCUTLASS_WAVE_SLEEP_ENABLED",
]

setup(
    name="probe_streamk",
    ext_modules=[
        CUDAExtension(
            name="probe_streamk",
            sources=["probe_streamk.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
