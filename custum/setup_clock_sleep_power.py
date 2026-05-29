"""Build script for the clock64/nanosleep power microbenchmark extension.

Usage:
    python setup_clock_sleep_power.py build_ext --inplace

Override the target architecture if needed:
    CUDA_ARCH=sm_80 python setup_clock_sleep_power.py build_ext --inplace
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


CUDA_ARCH = os.environ.get("CUDA_ARCH", "sm_120")


setup(
    name="clock_sleep_power",
    ext_modules=[
        CUDAExtension(
            name="clock_sleep_power",
            sources=["clock_sleep_power.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    f"-arch={CUDA_ARCH}",
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
