"""Build script for sleep_wmma CUDA extension."""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="sleep_wmma",
    ext_modules=[
        CUDAExtension(
            name="sleep_wmma",
            sources=["sleep_wmma.cu"],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_120",
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
