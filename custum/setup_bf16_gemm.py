"""Build script for bf16_wmma_sleep CUDA extension (Blackwell SM120)."""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="bf16_wmma_sleep",
    ext_modules=[
        CUDAExtension(
            name="bf16_wmma_sleep",
            sources=["bf16_wmma_sleep.cu"],
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
