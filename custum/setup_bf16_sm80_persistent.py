"""Build: python setup_bf16_sm80_persistent.py build_ext --inplace"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="bf16_gemm_sm80_persistent",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80_persistent",
            sources=["bf16_gemm_sm80_persistent.cu"],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_120",
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-Xptxas",
                    "-v",
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
