"""Build: python setup_bf16_sm80_v3_manual.py build_ext --inplace"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="bf16_gemm_sm80_v3_manual",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80_v3_manual",
            sources=["bf16_gemm_sm80_v3_manual.cu"],
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
