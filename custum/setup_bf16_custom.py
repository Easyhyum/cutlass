"""Build: python setup_bf16_custom.py build_ext --inplace"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="bf16_gemm_custom",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_custom",
            sources=["bf16_gemm_custom.cu"],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_120a",          # SM120 전용 (mma.sync, ldmatrix)
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-lineinfo",
                    "-Xptxas", "-v",           # 레지스터 사용량 출력
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
