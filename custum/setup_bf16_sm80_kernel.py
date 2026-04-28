"""
Build: python setup_bf16_sm80_kernel.py build_ext --inplace

bf16_gemm_sm80_kernel.cu  (v2):
  PTX inline-asm BF16 GEMM
    mma.sync.aligned.m16n8k16  (Tensor Core)
    ldmatrix.sync.aligned       (smem→reg 분배)
    cp.async.cg.shared.global   (double buffering)
  Block 128×128, BK=32, 8 warps (256 th), static smem ~37 KB
  M/N 는 128 배수, K 는 32 배수 권장
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="bf16_gemm_sm80_kernel",
    ext_modules=[
        CUDAExtension(
            name="bf16_gemm_sm80_kernel",
            sources=["bf16_gemm_sm80_kernel.cu"],
            extra_compile_args={
                "nvcc": [
                    "-arch=sm_120",
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                    "-DTORCH_EXTENSION_H",  # PyTorch 바인딩 활성화
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
