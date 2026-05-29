"""
Build:
  python setup_cta_probe.py build_ext --inplace
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

nvcc_args = [
    "-arch=sm_120",
    "-O3",
    "-lineinfo",
    "--expt-relaxed-constexpr",
]

setup(
    name="cta_probe",
    ext_modules=[
        CUDAExtension(
            name="cta_probe",
            sources=["cta_probe.cu"],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
