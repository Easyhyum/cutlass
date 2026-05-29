"""
Probe-only sm80_v3 build for test_cta_probe.

Compiles bf16_gemm_sm80.cu-style CUTLASS device::Gemm (128×128×64, 3-stage,
GemmIdentityThreadblockSwizzle<8>) with
  -DCUTLASS_CTA_PROBE_ENABLED
so MmaMultistage::operator() records (smid, globaltimer_start,
globaltimer_end, blockIdx) into device-side buffers (installed via
set_probe_buffers from Python).

The real GEMM mainloop still runs — the probe is just an extra per-CTA
single write at entry/exit.  C output is correct.

Build:
  python setup_probe_sm80_v3.py build_ext --inplace
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
    # Activates the (smid, globaltimer, blockIdx) recording inside
    # MmaMultistage::operator().
    "-DCUTLASS_CTA_PROBE_ENABLED",
]

setup(
    name="probe_sm80_v3",
    ext_modules=[
        CUDAExtension(
            name="probe_sm80_v3",
            sources=["probe_sm80_v3.cu"],
            include_dirs=[CUTLASS_INC],
            extra_compile_args={"nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
