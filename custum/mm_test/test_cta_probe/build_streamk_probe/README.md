# build_streamk_probe — Stream-K + CTA probe

Slim variant of `build_streamk/` for the CTA dispatch probe.  Same CUTLASS
Stream-K template, but built with `-DCUTLASS_CTA_PROBE_ENABLED` so that
`MmaMultistage::operator()` records per-CTA `(smid, globaltimer_start,
globaltimer_end, blockIdx)` into device buffers installed via
`set_probe_buffers()` from Python.

The real GEMM mainloop runs (output C is correct).  Probe is one per-CTA
write at entry + exit — no atomics, no shared memory.

## Files

### In this folder

| file | purpose |
|---|---|
| `probe_streamk.cu`        | source — exposes `gemm_streamk_probe(A, B, ...)` + `set_probe_buffers` / `clear_probe_buffers` |
| `setup_probe_streamk.py`  | build script — `-DCUTLASS_CTA_PROBE_ENABLED` only |

### Headers pulled in from `/workspace/include/` (CUTLASS_INC)

| header | role |
|---|---|
| `cutlass/cutlass.h`                                       | CUTLASS core |
| `cutlass/gemm/device/gemm_universal.h`                    | `GemmUniversal` device wrapper |
| `cutlass/epilogue/thread/linear_combination.h`            | linear-combination epilogue |
| `cutlass/gemm/threadblock/threadblock_swizzle_streamk.h`  | `ThreadblockSwizzleStreamK` swizzle |
| `cutlass/gemm/threadblock/mma_multistage.h` *(modified)*  | mainloop — contains the probe-write block guarded by `CUTLASS_CTA_PROBE_ENABLED` |
| `cutlass/cta_probe_globals.cuh`                           | `__constant__ g_cta_probe_smid_out` / `_start_out` / `_end_out` / `_bx_out` / `_by_out` / `_bz_out` / `kCtaProbeMaxCtas` |

### PyTorch / CUDA system headers

`torch/extension.h`, `c10/cuda/CUDAStream.h`, `ATen/cuda/CUDAContext.h`,
`cuda_runtime.h`, `cuda_bf16.h`, `<cstdint>`.

## Build

```bash
cd /workspace/custum/mm_test/test_cta_probe/build_streamk_probe
python setup_probe_streamk.py build_ext --inplace
```

→ produces `probe_streamk*.so` in this folder.  Loaded by
`../eval_cta_probe.py` when `MM_KERNEL=streamk`.

## CUTLASS template (matches build_streamk/)

```cpp
cutlass::gemm::device::GemmUniversal<
    bf16_t, RowMajor, bf16_t, RowMajor, bf16_t, RowMajor,
    float,
    OpClassTensorOp, Sm80,
    GemmShape<128, 128, 32>,           // threadblock tile  ← matches production
    GemmShape< 64,  64, 32>,           // warp tile
    GemmShape< 16,   8, 16>,           // mma instruction (HMMA m16n8k16)
    LinearCombination<...>,
    ThreadblockSwizzleStreamK,         // Stream-K swizzle
    /*Stages=*/ 4,
    /*AlignmentA=*/8, /*AlignmentB=*/8
>;
```

Note: this binary uses 4 stages (matches the existing probe build) while
`build_streamk/` production uses 3 stages.  Adjust if you want bit-exact
SASS parity.

## Exported Python symbols

| symbol | notes |
|---|---|
| `gemm_streamk_probe(A, B, split_k_factor=1, avail_sms=-1)` | runs real streamk GEMM with probe writes |
| `set_probe_buffers(smid, start_t, end_t, bx, by, bz, max_ctas)` | install device buffers |
| `clear_probe_buffers()` | detach buffers so subsequent kernels don't clobber |
