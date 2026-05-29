# build_sm80_v3_probe — sm80_v3 + CTA probe

Probe-enabled variant of `build_sm80_v3/`.  Same CUTLASS template
(`cutlass::gemm::device::Gemm` 128×128×64, 3-stage,
`GemmIdentityThreadblockSwizzle<8>`) but built with
`-DCUTLASS_CTA_PROBE_ENABLED` so `MmaMultistage::operator()` records per-CTA
`(smid, globaltimer_start, globaltimer_end, blockIdx)` into device buffers
installed via `set_probe_buffers()` from Python.

This is the **DP-side baseline** for the CTA dispatch comparison vs.
streamk:

- 1 CTA = 1 output tile (DP semantics) → `operator()` called exactly once
  per CTA → probe writes happen exactly once per CTA per launch.
- Same M×N tile shape (128×128) as streamk → identical grid dimensions for
  the same problem size.
- Replaces `basicdp` (which had a `GemmUniversal` overhead cliff at large M).

## Files

### In this folder

| file | purpose |
|---|---|
| `probe_sm80_v3.cu`         | source — exposes `gemm_sm80_v3_probe(A, B)` + `set_probe_buffers` / `clear_probe_buffers` |
| `setup_probe_sm80_v3.py`   | build script — `-DCUTLASS_CTA_PROBE_ENABLED` only |

### Headers pulled in from `/workspace/include/` (CUTLASS_INC)

| header | role |
|---|---|
| `cutlass/cutlass.h`                                       | CUTLASS core |
| `cutlass/gemm/device/gemm.h`                              | `cutlass::gemm::device::Gemm` (non-Universal) |
| `cutlass/epilogue/thread/linear_combination.h`            | linear-combination epilogue |
| `cutlass/gemm/threadblock/mma_multistage.h` *(shared)*    | mainloop — same modified header as streamk probe build, gated by `CUTLASS_CTA_PROBE_ENABLED` |
| `cutlass/cta_probe_globals.cuh`                           | `__constant__ g_cta_probe_*_out` + `kCtaProbeMaxCtas` |

The default `cutlass::gemm::device::Gemm` configuration here selects
`GemmIdentityThreadblockSwizzle<8>` as its swizzle (set explicitly in
`probe_sm80_v3.cu`'s template arguments).

### PyTorch / CUDA system headers

`torch/extension.h`, `c10/cuda/CUDAStream.h`, `ATen/cuda/CUDAContext.h`,
`cuda_runtime.h`, `cuda_bf16.h`, `<cstdint>`.

## Build

```bash
cd /workspace/custum/mm_test/test_cta_probe/build_sm80_v3_probe
python setup_probe_sm80_v3.py build_ext --inplace
```

→ produces `probe_sm80_v3*.so` in this folder.  Loaded by
`../eval_cta_probe.py` when `MM_KERNEL=sm80_v3`.

## CUTLASS template (matches build_sm80_v3/)

```cpp
cutlass::gemm::device::Gemm<
    bf16_t, RowMajor, bf16_t, RowMajor, bf16_t, RowMajor,
    float,
    OpClassTensorOp, Sm80,
    GemmShape<128, 128, 64>,                           // threadblock tile
    GemmShape< 64,  64, 32>,                           // warp tile
    GemmShape< 16,   8, 16>,                           // mma instruction
    LinearCombination<...>,
    GemmIdentityThreadblockSwizzle<8>,                 // identity swizzle, group=8
    /*Stages=*/ 3,
    /*AlignmentA=*/8, /*AlignmentB=*/8,
    /*SplitKSerial=*/false,
    OpMultiplyAdd
>;
```

## Exported Python symbols

| symbol | notes |
|---|---|
| `gemm_sm80_v3_probe(A, B)` | runs real device::Gemm with probe writes |
| `set_probe_buffers(smid, start_t, end_t, bx, by, bz, max_ctas)` | install device buffers |
| `clear_probe_buffers()` | detach buffers |
