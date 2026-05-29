# build_sm80_v3 — Pure CUTLASS SM80 BF16 GEMM extension

PyTorch C++ extension wrapping CUTLASS 2.x `cutlass::gemm::device::Gemm` for
bf16 × bf16 → bf16 on SM80 HMMA path (compiled with `-arch=sm_120` for
Blackwell, runs the legacy HMMA tensor-op path).

This is the **"clean" CUTLASS baseline** — no sleep / probe / ramp macros are
defined, so the binary has zero overhead from those code paths in SASS.
Used as the third reference kernel alongside cuBLAS and Stream-K in the
power / Nsight Compute comparisons.

## Files

### In this folder

| file | purpose |
|---|---|
| `bf16_gemm_sm80.cu`           | source — `gemm_sm80_v3(A, B)` PyTorch binding |
| `setup_cutlass_sm80_v3.py`    | build script — **no `-D` macros**, all sleep/probe code in `mma_multistage.h` is `#if`-out |

### Headers pulled in from `/workspace/include/` (CUTLASS_INC)

CUTLASS upstream:

| header | role |
|---|---|
| `cutlass/cutlass.h`                                       | CUTLASS core types / constants |
| `cutlass/gemm/device/gemm.h`                              | `cutlass::gemm::device::Gemm` (device-level wrapper, **non**-Universal) |
| `cutlass/epilogue/thread/linear_combination.h`            | `D = alpha·AB + beta·C` epilogue |
| `cutlass/gemm/threadblock/mma_multistage.h` *(shared)*    | Mainloop — same modified header as the streamk build, but with no sleep/probe `-D` macros set the guarded blocks all compile out → SASS has zero overhead from that code |

The default `cutlass::gemm::device::Gemm` configuration here picks
`GemmIdentityThreadblockSwizzle<8>` as its swizzle (set in
`bf16_gemm_sm80.cu` via the `Gemm<...>` template arguments).

Local custom headers under `cutlass/` (present in the tree and included
transitively by `mma_multistage.h`, but **none of their `__constant__`
symbols are touched** because the sleep/probe macros are not defined):

| header | role for this build |
|---|---|
| `cutlass/cutlass_sleep_globals.cuh`     | included by `bf16_gemm_sm80.cu` for `__constant__` scaffolding, but no setter is exposed and the runtime values stay at 0 → entire block is dead-code-eliminated |
| `cutlass/cta_probe_globals.cuh`         | included transitively via `mma_multistage.h`; all probe code `#if`-out |
| `cutlass/cta_wave_sleep_globals.cuh`    | included transitively via `mma_multistage.h`; all wave-sleep code `#if`-out |

Plus CUTLASS's own transitive headers (iterators, warp-level MMA,
`PredicatedTileAccessIterator`, `GemmIdentityThreadblockSwizzle`, …) pulled
in by the three upstream headers above.

### PyTorch / CUDA system headers

`torch/extension.h`, `c10/cuda/CUDAStream.h`, `ATen/cuda/CUDAContext.h`,
`cuda_runtime.h`, `cuda_bf16.h`, `<cstdint>`.

## CUTLASS template

```cpp
cutlass::gemm::device::Gemm<
    bf16_t, RowMajor,                                 // A
    bf16_t, RowMajor,                                 // B
    bf16_t, RowMajor,                                 // C
    float,                                            // accumulator
    OpClassTensorOp,
    Sm80,
    GemmShape<128, 128, 64>,                          // threadblock tile
    GemmShape< 64,  64, 64>,                          // warp tile
    GemmShape< 16,   8, 16>,                          // mma instruction (HMMA m16n8k16)
    LinearCombination<...>,
    GemmIdentityThreadblockSwizzle<8>,                // ← identity swizzle, group=8 (key)
    /*Stages=*/ 3,
    /*AlignmentA=*/8, /*AlignmentB=*/8
>;
```

Notable differences vs `build_streamk`:

|              | sm80_v3                          | stream_k                       |
|---           |---                               |---                             |
| device class | `cutlass::gemm::device::Gemm`    | `cutlass::gemm::device::GemmUniversal` |
| swizzle      | `GemmIdentityThreadblockSwizzle<8>` | `ThreadblockSwizzleStreamK`  |
| wave-residue | last wave partial (fractional)   | work-stealing → last wave full |
| tile / stages| **same** 128×128×64, 3-stage     | **same** 128×128×64, 3-stage   |

Distinguishing kernel-name fragment in ncu / nvprof:

```
... PredicatedTileAccessIterator ... GemmIdentityThreadblockSwizzle ...
```

## Build

```bash
cd /workspace/custum/build_sm80_v3
python setup_cutlass_sm80_v3.py build_ext --inplace
```
→ produces `cutlass_sm80_v3*.so` in this folder.
Python import: `import cutlass_sm80_v3 as ext_v3; ext_v3.gemm_sm80_v3(A, B)`.

No env vars — this build intentionally has none of:
- `-DCUTLASS_SLEEP_ENABLED`
- `-DCUTLASS_WAVE_SLEEP_ENABLED`
- `-DCUTLASS_CTA_PROBE_ENABLED`
- `-DCUTLASS_METHOD_A_BM_ENABLED`
- `-DCUTLASS_METHOD_A_RAMP_ENABLED`
- `-DCUTLASS_METHOD_A_RAMP_V9_ENABLED`

So the per-CTA / per-wave / per-iter sleep blocks in
`/workspace/include/cutlass/gemm/threadblock/mma_multistage.h` are all
preprocessed out.

## Exported Python function

`gemm_sm80_v3(A: Tensor[bf16, MxK], B: Tensor[bf16, KxN]) -> Tensor[bf16, MxN]`

Single kernel launch.  No `prime_wave_sleep` / `set_avail_sms` / chunked variants
(those live in `build_streamk`).

## How this binary is consumed

- M-sweep:        `mm_test/test_M_kernel_sweep/eval_M_kernel_sweep.py` → `import cutlass_sm80_v3`
- ncu profiling:  `mm_test/test_nsight_compute/eval_ncu_target.py`     → `import cutlass_sm80_v3`

Both scripts do `sys.path.insert(0, '/workspace/custum')` and import by
module name, so the `.so` must be discoverable via that path.

## Note about file duplication

`bf16_gemm_sm80.cu` and `setup_cutlass_sm80_v3.py` also live under
`/workspace/custum/` (the historical build location).  Single-source-of-truth
options:
1. Delete the originals at `/workspace/custum/`, then **add
   `/workspace/custum/build_sm80_v3` to `sys.path`** in test scripts so the
   `.so` built here is the one imported, OR
2. Keep both copies in sync manually.
