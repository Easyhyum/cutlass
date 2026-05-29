# build_streamk — CUTLASS Stream-K BF16 GEMM extension

PyTorch C++ extension wrapping CUTLASS's Stream-K GEMM for bf16 × bf16 → bf16
on SM80 HMMA path (compiled with `-arch=sm_120` for Blackwell, runs the legacy
HMMA tensor-op path).

## Files

### In this folder

| file | purpose |
|---|---|
| `bf16_gemm_sm80_streamk.cu`              | source — `gemm_streamk` + (optionally) `gemm_streamk_chunked`, `prime_wave_sleep`, etc. |
| `setup_bf16_sm80_streamk_baseline.py`    | **pristine baseline** — wave-sleep / probe / ramp macros all OFF (used by M-sweep and ncu tests as the apples-to-apples streamk binary) |
| `setup_bf16_sm80_streamk.py`             | **wave-sleep build** — `-DCUTLASS_SLEEP_ENABLED` + optional `WAVE_SLEEP=1` / `METHOD_A=1` / `METHOD_A_RAMP*` env vars to enable the per-CTA staircase, mid-wave sync, V7/V8/V9 ramps inside `mma_multistage.h` |

### Headers pulled in from `/workspace/include/` (CUTLASS_INC)

CUTLASS upstream:

| header | role |
|---|---|
| `cutlass/cutlass.h`                                           | CUTLASS core types / constants |
| `cutlass/gemm/device/gemm_universal.h`                        | `cutlass::gemm::device::GemmUniversal` (device-level wrapper) |
| `cutlass/epilogue/thread/linear_combination.h`                | `D = alpha·AB + beta·C` epilogue |
| `cutlass/gemm/threadblock/threadblock_swizzle_streamk.h`      | **`ThreadblockSwizzleStreamK`** — the Stream-K work-partition swizzle |
| `cutlass/gemm/threadblock/mma_multistage.h` *(modified)*      | Mainloop — contains the `#ifdef`-guarded sleep / probe / ramp blocks (all `#if`-out in the baseline build) |

Local custom headers under `cutlass/` (added for our experiments — present in
the source tree, included unconditionally by `mma_multistage.h` and the .cu,
but their effect is `#if`-guarded):

| header | role |
|---|---|
| `cutlass/cutlass_sleep_globals.cuh`     | `__constant__` / host setter scaffolding for nanosleep params (`CUTLASS_SLEEP_ENABLED`) |
| `cutlass/cta_probe_globals.cuh`         | per-CTA timestamp probe buffers (`CUTLASS_CTA_PROBE_ENABLED`) |
| `cutlass/cta_wave_sleep_globals.cuh`    | wave-aware per-CTA sleep params: mode, staircase shape, etc. (`CUTLASS_WAVE_SLEEP_ENABLED`) |

Plus CUTLASS's own transitive headers (iterators, warp-level MMA, predicated
tile iterators, threadblock swizzles, …) pulled in by the four upstream
headers above.

### PyTorch / CUDA system headers

`torch/extension.h`, `c10/cuda/CUDAStream.h`, `ATen/cuda/CUDAContext.h`,
`cuda_runtime.h`, `cuda_bf16.h`, plus `<chrono> <thread> <cstdint>`.

Both setups compile the same `bf16_gemm_sm80_streamk.cu` source — the
difference is which `#ifdef` macros are defined.  The wave-sleep code is
fully `#if`-guarded so the **baseline binary has zero overhead from the
sleep machinery in SASS** (verified by ncu).

## CUTLASS template (what makes this Stream-K)

```cpp
cutlass::gemm::device::GemmUniversal<
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
    ThreadblockSwizzleStreamK,                        // ← Stream-K swizzle (key)
    /*Stages=*/ 3,
    /*AlignmentA=*/8, /*AlignmentB=*/8,
    OpMultiplyAdd
>;
```

Distinguishing kernel-name fragment in ncu / nvprof:

```
... DefaultGemmUniversal< ... ThreadblockSwizzleStreamK ... >
```

## Build

### Baseline (the binary the M-sweep / ncu tests import)
```bash
cd /workspace/custum/build_streamk
python setup_bf16_sm80_streamk_baseline.py build_ext --inplace
```
→ produces `bf16_gemm_sm80_streamk_baseline*.so` in this folder.
Python import: `import bf16_gemm_sm80_streamk_baseline as ext; ext.gemm_streamk(A, B)`.

### Wave-sleep build (research variants)
```bash
cd /workspace/custum/build_streamk
WAVE_SLEEP=1                     python setup_bf16_sm80_streamk.py build_ext --inplace
METHOD_A=1                       python setup_bf16_sm80_streamk.py build_ext --inplace
METHOD_A_RAMP=1                  python setup_bf16_sm80_streamk.py build_ext --inplace
METHOD_A_RAMP_V9=1               python setup_bf16_sm80_streamk.py build_ext --inplace
```
→ produces `bf16_gemm_sm80_streamk*.so`.
Python: `import bf16_gemm_sm80_streamk as ext_ws; ext_ws.prime_wave_sleep(...); ext_ws.gemm_streamk(A, B)`.

Env-var matrix:

| env var               | effect (NVCC macro added) |
|---                    |---|
| `WAVE_SLEEP=1` (default) | `-DCUTLASS_WAVE_SLEEP_ENABLED` — wave-aware per-CTA staircase / mid-wave random sleep inside `MmaMultistage::operator()` |
| `METHOD_A=1`             | `-DCUTLASS_METHOD_A_BM_ENABLED` — V7 ring-rotating warp mis-aligned bubble sync inside `warp_mma_k` loop (baseline drops ~15%) |
| `METHOD_A_RAMP=1`        | `-DCUTLASS_METHOD_A_RAMP_ENABLED` — V8 soft-launch ramp at mainloop entry |
| `METHOD_A_RAMP_V9=1`     | `-DCUTLASS_METHOD_A_RAMP_V9_ENABLED` — V9 spatial SM ramp at `operator()` entry (one nanosleep per kernel launch) |

`-DCUTLASS_SLEEP_ENABLED` is always set in the non-baseline build (provides
host setter / `__constant__` symbol scaffolding).

## Exported Python functions (from `bf16_gemm_sm80_streamk.cu`)

| symbol                          | available in   | notes |
|---                              |---             |---|
| `gemm_streamk(A, B)`            | both           | bf16 mat-mul, returns C = A @ B |
| `gemm_streamk_chunked(A, B, M_chunk, chunk_idle_us)` | both | kernel-level M-chunking (each chunk = separate launch) with optional inter-chunk idle |
| `prime_wave_sleep(mode, ...)`   | wave-sleep only | configures the wave-aware sleep parameters in `__constant__` memory |
| `set_avail_sms(n)`              | both           | passes `--avail_sms n` into Stream-K so it pretends only n SMs exist |

(See the source for full signature lists.)

## How this binary is consumed

- M-sweep:        `mm_test/test_M_kernel_sweep/eval_M_kernel_sweep.py` → `import bf16_gemm_sm80_streamk_baseline`
- ncu profiling:  `mm_test/test_nsight_compute/eval_ncu_target.py`     → `import bf16_gemm_sm80_streamk_baseline`
- wave-sleep research / power experiments: scripts under `mm_test/` that
  `import bf16_gemm_sm80_streamk`.

These scripts do `sys.path.insert(0, '/workspace/custum')` and import the
extension by module name, so the `.so` must be discoverable via that path.

## Note about file duplication

The same `bf16_gemm_sm80_streamk.cu` and setup files also live under
`/workspace/custum/` (the historical build location).  If you want a single
source of truth, delete the originals and **add `/workspace/custum/build_streamk`
to `sys.path` in the test scripts** (so the `.so` built here is the one
imported).  Otherwise keep both in sync manually.
