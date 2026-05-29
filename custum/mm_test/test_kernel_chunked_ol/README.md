# test_kernel_chunked_ol — kernel-level M-chunking with tail OVERLAP

Same sweep as `test_kernel_chunked` but the C++ chunked function issues
chunks on **N alternating CUDA streams** so chunk N+1 can start when SMs
free up at the tail of chunk N.

## What changed in C++

New functions in the kernel binaries:

| binary | new function |
|---|---|
| `bf16_gemm_sm80_streamk_chunked_ol` | `gemm_streamk_chunked_ol(A, B, chunk_m, split_k=1, avail_sms=-1, n_streams=2)` |
| `bf16_gemm_sm80_v3_chunked_ol`      | `gemm_sm80_v3_chunked_ol(A, B, chunk_m, n_streams=2)` |

Each function:
1. Creates `n_streams` worker CUDA streams.
2. Issues chunk `c` on `streams[c % n_streams]` with a per-slot `GemmOp` so
   `initialize()` state doesn't race across streams.
3. Final sync: user's current stream `wait_event` on each worker stream's
   final event → returns `C` valid for downstream ops.

Why per-slot ops?  CUTLASS device::Gemm/GemmUniversal's `initialize()`
writes mutable state (workspace pointers etc.).  Sharing one op across
multiple streams would race on those writes.

## Files

```
test_kernel_chunked_ol/
├── README.md
├── build_streamk_chunked_ol/
│   ├── bf16_gemm_sm80_streamk.cu                NEW gemm_streamk_chunked_ol
│   └── setup_bf16_sm80_streamk_chunked_ol.py
├── build_sm80_v3_chunked_ol/
│   ├── bf16_gemm_sm80_v3_chunked.cu             NEW gemm_sm80_v3_chunked_ol
│   └── setup_bf16_sm80_v3_chunked_ol.py
├── eval_kernel_chunked.py                       reuses MM_IDLE_US_LIST slot as n_streams
├── plot_chunked.py
├── plot_timeline_v9.py
├── run.sh                                       tag prefix kchunk_ol_
├── watch_plot.sh
└── verify_chunked.py
```

## eval_kernel_chunked.py mapping

The existing `MM_IDLE_US_LIST` env var slot is **reused as n_streams** in
the `_ol` eval:

| `MM_IDLE_US_LIST` value | meaning |
|---|---|
| `0` (default) | → n_streams=2 (canonical overlap) |
| `1` | n_streams=1 (= sequential, sanity check, equivalent to non-ol) |
| `2,3,4` | sweep over different n_streams |

(The name MM_IDLE_US_LIST kept for backwards compat with the plot
scripts; rename later if confusing.)

## Build

```bash
cd /workspace/custum/mm_test/test_kernel_chunked_ol/build_streamk_chunked_ol
python setup_bf16_sm80_streamk_chunked_ol.py build_ext --inplace

cd /workspace/custum/mm_test/test_kernel_chunked_ol/build_sm80_v3_chunked_ol
python setup_bf16_sm80_v3_chunked_ol.py build_ext --inplace
```

Both `.so` produced in their own folders.

## Run (GPU 0 default)

```bash
cd /workspace/custum/mm_test/test_kernel_chunked_ol
./run.sh

# sweep n_streams
MM_IDLE_US_LIST=1,2,3,4 ./run.sh
```

Output in `logs/kchunk_ol_<TS>/`.

## Expected difference vs non-ol

| | non-ol (sequential, single stream) | _ol (n_streams=2) |
|---|---|---|
| MMA timeline | strictly back-to-back on one stream | chunks alternate streams |
| streamk last-wave | FULL (work-stealing) → no idle tail | overlap negligible |
| sm80_v3 last-wave | partial (DP quantization) → idle tail | **tail-overlap helps** |
| TFLOPS | baseline of kernel-chunked | sm80_v3 should see small TF lift |
| power | baseline | slightly higher (tail SMs busy) |

Streamk's work-stealing means its chunks fill SMs continuously up to the
last cycle, so there's little tail to overlap.  sm80_v3 (DP) has fractional
last wave → tail-overlap should be most visible.

## cublas?

cublas has no C++ chunked binding (it lives in PyTorch), so the eval falls
back to Python-loop chunking exactly as in `test_torch_chunked`.  Included
in the sweep for apples-to-apples comparison but the `_ol` modification
doesn't affect cublas.
