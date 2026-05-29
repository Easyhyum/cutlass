# test_kernel_chunked — kernel-level M-chunking sweep

**Goal**: same as torch-chunked — find the chunk_m that maximizes TFLOPS while
staying under 600 W — but the chunking happens **inside the C++ host
function**, not in Python.  One `gemm_*_chunked(A, B, chunk_m, idle_us)` call
issues N sequential GEMM launches on the same CUDA stream.  Python sees ONE
function call.

**Why**: torch-level loops add ~10-30 µs of Python overhead per chunk, which
naturally widens the inter-chunk power dip.  Kernel-level chunking removes
that overhead — chunks queue back-to-back on the stream.  Comparing the two
isolates "natural Python overhead" vs "explicit idle gap" as power-modulation
knobs.  `chunk_idle_us > 0` adds `cudaStreamSynchronize + usleep` between
chunks to recreate the torch-level behavior.

## Layout

```
test_kernel_chunked/
├── README.md
├── build_streamk_chunked/
│   ├── bf16_gemm_sm80_streamk.cu          (cp from production streamk source)
│   └── setup_bf16_sm80_streamk_chunked.py (pristine baseline, no sleep / probe)
├── build_sm80_v3_chunked/
│   ├── bf16_gemm_sm80_v3_chunked.cu       NEW — device::Gemm + gemm_sm80_v3_chunked
│   └── setup_bf16_sm80_v3_chunked.py
├── eval_kernel_chunked.py
├── plot_chunked.py
├── plot_timeline_v9.py
├── run.sh
└── logs/<TAG>/
    ├── segments.csv
    ├── gpu0_power.csv
    ├── chunk_tflops_vs_cm_<kernel>.png
    ├── chunk_power_vs_cm_<kernel>.png
    ├── chunk_pareto_<kernel>.png
    └── ws10_2d_timeline_v9_<kernel>.png
```

## Builds (one time)

```bash
cd /workspace/custum/mm_test/test_kernel_chunked/build_streamk_chunked
python setup_bf16_sm80_streamk_chunked.py build_ext --inplace

cd /workspace/custum/mm_test/test_kernel_chunked/build_sm80_v3_chunked
python setup_bf16_sm80_v3_chunked.py build_ext --inplace
```

Module names: `bf16_gemm_sm80_streamk_chunked`, `bf16_gemm_sm80_v3_chunked` —
both coexist with production binaries.

## Exposed functions

| binary | function | notes |
|---|---|---|
| `bf16_gemm_sm80_streamk_chunked` | `gemm_streamk(A, B, split_k_factor=1, avail_sms=-1)` | single-launch baseline |
| | `gemm_streamk_chunked(A, B, chunk_m, split_k_factor=1, avail_sms=-1, chunk_idle_us=0)` | kernel-side row-chunked |
| `bf16_gemm_sm80_v3_chunked`      | `gemm_sm80_v3(A, B)`                                                                  | single-launch baseline |
| | `gemm_sm80_v3_chunked(A, B, chunk_m, chunk_idle_us=0)`                                | kernel-side row-chunked |

## Run

```bash
cd /workspace/custum/mm_test/test_kernel_chunked

# Default sweep — 8 chunk × 2 kernels × 1 idle × 50 bursts ≈ 15 min
./run.sh

# Sweep idle as well
MM_IDLE_US_LIST=0,50,100,200 ./run.sh

# Smaller chunk list
MM_CHUNK_LIST=1024,2048,4096,8192 ./run.sh
```

## Env vars

| var               | default                                  |
|---                |---                                       |
| `MM_KERNELS`      | streamk,sm80_v3                          |
| `MM_CHUNK_LIST`   | 512,1024,1536,2048,3072,4096,6144,8192   |
| `MM_IDLE_US_LIST` | 0                                        |
| `MM_M`            | 8192                                     |
| `MM_K`, `MM_N`    | 25600, 5120                              |
| `MM_N_BURSTS`     | 50                                       |
| `MM_BURST_MS`     | 500                                      |
| `MM_BURST_GAP_MS` | 500                                      |
| `MM_PEAK_TFLOPS`  | 400                                      |
| `TAG`             | kchunk_<timestamp>                       |

## Comparison vs torch-chunked

Run the two sweeps with the **same chunk_list** to compare:
- torch-chunked: chunk transitions go through Python — natural ~10-30 µs gap
- kernel-chunked + idle_us=0 : chunks queue back-to-back on the stream (no host gap)
- kernel-chunked + idle_us=50/100 : explicit host gap, mimics torch-chunked overhead

Pareto scatter (`chunk_pareto_*.png`) lets you read off TF / peak-W
points side-by-side.
