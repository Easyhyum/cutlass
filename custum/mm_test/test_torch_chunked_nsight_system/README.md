# test_torch_chunked_nsight_system — Nsight Systems timeline profiling

Validate the chunked GEMM pipeline design by capturing CUDA + NVTX timeline.
One `nsys` capture per run; multiple (kernel × M × cfg) phases marked with
NVTX ranges so the timeline is easy to read.

## What this verifies

For each (kernel, M, cfg) section in the timeline:

| pattern | what to look for |
|---|---|
| **BASE** | single MMA kernel on default stream, no chunking activity |
| **cm == M, seq** | single MMA kernel on default stream (chunked-path wrapper but only 1 chunk) |
| **cm == M, pipe** | single MMA kernel on **s_mma** stream; one tail copy on **s_copy** stream |
| **cm < M, seq** | N MMAs on default stream, **MMA → copy → MMA → copy → ...** strictly serial |
| **cm < M, pipe** | N MMAs on **s_mma** stream, copies on **s_copy** stream; each copy overlaps with the NEXT MMA on s_mma |

Goal: confirm visually that pipe mode 의 MMAs **never** overlap in time on s_mma
(strictly sequential, single dedicated stream).  Copies overlap with the
immediately-following MMA.

## Layout

```
test_torch_chunked_nsight_system/
├── README.md
├── eval_nsys.py      NVTX-tagged eval — one process per nsys capture
├── run_nsys.sh       nsys profile wrapper (GPU 0 only)
└── logs/<TAG>/
    ├── profile.nsys-rep   nsys raw output — open with nsys-ui
    ├── stats.txt          text summary (kernel/NVTX breakdown)
    └── run.log            stdout
```

## Imported kernels

Reuses production baselines at `/workspace/custum/`:

| kernel | module |
|---|---|
| `cublas` | `torch.matmul` (uses `out=` for direct write) |
| `streamk` | `bf16_gemm_sm80_streamk_baseline` |
| `sm80_v3` | `cutlass_sm80_v3` |

No new C++ build required.

## Run

Default GPU = **3** (because GPU 0 is reserved for the long-running
torch_chunked sweep in this session).  Override with `GPU=<id>` env var.

```bash
cd /workspace/custum/mm_test/test_torch_chunked_nsight_system

# Default GPU 3, all 3 kernels × 9 M × 3 chunk_m (full ~15-25 min)
./run_nsys.sh

# Override GPU (e.g. back to 0 when its sweep is done)
GPU=0 ./run_nsys.sh

# Quick sanity (single kernel, fewer M values)
MM_KERNELS=cublas MM_M_LIST=1024,8192 ./run_nsys.sh

# Just one M to view a small timeline
MM_M_LIST=8192 ./run_nsys.sh

# Open with Nsight Systems UI
nsys-ui logs/<TAG>/profile.nsys-rep
```

## Env vars

| var               | default                                  |
|---                |---                                       |
| `MM_KERNELS`      | cublas,sm80_v3,streamk                   |
| `MM_M_LIST`       | 1024,2048,4096,8192,16384,32768,65536,131072,262144 |
| `MM_CHUNK_LIST`   | 1024,1280,2048                           |
| `MM_K`, `MM_N`    | 25600, 5120 (qwen3-32b down_proj)        |
| `N_WARMUP`        | 30 (per-kernel global warmup iters)      |
| `N_PROFILE`       | 3 (iters per cfg captured in nsys trace) |
| `MM_MEM_BUDGET_GB`| 40 (skip cfg above this)                 |
| `TAG`             | nsys_<timestamp>                         |

## What's in the timeline

### NVTX hierarchy
```
WARMUP_<kernel>                          ← 30 calls, ignored
<kernel>_M<M>                            ← outer band
  ├── <kernel>_M<M>_BASE                 ← 3 single-call BASE
  ├── <kernel>_M<M>_cm<cm>_seq           ← 3 seq calls
  └── <kernel>_M<M>_cm<cm>_pipe          ← 3 pipe calls
```

### Streams to inspect in Nsight Systems UI
- **Default stream**: BASE + all seq mode calls
- **Stream N** (s_mma): pipe mode MMAs — these should be **strictly sequential**
- **Stream N+1** (s_copy): pipe mode copies — only CUTLASS (sm80_v3, streamk).
                          cublas has no copy because `use_out=True` writes
                          directly to C.

## Sanity checks (from the UI)

1. Two MMA kernels on s_mma must **not** overlap in time. If they do →
   stream caching is broken.
2. For non-cublas pipe: each copy on s_copy should overlap with the NEXT
   chunk's MMA on s_mma (the whole point of pipelining).
3. For cublas pipe: s_copy should be unused (no copy step thanks to out=).
4. BASE and seq cm==M should look identical (single MMA on default stream).

## After capture — view text summary

```bash
cat logs/<TAG>/stats.txt
```

Includes per-kernel time, per-NVTX-range total, CUDA API breakdown.

## Notes

- `nsys profile` adds 10-30 % overhead during capture — wall time will be
  inflated, but kernel timings inside are still accurate.
- For per-kernel detail (SM utilization, memory throughput), use Nsight
  Compute (`ncu`) instead — see `mm_test/test_nsight_compute/` for that
  setup.  This folder focuses on timeline + stream / NVTX visualization.
