# Nsight Compute profiling — cublas vs stream_k vs sm80_v3

Capture per-kernel `ncu` profiles for the GEMM kernels running every Qwen3
op (5 ops × 2 models = 10 (K, N) combos) and an M sweep.

**All configs are profiled inside ONE `ncu` invocation** — Python loops over
(kernel × op × M), and the profile region per config is bounded by
`torch.cuda.profiler.start/stop` + NVTX range so the report groups launches
by `<kernel>__<op_short><model_short>__M<M>`.

## Layout

```
test_nsight_compute/
├── README.md             (this file)
├── eval_ncu_target.py    sweep runner — does warmup + per-config profile region
├── run.sh                ncu orchestrator (one invocation, Python sweeps inside)
└── profiles/             output (one folder per run)
    └── <TAG>/
        ├── run.log
        └── sweep.ncu-rep          ← single file with all (kernel × op × M) launches
```

## Run

```bash
cd /workspace/custum/mm_test/test_nsight_compute

# Full matrix — 3 kernels × 10 ops × N M values. Single ncu run, no per-cfg
# startup cost (much faster than spawning ncu per cfg).
./run.sh

# Single op, single kernel, one M
MM_KERNELS=stream_k MM_OPS=down_proj MM_M_LIST=8192 ./run.sh

# qwen3-8b only, three kernels, M = 1024..8192
MM_MODEL=qwen3-8b MM_M_LIST=1024,2048,4096,8192 ./run.sh

# GPU 3 with shell pre-pinned CUDA_VISIBLE_DEVICES=0,3
export CUDA_VISIBLE_DEVICES=0,3
GPU=3 MM_OPS=down_proj MM_M_LIST=8192 ./run.sh

# GPU 0 (same shell pre-pinned 0,3) — uses physical GPU 0 = cuda:0
export CUDA_VISIBLE_DEVICES=0,3
GPU=0 MM_OPS=down_proj MM_M_LIST=8192 ./run.sh

# Custom output folder
TAG=ncu_8b_lm ./run.sh
```

## Env vars

| var                    | default          | meaning |
|---                     |---               |---|
| `GPU`                  | 0                | physical GPU id (nvidia-smi indexing) |
| `MM_KERNELS`           | cublas,stream_k,sm80_v3 | kernels to profile |
| `MM_OPS`               | (all)            | op subset (qkv_proj,o_proj,up_proj,down_proj,lm_head) |
| `MM_MODEL`             | (both)           | qwen3-8b OR qwen3-32b |
| `MM_M_LIST`            | 1024..262144     | M values (comma list) |
| `MM_MEM_BUDGET_GB`     | 40               | skip cfg whose (A+B+C) bf16 estimate exceeds this |
| `NCU_SET`              | detailed         | ncu metric group (full / detailed / basic / roofline / source) |
| `NCU_INITIAL_WARMUP`   | 20               | one-time warmup launches at start (per kernel) |
| `NCU_WARMUP_PER_SHAPE` | 3                | per-shape warmup launches before each profile region |
| `NCU_PROFILE`          | 1                | profile launches per (kernel, op, M) — 1 is enough since `--replay-mode kernel` re-runs as needed |
| `TAG`                  | ncu_<timestamp>  | sub-folder name under `profiles/` |

`NCU_SET=full` collects every metric (very slow). `detailed` is a good
default for compute / memory breakdown. `roofline` focuses on roofline plot.

## How profiling region is controlled

`run.sh` invokes `ncu` with `--profile-from-start no --nvtx`, so:
- ncu boots, attaches, but **does not capture** anything yet.
- `eval_ncu_target.py` does initial warmup + per-shape warmup OUTSIDE the
  profile region (cudaProfilerStop state).
- For each (kernel, op, M), it pushes an NVTX range, calls
  `torch.cuda.profiler.start()`, runs `NCU_PROFILE` launches, syncs, then
  `torch.cuda.profiler.stop()`, pops NVTX.
- Only those launches end up in `sweep.ncu-rep`, each labeled with its
  NVTX context (`<kernel>__<op><model>__M<M>`).

## Inspecting profiles

Open in UI (NVTX-aware view groups launches by range name):
```bash
ncu-ui profiles/<TAG>/sweep.ncu-rep
```

CLI summary (whole sweep → CSV):
```bash
ncu --import profiles/<TAG>/sweep.ncu-rep \
    --print-summary per-gpu --csv > summary.csv
```

Filter by NVTX range when extracting metrics:
```bash
ncu --import profiles/<TAG>/sweep.ncu-rep \
    --nvtx-include "stream_k__down8b__M8192" \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_active,\
launch__waves_per_multiprocessor \
    --csv --page raw
```

## Adding kernels / ops

- New op (K, N): add a `(opname, model, K, N)` tuple to `OPS` in
  `eval_ncu_target.py`, and add `OP_SHORT` / `MODEL_SHORT` entries if needed.
- New kernel: extend the `kernel_fns` switch in `eval_ncu_target.py`
  (import + lambda).

## NVTX tag naming

Inside the profile, each kernel launch is labeled:

`<kernel>__<op_short><model_short>__M<M>`

Examples:
- `cublas__down8b__M8192`
- `stream_k__qkv32b__M2048`
- `sm80_v3__lm8b__M1024`
