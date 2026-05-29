# M-sweep across GEMM kernels

Sweep M from 1024 to 262144 for multiple GEMM backends and record per-burst
power / SM clock / TFLOPS. Output is a **multi-row timeline plot** with one
row per kernel (cublas → first row, stream_k → second row, …).

## Layout

```
test_M_kernel_sweep/
├── README.md                   (this file)
├── eval_M_kernel_sweep.py      sweep runner (cublas + streamk)
├── plot_kernel_timeline.py     multi-row timeline plot (v9-style)
├── run.sh                      orchestrator: nvidia-smi + eval + analyze + plot
└── logs/                       output (one **folder per run**)
    └── <TAG>/                  e.g. mks4_041316/
        ├── segments.csv               per-burst data
        ├── gpu<GPU>_power.csv         nvidia-smi 50 ms log
        ├── segments_with_power.csv    enriched (analyze_power.py output)
        ├── analysis.txt               analyze_power text summary
        ├── run.log                    eval stdout
        ├── timeline.png               single all-in-one timeline (rows = kernels)
        ├── by_op/                     one PNG per (op, model), rows = kernels
        │   ├── down8b_timeline.png
        │   ├── qkv8b_timeline.png
        │   └── ...
        └── by_kernel/                 one PNG per kernel, rows = (op, model)
            ├── cublas_timeline.png
            ├── stream_k_timeline.png
            ├── sm80_v3_timeline.png
            └── ...
```

## Configuration

- **M values**: 1024, 1536, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144
  (override via `MM_M_LIST=1024,2048,8192,...`)
- **Operations** (5 per model × 2 models = 10) — edit `OPS` list in
  `eval_M_kernel_sweep.py` to comment/select. Each op fixes (K, N):

  | model      | op            | K     | N      |
  |---         |---            |---:   |---:    |
  | qwen3-8b   | qkv_proj      | 4096  | 6144   |
  | qwen3-8b   | o_proj        | 4096  | 4096   |
  | qwen3-8b   | gate_up_proj  | 4096  | 24576  |
  | qwen3-8b   | down_proj     | 12288 | 4096   |
  | qwen3-8b   | lm_head       | 4096  | 151936 |
  | qwen3-32b  | qkv_proj      | 5120  | 10240  |
  | qwen3-32b  | o_proj        | 8192  | 5120   |
  | qwen3-32b  | gate_up_proj  | 5120  | 51200  |
  | qwen3-32b  | down_proj     | 25600 | 5120   |
  | qwen3-32b  | lm_head       | 5120  | 151936 |

  - Filter by env: `MM_OPS=down_proj,o_proj` or `MM_MODEL=qwen3-8b`.
- **dtype**: bfloat16
- **M_KERNELS** per cfg: **auto-computed** = `TARGET_BURST_MS / ms_per_kernel`,
  where `ms_per_kernel = 2·M·K·N / PEAK_TFLOPS / 1e12 × 1000`.
  Default target = 500 ms, PEAK_TFLOPS = 400. Override via
  `MM_TARGET_BURST_MS`, `MM_PEAK_TFLOPS`, `MM_M_KERNELS_MIN/MAX`.
- **Memory / OOM handling** — cfg auto-skip on:
  1. **pre-budget**: static `(A+B+C) BF16` estimate > `MM_MEM_BUDGET_GB` (default 40 GB)
  2. **live-free**: estimated × 3 + 0.5 GB > current `cuda.mem_get_info()` free
  3. **alloc OOM**: `torch.empty(...)` of A/B raises (caught, logged, continue)
  4. **runtime OOM**: any GEMM call inside a burst raises (caught, logged, skip rest of bursts for this cfg)

  Override pre-budget: `MM_MEM_BUDGET_GB=60`. lm_head + very large M is the
  typical case skipped — log entries make it explicit which (model, op, M)
  was dropped and why.
- **N_BURSTS**: 50 (env: `MM_N_BURSTS`)
- **burst gap**: 500 ms (env: `MM_BURST_GAP_MS`) — idle-to-burst spike measurement
- **cfg gap**: 1500 ms (env: `MM_CFG_GAP_MS`)
- **Kernels** (in plot row order):
   1. `cublas`   — `torch.matmul(A, B)`
   2. `basicdp`  — `ext.gemm_basicdp(A, B)`   (GemmIdentityThreadblockSwizzle, GemmUniversal)
   3. `stream_k` — `ext.gemm_streamk(A, B)`   (ThreadblockSwizzleStreamK, GemmUniversal)
   4. `sm80_v3`  — `ext_v3.gemm_sm80_v3(A, B)` (cutlass::gemm::device::Gemm, 128×128×64 tile,
       3-stage, GemmIdentityThreadblockSwizzle<8>)
       (binary built with ALL sleep / probe / ramp macros OFF — see
        `setup_cutlass_sm80_v3.py`)

  **Wave-count semantics** (encoded in cfg name as `_w<...>`):
  - `stream_k` : integer = `floor(tiles / n_sm)` (last wave is FULL — residue
                                                 absorbed via K-direction work-stealing).
  - `basicdp`  : fractional = `tiles / n_sm` (last wave is partially filled
                                              when `tiles` isn't a multiple of n_sm).
                e.g. M=8192 → tiles=2048, basicdp waves = 2048/188 = **10.89**.
  - `cublas`   : `NaN` (not introspected).

To add another kernel: edit `KERNELS` list in `eval_M_kernel_sweep.py`.
A new row will appear automatically in the timeline plot.

## Wave count column

cfg name encodes the expected wave count of the kernel for that (M, N):
- `streamk` : `floor(tiles / n_sm)` (integer — last wave is FULL by streamk
  work-stealing). Empirically confirmed (cta_probe) at occupancy=1.
- `basicdp` : `tiles / n_sm` (fractional — last wave partial). Occupancy=1
  assumed; actual SMEM allows up to 3 CTAs/SM but streamk-family swizzles
  observe occupancy=1 in practice.
- `sm80_v3` : `tiles / n_sm` (fractional). **128×128×64 tile → 96 KB SMEM/CTA →
  theoretical occupancy=2.** Reported wave count assumes occupancy=1 so it may
  overestimate by 2× in some cases.
- `cublas`  : `NaN` (we don't introspect cuBLAS internals).

tile count: `tiles = ceil(M/128) × ceil(N/128)` — depends on **N**, so wave
count differs across ops with the same M.

cfg name format:  `<op_short><model_short>_M<M>_w<waves>`
Examples: `down8b_M8192_w10`, `qkv32b_M2048_w0.55`, `lm8b_M1024_wNaN` (cublas).

## GPU selection

`GPU=<physical_id>` env-var picks the GPU. Works in either of:

1. **Shell pre-pinned**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,3   # only GPU 0 and 3 visible to torch
   GPU=3 ./run.sh                    # → uses physical GPU 3 (= cuda:1 within visible)
   GPU=0 ./run.sh                    # → uses physical GPU 0 (= cuda:0 within visible)
   ```
   `run.sh` finds the GPU's index inside `CUDA_VISIBLE_DEVICES` and exports
   `MM_GPU` = that cuda index (which torch then `set_device`'s).
   Errors out if `GPU` is not in the visible list.

2. **Shell unset**:
   ```bash
   GPU=3 ./run.sh                    # pins CUDA_VISIBLE_DEVICES=3, MM_GPU=0
   ```

`nvidia-smi -i $GPU` is always the **physical** GPU index, while
`torch.cuda.set_device(MM_GPU)` uses the visible-list index. The two are
disentangled automatically.

## Run

Default: ALL OPS (10) × ALL M (10) × ALL kernels (4) × 50 burst ≈ **7 hours**.
Edit `OPS` / `M_LIST` / `KERNELS_ALL` lists (or use env-vars) to reduce.

```bash
cd /workspace/custum/mm_test/test_M_kernel_sweep

# Full sweep (~7h) — edit OPS in eval_M_kernel_sweep.py first if needed
./run.sh

# qwen3-8b only (~3-4h)
MM_MODEL=qwen3-8b ./run.sh

# Specific ops only
MM_OPS=down_proj,o_proj ./run.sh

# Fewer M values
MM_M_LIST=1024,8192,65536 ./run.sh

# Fewer burst samples
MM_N_BURSTS=50 ./run.sh

# Different kernel subset
MM_KERNELS=cublas,stream_k ./run.sh

# Custom tag
TAG=qwen8b_full_$(date +%H%M%S) ./run.sh
```

Each invocation creates its own `logs/<TAG>/` folder so runs are easy to
browse and copy/zip in isolation.

## Adding more kernels

Edit `eval_M_kernel_sweep.py`:

```python
KERNELS = [
    ('cublas',   lambda A, B: torch.matmul(A, B),       waves_cublas),
    ('stream_k', lambda A, B: ext.gemm_streamk(A, B),   waves_streamk),
    ('my_new',   lambda A, B: my_ext.my_gemm(A, B),     my_waves_fn),
]
```

`plot_kernel_timeline.py` will pick up the new backend and render an extra
row (cublas first, stream_k second, then alphabetical for the rest).

## Run 
  cd /workspace/custum/mm_test/test_M_kernel_sweep && \
  GPU=3 MM_N_BURSTS=50 MM_BURST_GAP_MS=500 MM_CFG_GAP_MS=1000 \
  TAG=mks4_$(date +%H%M%S) ./run.sh