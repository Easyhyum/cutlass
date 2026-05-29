# test_torch_chunked — Python-level M-chunking sweep

**Goal**: find the chunk_m that maximizes TFLOPS while staying under 600 W.

**Approach**: no kernel modification at all.  Python loop slices A by M-rows
into `chunk_m`-row chunks, calls the existing baseline GEMM for each chunk,
writes back to C.  Natural Python-loop overhead between launches is the only
"gap" mechanism (`MM_CHUNK_GAP_US` can add an explicit `time.sleep` if you want
to widen it).

## Layout

```
test_torch_chunked/
├── README.md
├── eval_torch_chunked.py   # sweep (kernel × chunk_m), 50 bursts × full-M chunked GEMM
├── plot_chunked.py         # TF vs chunk_m + power vs chunk_m + Pareto scatter
├── plot_timeline_v9.py     # wall-time 3-axis (Power + SM + TF) timeline
├── run.sh                  # GPU 0 forced, nvidia-smi 50ms sampler, auto-plots
└── logs/<TAG>/
    ├── segments.csv
    ├── gpu0_power.csv
    ├── chunk_tflops_vs_cm_<kernel>.png
    ├── chunk_power_vs_cm_<kernel>.png
    ├── chunk_pareto_<kernel>.png
    └── ws10_2d_timeline_v9_<kernel>.png
```

## Imported kernels (production baselines, no rebuild needed)

| kernel | module                                | source .so                                                |
|---     |---                                    |---                                                        |
| streamk | `bf16_gemm_sm80_streamk_baseline`     | `/workspace/custum/bf16_gemm_sm80_streamk_baseline.so`   |
| sm80_v3 | `cutlass_sm80_v3`                     | `/workspace/custum/cutlass_sm80_v3.so`                   |

Both are already compiled in production.  If you want self-contained build
artifacts under this folder, re-run the matching setup scripts in
`build_streamk/` or `build_sm80_v3/` from the project root.

## Run

```bash
cd /workspace/custum/mm_test/test_torch_chunked

# Default sweep — 8 chunk values × 2 kernels × 50 bursts ≈ 15 min
./run.sh

# Add an explicit Python sleep between chunks within one chunked-GEMM
MM_CHUNK_GAP_US=50 ./run.sh

# Smaller chunk list
MM_CHUNK_LIST=1024,2048,4096,8192 ./run.sh
```

## Env vars

| var               | default                                  |
|---                |---                                       |
| `MM_KERNELS`      | streamk,sm80_v3                          |
| `MM_CHUNK_LIST`   | 512,1024,1536,2048,3072,4096,6144,8192   |
| `MM_CHUNK_GAP_US` | 0  (Python `time.sleep` between chunks)  |
| `MM_M`            | 8192                                     |
| `MM_K`, `MM_N`    | 25600, 5120 (qwen3-32b down_proj)        |
| `MM_N_BURSTS`     | 50                                       |
| `MM_BURST_MS`     | 500                                      |
| `MM_BURST_GAP_MS` | 500                                      |
| `MM_PEAK_TFLOPS`  | 400                                      |
| `TAG`             | tchunk_<timestamp>                       |

## Outputs

- `segments.csv` — per-burst (kernel, chunk_m, chunk_gap_us, t_start_ns, t_end_ns, elapsed_ms, tflops_obs)
- `chunk_tflops_vs_cm_<kernel>.png` — median TFLOPS vs chunk_m (log-x)
- `chunk_power_vs_cm_<kernel>.png` — peak / avg W vs chunk_m, with TDP 600W line
- `chunk_pareto_<kernel>.png` — TF vs peak-W scatter for all (chunk_m × gap_us) cfgs
- `ws10_2d_timeline_v9_<kernel>.png` — wall-time 3-axis plot (auto-generated)
