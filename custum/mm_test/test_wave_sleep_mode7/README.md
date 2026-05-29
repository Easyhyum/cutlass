# test_wave_sleep_mode7 — SM gating (mode 7) sweep: streamk vs sm80_v3

Compare wave-sleep **mode 7 (SM gating)** between Stream-K and clean
device::Gemm (sm80_v3) at a single (op, M):

| param        | value                                  |
|---           |---                                     |
| model        | qwen3-32b                              |
| op           | down_proj                              |
| M, K, N      | 8192, 25600, 5120                      |
| burst count  | **50 bursts / cfg**                    |
| burst length | **500 ms** kernel calls per burst      |
| burst gap    | **500 ms** idle between bursts         |
| active_pct   | sweep (default 100 / 90 / 80 / 70 / 60 / 50 / 40) |
| kernels      | streamk_ws, sm80_v3_ws                 |

`active_pct=100` is the baseline (no SM gating).  For other values,
mode 7 sets `first_smid_thr = n_sm × active_pct / 100`; SMs with smid ≥ thr
return immediately at operator() entry (MPS-like spatial gating without
nanosleep).

This is the first wave-sleep comparison where **both kernels** see the sleep
mechanism on the same `mma_multistage.h` mainloop, so the difference between
them isolates **streamk's multi-segment penalty** (a CTA crossing 2-3 tile
segments runs the sleep block per segment).

## Layout

```
test_wave_sleep_mode7/
├── README.md
├── build_streamk_ws/                  Stream-K + WAVE_SLEEP=1 (extension name: bf16_gemm_sm80_streamk_ws)
│   ├── bf16_gemm_sm80_streamk.cu      copy of production source
│   └── setup_bf16_sm80_streamk_ws.py  -DCUTLASS_SLEEP_ENABLED -DCUTLASS_WAVE_SLEEP_ENABLED
├── build_sm80_v3_ws/                  device::Gemm 128×128×64 + WAVE_SLEEP   (extension name: bf16_gemm_sm80_v3_ws)
│   ├── bf16_gemm_sm80_v3_ws.cu        NEW — sm80_v3 template + prime_wave_sleep host binding
│   └── setup_bf16_sm80_v3_ws.py       -DCUTLASS_SLEEP_ENABLED -DCUTLASS_WAVE_SLEEP_ENABLED
├── eval_wave_sleep.py                 50-burst eval per (kernel, active_pct), writes segments.csv
├── plot_wave_sleep.py                 TFLOPS-vs-pct, power-vs-pct, per-burst timeline
├── run.sh                             nvidia-smi 50ms sampler + sequential per-kernel eval + analysis + plots
└── logs/<TAG>/                        per-run sub-folder
    ├── run.log
    ├── segments.csv                   per-burst (kernel, active_pct, t_start, t_end, kernels, tf)
    ├── gpu0_power.csv                 nvidia-smi 50ms sample log
    ├── segments_with_power.csv        analyze_power.py output (if available)
    ├── analysis.txt
    ├── ws_mode7_tflops_vs_pct.png     median TFLOPS vs active_pct, both kernels
    ├── ws_mode7_power_vs_pct.png      median peak / avg W vs active_pct (if power enriched)
    ├── ws_mode7_timeline_streamk_ws.png    per-burst TF over time per active_pct
    └── ws_mode7_timeline_sm80_v3_ws.png
```

## Build (one time)

```bash
cd /workspace/custum/mm_test/test_wave_sleep_mode7/build_streamk_ws
python setup_bf16_sm80_streamk_ws.py build_ext --inplace

cd /workspace/custum/mm_test/test_wave_sleep_mode7/build_sm80_v3_ws
python setup_bf16_sm80_v3_ws.py build_ext --inplace
```

Both produce `.so` files in their own folders:
- `build_streamk_ws/bf16_gemm_sm80_streamk_ws.cpython-*.so`
- `build_sm80_v3_ws/bf16_gemm_sm80_v3_ws.cpython-*.so`

Neither overwrites the production `/workspace/custum/` artifacts.

## Run (GPU 0 only)

```bash
cd /workspace/custum/mm_test/test_wave_sleep_mode7

# Default sweep
./run.sh

# Custom active_pct list
MM_ACTIVE_PCTS=100,80,70,60 ./run.sh

# Different op / M (e.g., qwen3-8b down_proj)
MM_K=12288 MM_N=4096 MM_MODEL=qwen3-8b MM_M=8192 ./run.sh
```

The script forces `CUDA_VISIBLE_DEVICES=0` and `MM_GPU=0` regardless of the
caller's environment — isolates this run from anything on GPU 3.

## Env vars

| var                | default                  | meaning |
|---                 |---                       |---|
| `MM_KERNELS`       | streamk_ws,sm80_v3_ws    | comma list |
| `MM_ACTIVE_PCTS`   | 100,90,80,70,60,50,40    | mode-7 active SM percentages |
| `MM_M`             | 8192                     | M |
| `MM_K`, `MM_N`     | 25600, 5120              | K, N (qwen3-32b down_proj) |
| `MM_N_BURSTS`      | 50                       | bursts per (kernel, active_pct) |
| `MM_BURST_MS`      | 500                      | per-burst target ms |
| `MM_BURST_GAP_MS`  | 500                      | idle gap between bursts |
| `MM_PEAK_TFLOPS`   | 400                      | used to size kernels/burst |
| `TAG`              | ws7_<timestamp>          | sub-folder under `logs/` |

## What the plots show

- `ws_mode7_tflops_vs_pct.png` — median observed TFLOPS / burst as a function
  of active SM %. Expect both kernels to drop near-linearly with pct.
  The slope difference reveals whether streamk loses MORE than sm80_v3
  per gated SM (streamk's multi-segment penalty hypothesis).

- `ws_mode7_power_vs_pct.png` — median peak / average W per burst.  Lower
  pct should drive power down toward the 600 W TDP cap and below.

- `ws_mode7_timeline_<kernel>.png` — per-burst TFLOPS over time for each
  active_pct; reveals whether throughput stabilizes across the 50 bursts or
  drifts (e.g., thermal / clock effects).
