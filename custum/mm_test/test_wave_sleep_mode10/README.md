# test_wave_sleep_mode10 — Mode 10 separate-phase sweep

Wave-sleep **mode 10** evaluation: 3-phase pattern where SMs that sleep ROTATE
within a kernel launch so no single SM accumulates tail latency.

## Mode 10 phases (defined in `mma_multistage.h`)

| Phase | Wave         | Where fires                | Behavior |
|---    |---           |---                         |---|
| 1     | wave 0       | operator() entry (post-prologue) | smid-keyed staircase delay (first-wave burst prevention) |
| 2     | waves 1..N-2 | inside `mac_loop_iter` (per outer K iter) | hash-selected `mid_pct %` of CTAs sleep `mid_ns` ns; **hash mixes `gemm_k_iterations`** so the selected set ROTATES across iters → different SMs bubble at different times |
| 3     | wave N-1     | (skip)                     | no sleep, drain fast |

Two sticky-state fixes relative to the production wave-sleep build:

1. **Sticky `__constant__`** — `prime_wave_sleep(...)` installs the 9 sleep
   params; they stay in effect for ALL subsequent kernel launches until the
   next prime or an explicit `clear_wave_sleep()`.  The original
   "one-shot then zero out" semantic meant only the first kernel of a burst
   actually saw sleep; the rest were baseline.  Sticky fixes that.
2. **`clear_wave_sleep()`** — host setter to explicitly disable without
   re-priming all params.  Used between baseline and primed configs.

## Sweep (this folder)

Two phases run separately so the per-cell power profile is interpretable:

| Phase | What's swept                                  | Other params |
|---    |---                                            |---|
| A — `first` | `first_sleep_pct × first_ns` (9 × 5 = 45)     | mid disabled (mid_pct=0, mid_ns=0) |
| B — `mid`   | `mid_sleep_pct  × mid_ns`   (9 × 5 = 45)      | first disabled (first_smid_thr=n_sm → no SM sleeps) |

Plus one BASELINE burst (no sleep) per (kernel, phase) for comparison.

Semantics of `sleep_pct` (per user spec):
> *pct = % of SMs/CTAs that SLEEP in the affected wave-group*

| pct | first wave (mode 10 phase A) | mid wave (mode 10 phase B) |
|---  |---                           |---                         |
| 60  | first_smid_thr = n_sm × 40 / 100 → top 60% of SMs run the staircase | mid_pct (kernel) = 60 → 60% of mid-wave CTAs sleep |
| 100 | first_smid_thr = 0 → ALL SMs in wave-0 run the staircase | mid_pct (kernel) = 100 → all mid-wave CTAs sleep |

`first_ns` controls the staircase **step**; `mid_ns` is the constant nanosleep for the gated mid-wave CTAs.

## Layout

```
test_wave_sleep_mode10/
├── README.md
├── build_streamk_ws/
│   ├── bf16_gemm_sm80_streamk.cu          ← sticky + clear_wave_sleep
│   └── setup_bf16_sm80_streamk_ws.py      -DCUTLASS_SLEEP_ENABLED -DCUTLASS_WAVE_SLEEP_ENABLED
├── build_sm80_v3_ws/
│   ├── bf16_gemm_sm80_v3_ws.cu            ← sticky + clear_wave_sleep
│   └── setup_bf16_sm80_v3_ws.py
├── eval_wave_sleep_mode10.py              per-phase eval, sticky sleep across all 50 bursts of a cfg
├── plot_wave_sleep_mode10.py              heatmap + power-timeline grid per (kernel, phase)
├── run.sh                                 GPU-0-only orchestrator
└── logs/<TAG>/                            per-run outputs
    ├── run.log
    ├── segments.csv                       per-burst (kernel, phase, sleep_pct, sleep_ns, tf_obs, …)
    ├── gpu0_power.csv                     nvidia-smi 50ms instant power
    ├── segments_with_power.csv            analyze_power output (if available)
    ├── ws10_tflops_heatmap_<k>_<phase>.png       9 × 5 heatmap
    ├── ws10_power_heatmap_<k>_<phase>.png        peak/avg W heatmap (if enriched)
    └── ws10_power_timeline_<k>_<phase>.png       per-(pct,ns) cell, 50-burst power waveform
```

## Build (one time)

```bash
cd /workspace/custum/mm_test/test_wave_sleep_mode10/build_streamk_ws
python setup_bf16_sm80_streamk_ws.py build_ext --inplace

cd /workspace/custum/mm_test/test_wave_sleep_mode10/build_sm80_v3_ws
python setup_bf16_sm80_v3_ws.py build_ext --inplace
```

Both produce `.so` files local to their folders; production `/workspace/custum`
binaries are not touched.

## Run (GPU 0 only)

```bash
cd /workspace/custum/mm_test/test_wave_sleep_mode10

# Full sweep (~3 hours, see env table below)
./run.sh

# Smaller sweep to scope it
MM_PCT_LIST=60,80,100 MM_NS_LIST=500,1000,5000 ./run.sh   # 3×3×2×2=36 cfg ≈ 35 min
```

## Env vars

| var               | default                  | meaning |
|---                |---                       |---|
| `MM_KERNELS`      | streamk_ws,sm80_v3_ws    | comma list (2) |
| `MM_PHASES`       | first,mid                | which phase(s) to run (default: both) |
| `MM_PCT_LIST`     | 60,65,70,75,80,85,90,95,100 | sleep percentile sweep |
| `MM_NS_LIST`      | 250,500,750,1000,5000    | sleep ns sweep |
| `MM_M`            | 8192                     | M |
| `MM_K`, `MM_N`    | 25600, 5120              | K, N (qwen3-32b down_proj) |
| `MM_N_BURSTS`     | 50                       | bursts per cfg |
| `MM_BURST_MS`     | 500                      | per-burst target ms |
| `MM_BURST_GAP_MS` | 500                      | idle gap between bursts |
| `MM_PEAK_TFLOPS`  | 400                      | used to size kernels/burst |
| `TAG`             | ws10_<timestamp>         | sub-folder under `logs/` |

## Expected runtime

For the default sweep:
- Each cfg = 50 bursts × ~1 s (500 ms burst + 500 ms gap) + ~5 s cfg-gap ≈ **55 s**
- Per (kernel × phase) = 45 cfgs + 1 baseline = 46 cfgs × 55 s ≈ **42 min**
- 2 kernels × 2 phases = 4 (kernel × phase) tuples × 42 min ≈ **~2 h 50 min**

Plus build time (~2-3 min per kernel), nvidia-smi overhead, plot generation.
**Round to ~3 hours.**

To shrink:
- 5 pct × 3 ns = 15 cfgs / phase → ~50 min total
- 3 pct × 2 ns =  6 cfgs / phase → ~22 min total

## Output interpretation

- `ws10_tflops_heatmap_*.png` — median observed TFLOPS per (sleep_pct × sleep_ns).
  Bigger ns or higher pct → more idle → lower TFLOPS.  Compare streamk vs
  sm80_v3 to see whether streamk pays MORE per gated SM (multi-segment penalty).
- `ws10_power_heatmap_*.png` — median peak/avg W per cfg.  Tracks the trade-off
  with TFLOPS: target the 600 W TDP point.
- `ws10_power_timeline_*.png` — 9 × 5 grid of per-burst power waveforms.
  Shows the per-cell instantaneous power across the 50 bursts; lets you see
  whether power is stable, drifts, or oscillates.

## Notes

- This setup requires the `mma_multistage.h` updated with the mode 10 hook
  (this repo's version).  Mode 10 must be EXPLICITLY passed as `mode=10` to
  `prime_wave_sleep(...)` — otherwise the default `mode=0` (existing behavior)
  applies.
- `prime_wave_sleep` here is **sticky**: after one call, every subsequent
  kernel launch sees the same sleep params until the next prime or until
  `clear_wave_sleep()` is invoked.  This fixes a long-standing bug where only
  the first kernel of a burst actually saw sleep.
- The `_ws` extensions co-exist with the production builds (different module
  names), so the running M-sweep on GPU 3 is unaffected.
