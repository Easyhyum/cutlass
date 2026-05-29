# test_cta_probe — CTA dispatch / wave-monotonicity probe (streamk vs sm80_v3)

Per-CTA `(smid, globaltimer_start, globaltimer_end, blockIdx)` recording for
two CUTLASS kernels — **Stream-K** (`GemmUniversal + ThreadblockSwizzleStreamK`)
and **sm80_v3** (`device::Gemm + GemmIdentityThreadblockSwizzle<8>`) — to
verify CTA index → wave monotonicity, look at CTA→SM dispatch, and compare
per-CTA durations across the two kernels.

Replaces the legacy `basicdp` baseline in `/workspace/custum/cta_probe/` with
**sm80_v3** as the clean DP-side reference (no `GemmUniversal` overhead, no
large-M performance cliff, same tile/stages as streamk).

## Layout

```
test_cta_probe/
├── README.md                              (this file)
├── eval_cta_probe.py                      per-kernel probe runner (one process per kernel)
├── analyze_monotonicity.py                ρ(L, wave), monotone_frac across kernels
├── plot_cta_probe.py                      CTA→SM, dur-vs-idx, side-by-side plots
├── run.sh                                 orchestrator
├── build_streamk_probe/
│   ├── probe_streamk.cu                   slim streamk + probe (no basicdp half)
│   ├── setup_probe_streamk.py             -DCUTLASS_CTA_PROBE_ENABLED
│   └── README.md
├── build_sm80_v3_probe/
│   ├── probe_sm80_v3.cu                   device::Gemm 128×128×64 + probe
│   ├── setup_probe_sm80_v3.py             -DCUTLASS_CTA_PROBE_ENABLED
│   └── README.md
└── logs/
    └── <TAG>/                             (created by run.sh)
        ├── run.log
        ├── cta_probe_per_cta_<model>_<op>_streamk.csv
        ├── cta_probe_per_cta_<model>_<op>_sm80_v3.csv
        ├── cta_probe_summary_<model>_<op>_<kernel>.csv
        ├── cta_probe_monotonicity_<model>_<op>.csv
        ├── cta_probe_monotonicity_<model>_<op>_<kernel>.png
        ├── cta_probe_sm_<model>_<op>_<kernel>.png
        ├── cta_probe_dur_vs_idx_<model>_<op>_<kernel>.png
        └── cta_probe_streamk_vs_sm80_v3_<model>_<op>.png
```

## Why two separate probe binaries (not one combined .cu)

Each probe build defines its own `__constant__` symbol table from
`cutlass/cta_probe_globals.cuh`.  Loading two probe extensions into the same
Python process tripped CUTLASS's `Error Internal` in earlier experiments
(symbol-name collision in the device link).  **`run.sh` launches one
sub-process per kernel** so each loads only its own binary.  The cost is a
second Python startup; the benefit is a clean isolation that mirrors the
build_streamk / build_sm80_v3 layout.

## Build (one time)

```bash
cd /workspace/custum/mm_test/test_cta_probe/build_streamk_probe
python setup_probe_streamk.py build_ext --inplace

cd /workspace/custum/mm_test/test_cta_probe/build_sm80_v3_probe
python setup_probe_sm80_v3.py build_ext --inplace
```

Each produces a `.so` next to its setup script.  `eval_cta_probe.py` adds
the right `build_*_probe/` to `sys.path` based on `MM_KERNEL`.

## Run

```bash
cd /workspace/custum/mm_test/test_cta_probe

# Runs always pin physical GPU 0. The shell may have CUDA_VISIBLE_DEVICES=0,3
# pre-set; run.sh maps GPU=0 → MM_GPU=<cuda idx of physical 0 in the visible list>.
# Passing GPU=<anything other than 0> is rejected with an error.

# Default: down_proj qwen3-8b, both kernels, M = 32..131072
./run.sh

# Smaller M-set
MM_MS=8192,65536 ./run.sh

# Different op or model
MM_OP=qkv_proj MM_MODEL=qwen3-32b ./run.sh

# Only one kernel
MM_KERNELS=sm80_v3 ./run.sh
```

## Env vars

| var          | default                                  | meaning |
|---           |---                                       |---|
| `GPU`        | 0 (only allowed value)                   | physical GPU id (nvidia-smi indexing). `run.sh` rejects anything other than 0. |
| `MM_MODEL`   | qwen3-8b                                 | qwen3-8b or qwen3-32b |
| `MM_OP`      | down_proj                                | qkv_proj / o_proj / up_proj / down_proj / lm_head |
| `MM_MS`      | 32,256,1024,4096,8192,65536,131072       | M sweep |
| `MM_KERNELS` | streamk,sm80_v3                          | which probe binaries to invoke |
| `TAG`        | ctap_<timestamp>                         | sub-folder under `logs/` |

## What the analysis reports

`analyze_monotonicity.py` writes `cta_probe_monotonicity_<model>_<op>.csv`
with one row per (kernel, M).  Key columns:

| column          | meaning |
|---              |---|
| `rho(L,wave)`   | Spearman ρ between linear-blockIdx `L = (bz·gy + by)·gx + bx` and wave_idx.  Close to 1.0 ⇒ CTA index increases monotonically with wave. |
| `tau(L,wave)`   | Kendall τ (same idea, more robust on ties). |
| `monotone_frac` | fraction of adjacent (L-sorted) CTA pairs whose wave_idx is non-decreasing. 1.0 = perfect monotonicity. |
| `rho(bx,wave)`  | per-axis Spearman: shows which blockIdx axis carries the wave dimension. |
| `n_ctas`, `waves` | grid stats |

For DP-style kernels (sm80_v3) we expect `ρ(by, wave) ≈ 1` (rows scanned in
M-direction → that's the wave axis).  For streamk we expect `ρ(L, wave) ≈ 1`
because the streamk swizzle linearizes the 2D grid.

## What the plots show

- `cta_probe_monotonicity_<model>_<op>_<k>.png` — L vs wave scatter (one row
  per M).  Straight upward staircase = perfect monotonicity.
- `cta_probe_sm_<model>_<op>_<k>.png` — `cta_idx` vs `smid`, colored by
  wave_idx.  Reveals how the hardware dispatcher binds CTAs to SMs across
  waves.
- `cta_probe_dur_vs_idx_<model>_<op>_<k>.png` — per-CTA duration (ns) vs
  `cta_idx`.  Multi-segment CTAs (streamk only) show up as longer CTAs.
- `cta_probe_streamk_vs_sm80_v3_<model>_<op>.png` — side-by-side dispatch
  diagram for the two kernels at each M.

## Build files used by each probe binary

Each setup.py's nvcc invocation pulls in the same headers as the production
builds (`build_streamk/`, `build_sm80_v3/`), with the extra
`-DCUTLASS_CTA_PROBE_ENABLED` flag activating the probe block in
`/workspace/include/cutlass/gemm/threadblock/mma_multistage.h` (lines
996-1024).  See each `build_*_probe/README.md` for the full file list.
