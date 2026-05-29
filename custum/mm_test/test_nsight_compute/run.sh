#!/usr/bin/env bash
# Run `ncu` (Nsight Compute) over {cublas, stream_k, sm80_v3} × Qwen3-{8B,32B}
# ops × M sweep, *all inside ONE ncu invocation*.
#
# eval_ncu_target.py does the sweep internally; the profile region per launch
# is bounded by cudaProfilerStart/Stop (torch.cuda.profiler.start/stop) and
# labeled with NVTX ranges so the report groups by (kernel, op, M).
#
# Profile is written to profiles/<TAG>/sweep.ncu-rep (single file).
#
# Env vars (with defaults):
#   GPU                physical GPU id (nvidia-smi indexing).  default 0
#   NCU_SET            ncu metric set: full / detailed / basic / roofline / source.  default 'detailed'
#   NCU_INITIAL_WARMUP one-time warmup launches at start.  default 20
#   NCU_WARMUP_PER_SHAPE  per-shape warmup before each profile.  default 3
#   NCU_PROFILE        profile launches per (kernel, op, M).  default 1
#   TAG                output sub-folder name.  default ncu_<timestamp>
#   MM_KERNELS         comma list to limit kernels  (default: all 3)
#   MM_OPS             comma list to limit ops
#   MM_MODEL           single model filter (qwen3-8b | qwen3-32b)
#   MM_M_LIST          comma list of M values
#   MM_MEM_BUDGET_GB   skip cfgs whose (A+B+C) bf16 estimate exceeds this. default 40
#
# Usage:
#   ./run.sh                                     # full matrix in one ncu run
#   GPU=3 MM_OPS=down_proj ./run.sh
#   MM_KERNELS=stream_k MM_M_LIST=2048,8192 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-ncu_${TS}}"
NCU_SET="${NCU_SET:-detailed}"

OUT_DIR="profiles/$TAG"
mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/run.log"
OUT_FILE="$OUT_DIR/sweep.ncu-rep"

# ── GPU mapping (same logic as test_M_kernel_sweep) ─────────────────────────
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # `|| true` so a no-match in grep doesn't trip `set -e`/`pipefail` —
    # we want the explicit error message below to fire instead.
    cuda_idx=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' \
               | grep -n "^${GPU}$" | head -1 | cut -d: -f1 || true)
    if [ -z "$cuda_idx" ]; then
        echo "ERROR: GPU=$GPU not in CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        echo "       Either set CUDA_VISIBLE_DEVICES to include $GPU, or unset it." >&2
        exit 1
    fi
    export MM_GPU=$((cuda_idx - 1))
    VIS="$CUDA_VISIBLE_DEVICES"
else
    export CUDA_VISIBLE_DEVICES="$GPU"
    export MM_GPU=0
    VIS="$GPU"
fi

echo "[ncu] GPU=$GPU (physical)  CUDA_VISIBLE_DEVICES=$VIS  MM_GPU=$MM_GPU"
echo "[ncu] TAG=$TAG  NCU_SET=$NCU_SET"
echo "[ncu] out=$OUT_FILE"
echo "[ncu] log=$RUN_LOG"

# Single ncu invocation — Python loops over (kernel × op × M) and calls
# cudaProfilerStart/Stop only inside the profile region of each config.
#
#   --profile-from-start no  : do NOT capture until torch.cuda.profiler.start()
#   --nvtx                   : embed NVTX ranges in the report (so each kernel
#                              gets labeled "<kernel>__<op><model>__M<M>")
#   --target-processes application-only : skip child processes
#   --devices $MM_GPU        : profile only the cuda device PyTorch set_device()'d
#   --replay-mode kernel     : replay individual kernels for metric collection
ncu \
    --target-processes application-only \
    --devices "$MM_GPU" \
    --replay-mode kernel \
    --profile-from-start no \
    --nvtx \
    --set "$NCU_SET" \
    --force-overwrite \
    --export "$OUT_FILE" \
    python3 -u eval_ncu_target.py \
    2>&1 | tee "$RUN_LOG"

echo
echo "[ncu] DONE"
echo "  profile : $OUT_FILE"
echo "  log     : $RUN_LOG"
echo "  open    : ncu-ui $OUT_FILE"
echo "  summary : ncu --import $OUT_FILE --print-summary per-gpu --csv"
