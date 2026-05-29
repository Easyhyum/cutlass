#!/usr/bin/env bash
# M-sweep across (cublas, streamk) kernels.
# Drop-in compatible with analyze_power.py and the multi-row plot script.
#
# GPU selection:
#   GPU=<physical_id>   physical GPU index (used by nvidia-smi). default 0.
#
# Two cases:
#   (a) shell pre-set CUDA_VISIBLE_DEVICES (e.g. "0,3"):
#         run.sh keeps it intact and maps GPU=<physical> → MM_GPU=<cuda_idx
#         within the visible list>.  e.g. CUDA_VISIBLE_DEVICES=0,3 + GPU=3
#         → MM_GPU=1 (cuda:1 is physical GPU 3).
#
#   (b) shell did NOT set CUDA_VISIBLE_DEVICES:
#         run.sh sets it to just <GPU> so torch sees only that one device,
#         and MM_GPU=0.
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"                    # physical GPU id (nvidia-smi indexing)
LOG_ROOT="${LOG_DIR:-logs}"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-mkernel_${TS}}"

# Each run gets its own sub-folder: logs/<TAG>/
LOG_DIR="$LOG_ROOT/$TAG"
mkdir -p "$LOG_DIR"

# ── Map physical GPU → CUDA device index within CUDA_VISIBLE_DEVICES ────────
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Shell already pin a subset of devices visible. Find index of GPU within
    # the comma list (1-based → subtract 1).
    # `|| true` so a no-match in grep doesn't trip `set -e`/`pipefail` —
    # we want the explicit error message below to fire instead.
    cuda_idx=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' \
               | grep -n "^${GPU}$" | head -1 | cut -d: -f1 || true)
    if [ -z "$cuda_idx" ]; then
        echo "ERROR: GPU=$GPU is not in CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        echo "       Either unset CUDA_VISIBLE_DEVICES or add $GPU to it." >&2
        exit 1
    fi
    export MM_GPU=$((cuda_idx - 1))
    echo "[run] GPU=$GPU (physical) → MM_GPU=$MM_GPU (cuda)  "\
         "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
    # Pin this run to just that GPU and present it as cuda:0.
    export CUDA_VISIBLE_DEVICES="$GPU"
    export MM_GPU=0
    echo "[run] GPU=$GPU (physical)  CUDA_VISIBLE_DEVICES=$GPU  → MM_GPU=0"
fi

# File names inside the per-run folder (no TAG prefix needed — folder IS the tag).
POWER_CSV="$LOG_DIR/gpu${GPU}_power.csv"
RUN_LOG="$LOG_DIR/run.log"
SEG_CSV="$LOG_DIR/segments.csv"

# nvidia-smi sampler — 50 ms cadence (physical-GPU index)
nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 50 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

export MM_SEGMENTS="$SEG_CSV"
rm -f "$SEG_CSV"

echo "[run] GPU=$GPU  TAG=$TAG"
python3 -u eval_M_kernel_sweep.py 2>&1 | tee "$RUN_LOG"

sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

# Enrich with power (per-burst max/avg/percentile)
python3 analyze_power.py "$SEG_CSV" "$POWER_CSV" \
    --threshold 660 --drop-tol 50 \
    > "$LOG_DIR/analysis.txt" 2>&1 || true

# Plot 1 — single all-in-one timeline (rows = kernels, all ops in time axis)
python3 plot_kernel_timeline.py \
    --segments "$SEG_CSV" --power "$POWER_CSV" \
    --out "$LOG_DIR/timeline.png" \
    --title "M-sweep — $TAG"

# Plot 2 — one PNG per OP (rows = kernels)
mkdir -p "$LOG_DIR/by_op"
python3 plot_grouped.py \
    --segments "$SEG_CSV" --power "$POWER_CSV" \
    --out-dir "$LOG_DIR/by_op" --by op --tag "$TAG"

# Plot 3 — one PNG per KERNEL (rows = ops)
mkdir -p "$LOG_DIR/by_kernel"
python3 plot_grouped.py \
    --segments "$SEG_CSV" --power "$POWER_CSV" \
    --out-dir "$LOG_DIR/by_kernel" --by kernel --tag "$TAG"

echo
echo "[run] DONE — output folder: $LOG_DIR/"
echo "  segments      : $LOG_DIR/segments.csv"
echo "  power         : $LOG_DIR/gpu${GPU}_power.csv"
echo "  analysis      : $LOG_DIR/analysis.txt"
echo "  timeline.png  : $LOG_DIR/timeline.png            (all rows=kernels)"
echo "  by_op/        : $LOG_DIR/by_op/<op><model>_timeline.png   (per-op, rows=kernels)"
echo "  by_kernel/    : $LOG_DIR/by_kernel/<kernel>_timeline.png  (per-kernel, rows=ops)"
echo "  run log       : $LOG_DIR/run.log"
