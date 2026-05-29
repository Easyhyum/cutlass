#!/usr/bin/env bash
# RAMP power-spike comparison: each config runs N kernels with ramp applied
# to all kernels (measurement convenience — not deployment behavior).
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-3}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="ramp_power_${TS}"

POWER_CSV="$LOG_DIR/${TAG}_gpu${GPU}_power.csv"
RUN_LOG="$LOG_DIR/${TAG}.log"
SEG_CSV="$LOG_DIR/${TAG}_segments.csv"

# 50ms sampling — paired with N=500-kernel bursts (~1.1s) → ~22 samples/cfg
nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 50 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="$GPU"
export MM_GPU=0
export MM_SEGMENTS="$SEG_CSV"

echo "[run_ramp_power] GPU=$GPU"
python3 -u eval_ramp_with_power.py 2>&1 | tee "$RUN_LOG"

sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo
echo "[run_ramp_power] enriching segments with power stats..."
python3 analyze_power.py "$SEG_CSV" "$POWER_CSV" \
    --threshold 660 --drop-tol 50 \
    2>&1 | tee "$LOG_DIR/${TAG}_analysis.txt" || true

echo
echo "[run_ramp_power] DONE"
echo "  power csv  : $POWER_CSV"
echo "  segments   : $SEG_CSV"
echo "  enriched   : ${SEG_CSV%.csv}_with_power.csv"
