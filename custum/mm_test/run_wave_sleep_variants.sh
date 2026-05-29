#!/usr/bin/env bash
# Variant sweep:  A=baseline, B=wave0+mid_small, C=all-wave staircase, D=quartile
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-wsvar_${TS}}"

POWER_CSV="$LOG_DIR/${TAG}_gpu${GPU}_power.csv"
RUN_LOG="$LOG_DIR/${TAG}.log"
SEG_CSV="$LOG_DIR/${TAG}_segments.csv"

nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 50 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="$GPU"
export MM_GPU=0
export MM_SEGMENTS="$SEG_CSV"
rm -f "$SEG_CSV"

echo "[run_wsvar] GPU=$GPU TAG=$TAG"

echo
echo "[run_wsvar] PHASE 1: baseline binary"
MM_MODE=baseline python3 -u eval_wave_sleep_variants.py 2>&1 | tee -a "$RUN_LOG"

sleep 1.0

echo
echo "[run_wsvar] PHASE 2: wave-sleep binary (variants)"
MM_MODE=ws python3 -u eval_wave_sleep_variants.py 2>&1 | tee -a "$RUN_LOG"

sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo
echo "[run_wsvar] enriching..."
python3 analyze_power.py "$SEG_CSV" "$POWER_CSV" \
    --threshold 660 --drop-tol 50 \
    > "$LOG_DIR/${TAG}_analysis.txt" 2>&1 || true

echo
echo "[run_wsvar] DONE"
echo "  power csv  : $POWER_CSV"
echo "  segments   : $SEG_CSV"
echo "  -> python plot_wsvar.py $TAG"
