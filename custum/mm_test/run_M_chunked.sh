#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-mchunked_${TS}}"

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

echo "[run_chunk] GPU=$GPU  TAG=$TAG  CHUNK_M=${CHUNK_M:-2048}"
python3 -u eval_M_chunked.py 2>&1 | tee "$RUN_LOG"

sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

python3 analyze_power.py "$SEG_CSV" "$POWER_CSV" \
    --threshold 660 --drop-tol 50 \
    > "$LOG_DIR/${TAG}_analysis.txt" 2>&1 || true

echo
echo "[run_chunk] DONE"
echo "  -> python plot_wsvar.py $TAG"
