#!/usr/bin/env bash
# Focused Method A experiment: ONE (backend, op, M) sustained run across
# several configs, with concurrent nvidia-smi 50ms sampling for tighter spike
# capture, and a multi-config power-timeline plot.
#
# Usage:
#   ./run_focus.sh                                  # default: stream_k down_proj M=8192
#   GPU=3 MM_OP=qkv_proj MM_M=8192 ./run_focus.sh
#   MM_CONFIGS=B,P-500-16,P-2000-32 ./run_focus.sh

set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-3}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="focus_${MM_BACKEND:-stream_k}_${MM_OP:-down_proj}_M${MM_M:-8192}_${TS}"

POWER_CSV="$LOG_DIR/${TAG}_gpu${GPU}_power.csv"
RUN_LOG="$LOG_DIR/${TAG}.log"
SEG_CSV="$LOG_DIR/${TAG}_segments.csv"
PLOT_PNG="$LOG_DIR/${TAG}_timeline.png"

# 50ms nvidia-smi sampling for tighter spike capture
nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 50 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="$GPU"
export MM_GPU=0
export MM_SEGMENTS="$SEG_CSV"

echo "[run_focus] GPU=$GPU"
echo "  power log : $POWER_CSV  (50ms)"
echo "  segments  : $SEG_CSV"
echo "  plot      : $PLOT_PNG"
echo

python3 -u method_a_focus.py 2>&1 | tee "$RUN_LOG"

sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo
echo "[run_focus] enriching..."
python3 analyze_power.py "$SEG_CSV" "$POWER_CSV" \
    --threshold 660 --drop-tol 50 \
    2>&1 | tee "$LOG_DIR/${TAG}_analysis.txt"

echo
echo "[run_focus] rendering timeline plot..."
python3 plot_focus_timeline.py \
    --segments "$SEG_CSV" --power "$POWER_CSV" \
    --out "$PLOT_PNG"

echo
echo "[run_focus] DONE"
ls -la "$LOG_DIR"/${TAG}_*
