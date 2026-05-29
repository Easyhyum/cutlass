#!/usr/bin/env bash
# Method A (SM-staggered nanosleep) parameter sweep with concurrent
# nvidia-smi power+clock logging and Pareto-front analysis.
#
# Usage:
#   ./run_method_a.sh                           # GPU=3, default sweep
#   GPU=1 ./run_method_a.sh
#   MM_OPS=qkv_proj MM_M=8192 ./run_method_a.sh
#   MM_BACKENDS=stream_k ./run_method_a.sh
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-3}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="method_a_${TS}"

POWER_CSV="$LOG_DIR/${TAG}_gpu${GPU}_power.csv"
RUN_LOG="$LOG_DIR/${TAG}.log"
SEG_CSV="$LOG_DIR/${TAG}_segments.csv"
PARETO_CSV="$LOG_DIR/${TAG}_pareto.csv"
PLOT_PNG="$LOG_DIR/${TAG}_pareto.png"

nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 100 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="$GPU"
export MM_GPU=0
export MM_SEGMENTS="$SEG_CSV"

echo "[run_method_a] GPU=$GPU"
echo "  run log    : $RUN_LOG"
echo "  power log  : $POWER_CSV"
echo "  segments   : $SEG_CSV"
echo "  pareto csv : $PARETO_CSV"
echo "  pareto png : $PLOT_PNG"
echo

python3 -u method_a_sweep.py 2>&1 | tee "$RUN_LOG"

sleep 1.0
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo
echo "[run_method_a] enriching segments with power stats..."
python3 analyze_power.py "$SEG_CSV" "$POWER_CSV" \
    --threshold 660 --drop-tol 50 \
    2>&1 | tee "$LOG_DIR/${TAG}_analysis.txt"

echo
echo "[run_method_a] rendering Pareto plot + summary CSV..."
python3 method_a_pareto.py \
    --segments-with-power "${SEG_CSV%.csv}_with_power.csv" \
    --pareto-csv "$PARETO_CSV" \
    --plot "$PLOT_PNG"

echo
echo "[run_method_a] DONE"
echo "  segments csv (w/ power) : ${SEG_CSV%.csv}_with_power.csv"
echo "  pareto csv              : $PARETO_CSV"
echo "  pareto plot             : $PLOT_PNG"
