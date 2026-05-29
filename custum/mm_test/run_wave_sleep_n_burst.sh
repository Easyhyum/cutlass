#!/usr/bin/env bash
# Drop-in replacement for run_v9_n_burst.sh.
# Splits work into TWO Python processes so the baseline binary and the
# wave-sleep binary never share a process — avoids CUTLASS module
# cross-talk we observed when both .so were imported into the same proc.
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-ws_nburst_${TS}}"

POWER_CSV="$LOG_DIR/${TAG}_gpu${GPU}_power.csv"
RUN_LOG="$LOG_DIR/${TAG}.log"
SEG_CSV="$LOG_DIR/${TAG}_segments.csv"

# Start power logger
nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 50 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="$GPU"
export MM_GPU=0
export MM_SEGMENTS="$SEG_CSV"

# Make sure segments CSV starts fresh
rm -f "$SEG_CSV"

echo "[run_ws_n_burst] GPU=$GPU  TAG=$TAG"

# Phase 1: baseline binary in its own Python process.
echo
echo "[run_ws_n_burst] ── PHASE 1: baseline binary (no wave-sleep code) ──"
MM_MODE=baseline python3 -u eval_wave_sleep_n_burst.py 2>&1 | tee -a "$RUN_LOG"

# brief settling between processes
sleep 1.0

# Phase 2: wave-sleep binary in its own Python process.
echo
echo "[run_ws_n_burst] ── PHASE 2: wave-sleep binary (sweep) ──"
MM_MODE=ws python3 -u eval_wave_sleep_n_burst.py 2>&1 | tee -a "$RUN_LOG"

sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo
echo "[run_ws_n_burst] enriching..."
python3 analyze_power.py "$SEG_CSV" "$POWER_CSV" \
    --threshold 660 --drop-tol 50 \
    > "$LOG_DIR/${TAG}_analysis.txt" 2>&1 || true

echo
echo "[run_ws_n_burst] DONE"
echo "  power csv  : $POWER_CSV"
echo "  segments   : $SEG_CSV"
echo "  -> ./make_plots.sh ${TAG}"
