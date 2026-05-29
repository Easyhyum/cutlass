#!/usr/bin/env bash
# Run mm_test on GPU 3 with concurrent nvidia-smi power/clock logging.
# Usage: ./run.sh                  # default M sweep
#        ./run.sh 2048 4096 8192   # custom M values
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-3}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

POWER_CSV="$LOG_DIR/gpu${GPU}_power_${TS}.csv"
RUN_LOG="$LOG_DIR/mm_test_${TS}.log"
SEG_CSV="$LOG_DIR/segments_${TS}.csv"

# Start nvidia-smi sampler before kernel launches
nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 100 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

# Pin to GPU 3 — program then sees device 0
export CUDA_VISIBLE_DEVICES="$GPU"
export MM_SEGMENTS="$SEG_CSV"

echo "[run.sh] GPU=$GPU"
echo "  run log   : $RUN_LOG"
echo "  power log : $POWER_CSV"
echo "  segments  : $SEG_CSV"
./mm_test "$@" 2>&1 | tee "$RUN_LOG"

# Brief tail to overlap measurement with cooldown
sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo "[run.sh] DONE"
echo "  run log   : $RUN_LOG"
echo "  power log : $POWER_CSV"
echo "  segments  : $SEG_CSV"
