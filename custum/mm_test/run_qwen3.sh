#!/usr/bin/env bash
# Qwen3-8B GEMM sweep on a single GPU.
# Concurrent nvidia-smi power+clock logging, post-run spike / clock-drop
# analysis, and two-panel time-series plot. All output files are tagged
# with the backend (cublas | cutlass_sm80 | stream_k).
#
# Usage:
#   ./run_qwen3.sh                              # cublas, full M sweep
#   BACKEND=cutlass_sm80 ./run_qwen3.sh         # CUTLASS gemm_sm80_v3
#   BACKEND=stream_k ./run_qwen3.sh             # CUTLASS Stream-K
#   GPU=2 ./run_qwen3.sh                        # different GPU
#   MM_OPS=qkv_proj,o_proj ./run_qwen3.sh       # subset
#   MM_M=4096,16384,65536 ./run_qwen3.sh        # subset of Ms
#   MM_MIN_MS=4000 MM_GAP_MS=2000 ./run_qwen3.sh
#   SPIKE_W=660 DROP_TOL=50 ./run_qwen3.sh

set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-3}"
BACKEND="${BACKEND:-cublas}"      # cublas | cutlass_sm80 | stream_k
REVERSE="${REVERSE:-0}"           # 1 = walk M large -> small
LOG_DIR="${LOG_DIR:-logs}"
SPIKE_W="${SPIKE_W:-660}"
DROP_TOL="${DROP_TOL:-50}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
ORDER_TAG=""
if [[ "$REVERSE" == "1" || "$REVERSE" == "true" ]]; then
    ORDER_TAG="_reverse"
fi
TAG="${BACKEND}${ORDER_TAG}_${TS}"

POWER_CSV="$LOG_DIR/qwen3_${TAG}_gpu${GPU}_power.csv"
RUN_LOG="$LOG_DIR/qwen3_${TAG}.log"
SEG_CSV="$LOG_DIR/qwen3_${TAG}_segments.csv"
ANALYSIS_TXT="$LOG_DIR/qwen3_${TAG}_analysis.txt"
PLOT_PNG="$LOG_DIR/qwen3_${TAG}_plot.png"

# Start the nvidia-smi sampler first so we don't miss the first kernel.
nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 100 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="$GPU"
export MM_GPU=0
export MM_SEGMENTS="$SEG_CSV"
export MM_BACKEND="$BACKEND"
export MM_REVERSE="$REVERSE"

echo "[run_qwen3.sh] GPU=$GPU  BACKEND=$BACKEND  REVERSE=$REVERSE"
echo "  python    : $(command -v python3)"
echo "  run log   : $RUN_LOG"
echo "  power log : $POWER_CSV"
echo "  segments  : $SEG_CSV"
echo "  analysis  : $ANALYSIS_TXT"
echo "  plot      : $PLOT_PNG"
echo "  thresholds: spike=${SPIKE_W}W  sm_drop_tol=${DROP_TOL}MHz"
echo

python3 -u qwen3_8b_sweep.py 2>&1 | tee "$RUN_LOG"

sleep 1.0
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo
echo "[run_qwen3.sh] running analyzer..."
python3 analyze_power.py "$SEG_CSV" "$POWER_CSV" \
    --threshold "$SPIKE_W" --drop-tol "$DROP_TOL" \
    2>&1 | tee "$ANALYSIS_TXT"

echo
echo "[run_qwen3.sh] rendering plot..."
python3 plot_power.py "$SEG_CSV" "$POWER_CSV" \
    --out "$PLOT_PNG" --backend "$BACKEND"

echo
echo "[run_qwen3.sh] DONE"
echo "  segments csv (raw)      : $SEG_CSV"
echo "  segments csv (w/ power) : ${SEG_CSV%.csv}_with_power.csv"
echo "  power csv               : $POWER_CSV"
echo "  run log                 : $RUN_LOG"
echo "  analysis                : $ANALYSIS_TXT"
echo "  plot                    : $PLOT_PNG"
