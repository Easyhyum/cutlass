#!/usr/bin/env bash
# ============================================================================
# Method A V9 (spatial SM ramp) evaluation script
#
# Usage:
#   ./run_eval.sh                                   # default sweep on GPU 3
#   GPU=0 ./run_eval.sh                             # different GPU
#   MM_OP=qkv_proj ./run_eval.sh                    # different op
#   MM_N_BURSTS=50 ./run_eval.sh                    # fewer bursts (faster)
#   MM_V9_STARTS=70,80,90 ./run_eval.sh             # subset of start_pct
#   MM_V9_STEPS_NS=500,5000 ./run_eval.sh           # subset of step_ns
#   MM_BURST_GAP_MS=600 ./run_eval.sh               # idle gap between bursts
#   TAG_PREFIX=myexp ./run_eval.sh                  # custom output tag
#
# Output: logs/v9_<tag>_<timestamp>_{segments,gpu*_power,summary}.csv/png
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ── User-tunable parameters (env var) ───────────────────────────────────────
GPU="${GPU:-3}"
LOG_DIR="${LOG_DIR:-logs}"
TAG_PREFIX="${TAG_PREFIX:-v9}"

# Workload
export MM_OP="${MM_OP:-down_proj}"
export MM_M="${MM_M:-8192}"

# Burst structure
export MM_N_BURSTS="${MM_N_BURSTS:-100}"        # bursts per config
export MM_M_KERNELS="${MM_M_KERNELS:-150}"      # kernels per burst
export MM_BURST_GAP_MS="${MM_BURST_GAP_MS:-600}"  # idle gap (was 200→clock stuck)
export MM_CFG_GAP_MS="${MM_CFG_GAP_MS:-500}"
export MM_GLOBAL_WARMUP_MS="${MM_GLOBAL_WARMUP_MS:-3000}"

# V9 sweep ranges
export MM_V9_STARTS="${MM_V9_STARTS:-60,65,70,75,80,85,90,95,100}"
export MM_V9_STEPS_NS="${MM_V9_STEPS_NS:-500,2000,5000,10000,20000,50000}"

# ── output paths ────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG_PREFIX}_${TS}"
POWER_CSV="$LOG_DIR/${TAG}_gpu${GPU}_power.csv"
RUN_LOG="$LOG_DIR/${TAG}.log"
SEG_CSV="$LOG_DIR/${TAG}_segments.csv"

# ── start nvidia-smi power logger (50ms) ────────────────────────────────────
nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 50 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="$GPU"
export MM_GPU=0
export MM_SEGMENTS="$SEG_CSV"

echo "=========================================================================="
echo "[run_eval] GPU=$GPU  op=$MM_OP  M=$MM_M"
echo "[run_eval] N_BURSTS=$MM_N_BURSTS  M_KERNELS=$MM_M_KERNELS"
echo "[run_eval] BURST_GAP=$MM_BURST_GAP_MS ms  CFG_GAP=$MM_CFG_GAP_MS ms"
echo "[run_eval] V9_STARTS=$MM_V9_STARTS"
echo "[run_eval] V9_STEPS_NS=$MM_V9_STEPS_NS"
echo "[run_eval] TAG=$TAG"
echo "=========================================================================="

python3 -u eval_v9_n_burst.py 2>&1 | tee "$RUN_LOG"

sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo
echo "[run_eval] Enriching segments with power stats..."
python3 analyze_power.py "$SEG_CSV" "$POWER_CSV" \
    --threshold 660 --drop-tol 50 \
    > "$LOG_DIR/${TAG}_analysis.txt" 2>&1 || true

echo "[run_eval] Auto-generating plots..."
./make_plots.sh "$TAG"

echo
echo "================ DONE ================"
echo "  segments  : $SEG_CSV"
echo "  power     : $POWER_CSV"
echo "  enriched  : ${SEG_CSV%.csv}_with_power.csv"
echo "  analysis  : $LOG_DIR/${TAG}_analysis.txt"
echo "  log       : $RUN_LOG"
echo "  plots     : $LOG_DIR/${TAG}_{timeline,summary}.png"
