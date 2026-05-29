#!/usr/bin/env bash
# Mode 10 4D cross-product sweep — (first_pct × first_ns × mid_pct × mid_ns).
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
if [ "$GPU" != "0" ]; then
    echo "ERROR: GPU=0 only." >&2; exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-ws10_4d_${TS}}"
OUT_DIR="logs/$TAG"
mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES=0
export MM_GPU=0

KERNELS="${MM_KERNELS:-streamk_ws,sm80_v3_ws}"
FIRST_PCTS="${MM_FIRST_PCTS:-60,80,100}"
FIRST_NS_LIST="${MM_FIRST_NS_LIST:-500,1000}"
MID_PCTS="${MM_MID_PCTS:-60,80,100}"
MID_NS_LIST="${MM_MID_NS_LIST:-500,1000}"

MM_M="${MM_M:-8192}"
MM_K="${MM_K:-25600}"
MM_N="${MM_N:-5120}"
MM_N_BURSTS="${MM_N_BURSTS:-50}"
MM_BURST_MS="${MM_BURST_MS:-500}"
MM_BURST_GAP_MS="${MM_BURST_GAP_MS:-500}"
MM_PEAK_TFLOPS="${MM_PEAK_TFLOPS:-400}"

SEG_CSV="$OUT_DIR/segments_4d.csv"
POWER_CSV="$OUT_DIR/gpu0_power.csv"
RUN_LOG="$OUT_DIR/run.log"
rm -f "$SEG_CSV"

{
    echo "[ws10-4d] GPU=0  TAG=$TAG  kernels=$KERNELS"
    echo "[ws10-4d] first_pcts=$FIRST_PCTS  first_nss=$FIRST_NS_LIST"
    echo "[ws10-4d] mid_pcts  =$MID_PCTS   mid_nss  =$MID_NS_LIST"
    echo "[ws10-4d] M=$MM_M K=$MM_K N=$MM_N  bursts=$MM_N_BURSTS"
    echo
} | tee "$RUN_LOG"

nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 50 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

sleep 0.5

IFS=',' read -ra KS <<< "$KERNELS"
for K in "${KS[@]}"; do
    echo "===== kernel=$K =====" | tee -a "$RUN_LOG"
    MM_KERNEL="$K" \
        MM_FIRST_PCTS="$FIRST_PCTS" MM_FIRST_NS_LIST="$FIRST_NS_LIST" \
        MM_MID_PCTS="$MID_PCTS"     MM_MID_NS_LIST="$MID_NS_LIST" \
        MM_M="$MM_M" MM_K="$MM_K" MM_N="$MM_N" \
        MM_OP_NAME=down_proj MM_MODEL=qwen3-32b \
        MM_N_BURSTS="$MM_N_BURSTS" \
        MM_BURST_MS="$MM_BURST_MS" \
        MM_BURST_GAP_MS="$MM_BURST_GAP_MS" \
        MM_PEAK_TFLOPS="$MM_PEAK_TFLOPS" \
        MM_SEGMENTS="$SEG_CSV" \
        python3 -u eval_wave_sleep_mode10_4d.py 2>&1 | tee -a "$RUN_LOG"
done

sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo "===== plot_wave_sleep_mode10_4d (heatmap grid + Pareto) =====" | tee -a "$RUN_LOG"
IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" \
    python3 -u plot_wave_sleep_mode10_4d.py 2>&1 | tee -a "$RUN_LOG"

echo "===== plot_timeline_v9 (wall-time Power + SM + TFLOPS) =====" | tee -a "$RUN_LOG"
IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" \
    python3 -u plot_timeline_v9.py 2>&1 | tee -a "$RUN_LOG"

echo "[ws10-4d] DONE — outputs in $OUT_DIR/" | tee -a "$RUN_LOG"
