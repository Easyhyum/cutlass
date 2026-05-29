#!/usr/bin/env bash
# Torch-level M-chunking sweep — streamk + sm80_v3 baselines.
# No new C++ build; imports production baselines from /workspace/custum.
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
if [ "$GPU" != "0" ]; then echo "GPU=0 only" >&2; exit 1; fi

TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-tchunk_${TS}}"
OUT_DIR="logs/$TAG"
mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES=0
export MM_GPU=0

KERNELS="${MM_KERNELS:-cublas,sm80_v3,streamk}"
MM_CHUNK_LIST="${MM_CHUNK_LIST:-1024,1280,2048}"
MM_CHUNK_GAP_US="${MM_CHUNK_GAP_US:-0}"
MM_M_LIST="${MM_M_LIST:-1024,2048,4096,8192,16384,32768,65536,131072,262144}"
MM_N_BURSTS="${MM_N_BURSTS:-50}"
MM_BURST_MS="${MM_BURST_MS:-500}"
MM_BURST_GAP_MS="${MM_BURST_GAP_MS:-500}"
MM_PEAK_TFLOPS="${MM_PEAK_TFLOPS:-400}"

SEG_CSV="$OUT_DIR/segments.csv"
POWER_CSV="$OUT_DIR/gpu0_power.csv"
RUN_LOG="$OUT_DIR/run.log"
rm -f "$SEG_CSV"

{
    echo "[torch-chunk] GPU=0  TAG=$TAG  kernels=$KERNELS"
    echo "[torch-chunk] M_list=$MM_M_LIST"
    echo "[torch-chunk] chunk_list=$MM_CHUNK_LIST  chunk_gap_us=$MM_CHUNK_GAP_US"
    echo "[torch-chunk] bursts=$MM_N_BURSTS"
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
        MM_M_LIST="$MM_M_LIST" \
        MM_CHUNK_LIST="$MM_CHUNK_LIST" MM_CHUNK_GAP_US="$MM_CHUNK_GAP_US" \
        MM_N_BURSTS="$MM_N_BURSTS" \
        MM_BURST_MS="$MM_BURST_MS" \
        MM_BURST_GAP_MS="$MM_BURST_GAP_MS" \
        MM_PEAK_TFLOPS="$MM_PEAK_TFLOPS" \
        MM_SEGMENTS="$SEG_CSV" \
        python3 -u eval_torch_chunked.py 2>&1 | tee -a "$RUN_LOG"
done

sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo "===== plot_chunked =====" | tee -a "$RUN_LOG"
IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" \
    python3 -u plot_chunked.py 2>&1 | tee -a "$RUN_LOG"

echo "===== plot_timeline_v9 =====" | tee -a "$RUN_LOG"
IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" \
    python3 -u plot_timeline_v9.py 2>&1 | tee -a "$RUN_LOG"

echo "[torch-chunk] DONE — $OUT_DIR/" | tee -a "$RUN_LOG"
