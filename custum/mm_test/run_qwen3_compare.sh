#!/usr/bin/env bash
# Qwen3 2-backend compare eval on GPU 0.
#
# Runs MM_BACKEND in {cublas, stream_k} sequentially, each paired with its
# own 100 ms nvidia-smi power+clock logger. After both backends finish, the
# 2-way compare plot and the TFLOPS/per-cycle line chart are rendered.
#
# Usage:
#   MODEL=qwen3-8b  ./run_qwen3_compare.sh
#   MODEL=qwen3-32b ./run_qwen3_compare.sh
#
# Optional env:
#   GPU         (default 0; also exported as CUDA_VISIBLE_DEVICES)
#   COOL_S      cooldown between backends (default 20s)
#   MM_CYCLE_MS work cycle target (default 320)
#   MM_REST_MS  rest gap (default 500)
#   MM_MS       comma-list, override M sweep
#   MM_OPS      comma-list, op subset
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-qwen3-8b}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
COOL_S="${COOL_S:-20}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${MODEL//-/_}_compare_${TS}"

echo "[compare] MODEL=$MODEL  GPU=$GPU  TAG=$TAG"
echo

for BE in cublas stream_k; do
    SEG_CSV="$LOG_DIR/${TAG}_${BE}_segments.csv"
    PWR_CSV="$LOG_DIR/${TAG}_${BE}_power.csv"
    RUN_LOG="$LOG_DIR/${TAG}_${BE}.log"

    echo "============================================================"
    echo "[compare] backend=$BE"
    echo "  segments : $SEG_CSV"
    echo "  power    : $PWR_CSV"
    echo "  log      : $RUN_LOG"
    echo "============================================================"

    # Start the 100 ms power+clock sampler BEFORE the kernel run starts
    nvidia-smi -i 0 \
      --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
      --format=csv -lms 100 \
      > "$PWR_CSV" &
    SMI_PID=$!
    # Make sure we never leak the sampler
    trap "kill $SMI_PID 2>/dev/null || true" EXIT

    sleep 0.4  # let a few baseline rows land before kernels start

    MM_GPU=0 \
    MM_MODEL="$MODEL" \
    MM_BACKEND="$BE" \
    MM_SEGMENTS="$SEG_CSV" \
    MM_CYCLE_MS="${MM_CYCLE_MS:-320}" \
    MM_REST_MS="${MM_REST_MS:-500}" \
    ${MM_MS:+MM_MS="$MM_MS"} \
    ${MM_OPS:+MM_OPS="$MM_OPS"} \
        python3 -u eval_qwen3_compare.py 2>&1 | tee "$RUN_LOG"

    sleep 0.6
    kill $SMI_PID 2>/dev/null || true
    wait $SMI_PID 2>/dev/null || true
    trap - EXIT

    echo
    echo "[compare] $BE done."
    echo

    if [ "$BE" = "cublas" ]; then
        echo "[compare] cooldown ${COOL_S}s before stream_k ..."
        sleep "$COOL_S"
    fi
done

# ---------- render plots ----------
SEG_CU="$LOG_DIR/${TAG}_cublas_segments.csv"
SEG_SK="$LOG_DIR/${TAG}_stream_k_segments.csv"
PWR_CU="$LOG_DIR/${TAG}_cublas_power.csv"
PWR_SK="$LOG_DIR/${TAG}_stream_k_power.csv"

OUT_CMP="$LOG_DIR/${TAG}_compare.png"
OUT_LINE="$LOG_DIR/${TAG}_lines.png"
OUT_MD="$LOG_DIR/${TAG}_table.md"

echo
echo "[compare] rendering 2-way comparison plot ..."
python3 plot_qwen3_compare_2way.py \
    --segments-cublas   "$SEG_CU" --power-cublas   "$PWR_CU" \
    --segments-streamk  "$SEG_SK" --power-streamk  "$PWR_SK" \
    --out "$OUT_CMP" \
    --title "${MODEL} GEMM sweep  —  cuBLAS vs CUTLASS Stream-K  (GPU $GPU)"

echo
echo "[compare] rendering line chart ..."
python3 plot_qwen3_lines.py \
    --segments-cublas  "$SEG_CU" \
    --segments-streamk "$SEG_SK" \
    --out "$OUT_LINE" \
    --out-md "$OUT_MD" \
    --title "${MODEL} TFLOPS & per-cycle time"

echo
echo "[compare] DONE — artifacts:"
echo "  segments  : $SEG_CU"
echo "              $SEG_SK"
echo "  power     : $PWR_CU"
echo "              $PWR_SK"
echo "  compare   : $OUT_CMP"
echo "  lines     : $OUT_LINE"
echo "  table     : $OUT_MD"
