#!/usr/bin/env bash
# Qwen3-32B per-call timing eval on GPU 0.
# Sweeps (op, M) x backend in {cublas, stream_k} with 50 bursts per cfg.
# M values are reduced relative to 8B because hidden / intermediate / Q_DIM
# are larger.
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0
GPU=0
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="qwen3_32b_timing_${TS}"

SEG_CSV="$LOG_DIR/${TAG}_segments.csv"
RUN_LOG="$LOG_DIR/${TAG}.log"

export MM_GPU=0
export MM_MODEL=qwen3-32b
export MM_SEGMENTS="$SEG_CSV"
export MM_N_BURSTS="${MM_N_BURSTS:-50}"
export MM_M_KERNELS="${MM_M_KERNELS:-50}"
export MM_BACKENDS="${MM_BACKENDS:-cublas,stream_k}"

echo "[run_qwen3_32b_timing] GPU=$GPU  TAG=$TAG"
echo "  log      : $RUN_LOG"
echo "  segments : $SEG_CSV"
echo

python3 -u eval_qwen3_timing.py 2>&1 | tee "$RUN_LOG"

echo
echo "[run_qwen3_32b_timing] rendering plot + table ..."
python3 plot_percall_time.py \
    --segments "$SEG_CSV" \
    --out-png  "$LOG_DIR/${TAG}_percall.png" \
    --out-md   "$LOG_DIR/${TAG}_table.md" \
    --title    "Qwen3-32B per-MatMul time  (50 bursts, GPU 0)"

echo
echo "[run_qwen3_32b_timing] DONE"
echo "  segments : $SEG_CSV"
echo "  plot     : $LOG_DIR/${TAG}_percall.png"
echo "  table    : $LOG_DIR/${TAG}_table.md"
