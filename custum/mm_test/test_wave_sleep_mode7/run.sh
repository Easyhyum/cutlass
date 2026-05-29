#!/usr/bin/env bash
# Wave-sleep mode 7 (SM gating) sweep — streamk_ws vs sm80_v3_ws
#   qwen3-32b down_proj  M=8192 K=25600 N=5120
#   50 bursts × 500ms burst × 500ms gap × {active_pct sweep}
#
# Two processes (one per kernel) so each loads only its own ws .so and
# __constant__ symbols stay isolated.  nvidia-smi sampler runs in parallel
# for power capture, same way test_M_kernel_sweep does it.
#
# GPU: always physical GPU 0 (project rule).  Rejects any other GPU.
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
if [ "$GPU" != "0" ]; then
    echo "ERROR: this project runs only on GPU 0 (got GPU=$GPU)." >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-ws7_${TS}}"
OUT_DIR="logs/$TAG"
mkdir -p "$OUT_DIR"

# Always force CUDA_VISIBLE_DEVICES to ONLY GPU 0 — keeps probe + other-job
# GPU 3 (if any) completely out of this run's CUDA context.
export CUDA_VISIBLE_DEVICES=0
export MM_GPU=0

# Config knobs — defaults match the spec the user asked for
KERNELS="${MM_KERNELS:-streamk_ws,sm80_v3_ws}"
ACTIVE_PCTS="${MM_ACTIVE_PCTS:-100,90,80,70,60,50,40}"
MM_M="${MM_M:-8192}"
MM_K="${MM_K:-25600}"
MM_N="${MM_N:-5120}"
MM_N_BURSTS="${MM_N_BURSTS:-50}"
MM_BURST_MS="${MM_BURST_MS:-500}"
MM_BURST_GAP_MS="${MM_BURST_GAP_MS:-500}"
MM_PEAK_TFLOPS="${MM_PEAK_TFLOPS:-400}"

SEG_CSV="$OUT_DIR/segments.csv"
POWER_CSV="$OUT_DIR/gpu0_power.csv"
RUN_LOG="$OUT_DIR/run.log"
rm -f "$SEG_CSV"

{
    echo "[ws7] GPU=0  TAG=$TAG"
    echo "[ws7] kernels=$KERNELS  active_pcts=$ACTIVE_PCTS"
    echo "[ws7] M=$MM_M K=$MM_K N=$MM_N (qwen3-32b down_proj)"
    echo "[ws7] bursts=$MM_N_BURSTS  burst_ms=$MM_BURST_MS  gap_ms=$MM_BURST_GAP_MS"
    echo "[ws7] OUT_DIR=$OUT_DIR"
    echo
} | tee "$RUN_LOG"

# ── nvidia-smi 50ms sampler ─────────────────────────────────────────────────
nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 50 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

sleep 0.5

# ── Sequential per-kernel eval, separate process each ───────────────────────
IFS=',' read -ra KS <<< "$KERNELS"
for K in "${KS[@]}"; do
    echo "===== kernel=$K =====" | tee -a "$RUN_LOG"
    MM_KERNEL="$K" \
        MM_ACTIVE_PCTS="$ACTIVE_PCTS" \
        MM_M="$MM_M" MM_K="$MM_K" MM_N="$MM_N" \
        MM_OP_NAME=down_proj MM_MODEL=qwen3-32b \
        MM_N_BURSTS="$MM_N_BURSTS" \
        MM_BURST_MS="$MM_BURST_MS" \
        MM_BURST_GAP_MS="$MM_BURST_GAP_MS" \
        MM_PEAK_TFLOPS="$MM_PEAK_TFLOPS" \
        MM_SEGMENTS="$SEG_CSV" \
        python3 -u eval_wave_sleep.py 2>&1 | tee -a "$RUN_LOG"
done

# ── Stop sampler ─────────────────────────────────────────────────────────────
sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

# ── Optional: enrich segments with power via test_M_kernel_sweep's analyzer ─
ANALYZE_PWR="../test_M_kernel_sweep/analyze_power.py"
if [ -f "$ANALYZE_PWR" ]; then
    echo "===== analyze_power =====" | tee -a "$RUN_LOG"
    python3 "$ANALYZE_PWR" "$SEG_CSV" "$POWER_CSV" \
        --threshold 660 --drop-tol 50 \
        > "$OUT_DIR/analysis.txt" 2>&1 || true
fi

# ── Plots ────────────────────────────────────────────────────────────────────
echo "===== plot_wave_sleep =====" | tee -a "$RUN_LOG"
IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" python3 -u plot_wave_sleep.py 2>&1 | tee -a "$RUN_LOG"

echo | tee -a "$RUN_LOG"
echo "[ws7] DONE — outputs in $OUT_DIR/" | tee -a "$RUN_LOG"
