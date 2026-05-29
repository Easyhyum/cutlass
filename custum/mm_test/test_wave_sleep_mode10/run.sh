#!/usr/bin/env bash
# Wave-sleep mode 10 — SEPARATE phase sweep
#   Phase A: first-wave only  (vary first_sleep_pct × first_ns)
#   Phase B: mid-wave only    (vary mid_sleep_pct  × mid_ns)
#
# Both kernels (streamk_ws, sm80_v3_ws) × both phases × 45 cfgs each ≈ ~3 h.
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
TAG="${TAG:-ws10_${TS}}"
OUT_DIR="logs/$TAG"
mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES=0
export MM_GPU=0

# Sweep knobs
KERNELS="${MM_KERNELS:-streamk_ws,sm80_v3_ws}"
PHASES="${MM_PHASES:-first,mid}"
PCT_LIST="${MM_PCT_LIST:-60,65,70,75,80,85,90,95,100}"
NS_LIST="${MM_NS_LIST:-250,500,750,1000,5000}"

# Problem & burst profile
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
    echo "[ws10] GPU=0  TAG=$TAG"
    echo "[ws10] kernels=$KERNELS  phases=$PHASES"
    echo "[ws10] pct_list=$PCT_LIST  ns_list=$NS_LIST"
    echo "[ws10] M=$MM_M K=$MM_K N=$MM_N (qwen3-32b down_proj)"
    echo "[ws10] bursts=$MM_N_BURSTS  burst_ms=$MM_BURST_MS  gap_ms=$MM_BURST_GAP_MS"
    echo "[ws10] OUT_DIR=$OUT_DIR"
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

# ── Sequential per (kernel × phase) eval, separate process each ─────────────
IFS=',' read -ra KS <<< "$KERNELS"
IFS=',' read -ra PH <<< "$PHASES"

for K in "${KS[@]}"; do
    for P in "${PH[@]}"; do
        echo "===== kernel=$K  phase=$P =====" | tee -a "$RUN_LOG"
        MM_KERNEL="$K" MM_PHASE="$P" \
            MM_PCT_LIST="$PCT_LIST" MM_NS_LIST="$NS_LIST" \
            MM_M="$MM_M" MM_K="$MM_K" MM_N="$MM_N" \
            MM_OP_NAME=down_proj MM_MODEL=qwen3-32b \
            MM_N_BURSTS="$MM_N_BURSTS" \
            MM_BURST_MS="$MM_BURST_MS" \
            MM_BURST_GAP_MS="$MM_BURST_GAP_MS" \
            MM_PEAK_TFLOPS="$MM_PEAK_TFLOPS" \
            MM_SEGMENTS="$SEG_CSV" \
            python3 -u eval_wave_sleep_mode10.py 2>&1 | tee -a "$RUN_LOG"
    done
done

# ── Stop sampler ─────────────────────────────────────────────────────────────
sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

# ── Power enrichment ─────────────────────────────────────────────────────────
ANALYZE_PWR="../test_M_kernel_sweep/analyze_power.py"
if [ -f "$ANALYZE_PWR" ]; then
    echo "===== analyze_power =====" | tee -a "$RUN_LOG"
    python3 "$ANALYZE_PWR" "$SEG_CSV" "$POWER_CSV" \
        --threshold 660 --drop-tol 50 \
        > "$OUT_DIR/analysis.txt" 2>&1 || true
fi

# ── Plots ────────────────────────────────────────────────────────────────────
echo "===== plot_wave_sleep_mode10 (heatmap + per-cfg power timeline) =====" | tee -a "$RUN_LOG"
IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" \
    python3 -u plot_wave_sleep_mode10.py 2>&1 | tee -a "$RUN_LOG"

echo "===== plot_timeline_v9 (wall-time Power + SM + TFLOPS) =====" | tee -a "$RUN_LOG"
IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" \
    python3 -u plot_timeline_v9.py 2>&1 | tee -a "$RUN_LOG"

echo | tee -a "$RUN_LOG"
echo "[ws10] DONE — outputs in $OUT_DIR/" | tee -a "$RUN_LOG"
