#!/usr/bin/env bash
# CTA dispatch / monotonicity probe.
#
# Runs the per-CTA (smid, globaltimer, blockIdx) probe for streamk and
# sm80_v3 in SEPARATE PROCESSES (one kernel per process) so that the two
# probe binaries — each with their own __constant__ symbol set — never live
# in the same CUDA context.  Without the split we hit "streamk run: Error
# Internal" when two TUs touch the same symbol table.
#
# Outputs (under logs/<TAG>/):
#   cta_probe_per_cta_<model>_<op>_<kernel>.csv     per-CTA records
#   cta_probe_summary_<model>_<op>_<kernel>.csv     per-wave summary
#   cta_probe_monotonicity_<model>_<op>.csv         summary across kernels
#   cta_probe_monotonicity_<model>_<op>_<k>.png     L vs wave plot
#   cta_probe_sm_<model>_<op>_<k>.png               CTA → SM scatter
#   cta_probe_dur_vs_idx_<model>_<op>_<k>.png       duration vs index
#   cta_probe_streamk_vs_sm80_v3_<model>_<op>.png   side-by-side
#
# Env:
#   GPU       physical GPU id (nvidia-smi indexing).  default 0
#   MM_MODEL  qwen3-8b (default) | qwen3-32b
#   MM_OP     down_proj (default) | qkv_proj | o_proj | up_proj | lm_head
#   MM_MS     comma list of M (default: 32,256,1024,4096,8192,65536,131072)
#   MM_KERNELS  comma list, default: streamk,sm80_v3
#   TAG       output sub-folder under logs/  (default: ctap_<timestamp>)
set -euo pipefail
cd "$(dirname "$0")"

# Always GPU 0 for this project — see /root/.claude/projects/-workspace/memory/feedback_gpu0_only.md
GPU="${GPU:-0}"
if [ "$GPU" != "0" ]; then
    echo "ERROR: this project runs only on GPU 0 (got GPU=$GPU). Re-run with GPU=0 (or omit GPU=)." >&2
    exit 1
fi
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-ctap_${TS}}"
MODEL="${MM_MODEL:-qwen3-8b}"
OP="${MM_OP:-down_proj}"
MS="${MM_MS:-32,256,1024,4096,8192,65536,131072}"
KERNELS="${MM_KERNELS:-streamk,sm80_v3}"

# ── GPU mapping (consistent with other mm_test/* run.sh scripts) ─────────────
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    cuda_idx=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' \
               | grep -n "^${GPU}$" | head -1 | cut -d: -f1 || true)
    if [ -z "$cuda_idx" ]; then
        echo "ERROR: GPU=$GPU not in CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        exit 1
    fi
    export MM_GPU=$((cuda_idx - 1))
    VIS="$CUDA_VISIBLE_DEVICES"
else
    export CUDA_VISIBLE_DEVICES="$GPU"
    export MM_GPU=0
    VIS="$GPU"
fi

OUT_DIR="logs/$TAG"
mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/run.log"

{
    echo "[ctap] GPU=$GPU  CUDA_VISIBLE_DEVICES=$VIS  MM_GPU=$MM_GPU"
    echo "[ctap] TAG=$TAG  MODEL=$MODEL  OP=$OP  M_LIST=$MS  KERNELS=$KERNELS"
    echo "[ctap] OUT_DIR=$OUT_DIR"
    echo
} | tee "$RUN_LOG"

# ── Per-kernel probe — separate process per kernel ───────────────────────────
IFS=',' read -ra KS <<< "$KERNELS"
for K in "${KS[@]}"; do
    echo "===== kernel=$K =====" | tee -a "$RUN_LOG"
    MM_KERNEL="$K" MM_MODEL="$MODEL" MM_OP="$OP" MM_MS="$MS" \
        OUT_DIR="$OUT_DIR" \
        python3 -u eval_cta_probe.py 2>&1 | tee -a "$RUN_LOG"
done

# ── Monotonicity analysis ────────────────────────────────────────────────────
echo "===== analyze_monotonicity =====" | tee -a "$RUN_LOG"
MM_MODEL="$MODEL" MM_OP="$OP" OUT_DIR="$OUT_DIR" \
    python3 -u analyze_monotonicity.py 2>&1 | tee -a "$RUN_LOG"

# ── Plots ────────────────────────────────────────────────────────────────────
echo "===== plot_cta_probe =====" | tee -a "$RUN_LOG"
MM_MODEL="$MODEL" MM_OP="$OP" OUT_DIR="$OUT_DIR" \
    python3 -u plot_cta_probe.py 2>&1 | tee -a "$RUN_LOG"

echo | tee -a "$RUN_LOG"
echo "[ctap] DONE — outputs in $OUT_DIR/" | tee -a "$RUN_LOG"
