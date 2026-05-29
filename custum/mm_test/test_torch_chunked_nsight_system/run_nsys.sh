#!/usr/bin/env bash
# Nsight Systems profiling wrapper for torch-chunked GEMM eval.
# Captures CUDA + NVTX timeline so we can verify:
#   - pipe mode MMAs are strictly sequential on s_mma stream
#   - copy stream s_copy runs in parallel with next MMA
#   - Streams / kernel concurrency exactly as designed
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-3}"

# GPU mapping — supports both:
#   shell-pinned CUDA_VISIBLE_DEVICES (find GPU's index inside the list)
#   shell-unpinned (pin to just $GPU, MM_GPU=0)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    cuda_idx=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' \
               | grep -n "^${GPU}$" | head -1 | cut -d: -f1 || true)
    if [ -z "$cuda_idx" ]; then
        echo "ERROR: GPU=$GPU not in CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
        exit 1
    fi
    export MM_GPU=$((cuda_idx - 1))
else
    export CUDA_VISIBLE_DEVICES="$GPU"
    export MM_GPU=0
fi
echo "[nsys] GPU=$GPU (physical)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MM_GPU=$MM_GPU"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-nsys_${TS}}"
OUT_DIR="logs/$TAG"
mkdir -p "$OUT_DIR"

# Tunables (forwarded to eval_nsys.py)
KERNELS="${MM_KERNELS:-cublas,sm80_v3,streamk}"
M_LIST="${MM_M_LIST:-1024,2048,4096,8192,16384,32768,65536,131072,262144}"
CHUNK_LIST="${MM_CHUNK_LIST:-1024,1280,2048}"
N_WARMUP="${N_WARMUP:-30}"
N_PROFILE="${N_PROFILE:-3}"

RUN_LOG="$OUT_DIR/run.log"

{
    echo "[nsys] TAG=$TAG"
    echo "[nsys] KERNELS=$KERNELS"
    echo "[nsys] M_LIST=$M_LIST"
    echo "[nsys] CHUNK_LIST=$CHUNK_LIST"
    echo "[nsys] N_WARMUP=$N_WARMUP  N_PROFILE=$N_PROFILE"
    echo "[nsys] output: $OUT_DIR/profile.nsys-rep"
    echo
} | tee "$RUN_LOG"

# ── Run nsys profile ─────────────────────────────────────────────────────────
#   -t cuda,nvtx,osrt          : capture CUDA API, NVTX ranges, OS runtime
#   --stats=true               : produce text summary
#   --cuda-graph-trace=node    : node-level granularity for CUDA graph (if any)
#   --force-overwrite=true     : overwrite existing .nsys-rep
# Forward eval-config env vars (exports above are also inherited; -e is redundant
# but harmless and makes the actual values visible in nsys's log).
export MM_KERNELS="$KERNELS"
export MM_M_LIST="$M_LIST"
export MM_CHUNK_LIST="$CHUNK_LIST"
export N_WARMUP N_PROFILE

nsys profile \
    -o "$OUT_DIR/profile" \
    -t cuda,nvtx,osrt \
    --stats=true \
    --cuda-graph-trace=node \
    --force-overwrite=true \
    --gpu-metrics-devices="$GPU" \
    --gpu-metrics-frequency=100 \
    python3 -u eval_nsys.py 2>&1 | tee -a "$RUN_LOG"

# ── Stats summary (nsys CLI digest) ──────────────────────────────────────────
echo | tee -a "$RUN_LOG"
echo "===== nsys stats =====" | tee -a "$RUN_LOG"
nsys stats "$OUT_DIR/profile.nsys-rep" \
    --report cuda_api_sum,gpukernsum,nvtxsum \
    > "$OUT_DIR/stats.txt" 2>&1 || true

tail -50 "$OUT_DIR/stats.txt" 2>/dev/null | tee -a "$RUN_LOG"

echo | tee -a "$RUN_LOG"
echo "[nsys] DONE — outputs in $OUT_DIR/" | tee -a "$RUN_LOG"
echo "  profile.nsys-rep : open with 'nsys-ui $OUT_DIR/profile.nsys-rep'" | tee -a "$RUN_LOG"
echo "  stats.txt        : text summary (kernel breakdown, NVTX time)" | tee -a "$RUN_LOG"
echo "  run.log          : eval stdout" | tee -a "$RUN_LOG"
