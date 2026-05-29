#!/usr/bin/env bash
# Run Qwen3-8B GEMM sweep for all 3 backends (cublas, cutlass_sm80, stream_k)
# sequentially. Each backend produces its own power/segments/analysis/plot
# files tagged with the backend name, so they can be compared post-hoc.
#
# Usage:
#   ./run_qwen3_all.sh                          # GPU=1, default M sweep
#   GPU=2 ./run_qwen3_all.sh                    # different GPU
#   MM_OPS=qkv_proj ./run_qwen3_all.sh          # subset of ops
#   MM_M=2048,8192,32768 ./run_qwen3_all.sh     # subset of Ms
#   BACKENDS="cublas stream_k" ./run_qwen3_all.sh   # subset of backends
#
# Between backends there is a cooldown sleep (COOL_S, default 30s) so the
# GPU temperature / leakage state does not bias the next sweep.

set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-1}"
COOL_S="${COOL_S:-30}"
BACKENDS="${BACKENDS:-cublas cutlass_sm80 stream_k}"

echo "[run_qwen3_all.sh] GPU=$GPU  backends=[$BACKENDS]  cooldown=${COOL_S}s"
echo

TS_GROUP="$(date +%Y%m%d_%H%M%S)"
GROUP_LOG="logs/qwen3_all_${TS_GROUP}.log"
mkdir -p logs
echo "[run_qwen3_all.sh] group log: $GROUP_LOG"

{
    echo "=== Qwen3 GEMM sweep — all backends ==="
    echo "started:  $(date -Iseconds)"
    echo "GPU:      $GPU"
    echo "backends: $BACKENDS"
    echo
} | tee "$GROUP_LOG"

for BE in $BACKENDS; do
    echo
    echo "======================================================================"
    echo "[run_qwen3_all.sh] BACKEND=$BE  (GPU=$GPU)"
    echo "======================================================================"
    {
        echo
        echo "--- BACKEND=$BE  $(date -Iseconds) ---"
    } >> "$GROUP_LOG"

    GPU="$GPU" BACKEND="$BE" ./run_qwen3.sh

    echo "[run_qwen3_all.sh] $BE done at $(date -Iseconds)" | tee -a "$GROUP_LOG"

    if [[ "$BE" != "$(echo $BACKENDS | awk '{print $NF}')" ]]; then
        echo "[run_qwen3_all.sh] cooldown ${COOL_S}s before next backend ..."
        sleep "$COOL_S"
    fi
done

echo
echo "[run_qwen3_all.sh] ALL DONE — list of segment CSVs:"
ls -1 logs/qwen3_*_segments.csv | tail -10
echo
echo "group log: $GROUP_LOG"
