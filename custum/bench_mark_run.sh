#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/bench_mark.py"

# Pass any extra CLI args (except --pm) through to run_batch_inference.py
EXTRA_ARGS=("$@")

# Output / log root (relative to SCRIPT_DIR)
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── helper ────────────────────────────────────────────────────────────────────
run_once() {
    python "${SCRIPT}" \
        "${EXTRA_ARGS[@]}"
}

# ── GPU power monitoring ──────────────────────────────────────────────────────
GPU_LOG="${SCRIPT_DIR}/profile/power_profile_$(date +%Y%m%d_%H%M%S).csv"

nvidia-smi -i 3 \
    --query-gpu=timestamp,clocks.current.sm,power.draw.instant,power.draw.average,power.limit,clocks_event_reasons.sw_power_cap,clocks_event_reasons_counters.sw_power_cap \
    --format=csv \
    -lms 100 > "${GPU_LOG}" &
NVSMI_PID=$!
echo "▶ nvidia-smi started (PID=${NVSMI_PID}) → ${GPU_LOG}"

# Wait 3 seconds before first inference run
sleep 3

# ── sweep ─────────────────────────────────────────────────────────────────────
#  tag                sleep_ns  (empty string = cuBLAS, no env var)
run_once

sleep 3
kill "${NVSMI_PID}" 2>/dev/null && wait "${NVSMI_PID}" 2>/dev/null || true
echo "■ nvidia-smi stopped"

echo ""
echo "All runs complete. Results in ${LOG_DIR}/"
echo "GPU power log: ${GPU_LOG}"
