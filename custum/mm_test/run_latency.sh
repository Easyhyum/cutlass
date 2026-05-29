#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

TS="$(date +%Y%m%d_%H%M%S)"
TAG="${TAG:-latency_${TS}}"
BASE_CSV="logs/${TAG}_baseline.csv"
WS_CSV="logs/${TAG}_wsdisabled.csv"

echo "[latency] phase 1: baseline binary (no wave-sleep code)"
MEAS_MODE=baseline MEAS_OUT="$BASE_CSV" python3 -u measure_latency.py 2>&1

sleep 1

echo
echo "[latency] phase 2: wave-sleep binary (no prime, gate=false)"
MEAS_MODE=wsdisabled MEAS_OUT="$WS_CSV" python3 -u measure_latency.py 2>&1

echo
echo "[latency] comparing..."
python3 plot_latency.py "$BASE_CSV" "$WS_CSV" --tag "$TAG"
