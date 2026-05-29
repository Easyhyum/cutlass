#!/usr/bin/env bash
# Re-plot every N seconds while a sweep is running.
#
# Usage:
#   ./watch_plot.sh                  # plot the latest logs/kchunk_*/  every 60s
#   ./watch_plot.sh logs/kchunk_XYZ  # plot a specific tag dir every 60s
#   INTERVAL=30 ./watch_plot.sh      # 30s cadence
#
# Stop with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")"

INTERVAL="${INTERVAL:-60}"
if [ -n "${1:-}" ]; then
    TAG_DIR="$1"
else
    TAG_DIR=$(ls -d logs/kchunk_* 2>/dev/null | tail -1 || true)
fi
if [ -z "$TAG_DIR" ] || [ ! -d "$TAG_DIR" ]; then
    echo "ERROR: no tag dir found under logs/" >&2
    exit 1
fi

echo "[watch] target=$TAG_DIR  interval=${INTERVAL}s  PID=$$"
echo "[watch] stop with Ctrl-C (foreground), 'kill $$', or 'pkill -f watch_plot.sh'"
trap 'echo; echo "[watch] stopping (PID=$$)"; exit 0' INT TERM
while true; do
    if [ -f "$TAG_DIR/segments.csv" ]; then
        nrows=$(($(wc -l < "$TAG_DIR/segments.csv") - 1))
        ts=$(date +%H:%M:%S)
        echo "[$ts] segments.csv rows=$nrows — re-plot"
        IN_DIR="$TAG_DIR" OUT_DIR="$TAG_DIR" \
            python3 -u plot_chunked.py    >/dev/null 2>&1 || true
        IN_DIR="$TAG_DIR" OUT_DIR="$TAG_DIR" \
            python3 -u plot_timeline_v9.py >/dev/null 2>&1 || true
    else
        echo "[$(date +%H:%M:%S)] waiting for $TAG_DIR/segments.csv ..."
    fi
    sleep "$INTERVAL"
done
