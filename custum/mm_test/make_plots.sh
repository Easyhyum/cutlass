#!/usr/bin/env bash
# ============================================================================
# Generate timeline + per-config summary plots from a completed V9 sweep.
#
# Usage:
#   ./make_plots.sh <tag>
#
# Where <tag> is the experiment tag (e.g. v9_20260522_043405) — i.e. files
#   logs/<tag>_segments.csv
#   logs/<tag>_gpu*_power.csv
#   logs/<tag>_segments_with_power.csv
# must exist.
#
# Examples:
#   ./make_plots.sh v9_20260522_043405
#   ./make_plots.sh myexp_20260523_100000
#
# Output:
#   logs/<tag>_timeline.png   — wide time-axis plot, all configs back-to-back
#   logs/<tag>_summary.png    — 3-panel bar chart (max_W±σ, sm_p10, TFLOPS)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <tag>"
    echo "  tag examples: v9_20260522_043405, gap_recovery_20260522_042613"
    exit 1
fi
TAG="$1"
LOG_DIR="${LOG_DIR:-logs}"

SEG="$LOG_DIR/${TAG}_segments.csv"
ENR="$LOG_DIR/${TAG}_segments_with_power.csv"
PWR=$(ls "$LOG_DIR/${TAG}"_gpu*_power.csv 2>/dev/null | head -1)

if [ ! -f "$SEG" ] || [ ! -f "$ENR" ] || [ -z "$PWR" ]; then
    echo "ERROR: missing files for tag '$TAG'"
    echo "  expected: $SEG, $ENR, $LOG_DIR/${TAG}_gpu*_power.csv"
    exit 1
fi

# Rename "sXX_stepYY" → "sXX_pYY" for plot script compatibility
SEG_R="${SEG%.csv}_renamed.csv"
ENR_R="${ENR%.csv}_renamed.csv"
python3 - "$SEG" "$SEG_R" "$ENR" "$ENR_R" <<'PY'
import sys, csv
def rn(t): return t.replace('_step', '_p')
for src, dst in [(sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])]:
    with open(src) as f, open(dst, 'w', newline='') as g:
        r = csv.reader(f); w = csv.writer(g)
        hdr = next(r); w.writerow(hdr)
        for row in r:
            row[0] = rn(row[0])
            w.writerow(row)
print(f'renamed: {sys.argv[2]}')
print(f'renamed: {sys.argv[4]}')
PY

OUT_TL="$LOG_DIR/${TAG}_timeline.png"
OUT_SM="$LOG_DIR/${TAG}_summary.png"

python3 plot_nburst_timeline.py \
  --segments "$SEG_R" --power "$PWR" --out "$OUT_TL" \
  --title "Method A — ${TAG}   (wide time axis)" 2>&1 | tail -3

python3 plot_nburst_summary.py \
  --segments-with-power "$ENR_R" --segments "$SEG_R" --out "$OUT_SM" \
  --title "Method A — ${TAG}.   Green bar = max_W < baseline-2σ (statistically significant)" 2>&1 | tail -3

echo
echo "Plots generated:"
echo "  timeline : $OUT_TL"
echo "  summary  : $OUT_SM"
