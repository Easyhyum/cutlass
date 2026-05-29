#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

GPU="${GPU:-0}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="v9_nburst_${TS}"

POWER_CSV="$LOG_DIR/${TAG}_gpu${GPU}_power.csv"
RUN_LOG="$LOG_DIR/${TAG}.log"
SEG_CSV="$LOG_DIR/${TAG}_segments.csv"

nvidia-smi -i "$GPU" \
  --query-gpu=timestamp,clocks.current.sm,clocks.current.memory,power.draw.instant,power.draw.average,power.limit,temperature.gpu,utilization.gpu \
  --format=csv -lms 50 \
  > "$POWER_CSV" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

export CUDA_VISIBLE_DEVICES="$GPU"
export MM_GPU=0
export MM_SEGMENTS="$SEG_CSV"

echo "[run_v9_n_burst] GPU=$GPU"
python3 -u eval_v9_n_burst.py 2>&1 | tee "$RUN_LOG"

sleep 0.5
kill $SMI_PID 2>/dev/null || true
wait $SMI_PID 2>/dev/null || true

echo
echo "[run_v9_n_burst] enriching..."
python3 analyze_power.py "$SEG_CSV" "$POWER_CSV" \
    --threshold 660 --drop-tol 50 \
    > "$LOG_DIR/${TAG}_analysis.txt" 2>&1 || true

echo
echo "[run_v9_n_burst] aggregating per-config across bursts..."
python3 - <<PY
import csv
from collections import defaultdict
seg  = list(csv.DictReader(open("$SEG_CSV")))
enr  = list(csv.DictReader(open("${SEG_CSV%.csv}_with_power.csv")))
assert len(seg) == len(enr)
for s, e in zip(seg, enr):
    tag = s['backend']
    cfg = tag.split(':',1)[1].split('#',1)[0] if ':' in tag else tag
    e['cfg'] = cfg

groups = defaultdict(list)
for e in enr:
    groups[e['cfg']].append(e)

def mean(xs): return sum(xs)/len(xs)
def stdv(xs): m=mean(xs); return (sum((x-m)**2 for x in xs)/max(len(xs)-1,1))**0.5

# Find baseline: s100_step* (100% start = all SMs run immediately = baseline)
base_rows = [k for k in groups if k.startswith('s100')]
if not base_rows:
    print('no s100_* baseline found; aborting summary')
    raise SystemExit(0)
base_key = max(base_rows, key=lambda k: mean([float(r['tflops']) for r in groups[k]]))
base = groups[base_key]
base_tf  = mean([float(r['tflops']) for r in base])
base_mxw = mean([float(r['max_W']) for r in base])
base_avw = mean([float(r['avg_W']) for r in base])
base_p10 = mean([float(r['sm_p10']) for r in base])
print(f"\nBASELINE ({base_key}, N={len(base)}): tflops={base_tf:.1f}  "
      f"max_W={base_mxw:.1f}  avg_W={base_avw:.1f}  sm_p10={base_p10:.0f}\n")

def cfg_key(c):
    s = int(c.split('_')[0][1:])
    stp = int(c.split('_step')[1])
    return (s, stp)

print(f"{'cfg':>14s} {'N':>3s}  {'TFLOPS':>7s} {'%base':>5s}  "
      f"{'max_W μ':>8s} {'σ':>5s} {'Δμ':>7s}  "
      f"{'avg_W μ':>8s} {'Δμ':>6s}  {'sm_p10 μ':>8s} {'+MHz':>5s}")
print('-' * 105)
for cfg in sorted(groups.keys(), key=cfg_key):
    rows = groups[cfg]
    tfs = [float(r['tflops']) for r in rows]
    mxs = [float(r['max_W']) for r in rows]
    avs = [float(r['avg_W']) for r in rows]
    p10s = [float(r['sm_p10']) for r in rows]
    print(f"{cfg:>14s} {len(rows):>3d}  "
          f"{mean(tfs):>7.1f} {mean(tfs)/base_tf*100:>4.1f}%  "
          f"{mean(mxs):>8.1f} {stdv(mxs):>5.1f} {mean(mxs)-base_mxw:>+7.1f}  "
          f"{mean(avs):>8.1f} {mean(avs)-base_avw:>+6.1f}  "
          f"{mean(p10s):>8.0f} {mean(p10s)-base_p10:>+5.0f}")
PY

echo
echo "[run_v9_n_burst] DONE"
echo "  power csv  : $POWER_CSV"
echo "  segments   : $SEG_CSV"
