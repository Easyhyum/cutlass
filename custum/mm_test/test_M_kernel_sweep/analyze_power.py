#!/usr/bin/env python3
"""
Join segments.csv (per (op, M) wall-clock brackets) with nvidia-smi power
CSV and report:
  1. Per-segment table: avg / max instant power, SM clock min / avg / max.
  2. First M per operator where instant power exceeds the spike threshold
     (default 660 W = 110 % of 600 W TDP).
  3. Best M per operator: highest TFLOPS achieved while the SM clock did
     NOT drop more than `--drop-tol` MHz below that operator's peak clock.

Usage:
  python3 analyze_power.py <segments.csv> <power.csv> \
      [--threshold W] [--drop-tol MHz]
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime


def parse_ts(s):
    # "2026/05/19 09:00:16.847"
    return datetime.strptime(s.strip(), '%Y/%m/%d %H:%M:%S.%f')


def strip_unit(s):
    return float(s.strip().split()[0])


def load_power(path):
    """Returns list of dict rows sorted by timestamp."""
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = [c.strip() for c in next(reader)]
        # Resolve column indices
        idx = {h: i for i, h in enumerate(header)}
        col_ts   = idx['timestamp']
        col_sm   = idx.get('clocks.current.sm [MHz]')
        col_pw   = idx.get('power.draw.instant [W]')
        col_avg  = idx.get('power.draw.average [W]')
        col_util = idx.get('utilization.gpu [%]')
        for r in reader:
            if not r:
                continue
            try:
                ts = parse_ts(r[col_ts])
            except Exception:
                continue
            rows.append({
                'ts':   ts,
                'sm':   strip_unit(r[col_sm])   if col_sm   is not None else float('nan'),
                'pw':   strip_unit(r[col_pw])   if col_pw   is not None else float('nan'),
                'pavg': strip_unit(r[col_avg])  if col_avg  is not None else float('nan'),
                'util': strip_unit(r[col_util]) if col_util is not None else float('nan'),
            })
    rows.sort(key=lambda x: x['ts'])
    return rows


def load_segments(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # backend column is optional (older runs may not have it)
    for r in rows:
        r.setdefault('backend', 'unknown')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('segments_csv')
    ap.add_argument('power_csv')
    ap.add_argument('--threshold', type=float, default=660.0,
                    help='Spike threshold in watts (default 660)')
    ap.add_argument('--drop-tol', type=float, default=50.0,
                    help='SM clock drop tolerance in MHz for "best M" (default 50)')
    ap.add_argument('--trim-ms', type=float, default=100.0,
                    help='Trim this many ms off each side of every segment '
                         'before computing stats, to skip idle boundary '
                         'samples (default 100)')
    args = ap.parse_args()

    power = load_power(args.power_csv)
    segs  = load_segments(args.segments_csv)

    if not power:
        print('No power samples loaded.', file=sys.stderr)
        sys.exit(1)
    if not segs:
        print('No segments loaded.', file=sys.stderr)
        sys.exit(1)

    print(f'[analyze] {len(segs)} segments, {len(power)} power samples')
    print(f'[analyze] spike threshold = {args.threshold:.0f} W   '
          f'sm drop tolerance = {args.drop_tol:.0f} MHz\n')

    hdr_fmt = ('{op:13s} {M:>7s} {K:>6s} {N:>7s} {tflops:>8s} '
               '{n:>3s} {pavg:>7s} {pmax:>7s} '
               '{smp10:>6s} {smp50:>6s} {smp90:>6s} {util:>5s} {spk:>5s}')
    row_fmt = ('{op:13s} {M:>7d} {K:>6d} {N:>7d} {tflops:>8.2f} '
               '{n:>3d} {pavg:>7.1f} {pmax:>7.1f} '
               '{smp10:>6.0f} {smp50:>6.0f} {smp90:>6.0f} {util:>5.0f} {spk:>5s}')

    print(hdr_fmt.format(op='operator', M='M', K='K', N='N', tflops='TFLOPS',
                         n='n', pavg='avg_W', pmax='max_W',
                         smp10='smP10', smp50='smP50', smp90='smP90',
                         util='util', spk='spike'))
    print('-' * 115)

    # Per-operator stats
    per_op = defaultdict(list)
    first_spike = {}      # op -> tuple(M,K,N,max_W,tflops)
    op_peak_sm = {}       # op -> max sm observed across all M

    enriched = []         # list of dict, one per segment

    from datetime import timedelta
    trim = timedelta(milliseconds=args.trim_ms)

    def pct(vals, p):
        if not vals:
            return float('nan')
        s = sorted(vals)
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c:
            return s[f]
        return s[f] + (s[c] - s[f]) * (k - f)

    for seg in segs:
        t0 = parse_ts(seg['t_start'])
        t1 = parse_ts(seg['t_end'])
        # Trim segment boundaries to skip idle-clock samples at the edges.
        # If the segment is shorter than 2x trim, fall back to full window.
        if (t1 - t0) > 2 * trim:
            t0t, t1t = t0 + trim, t1 - trim
        else:
            t0t, t1t = t0, t1
        ps = [r for r in power if t0t <= r['ts'] <= t1t]
        if not ps:
            ps = [r for r in power if t0 <= r['ts'] <= t1]
        n = len(ps)
        if n == 0:
            avg_w = max_w = sm_p10 = sm_p50 = sm_p90 = sm_min = sm_max = util_avg = float('nan')
        else:
            ws  = [p['pw']   for p in ps]
            sms = [p['sm']   for p in ps]
            uts = [p['util'] for p in ps]
            avg_w    = sum(ws) / n
            max_w    = max(ws)
            sm_p10   = pct(sms, 0.10)
            sm_p50   = pct(sms, 0.50)
            sm_p90   = pct(sms, 0.90)
            sm_min   = min(sms)
            sm_max   = max(sms)
            util_avg = sum(uts) / n

        op     = seg['operator']
        M      = int(seg['M'])
        K      = int(seg['K'])
        N      = int(seg['N'])
        tflops = float(seg['tflops'])
        spike = 'Y' if (n > 0 and max_w > args.threshold) else 'N'

        print(row_fmt.format(op=op, M=M, K=K, N=N, tflops=tflops,
                             n=n, pavg=avg_w, pmax=max_w,
                             smp10=sm_p10, smp50=sm_p50, smp90=sm_p90,
                             util=util_avg, spk=spike))

        d = dict(op=op, M=M, K=K, N=N, tflops=tflops,
                 n=n, avg_w=avg_w, max_w=max_w,
                 sm_min=sm_min, sm_max=sm_max,
                 sm_p10=sm_p10, sm_p50=sm_p50, sm_p90=sm_p90,
                 util=util_avg)
        enriched.append(d)
        per_op[op].append(d)

        if n > 0 and max_w > args.threshold and op not in first_spike:
            first_spike[op] = d
        if n > 0:
            op_peak_sm[op] = max(op_peak_sm.get(op, 0.0), sm_p90)

    # ---- Summary 1: first spike per operator ----
    print()
    print(f'=== First M where instant power > {args.threshold:.0f} W ===')
    if not first_spike:
        print('  (no operator crossed the threshold)')
    else:
        print(f'  {"op":12s} {"M":>7s}  {"K":>6s}  {"N":>7s}  '
              f'{"max_W":>7s}  {"TFLOPS":>8s}')
        for op, d in first_spike.items():
            print(f'  {op:12s} {d["M"]:>7d}  {d["K"]:>6d}  {d["N"]:>7d}  '
                  f'{d["max_w"]:>7.1f}  {d["tflops"]:>8.2f}')

    # ---- Summary 2: best M per operator ----
    # "Best" = max TFLOPS while the sustained-clock proxy (sm_p10) stays
    # within `drop_tol` MHz of that operator's per-op peak (sm_p90 across Ms).
    print()
    print(f'=== Best M (max TFLOPS while sm_p10 >= peak_sm - '
          f'{args.drop_tol:.0f} MHz) ===')
    print(f'  {"op":13s} {"M":>7s}  {"TFLOPS":>8s}  {"sm_p10":>7s}  '
          f'{"sm_p50":>7s}  {"peak_sm":>8s}  {"avg_W":>7s}  {"max_W":>7s}')
    for op, rows in per_op.items():
        peak_sm = op_peak_sm.get(op, 0.0)
        floor   = peak_sm - args.drop_tol
        candidates = [r for r in rows
                      if r['n'] > 0 and r['sm_p10'] >= floor]
        if not candidates:
            print(f'  {op:13s}  -- no M kept clock within tolerance --')
            continue
        best = max(candidates, key=lambda r: r['tflops'])
        print(f'  {op:13s} {best["M"]:>7d}  {best["tflops"]:>8.2f}  '
              f'{best["sm_p10"]:>7.0f}  {best["sm_p50"]:>7.0f}  '
              f'{peak_sm:>8.0f}  {best["avg_w"]:>7.1f}  {best["max_w"]:>7.1f}')

    # ---- Summary 2b: knee = largest M BEFORE the first spike ----
    # Pareto-optimal operating point: highest TFLOPS still in the
    # non-throttled boost regime, before the GPU starts overshooting the
    # 660 W spike line and (at larger M) settles into 600 W steady-state.
    print()
    print(f'=== Knee M (largest M before first spike >{args.threshold:.0f} W) ===')
    print(f'  {"op":13s} {"M":>7s}  {"TFLOPS":>8s}  {"sm_p10":>7s}  '
          f'{"sm_p50":>7s}  {"avg_W":>7s}  {"max_W":>7s}')
    for op, rows in per_op.items():
        active = sorted([r for r in rows if r['n'] > 0], key=lambda r: r['M'])
        if not active:
            continue
        # M values that come before the first spike
        before = []
        for r in active:
            if r['max_w'] > args.threshold:
                break
            before.append(r)
        if not before:
            print(f'  {op:13s}  -- first measured M already spiked --')
            continue
        knee = before[-1]
        print(f'  {op:13s} {knee["M"]:>7d}  {knee["tflops"]:>8.2f}  '
              f'{knee["sm_p10"]:>7.0f}  {knee["sm_p50"]:>7.0f}  '
              f'{knee["avg_w"]:>7.1f}  {knee["max_w"]:>7.1f}')

    # ---- Summary 2c: throttled steady-state = largest M tested, no spike ----
    # At very large M the GPU power-caps to 600 W and clock throttles
    # down — different regime from "knee".
    print()
    print('=== Throttled steady-state (largest M, max_W ~ TDP) ===')
    print(f'  {"op":13s} {"M":>7s}  {"TFLOPS":>8s}  {"sm_p10":>7s}  '
          f'{"sm_p50":>7s}  {"avg_W":>7s}  {"max_W":>7s}')
    for op, rows in per_op.items():
        active = [r for r in rows if r['n'] > 0]
        if not active:
            continue
        big = max(active, key=lambda r: r['M'])
        print(f'  {op:13s} {big["M"]:>7d}  {big["tflops"]:>8.2f}  '
              f'{big["sm_p10"]:>7.0f}  {big["sm_p50"]:>7.0f}  '
              f'{big["avg_w"]:>7.1f}  {big["max_w"]:>7.1f}')

    # ---- Summary 3: largest frequency drop per operator ----
    print()
    print('=== Largest sustained SM clock drop per operator '
          '(peak_sm - sm_p10) ===')
    print(f'  {"op":13s} {"peak_sm":>8s}  {"sm_p10":>7s}  '
          f'{"drop":>6s}  {"@ M":>7s}  {"TFLOPS":>8s}')
    for op, rows in per_op.items():
        active = [r for r in rows if r['n'] > 0]
        if not active:
            continue
        peak_sm = op_peak_sm.get(op, 0.0)
        worst = min(active, key=lambda r: r['sm_p10'])
        drop = peak_sm - worst['sm_p10']
        print(f'  {op:13s} {peak_sm:>8.0f}  {worst["sm_p10"]:>7.0f}  '
              f'{drop:>6.0f}  {worst["M"]:>7d}  {worst["tflops"]:>8.2f}')

    # ---- Write enriched csv next to inputs ----
    out_csv = args.segments_csv.replace('.csv', '_with_power.csv')
    if out_csv == args.segments_csv:
        out_csv = args.segments_csv + '.enriched.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['operator', 'M', 'K', 'N', 'tflops', 'n_samples',
                    'avg_W', 'max_W',
                    'sm_min', 'sm_p10', 'sm_p50', 'sm_p90', 'sm_max',
                    'util_avg', 'spike'])
        for d in enriched:
            w.writerow([d['op'], d['M'], d['K'], d['N'], f'{d["tflops"]:.4f}',
                        d['n'], f'{d["avg_w"]:.2f}', f'{d["max_w"]:.2f}',
                        f'{d["sm_min"]:.0f}', f'{d["sm_p10"]:.0f}',
                        f'{d["sm_p50"]:.0f}', f'{d["sm_p90"]:.0f}',
                        f'{d["sm_max"]:.0f}', f'{d["util"]:.1f}',
                        'Y' if (d['n'] > 0 and d['max_w'] > args.threshold) else 'N'])
    print(f'\n[analyze] enriched csv: {out_csv}')


if __name__ == '__main__':
    main()
