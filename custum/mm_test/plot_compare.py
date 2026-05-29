#!/usr/bin/env python3
"""
Side-by-side comparison plot for two Qwen3-8B GEMM sweeps.

Use it to compare any two runs: cuBLAS vs CUTLASS, forward vs reverse M
order, GPU A vs GPU B, etc.

Layout (single figure, 4 rows):
  Row 1: Run A   Power (W, left) + SM clock (MHz, right) + TFLOPS (3rd axis)
  Row 2: Run A   Temperature (left) + Utilization (right)
  Row 3: Run B   Power + SM clock + TFLOPS
  Row 4: Run B   Temperature + Utilization

Power axis 0-1000 W with dashed reference at 600 W (TDP).
Clock axis 0-3000 MHz with dashed reference at 2430 MHz (boost target).
TFLOPS axis 0-500 (step segments matching each (op, M) interval).

Each operator group is shaded and labeled with its dims "<op>\\nK=K, N=N".

Usage:
  python3 plot_compare.py \
      --segments-a logs/qwen3_cublas_..._segments.csv \
      --power-a    logs/qwen3_cublas_..._gpu3_power.csv \
      --label-a    "cuBLAS forward (small->large)" \
      --segments-b logs/qwen3_cublas_reverse_..._segments.csv \
      --power-b    logs/qwen3_cublas_reverse_..._gpu3_power.csv \
      --label-b    "cuBLAS reverse (large->small)" \
      --out logs/qwen3_fwd_vs_rev.png

The older flag set --cublas-segments / --cublas-power / --cutlass-segments
/ --cutlass-power is still accepted as a convenience.
"""

import argparse
import csv
import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def parse_ts(s):
    return datetime.strptime(s.strip(), '%Y/%m/%d %H:%M:%S.%f')


def strip_unit(s):
    return float(s.strip().split()[0])


def load_power(path):
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        header = [c.strip() for c in next(reader)]
        idx = {h: i for i, h in enumerate(header)}
        cmap = {
            'ts':   idx.get('timestamp'),
            'sm':   idx.get('clocks.current.sm [MHz]'),
            'pw':   idx.get('power.draw.instant [W]'),
            'temp': idx.get('temperature.gpu'),
            'util': idx.get('utilization.gpu [%]'),
        }
        for r in reader:
            if not r:
                continue
            try:
                ts = parse_ts(r[cmap['ts']])
            except Exception:
                continue

            def gv(k):
                ci = cmap[k]
                return strip_unit(r[ci]) if ci is not None else float('nan')
            rows.append({'ts': ts, 'sm': gv('sm'), 'pw': gv('pw'),
                         'temp': gv('temp'), 'util': gv('util')})
    rows.sort(key=lambda x: x['ts'])
    return rows


def load_segments(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def crop_to_sweep(power, segs, pad_s=1.0):
    if not segs:
        return power
    t0 = parse_ts(segs[0]['t_start'])
    t1 = parse_ts(segs[-1]['t_end'])
    pad = timedelta(seconds=pad_s)
    return [p for p in power if (t0 - pad) <= p['ts'] <= (t1 + pad)]


def op_groups_from_segments(segs):
    """List of dicts: {op, K, N, t0, t1, segments: [...]}."""
    groups = []
    cur = None
    for s in segs:
        op = s['operator']
        t0 = parse_ts(s['t_start'])
        t1 = parse_ts(s['t_end'])
        if cur is None or cur['op'] != op:
            if cur is not None:
                groups.append(cur)
            cur = dict(op=op, K=int(s['K']), N=int(s['N']),
                       t0=t0, t1=t1, segments=[])
        cur['t1'] = t1
        cur['segments'].append(s)
    if cur is not None:
        groups.append(cur)
    return groups


def render_top(ax, power, segs, panel_label):
    """Top panel: power(W) + SM clock(MHz) + TFLOPS step."""
    if not power:
        ax.set_title(panel_label + '  (no data)', loc='left', fontsize=11)
        return

    ts = [p['ts'] for p in power]
    pw = [p['pw'] for p in power]
    sm = [p['sm'] for p in power]

    c_pw, c_sm, c_tf = 'tab:blue', 'tab:red', 'purple'

    ax.plot(ts, pw, color=c_pw, linewidth=1.1,
            label='Power instant (W)', zorder=3)
    ax.axhline(600, linestyle='--', color=c_pw, alpha=0.45, label='TDP 600 W')
    ax.set_ylim(0, 1000)
    ax.set_ylabel('Power (W)', color=c_pw)
    ax.tick_params(axis='y', labelcolor=c_pw)
    ax.grid(True, alpha=0.25)

    ax_r = ax.twinx()
    ax_r.plot(ts, sm, color=c_sm, linewidth=1.1,
              label='SM clock (MHz)', zorder=4)
    ax_r.axhline(2430, linestyle='--', color=c_sm, alpha=0.45,
                 label='Boost 2430 MHz')
    ax_r.set_ylim(0, 3000)
    ax_r.set_ylabel('SM clock (MHz)', color=c_sm)
    ax_r.tick_params(axis='y', labelcolor=c_sm)

    # Third axis: TFLOPS step plot per segment, offset outward
    ax_tf = ax.twinx()
    ax_tf.spines.right.set_position(('axes', 1.06))
    seg_xs, seg_ys = [], []
    for s in segs:
        t0 = parse_ts(s['t_start'])
        t1 = parse_ts(s['t_end'])
        tf = float(s['tflops'])
        seg_xs += [t0, t1, t1]
        seg_ys += [tf, tf, float('nan')]
    ax_tf.plot(seg_xs, seg_ys, color=c_tf, linewidth=2.2,
               label='TFLOPS', zorder=5)
    ax_tf.set_ylim(0, 500)
    ax_tf.set_ylabel('TFLOPS', color=c_tf)
    ax_tf.tick_params(axis='y', labelcolor=c_tf)

    # Operator group shading + dim labels
    groups = op_groups_from_segments(segs)
    palette = plt.cm.tab10.colors
    for i, g in enumerate(groups):
        c = palette[i % len(palette)]
        ax.axvspan(g['t0'], g['t1'], alpha=0.08, color=c, zorder=1)
        mid = g['t0'] + (g['t1'] - g['t0']) / 2
        label = f"{g['op']}\nK={g['K']}, N={g['N']}"
        ax.text(mid, 975, label, ha='center', va='top',
                fontsize=9, color=c, fontweight='bold', zorder=6)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    h3, l3 = ax_tf.get_legend_handles_labels()
    ax.legend(h1 + h2 + h3, l1 + l2 + l3, loc='lower left',
              fontsize=7.5, framealpha=0.85, ncol=2)

    ax.set_title(panel_label, loc='left', fontsize=11, fontweight='bold')


def render_bot(ax, power, segs, panel_label=''):
    """Bottom panel: temperature + utilization."""
    if not power:
        return
    ts = [p['ts'] for p in power]
    temp = [p['temp'] for p in power]
    util = [p['util'] for p in power]
    c_t, c_u = 'tab:orange', 'tab:green'

    ax.plot(ts, temp, color=c_t, linewidth=1.1, label='Temperature (C)')
    ax.set_ylabel('Temperature (C)', color=c_t)
    ax.tick_params(axis='y', labelcolor=c_t)
    ax.set_ylim(20, 100)
    ax.grid(True, alpha=0.25)

    ax_r = ax.twinx()
    ax_r.plot(ts, util, color=c_u, linewidth=1.1, label='Utilization (%)')
    ax_r.set_ylabel('Utilization (%)', color=c_u)
    ax_r.tick_params(axis='y', labelcolor=c_u)
    ax_r.set_ylim(0, 105)

    groups = op_groups_from_segments(segs)
    palette = plt.cm.tab10.colors
    for i, g in enumerate(groups):
        c = palette[i % len(palette)]
        ax.axvspan(g['t0'], g['t1'], alpha=0.08, color=c)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower left', fontsize=7.5, framealpha=0.85)


def main():
    ap = argparse.ArgumentParser()
    # Generic A/B/C interface
    ap.add_argument('--segments-a')
    ap.add_argument('--power-a')
    ap.add_argument('--label-a', default=None)
    ap.add_argument('--segments-b')
    ap.add_argument('--power-b')
    ap.add_argument('--label-b', default=None)
    ap.add_argument('--segments-c', default=None)
    ap.add_argument('--power-c', default=None)
    ap.add_argument('--label-c', default=None)
    # Back-compat: cublas/cutlass-specific aliases
    ap.add_argument('--cublas-segments')
    ap.add_argument('--cublas-power')
    ap.add_argument('--cutlass-segments')
    ap.add_argument('--cutlass-power')
    ap.add_argument('--out',   required=True)
    ap.add_argument('--title', default='Qwen3-8B GEMM sweep on RTX PRO 6000 Blackwell')
    args = ap.parse_args()

    # Resolve A side
    if args.segments_a:
        seg_a, pw_a = args.segments_a, args.power_a
        lab_a = args.label_a or 'Run A'
    elif args.cublas_segments:
        seg_a, pw_a = args.cublas_segments, args.cublas_power
        lab_a = args.label_a or 'cuBLAS — torch.matmul (cublasGemmEx, BF16+FP32 accum)'
    else:
        ap.error('provide --segments-a/--power-a or --cublas-segments/--cublas-power')

    # Resolve B side
    if args.segments_b:
        seg_b, pw_b = args.segments_b, args.power_b
        lab_b = args.label_b or 'Run B'
    elif args.cutlass_segments:
        seg_b, pw_b = args.cutlass_segments, args.cutlass_power
        lab_b = args.label_b or 'CUTLASS SM80 — gemm_sm80_v3 (128x128x64, 3-stage, HMMA)'
    else:
        ap.error('provide --segments-b/--power-b or --cutlass-segments/--cutlass-power')

    panels = [(seg_a, pw_a, lab_a), (seg_b, pw_b, lab_b)]
    if args.segments_c:
        panels.append((args.segments_c, args.power_c,
                       args.label_c or 'Run C'))

    loaded = []
    for seg, pw, lab in panels:
        segs = load_segments(seg)
        pwr  = crop_to_sweep(load_power(pw), segs)
        loaded.append((segs, pwr, lab))

    n = len(loaded)
    height_ratios = []
    for _ in range(n):
        height_ratios += [1.4, 1.0]
    fig, axes = plt.subplots(
        2 * n, 1, figsize=(17, 5.5 * n),
        gridspec_kw={'height_ratios': height_ratios})

    for i, (segs, pwr, lab) in enumerate(loaded):
        render_top(axes[2*i],   pwr, segs, lab)
        render_bot(axes[2*i+1], pwr, segs)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate(rotation=0)

    fig.suptitle(args.title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.94, 0.97])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f'plot_compare: saved {args.out}')


if __name__ == '__main__':
    main()
