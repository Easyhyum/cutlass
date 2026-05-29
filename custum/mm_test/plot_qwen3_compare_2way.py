#!/usr/bin/env python3
"""
2-way compare plot for Qwen3 GEMM sweeps, modeled after qwen3_compare_3way.png.

Renders two stacked panels (one per backend, top=cublas, bottom=stream_k).
Each panel shows on a shared time axis:
  - left  y-axis (0-1000 W)  : power.draw.instant  + dashed 600 W cap
  - right y-axis (0-3000 MHz): clocks.current.sm   + dashed 2430 MHz cap
  - offset y-axis (0-500 TF) : per-segment TFLOPS  (step plot)
Operator groups are shaded with op name + K, N labels at the top.
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
        cols = {
            'ts': 'timestamp',
            'sm': 'clocks.current.sm [MHz]',
            'pw': 'power.draw.instant [W]',
        }
        cmap = {k: idx.get(v) for k, v in cols.items()}
        for r in reader:
            if not r: continue
            try:
                ts = parse_ts(r[cmap['ts']])
            except Exception:
                continue
            def gv(k):
                ci = cmap[k]
                return strip_unit(r[ci]) if ci is not None else float('nan')
            rows.append({'ts': ts, 'sm': gv('sm'), 'pw': gv('pw')})
    rows.sort(key=lambda x: x['ts'])
    return rows


def load_segments(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def crop_power(power, segs, pad_s=1.0):
    if not segs or not power:
        return power
    t_first = parse_ts(segs[0]['t_start'])
    t_last  = parse_ts(segs[-1]['t_end'])
    pad = timedelta(seconds=pad_s)
    return [p for p in power if (t_first - pad) <= p['ts'] <= (t_last + pad)]


def operator_groups(segs):
    groups = []
    cur = None
    for s in segs:
        t0 = parse_ts(s['t_start'])
        t1 = parse_ts(s['t_end'])
        if cur is None or s['operator'] != cur['op']:
            if cur is not None:
                groups.append(cur)
            cur = dict(op=s['operator'],
                       K=int(s['K']), N=int(s['N']),
                       t0=t0, t1=t1)
        cur['t1'] = t1
    if cur is not None:
        groups.append(cur)
    return groups


def draw_panel(ax, segs, power, label):
    """Draw the power+SM+TFLOPS panel for one backend onto axis `ax`."""
    color_pw = 'tab:blue'
    color_sm = 'tab:red'
    color_tf = 'purple'

    if power:
        ts = [p['ts'] for p in power]
        pw = [p['pw'] for p in power]
        sm = [p['sm'] for p in power]
        ax.plot(ts, pw, color=color_pw, linewidth=1.0,
                label='Power instant (W)', zorder=3)
    ax.axhline(600, linestyle='--', color=color_pw, alpha=0.55, linewidth=1.2,
               label='TDP 600 W')
    ax.set_ylim(0, 1000)
    ax.set_ylabel('Power (W)', color=color_pw)
    ax.tick_params(axis='y', labelcolor=color_pw, labelsize=9)
    ax.grid(True, alpha=0.25)

    ax_r = ax.twinx()
    if power:
        ax_r.plot(ts, sm, color=color_sm, linewidth=1.0,
                  label='SM clock (MHz)', zorder=2, alpha=0.85)
    ax_r.axhline(2430, linestyle='--', color=color_sm, alpha=0.55, linewidth=1.2,
                 label='Boost 2430 MHz')
    ax_r.set_ylim(0, 3000)
    ax_r.set_ylabel('SM clock (MHz)', color=color_sm)
    ax_r.tick_params(axis='y', labelcolor=color_sm, labelsize=9)

    ax_tf = ax.twinx()
    ax_tf.spines.right.set_position(('axes', 1.06))
    seg_xs, seg_ys = [], []
    for s in segs:
        try:
            t0 = parse_ts(s['t_start'])
            t1 = parse_ts(s['t_end'])
            tf = float(s['tflops'])
        except (KeyError, ValueError):
            continue
        seg_xs += [t0, t1, t1]
        seg_ys += [tf, tf, float('nan')]
    if seg_xs:
        ax_tf.plot(seg_xs, seg_ys, color=color_tf, linewidth=1.8,
                   label='TFLOPS', zorder=5)
    ax_tf.set_ylim(0, 500)
    ax_tf.set_ylabel('TFLOPS', color=color_tf)
    ax_tf.tick_params(axis='y', labelcolor=color_tf, labelsize=9)

    # operator group bands + labels
    palette = plt.cm.tab10.colors
    groups = operator_groups(segs)
    for i, g in enumerate(groups):
        c = palette[i % len(palette)]
        ax.axvspan(g['t0'], g['t1'], alpha=0.10, color=c, zorder=1)
        mid = g['t0'] + (g['t1'] - g['t0']) / 2
        ax.text(mid, 975,
                f"{g['op']}\nK={g['K']}, N={g['N']}",
                ha='center', va='top', fontsize=8.5,
                color=c, fontweight='bold', zorder=6)

    # title strip
    ax.text(0.005, 0.97, label, transform=ax.transAxes,
            ha='left', va='top', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, pad=2))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    h3, l3 = ax_tf.get_legend_handles_labels()
    ax.legend(h1 + h2 + h3, l1 + l2 + l3,
              loc='lower right', fontsize=8, ncol=2, framealpha=0.85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments-cublas',  required=True)
    ap.add_argument('--power-cublas',     required=True)
    ap.add_argument('--segments-streamk', required=True)
    ap.add_argument('--power-streamk',    required=True)
    ap.add_argument('--out',              required=True)
    ap.add_argument('--title',            default='')
    args = ap.parse_args()

    seg_cu = load_segments(args.segments_cublas)
    pwr_cu = crop_power(load_power(args.power_cublas), seg_cu)
    seg_sk = load_segments(args.segments_streamk)
    pwr_sk = crop_power(load_power(args.power_streamk), seg_sk)

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    draw_panel(axes[0], seg_cu, pwr_cu, 'cuBLAS — torch.matmul (cublasGemmEx, BF16xBF16+FP32 accum)')
    draw_panel(axes[1], seg_sk, pwr_sk, 'CUTLASS Stream-K — gemm_streamk (128x128x32, 4-stage, work-stealing)')

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate(rotation=0)

    if args.title:
        fig.suptitle(args.title, fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f'compare_2way: saved {args.out}')


if __name__ == '__main__':
    main()
