#!/usr/bin/env python3
"""
Two-panel time-series plot for the Qwen3-8B GEMM sweep.

Top panel
  left  y-axis (0 ~ 1000 W) : power.draw.instant  + dashed reference at 600 W
  right y-axis (0 ~ 3000 MHz): clocks.current.sm  + dashed reference at 2430 MHz
  3rd  y-axis (0 ~ 500 TF) : per-segment TFLOPS (step plot, purple)

Bottom panel
  left  y-axis : temperature.gpu (deg C)
  right y-axis : utilization.gpu (%)

Each segment from segments.csv is overlaid as a light vertical band, and
operator groups are labeled along the top. The plot title and the output
filename are tagged with the backend (cublas | cutlass_sm80 | stream_k).

Usage:
  python3 plot_power.py <segments.csv> <power.csv> --out plot.png \
      [--backend cublas|cutlass_sm80|stream_k] [--title "extra title text"]
"""

import argparse
import csv
import os
from datetime import datetime

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
            'ts':   'timestamp',
            'sm':   'clocks.current.sm [MHz]',
            'pw':   'power.draw.instant [W]',
            'temp': 'temperature.gpu',
            'util': 'utilization.gpu [%]',
        }
        cmap = {k: idx.get(v) for k, v in cols.items()}
        for r in reader:
            if not r:
                continue
            try:
                ts = parse_ts(r[cmap['ts']])
            except Exception:
                continue
            def gv(key):
                ci = cmap[key]
                return strip_unit(r[ci]) if ci is not None else float('nan')
            rows.append({
                'ts': ts, 'sm': gv('sm'), 'pw': gv('pw'),
                'temp': gv('temp'), 'util': gv('util'),
            })
    rows.sort(key=lambda x: x['ts'])
    return rows


def load_segments(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('segments_csv')
    ap.add_argument('power_csv')
    ap.add_argument('--out', required=True, help='output png path')
    ap.add_argument('--backend', default='',
                    help='backend label for title and filename')
    ap.add_argument('--title', default='', help='extra title text')
    args = ap.parse_args()

    power = load_power(args.power_csv)
    segs  = load_segments(args.segments_csv)

    if not power:
        print('plot_power: no power samples', flush=True)
        return

    # Crop power samples to the sweep time range (skip leading idle).
    if segs:
        t_first = parse_ts(segs[0]['t_start'])
        t_last  = parse_ts(segs[-1]['t_end'])
        from datetime import timedelta
        pad = timedelta(seconds=1)
        power = [p for p in power if (t_first - pad) <= p['ts'] <= (t_last + pad)]

    if not power:
        print('plot_power: no power samples in sweep range', flush=True)
        return

    ts   = [p['ts']   for p in power]
    pw   = [p['pw']   for p in power]
    sm   = [p['sm']   for p in power]
    temp = [p['temp'] for p in power]
    util = [p['util'] for p in power]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={'height_ratios': [1.4, 1.0]})

    # ---- TOP: power + sm clock ----
    color_pw = 'tab:blue'
    color_sm = 'tab:red'
    ax_top.plot(ts, pw, color=color_pw, linewidth=1.2,
                label='power.draw.instant (W)')
    ax_top.axhline(600, linestyle='--', color=color_pw, alpha=0.45,
                   label='TDP 600 W')
    ax_top.set_ylim(0, 1000)
    ax_top.set_ylabel('Power (W)', color=color_pw)
    ax_top.tick_params(axis='y', labelcolor=color_pw)
    ax_top.grid(True, alpha=0.25)

    ax_top_r = ax_top.twinx()
    ax_top_r.plot(ts, sm, color=color_sm, linewidth=1.2,
                  label='clocks.current.sm (MHz)')
    ax_top_r.axhline(2430, linestyle='--', color=color_sm, alpha=0.45,
                     label='Boost 2430 MHz')
    ax_top_r.set_ylim(0, 3000)
    ax_top_r.set_ylabel('SM clock (MHz)', color=color_sm)
    ax_top_r.tick_params(axis='y', labelcolor=color_sm)

    # 3rd axis: per-segment TFLOPS (step plot, offset outward)
    color_tf = 'purple'
    ax_top_tf = ax_top.twinx()
    ax_top_tf.spines.right.set_position(('axes', 1.06))
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
        ax_top_tf.plot(seg_xs, seg_ys, color=color_tf, linewidth=2.2,
                       label='TFLOPS', zorder=5)
    ax_top_tf.set_ylim(0, 500)
    ax_top_tf.set_ylabel('TFLOPS', color=color_tf)
    ax_top_tf.tick_params(axis='y', labelcolor=color_tf)

    # Combined legend for top panel
    l1, lab1 = ax_top.get_legend_handles_labels()
    l2, lab2 = ax_top_r.get_legend_handles_labels()
    l3, lab3 = ax_top_tf.get_legend_handles_labels()
    ax_top.legend(l1 + l2 + l3, lab1 + lab2 + lab3,
                  loc='upper right', fontsize=8, ncol=2, framealpha=0.85)

    # ---- BOTTOM: temperature + utilization ----
    color_t = 'tab:orange'
    color_u = 'tab:green'
    ax_bot.plot(ts, temp, color=color_t, linewidth=1.2,
                label='temperature.gpu (C)')
    ax_bot.set_ylabel('Temperature (C)', color=color_t)
    ax_bot.tick_params(axis='y', labelcolor=color_t)
    ax_bot.set_ylim(20, 100)
    ax_bot.grid(True, alpha=0.25)

    ax_bot_r = ax_bot.twinx()
    ax_bot_r.plot(ts, util, color=color_u, linewidth=1.2,
                  label='utilization.gpu (%)')
    ax_bot_r.set_ylabel('Utilization (%)', color=color_u)
    ax_bot_r.tick_params(axis='y', labelcolor=color_u)
    ax_bot_r.set_ylim(0, 105)

    l3, lab3 = ax_bot.get_legend_handles_labels()
    l4, lab4 = ax_bot_r.get_legend_handles_labels()
    ax_bot.legend(l3 + l4, lab3 + lab4, loc='upper right', fontsize=8)

    # ---- Overlay operator groups ----
    if segs:
        # Build (op, K, N, group_start, group_end) tuples by walking segments
        op_groups = []
        cur = None
        for s in segs:
            t0 = parse_ts(s['t_start'])
            t1 = parse_ts(s['t_end'])
            if cur is None or s['operator'] != cur['op']:
                if cur is not None:
                    op_groups.append(cur)
                cur = dict(op=s['operator'],
                           K=int(s['K']), N=int(s['N']),
                           t0=t0, t1=t1)
            cur['t1'] = t1
        if cur is not None:
            op_groups.append(cur)

        palette = plt.cm.tab10.colors
        for i, g in enumerate(op_groups):
            c = palette[i % len(palette)]
            ax_top.axvspan(g['t0'], g['t1'], alpha=0.07, color=c)
            ax_bot.axvspan(g['t0'], g['t1'], alpha=0.07, color=c)
            mid = g['t0'] + (g['t1'] - g['t0']) / 2
            label = f"{g['op']}\nK={g['K']}, N={g['N']}"
            ax_top.text(mid, 975, label, ha='center', va='top',
                        fontsize=9, color=c, fontweight='bold', zorder=6)

    # ---- Cosmetics ----
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    backend = args.backend or 'unknown'
    title_parts = [f'Qwen3-8B GEMM sweep — backend={backend}']
    if args.title:
        title_parts.append(args.title)
    fig.suptitle('  |  '.join(title_parts), fontsize=13)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f'plot_power: saved {args.out}')


if __name__ == '__main__':
    main()
