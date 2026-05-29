#!/usr/bin/env python3
"""
Focused timeline plot: overlay per-config power+clock traces side-by-side.

Each config is plotted in its own column-band, with shared y-axes for direct
visual comparison. Shows:
  - power.draw.instant (W) on left y
  - clocks.current.sm (MHz) on right y (twin)
  - operator group shading + cfg label

Usage:
  python3 plot_focus_timeline.py --segments seg.csv --power pwr.csv --out plot.png
"""
import argparse
import csv
import os
from collections import defaultdict
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
        m = {
            'ts':   idx.get('timestamp'),
            'sm':   idx.get('clocks.current.sm [MHz]'),
            'pw':   idx.get('power.draw.instant [W]'),
        }
        for r in reader:
            if not r: continue
            try:
                ts = parse_ts(r[m['ts']])
            except Exception:
                continue
            def gv(k):
                ci = m[k]
                return strip_unit(r[ci]) if ci is not None else float('nan')
            rows.append({'ts': ts, 'sm': gv('sm'), 'pw': gv('pw')})
    rows.sort(key=lambda x: x['ts'])
    return rows


def load_segments(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def split_cfg(tag):
    if ':' in tag:
        return tag.split(':', 1)
    return tag, 'B'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', required=True)
    ap.add_argument('--power',    required=True)
    ap.add_argument('--out',      required=True)
    args = ap.parse_args()

    segs = load_segments(args.segments)
    if not segs:
        print('no segments'); return
    power = load_power(args.power)

    # group segments by cfg (each cfg has 1 row)
    cfg_segs = []
    for s in segs:
        be, cfg = split_cfg(s['backend'])
        s['be_base'] = be; s['cfg'] = cfg
        s['t0'] = parse_ts(s['t_start']); s['t1'] = parse_ts(s['t_end'])
        cfg_segs.append(s)

    # Multi-panel: one row per cfg, stacked vertically with shared time axis
    # NO - actually different cfgs are at different timestamps.
    # Better: single big plot with all cfgs side-by-side, vertical bands.

    fig, ax = plt.subplots(figsize=(16, 6))
    ax_r = ax.twinx()

    # plot full power+clock trace
    if power:
        ts_all = [p['ts'] for p in power]
        pw_all = [p['pw'] for p in power]
        sm_all = [p['sm'] for p in power]
        ax.plot(ts_all, pw_all, color='tab:blue', lw=0.9, label='Power (W)', zorder=3)
        ax.axhline(600, ls='--', color='tab:blue', alpha=0.4, label='TDP 600W')
        ax.axhline(660, ls='--', color='red', alpha=0.5, label='spike 660W')
        ax_r.plot(ts_all, sm_all, color='tab:red', lw=0.8, label='SM clock (MHz)', alpha=0.7, zorder=2)
        ax_r.axhline(2430, ls='--', color='tab:red', alpha=0.3, label='boost 2430')

    # Shade each cfg's window and label + per-window stats
    palette = plt.cm.tab10.colors
    stats_text = []
    for i, s in enumerate(cfg_segs):
        c = palette[i % len(palette)]
        ax.axvspan(s['t0'], s['t1'], alpha=0.10, color=c, zorder=1)
        mid = s['t0'] + (s['t1'] - s['t0'])/2
        # cfg label at top
        ax.text(mid, 990, s['cfg'], ha='center', va='top',
                fontsize=10, color=c, fontweight='bold', zorder=5)
        # compute per-window stats from power trace
        win_pw = [p['pw'] for p in power if s['t0'] <= p['ts'] <= s['t1']]
        win_sm = [p['sm'] for p in power if s['t0'] <= p['ts'] <= s['t1']]
        if win_pw:
            mx = max(win_pw); mn = sum(win_pw)/len(win_pw)
            sm10 = sorted(win_sm)[max(0, int(0.10*(len(win_sm)-1)))] if win_sm else 0
            stats_text.append(
                f'{s["cfg"]:>14s}: tflops={float(s["tflops"]):6.1f}  '
                f'max={mx:6.1f}W  avg={mn:6.1f}W  sm_p10={sm10:.0f}MHz')

    ax.set_ylim(0, 1000)
    ax.set_ylabel('Power (W)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xlabel('time')
    ax.grid(True, alpha=0.3)
    ax_r.set_ylim(0, 3000)
    ax_r.set_ylabel('SM clock (MHz)', color='tab:red')
    ax_r.tick_params(axis='y', labelcolor='tab:red')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='lower right', fontsize=8)

    if cfg_segs:
        be_op = f'{cfg_segs[0]["be_base"]}  {cfg_segs[0]["operator"]}  M={cfg_segs[0]["M"]}  K={cfg_segs[0]["K"]}  N={cfg_segs[0]["N"]}'
        title = f'Method A focus: {be_op}    (50ms nvidia-smi sampling)'
        fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(args.out, dpi=130)
    print(f'plot: {args.out}')
    print()
    print('=== per-config stats (from 50ms power trace within window) ===')
    for line in stats_text:
        print(line)


if __name__ == '__main__':
    main()
