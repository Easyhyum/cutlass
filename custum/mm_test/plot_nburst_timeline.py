#!/usr/bin/env python3
"""
Plot N-burst RAMP power evaluation results.

Groups bursts by config (strips '#N' suffix from backend tag), shows:
  - Left Y : Power instant (W), 0-1000, with TDP 600W guide
  - Right Y: SM clock (MHz),    0-3000, with boost 2430MHz guide
  - 3rd Y  : TFLOPS per-burst,  0-500, step plot
Each config gets one shaded band labeled at top.

Usage:
  python3 plot_nburst_timeline.py --segments seg.csv --power pwr.csv --out plot.png
"""
import argparse
import csv
from collections import defaultdict
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
        m = {
            'ts':   idx.get('timestamp'),
            'sm':   idx.get('clocks.current.sm [MHz]'),
            'pw':   idx.get('power.draw.instant [W]'),
        }
        for r in reader:
            if not r: continue
            try:
                ts = parse_ts(r[m['ts']])
            except Exception: continue
            def gv(k):
                ci = m[k]
                return strip_unit(r[ci]) if ci is not None else float('nan')
            rows.append({'ts': ts, 'sm': gv('sm'), 'pw': gv('pw')})
    rows.sort(key=lambda x: x['ts'])
    return rows


def split_cfg(tag):
    """'stream_k:s70_p1#3' -> ('stream_k', 's70_p1', 3)"""
    rep = 0
    if '#' in tag:
        tag, rs = tag.rsplit('#', 1)
        try: rep = int(rs)
        except ValueError: pass
    if ':' in tag:
        be, cfg = tag.split(':', 1)
        return be, cfg, rep
    return tag, 'B', rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', required=True)
    ap.add_argument('--power',    required=True)
    ap.add_argument('--out',      required=True)
    ap.add_argument('--title',    default=None)
    args = ap.parse_args()

    with open(args.segments) as f:
        segs = list(csv.DictReader(f))
    for s in segs:
        be, cfg, rep = split_cfg(s['backend'])
        s['be'] = be; s['cfg'] = cfg; s['rep'] = rep
        s['t0'] = parse_ts(s['t_start'])
        s['t1'] = parse_ts(s['t_end'])
        s['tflops'] = float(s['tflops'])
    if not segs:
        print('no segments'); return

    # Group by cfg, preserve order
    cfg_order = []
    cfg_bursts = defaultdict(list)
    for s in segs:
        if s['cfg'] not in cfg_bursts:
            cfg_order.append(s['cfg'])
        cfg_bursts[s['cfg']].append(s)

    power = load_power(args.power)

    # Use a fixed aspect — 22"×9" works well for 28 configs in a single timeline
    fig_w = max(18, min(30, 0.85 * len(cfg_order)))
    fig, ax = plt.subplots(figsize=(fig_w, 9))
    ax_r = ax.twinx()
    ax_tf = ax.twinx()
    ax_tf.spines.right.set_position(('axes', 1.04))

    # Plot continuous power+clock traces over the full window
    if power:
        t_first = segs[0]['t0']
        t_last  = segs[-1]['t1']
        from datetime import timedelta
        pad = timedelta(seconds=1)
        pw_window = [p for p in power if (t_first - pad) <= p['ts'] <= (t_last + pad)]
        if pw_window:
            ts_ = [p['ts'] for p in pw_window]
            pw_ = [p['pw'] for p in pw_window]
            sm_ = [p['sm'] for p in pw_window]
            ax.plot(ts_, pw_, color='tab:blue', lw=0.7, label='Power instant (W)',
                    zorder=3)
            ax_r.plot(ts_, sm_, color='tab:red', lw=0.6,
                      label='SM clock (MHz)', alpha=0.75, zorder=2)

    # Guide lines — power
    ax.axhline(600, ls='--', color='tab:blue', alpha=0.6,  lw=1.2,
               label='TDP 600 W')
    # spike thresholds with progressively-darker red
    spike_levels = [(630, 0.40), (660, 0.55), (690, 0.70), (720, 0.85)]
    for lvl, alpha in spike_levels:
        ax.axhline(lvl, ls=':', color='red', alpha=alpha, lw=1.0,
                   label=f'spike {lvl} W')
    ax_r.axhline(2430, ls='--', color='tab:red', alpha=0.35,
                 label='Boost 2430 MHz')

    # TFLOPS step plot per burst
    tf_xs = []; tf_ys = []
    for s in segs:
        tf_xs += [s['t0'], s['t1'], s['t1']]
        tf_ys += [s['tflops'], s['tflops'], float('nan')]
    ax_tf.plot(tf_xs, tf_ys, color='purple', lw=1.5, label='TFLOPS', zorder=4)

    # Shade per-config groups + annotate stats
    # Layout (stat labels 아래부터 중간까지 분산, 흰색 bbox로 분리):
    #    y=985  cfg name (BLACK BOLD, top)
    #    y=460  max=XW   (middle, darkred bold)
    #    y=300  XTF       (middle-low, purple bold)
    #    y=140  p10=X    (bottom-low, red bold)
    palette = plt.cm.tab20.colors
    from datetime import timedelta
    bbox_white = dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor='none', alpha=0.82)
    for i, cfg in enumerate(cfg_order):
        bursts = cfg_bursts[cfg]
        t0 = bursts[0]['t0']; t1 = bursts[-1]['t1']
        c = palette[i % len(palette)]
        ax.axvspan(t0, t1, alpha=0.12, color=c, zorder=1)
        mid = t0 + (t1 - t0) / 2
        # per-cfg stats
        pw_in = [p['pw'] for p in power if t0 <= p['ts'] <= t1]
        sm_in = [p['sm'] for p in power if t0 <= p['ts'] <= t1]
        mx_w  = max(pw_in) if pw_in else float('nan')
        mn_p10 = min(sm_in) if sm_in else float('nan')
        tfs = [b['tflops'] for b in bursts]
        mean_tf = sum(tfs)/len(tfs)

        # TOP — cfg name (BLACK BOLD, top)
        ax.text(mid, 985, cfg, ha='center', va='top',
                fontsize=10, color='black', fontweight='bold',
                rotation=90, zorder=11)

        # Stat labels — 아래(p10) → 중간(max), spacing 160W
        ax.text(mid, 460, f'max={mx_w:.0f}W',
                ha='center', va='center', fontsize=8,
                color='darkred', fontweight='bold', rotation=90,
                bbox=bbox_white, zorder=11)
        ax.text(mid, 300, f'{mean_tf:.0f} TF',
                ha='center', va='center', fontsize=8,
                color='purple', fontweight='bold', rotation=90,
                bbox=bbox_white, zorder=11)
        ax.text(mid, 140, f'p10={mn_p10:.0f}',
                ha='center', va='center', fontsize=8,
                color='tab:red', fontweight='bold', rotation=90,
                bbox=bbox_white, zorder=11)

    # Axes formatting
    ax.set_ylim(0, 1000)
    ax.set_ylabel('Power (W)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.grid(True, alpha=0.25)
    ax_r.set_ylim(0, 3000)
    ax_r.set_ylabel('SM clock (MHz)', color='tab:red')
    ax_r.tick_params(axis='y', labelcolor='tab:red')
    ax_tf.set_ylim(0, 500)
    ax_tf.set_ylabel('TFLOPS', color='purple')
    ax_tf.tick_params(axis='y', labelcolor='purple')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate(rotation=0)

    # Legend — 데이터 영역 밖 (axes 위쪽 outside)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    h3, l3 = ax_tf.get_legend_handles_labels()
    ax.legend(h1+h2+h3, l1+l2+l3,
              loc='upper left', bbox_to_anchor=(0.0, -0.05),
              fontsize=8, ncol=8, framealpha=0.9)

    title = args.title or (
        f'RAMP one-shot eval — {len(cfg_order)} configs × '
        f'{len(cfg_bursts[cfg_order[0]])} bursts/cfg')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.94, 0.96])
    fig.savefig(args.out, dpi=110)
    print(f'plot: {args.out}')


if __name__ == '__main__':
    main()
