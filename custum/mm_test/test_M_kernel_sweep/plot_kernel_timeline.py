#!/usr/bin/env python3
"""
Multi-row timeline plot for a cross-kernel M-sweep.

Each backend ('cublas', 'stream_k', ...) becomes its own row in the figure;
within each row we plot the wide time-axis view used in v9_n100_gap600_*
(Power instant + SM clock + TFLOPS), with cfg-name + max_W/p10/mean_TF
annotations per config band.

Tag format expected:  '<backend>:<cfg>#<burst_idx>'
   <cfg> typically 'M<M>_w<wavecount>' (wave count may be 'NaN' for cuBLAS).

Usage:
  python3 plot_kernel_timeline.py --segments seg.csv --power pwr.csv --out plot.png [--title T]
"""
import argparse
import csv
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
            'ts': idx.get('timestamp'),
            'sm': idx.get('clocks.current.sm [MHz]'),
            'pw': idx.get('power.draw.instant [W]'),
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


def split_cfg(tag):
    rep = 0
    if '#' in tag:
        tag, rs = tag.rsplit('#', 1)
        try: rep = int(rs)
        except ValueError: pass
    if ':' in tag:
        be, cfg = tag.split(':', 1)
        return be, cfg, rep
    return tag, 'B', rep


def draw_row(ax, ax_r, ax_tf, segs, power, kernel_label):
    """Render one timeline row for one backend."""
    if not segs:
        ax.text(0.5, 0.5, f'no data for {kernel_label}',
                ha='center', va='center', transform=ax.transAxes)
        return

    # Group by cfg (preserve order of first appearance)
    cfg_order = []
    cfg_bursts = defaultdict(list)
    for s in segs:
        if s['cfg'] not in cfg_bursts:
            cfg_order.append(s['cfg'])
        cfg_bursts[s['cfg']].append(s)

    t_first = segs[0]['t0']
    t_last  = segs[-1]['t1']
    pad = timedelta(seconds=1)
    pw_window = [p for p in power if (t_first - pad) <= p['ts'] <= (t_last + pad)]
    if pw_window:
        ts_ = [p['ts'] for p in pw_window]
        pw_ = [p['pw'] for p in pw_window]
        sm_ = [p['sm'] for p in pw_window]
        ax.plot(ts_, pw_, color='tab:blue', lw=0.7, zorder=3,
                label='Power instant (W)')
        ax_r.plot(ts_, sm_, color='tab:red', lw=0.6, alpha=0.75, zorder=2,
                  label='SM clock (MHz)')

    # Power guides
    ax.axhline(600, ls='--', color='tab:blue', alpha=0.6, lw=1.2,
               label='TDP 600 W')
    for lvl, alpha in [(630, 0.40), (660, 0.55), (690, 0.70), (720, 0.85)]:
        ax.axhline(lvl, ls=':', color='red', alpha=alpha, lw=1.0)
    ax_r.axhline(2430, ls='--', color='tab:red', alpha=0.35,
                 label='Boost 2430 MHz')

    # TFLOPS step plot
    tf_xs, tf_ys = [], []
    for s in segs:
        tf_xs += [s['t0'], s['t1'], s['t1']]
        tf_ys += [s['tflops'], s['tflops'], float('nan')]
    ax_tf.plot(tf_xs, tf_ys, color='purple', lw=1.4, zorder=4, label='TFLOPS')

    # Shading + cfg labels
    palette = plt.cm.tab20.colors
    bbox_white = dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor='none', alpha=0.85)
    for i, cfg in enumerate(cfg_order):
        bursts = cfg_bursts[cfg]
        t0 = bursts[0]['t0']; t1 = bursts[-1]['t1']
        c = palette[i % len(palette)]
        ax.axvspan(t0, t1, alpha=0.12, color=c, zorder=1)
        mid = t0 + (t1 - t0) / 2

        pw_in = [p['pw'] for p in power if t0 <= p['ts'] <= t1]
        sm_in = [p['sm'] for p in power if t0 <= p['ts'] <= t1]
        mx_w   = max(pw_in) if pw_in else float('nan')
        mn_p10 = min(sm_in) if sm_in else float('nan')
        tfs    = [b['tflops'] for b in bursts]
        mean_tf = sum(tfs) / len(tfs) if tfs else float('nan')

        ax.text(mid, 985, cfg, ha='center', va='top',
                fontsize=8, color='black', fontweight='bold',
                rotation=90, zorder=11)
        ax.text(mid, 460, f'max={mx_w:.0f}W',
                ha='center', va='center', fontsize=7,
                color='darkred', fontweight='bold', rotation=90,
                bbox=bbox_white, zorder=11)
        ax.text(mid, 300, f'{mean_tf:.0f} TF',
                ha='center', va='center', fontsize=7,
                color='purple', fontweight='bold', rotation=90,
                bbox=bbox_white, zorder=11)
        ax.text(mid, 140, f'p10={mn_p10:.0f}',
                ha='center', va='center', fontsize=7,
                color='tab:red', fontweight='bold', rotation=90,
                bbox=bbox_white, zorder=11)

    # Row axes setup
    ax.set_ylim(0, 1000)
    ax.set_ylabel(f'{kernel_label}\nPower (W)', color='tab:blue', fontsize=9)
    ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=8)
    ax.grid(True, alpha=0.25)
    ax_r.set_ylim(0, 3000)
    ax_r.set_ylabel('SM clock (MHz)', color='tab:red', fontsize=8)
    ax_r.tick_params(axis='y', labelcolor='tab:red', labelsize=8)
    ax_tf.set_ylim(0, 500)
    ax_tf.set_ylabel('TFLOPS', color='purple', fontsize=8)
    ax_tf.tick_params(axis='y', labelcolor='purple', labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))


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

    power = load_power(args.power)

    # Group by backend
    be_order = []
    by_be = defaultdict(list)
    for s in segs:
        if s['be'] not in by_be:
            be_order.append(s['be'])
        by_be[s['be']].append(s)

    # Fixed order: cublas → basicdp → stream_k → sm80_v3 → others (alphabetical).
    preferred = ['cublas', 'basicdp', 'stream_k', 'sm80_v3']
    be_sorted = [b for b in preferred if b in by_be] + \
                [b for b in be_order if b not in preferred]

    n_rows = len(be_sorted)
    # rough width — each cfg contributes ~0.9 inches
    n_cfgs_max = max(len({s['cfg'] for s in by_be[b]}) for b in be_sorted)
    fig_w = max(20, min(36, 1.4 * n_cfgs_max))
    fig_h = 6.0 * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False)

    for row, be in enumerate(be_sorted):
        ax = axes[row, 0]
        ax_r = ax.twinx()
        ax_tf = ax.twinx()
        ax_tf.spines.right.set_position(('axes', 1.05))
        draw_row(ax, ax_r, ax_tf, by_be[be], power, kernel_label=be)
        # Legend only on first row
        if row == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_r.get_legend_handles_labels()
            h3, l3 = ax_tf.get_legend_handles_labels()
            ax.legend(h1+h2+h3, l1+l2+l3,
                      loc='upper left', bbox_to_anchor=(0.0, -0.18),
                      fontsize=8, ncol=8, framealpha=0.9)

    title = args.title or f'M-sweep — {", ".join(be_sorted)}'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.94, 0.97])
    fig.savefig(args.out, dpi=110, bbox_inches='tight')
    print(f'plot: {args.out}')


if __name__ == '__main__':
    main()
