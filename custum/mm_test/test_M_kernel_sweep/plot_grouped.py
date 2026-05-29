#!/usr/bin/env python3
"""
Grouped multi-row timeline plot for the M-sweep.

Two grouping modes:

  --by op       For each (model, op) → emit one PNG named
                "<op_short><model_short>_timeline.png" with rows=kernels.
                e.g. down8b_timeline.png, qkv32b_timeline.png

  --by kernel   For each kernel → emit one PNG named
                "<kernel>_timeline.png" with rows=ops.
                e.g. cublas_timeline.png, stream_k_timeline.png

Within each row we keep the v9-style wide time-axis view used by
plot_kernel_timeline.py (Power instant + SM clock + TFLOPS, with cfg-band
annotations).

Tag format:  '<kernel>:<op_short><model_short>_M<M>_w<wavecount>#<burst>'
"""
import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from bisect import bisect_right


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


def split_tag(tag):
    rep = 0
    if '#' in tag:
        tag, rs = tag.rsplit('#', 1)
        try: rep = int(rs)
        except ValueError: pass
    if ':' in tag:
        be, cfg = tag.split(':', 1)
        return be, cfg, rep
    return tag, 'B', rep


_CFG_RE = re.compile(r'^([a-z]+)(8b|32b)_M(\d+)_w(.+)$')


def parse_cfg(cfg):
    """Return (op_short, model_short, M, wave_str) or (None, None, None, None)."""
    m = _CFG_RE.match(cfg)
    if not m:
        return None, None, None, None
    return m.group(1), m.group(2), int(m.group(3)), m.group(4)


def build_time_compressor(segs, pad_seconds=0.3):
    """Build a piecewise-linear datetime → compressed-seconds mapper.

    Intervals are built **per cfg in the order cfgs first appear in segs**
    (so callers control ordering — e.g. sort segs by M to plot M ascending
    even if the data was collected in a different chronological order, as
    when concatenating an early sweep with a later small-M sweep).
    Burst times within a cfg are expanded by ±pad and merged.
    Time gaps between cfgs (other kernels/ops running on the same GPU,
    or a different sweep run entirely) are removed from the x-axis."""
    if not segs:
        return None, 0.0, []
    pad = timedelta(seconds=pad_seconds)
    # Build per-cfg [start, end] in cfg first-appearance order
    seen = {}
    order = []
    for s in segs:
        cfg = s['cfg']
        a = s['t0'] - pad; b = s['t1'] + pad
        if cfg not in seen:
            seen[cfg] = [a, b]; order.append(cfg)
        else:
            if a < seen[cfg][0]: seen[cfg][0] = a
            if b > seen[cfg][1]: seen[cfg][1] = b
    merged = [tuple(seen[c]) for c in order]   # cfg-ordered, NOT time-ordered
    offsets = []
    cum = 0.0
    for a, b in merged:
        offsets.append(cum)
        cum += (b - a).total_seconds()
    total = cum

    def to_compressed(t):
        # Linear scan: intervals are not time-sorted, so bisect doesn't apply.
        # N_cfgs is small (≤ ~15 M values), so O(N) per sample is fine.
        for i, (a, b) in enumerate(merged):
            if a <= t <= b:
                return offsets[i] + (t - a).total_seconds()
        return float('nan')

    return to_compressed, total, merged


def draw_row(ax, ax_r, ax_tf, segs, power, row_label):
    """Render one timeline row. segs = list of burst dicts.

    The x-axis is a *compressed* seconds axis: only periods where this
    row's bursts are active appear on the axis. Times when other kernels
    or other ops ran on the same GPU (which would otherwise bleed power
    samples into this plot) are removed."""
    if not segs:
        ax.text(0.5, 0.5, f'no data', ha='center', va='center',
                transform=ax.transAxes)
        return

    # Group by cfg in order of first appearance
    cfg_order = []
    cfg_bursts = defaultdict(list)
    for s in segs:
        if s['cfg'] not in cfg_bursts:
            cfg_order.append(s['cfg'])
        cfg_bursts[s['cfg']].append(s)

    to_comp, total_comp, merged = build_time_compressor(segs, pad_seconds=0.3)

    # Map every power sample through the compressor. Samples outside any
    # interval return NaN, which breaks the line cleanly. We sort the
    # resulting (x, y) points by x so the polyline draws in axis order
    # (intervals are in cfg order — i.e. M-order — not chronological).
    pts = []
    for p in power:
        x = to_comp(p['ts'])
        if x == x:   # not NaN
            pts.append((x, p['pw'], p['sm']))
    pts.sort(key=lambda r: r[0])
    # Insert NaN breaks where the compressed x jumps by more than a small
    # threshold (cfg boundary), so the polyline doesn't bridge between cfgs.
    xs_p, ys_p, ys_sm = [], [], []
    prev_x = None
    for x, pw, sm in pts:
        if prev_x is not None and (x - prev_x) > 0.5:
            xs_p.append(float('nan')); ys_p.append(float('nan')); ys_sm.append(float('nan'))
        xs_p.append(x); ys_p.append(pw); ys_sm.append(sm)
        prev_x = x

    if xs_p:
        ax.plot(xs_p, ys_p, color='tab:blue', lw=0.7, zorder=3,
                label='Power instant (W)')
        ax_r.plot(xs_p, ys_sm, color='tab:red', lw=0.6, alpha=0.75, zorder=2,
                  label='SM clock (MHz)')

    ax.axhline(600, ls='--', color='tab:blue', alpha=0.6, lw=1.2,
               label='TDP 600 W')
    for lvl, alpha in [(630, 0.40), (660, 0.55), (690, 0.70), (720, 0.85)]:
        ax.axhline(lvl, ls=':', color='red', alpha=alpha, lw=1.0)
    ax_r.axhline(2430, ls='--', color='tab:red', alpha=0.35,
                 label='Boost 2430 MHz')

    tf_xs, tf_ys = [], []
    for s in segs:
        x0 = to_comp(s['t0']); x1 = to_comp(s['t1'])
        if x0 != x0 or x1 != x1:  # NaN check
            continue
        tf_xs += [x0, x1, x1]
        tf_ys += [s['tflops'], s['tflops'], float('nan')]
    ax_tf.plot(tf_xs, tf_ys, color='purple', lw=1.4, zorder=4, label='TFLOPS')

    palette = plt.cm.tab20.colors
    bbox_white = dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor='none', alpha=0.85)
    # Vertical separators between merged active intervals (where time was compressed out)
    for off in [sum((merged[k][1]-merged[k][0]).total_seconds() for k in range(i+1))
                for i in range(len(merged)-1)]:
        ax.axvline(off, color='gray', ls=':', lw=0.6, alpha=0.6, zorder=0)

    for i, cfg in enumerate(cfg_order):
        bursts = cfg_bursts[cfg]
        t0 = bursts[0]['t0']; t1 = bursts[-1]['t1']
        x0 = to_comp(t0); x1 = to_comp(t1)
        if x0 != x0 or x1 != x1:
            continue
        c = palette[i % len(palette)]
        ax.axvspan(x0, x1, alpha=0.12, color=c, zorder=1)
        mid_x = (x0 + x1) / 2

        pw_in = [p['pw'] for p in power if t0 <= p['ts'] <= t1]
        sm_in = [p['sm'] for p in power if t0 <= p['ts'] <= t1]
        mx_w   = max(pw_in) if pw_in else float('nan')
        mn_p10 = min(sm_in) if sm_in else float('nan')
        tfs    = [b['tflops'] for b in bursts]
        mean_tf = sum(tfs) / len(tfs) if tfs else float('nan')

        ax.text(mid_x, 985, cfg, ha='center', va='top', fontsize=8,
                color='black', fontweight='bold', rotation=90, zorder=11)
        ax.text(mid_x, 460, f'max={mx_w:.0f}W', ha='center', va='center',
                fontsize=7, color='darkred', fontweight='bold', rotation=90,
                bbox=bbox_white, zorder=11)
        ax.text(mid_x, 300, f'{mean_tf:.0f} TF', ha='center', va='center',
                fontsize=7, color='purple', fontweight='bold', rotation=90,
                bbox=bbox_white, zorder=11)
        ax.text(mid_x, 140, f'p10={mn_p10:.0f}', ha='center', va='center',
                fontsize=7, color='tab:red', fontweight='bold', rotation=90,
                bbox=bbox_white, zorder=11)

    ax.set_xlim(0, max(total_comp, 1.0))
    ax.set_ylim(0, 1000)
    ax.set_ylabel(f'{row_label}\nPower (W)', color='tab:blue', fontsize=9)
    ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=8)
    ax.grid(True, alpha=0.25)
    ax_r.set_ylim(0, 3000)
    ax_r.set_ylabel('SM clock (MHz)', color='tab:red', fontsize=8)
    ax_r.tick_params(axis='y', labelcolor='tab:red', labelsize=8)
    ax_tf.set_ylim(0, 500)
    ax_tf.set_ylabel('TFLOPS', color='purple', fontsize=8)
    ax_tf.tick_params(axis='y', labelcolor='purple', labelsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=12, integer=False))
    ax.set_xlabel('elapsed time (s, compressed — inter-kernel gaps removed)',
                  fontsize=8)


def render_figure(rows_data, power, out_path, title, row_label_fn):
    """rows_data: list of (row_key, list of segs)."""
    if not rows_data:
        print(f'  [skip] no rows for {out_path}')
        return
    n_rows = len(rows_data)
    n_cfgs_max = max(len({s['cfg'] for s in segs}) for _, segs in rows_data)
    fig_w = max(20, min(36, 1.4 * n_cfgs_max))
    fig_h = 5.5 * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False)
    for row, (key, segs) in enumerate(rows_data):
        ax = axes[row, 0]
        ax_r = ax.twinx()
        ax_tf = ax.twinx()
        ax_tf.spines.right.set_position(('axes', 1.05))
        draw_row(ax, ax_r, ax_tf, segs, power, row_label=row_label_fn(key))
        if row == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax_r.get_legend_handles_labels()
            h3, l3 = ax_tf.get_legend_handles_labels()
            ax.legend(h1+h2+h3, l1+l2+l3,
                      loc='upper left', bbox_to_anchor=(0.0, -0.18),
                      fontsize=8, ncol=8, framealpha=0.9)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.94, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  plot: {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', required=True)
    ap.add_argument('--power',    required=True)
    ap.add_argument('--out-dir',  required=True)
    ap.add_argument('--by',       choices=['op', 'kernel'], required=True)
    ap.add_argument('--tag',      default='')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.segments) as f:
        segs_raw = list(csv.DictReader(f))
    segs = []
    for s in segs_raw:
        be, cfg, rep = split_tag(s['backend'])
        op_s, mdl_s, M, w = parse_cfg(cfg)
        if op_s is None:
            continue   # skip unparseable cfg (e.g. legacy 'M8192_w...' form)
        segs.append({
            'be': be, 'cfg': cfg, 'rep': rep,
            'op_s': op_s, 'mdl_s': mdl_s, 'M': M, 'w': w,
            'K': int(s['K']), 'N': int(s['N']),
            'operator': s['operator'],
            't0': parse_ts(s['t_start']),
            't1': parse_ts(s['t_end']),
            'tflops': float(s['tflops']),
        })
    if not segs:
        print('no parseable segments'); return

    # Sort by (M, t0) so cfgs appear in ascending-M order on the x-axis,
    # even when concatenating sweeps that were collected at different times.
    segs.sort(key=lambda x: (x['M'], x['t0']))
    power = load_power(args.power)

    # Fixed order for kernels (rows when --by op):
    KERNEL_ORDER = ['cublas', 'basicdp', 'stream_k', 'sm80_v3']
    # Fixed order for ops (rows when --by kernel):
    OP_ORDER = ['qkv', 'o', 'gu', 'up', 'down', 'lm']

    if args.by == 'op':
        # Group by (op_short, model_short); within each, rows = kernels.
        groups = defaultdict(list)
        for s in segs:
            groups[(s['op_s'], s['mdl_s'])].append(s)
        for (op_s, mdl_s), grp_segs in groups.items():
            row_groups = defaultdict(list)
            for s in grp_segs:
                row_groups[s['be']].append(s)
            preferred = [b for b in KERNEL_ORDER if b in row_groups]
            extra = sorted([b for b in row_groups if b not in preferred])
            order = preferred + extra
            rows_data = [(b, row_groups[b]) for b in order]
            out_path = os.path.join(args.out_dir,
                                    f'{op_s}{mdl_s}_timeline.png')
            K0 = grp_segs[0]['K']; N0 = grp_segs[0]['N']
            op_full = grp_segs[0]['operator']
            title = (f'M-sweep — {op_s}{mdl_s}  ({op_full}, K={K0}, N={N0})'
                     + (f'  [{args.tag}]' if args.tag else '')
                     + f'   rows = kernels')
            render_figure(rows_data, power, out_path, title,
                          row_label_fn=lambda k: k)

    elif args.by == 'kernel':
        # Group by kernel; within each, rows = (op_short, model_short).
        groups = defaultdict(list)
        for s in segs:
            groups[s['be']].append(s)
        for be, grp_segs in groups.items():
            row_groups = defaultdict(list)
            for s in grp_segs:
                row_groups[(s['op_s'], s['mdl_s'])].append(s)
            # row order: preferred ops × (8b, 32b)
            def opkey(om):
                op, mdl = om
                try: oi = OP_ORDER.index(op)
                except ValueError: oi = 99
                mi = 0 if mdl == '8b' else 1
                return (oi, mi)
            order = sorted(row_groups.keys(), key=opkey)
            rows_data = [(om, row_groups[om]) for om in order]
            # K,N per row (same across all bursts of the same op)
            kn_by_om = {om: (row_groups[om][0]['K'], row_groups[om][0]['N'])
                        for om in order}
            out_path = os.path.join(args.out_dir,
                                    f'{be}_timeline.png')
            title = (f'M-sweep — {be}'
                     + (f'  ({args.tag})' if args.tag else '')
                     + f'   rows = (op, model)  [label: opModel  K=…  N=…]')
            def label(om, _kn=kn_by_om):
                K, N = _kn[om]
                return f'{om[0]}{om[1]}\nK={K}\nN={N}'
            render_figure(rows_data, power, out_path, title,
                          row_label_fn=label)

    print('[plot_grouped] DONE')


if __name__ == '__main__':
    main()
