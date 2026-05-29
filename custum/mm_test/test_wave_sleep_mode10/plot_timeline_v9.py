#!/usr/bin/env python3
"""
v9-style 3-axis wall-time timeline for mode 10 sweeps.

Shows power, SM clock, and TFLOPS all in one plot (matching
ws_044609_timeline.png / v9_n100_gap600_timeline_v5.png format):
  Left  Y : Power instant (W), 0–1000, with TDP 600W + spike guides
  Right Y : SM clock (MHz), 0–3000, with boost 2430 MHz guide
  3rd   Y : TFLOPS per-burst (step plot), 0–500

Auto-detects 2D (segments.csv with sleep_pct, sleep_ns + phase) vs.
4D (segments_4d.csv with first_pct, first_ns, mid_pct, mid_ns) input.

Usage:
  IN_DIR=logs/<TAG>  python3 plot_timeline_v9.py
or
  python3 plot_timeline_v9.py --segments <path> --power <path> --out <png>
"""
import os, sys, argparse, csv
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ─── data loading ────────────────────────────────────────────────────────────
def parse_ts(s):
    return datetime.strptime(s.strip(), '%Y/%m/%d %H:%M:%S.%f')


def strip_unit(s):
    return float(s.strip().split()[0])


def load_power(path):
    rows = []
    with open(path) as f:
        rdr = csv.reader(f)
        hdr = [c.strip() for c in next(rdr)]
        idx = {h: i for i, h in enumerate(hdr)}
        ts_i = idx.get('timestamp')
        sm_i = idx.get('clocks.current.sm [MHz]')
        pw_i = idx.get('power.draw.instant [W]')
        for r in rdr:
            if not r:
                continue
            try:
                ts = parse_ts(r[ts_i])
            except Exception:
                continue
            try:
                sm = strip_unit(r[sm_i]) if sm_i is not None else float('nan')
                pw = strip_unit(r[pw_i]) if pw_i is not None else float('nan')
            except Exception:
                continue
            rows.append({'ts': ts, 'sm': sm, 'pw': pw})
    rows.sort(key=lambda x: x['ts'])
    return rows


def load_segments(path):
    """Auto-detect mode10 4D / mode10 2D / chunked formats; add 'cfg' label.

    Also computes per-cfg baseline TFLOPS so each non-baseline cfg can be
    labelled with its TF / baseline_TF ratio (% of baseline).
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], 'none'
    sample = rows[0]
    if 'first_pct' in sample:
        fmt = '4d'
    elif 'sleep_pct' in sample:
        fmt = '2d'
    elif 'chunk_m' in sample:
        fmt = 'chunk'
    else:
        fmt = 'unknown'

    # ── Pre-compute baseline TF per group ────────────────────────────────────
    # chunk : key = (kernel, op, model, M)   → BASE row has chunk_m == 0
    # 2d    : key = (kernel, phase)          → BASE row has sleep_pct == 0
    # 4d    : key = (kernel,)                → BASE row has first_pct == 0 and mid_pct == 0
    from collections import defaultdict
    base_tfs = defaultdict(list)
    def base_key(s):
        if fmt == 'chunk':
            return ('chunk', s['kernel'], s['op'], s['model'], s['M'])
        if fmt == '2d':
            return ('2d', s['kernel'], s.get('phase', ''))
        if fmt == '4d':
            return ('4d', s['kernel'])
        return None
    def is_base(s):
        if fmt == 'chunk':
            return int(s.get('chunk_m', '0')) == 0
        if fmt == '2d':
            return int(s.get('sleep_pct', '0')) == 0 and int(s.get('sleep_ns', '0')) == 0
        if fmt == '4d':
            return int(s.get('first_pct', '0')) == 0 and int(s.get('mid_pct', '0')) == 0
        return False
    for s in rows:
        if is_base(s):
            k = base_key(s)
            if k is not None:
                base_tfs[k].append(float(s['tflops_obs']))
    base_med = {}
    for k, vs in base_tfs.items():
        vs_sorted = sorted(vs)
        base_med[k] = vs_sorted[len(vs_sorted) // 2]

    for s in rows:
        s['t0'] = datetime.fromtimestamp(int(s['t_start_ns']) / 1e9)
        s['t1'] = datetime.fromtimestamp(int(s['t_end_ns']) / 1e9)
        s['tflops'] = float(s['tflops_obs'])
        if is_base(s):
            s['base_tf'] = None   # don't show 100% on the baseline itself
        else:
            k = base_key(s)
            s['base_tf'] = base_med.get(k)
        if fmt == '4d':
            fp = int(s['first_pct']); fns = int(s['first_ns'])
            mp = int(s['mid_pct']);   mns = int(s['mid_ns'])
            s['cfg'] = ('BASELINE' if (fp == 0 and mp == 0)
                        else f'f{fp}_{fns}_m{mp}_{mns}')
        elif fmt == '2d':
            ph = s.get('phase', '?')
            pct = int(s['sleep_pct']); ns_ = int(s['sleep_ns'])
            s['cfg'] = (f'{ph}_BASE' if (pct == 0 and ns_ == 0)
                        else f'{ph}_p{pct}_n{ns_}')
        elif fmt == 'chunk':
            op    = s.get('op', '?')
            model = s.get('model', '?').replace('qwen3-', '')
            M  = s.get('M', '?'); cm = s.get('chunk_m', '?')
            extra = s.get('chunk_idle_us', s.get('chunk_gap_us', '0'))
            mode  = s.get('mode', '')
            if mode and mode not in ('', 'seq'):
                s['cfg'] = f'{op}{model}_M{M}_cm{cm}_{mode}_x{extra}'
            elif mode == 'seq' and cm != '0':
                s['cfg'] = f'{op}{model}_M{M}_cm{cm}_seq_x{extra}'
            else:
                # BASE (cm=0) or legacy CSV without mode column
                s['cfg'] = f'{op}{model}_M{M}_cm{cm}_x{extra}'
        else:
            s['cfg'] = s.get('cfg_name', 'cfg')
    return rows, fmt


# ─── plot ────────────────────────────────────────────────────────────────────
def plot_one_kernel(rows, power, kernel, out_path, fmt):
    rows = [r for r in rows if r.get('kernel') == kernel]
    if not rows:
        print(f'[plot] no rows for kernel={kernel}'); return
    # Group rows by cfg, preserving wall-time order
    rows.sort(key=lambda r: r['t0'])
    cfg_order, cfg_bursts = [], defaultdict(list)
    for r in rows:
        if r['cfg'] not in cfg_bursts:
            cfg_order.append(r['cfg'])
        cfg_bursts[r['cfg']].append(r)

    fig_w = max(20, min(36, 0.55 * len(cfg_order) + 5))
    fig, ax = plt.subplots(figsize=(fig_w, 9))
    ax_r = ax.twinx()
    ax_tf = ax.twinx()
    ax_tf.spines.right.set_position(('axes', 1.04))

    # continuous traces
    t_first = rows[0]['t0']; t_last = rows[-1]['t1']
    pad = timedelta(seconds=1)
    pw_in = [p for p in power if (t_first - pad) <= p['ts'] <= (t_last + pad)]
    if pw_in:
        ts_ = [p['ts'] for p in pw_in]
        ax.plot(ts_, [p['pw'] for p in pw_in], color='tab:blue', lw=0.7,
                label='Power instant (W)', zorder=3)
        ax_r.plot(ts_, [p['sm'] for p in pw_in], color='tab:red', lw=0.6,
                  label='SM clock (MHz)', alpha=0.75, zorder=2)

    # guides
    ax.axhline(600, ls='--', color='tab:blue', alpha=0.6, lw=1.2, label='TDP 600 W')
    for lvl, alpha in [(630, 0.40), (660, 0.55), (690, 0.70), (720, 0.85)]:
        ax.axhline(lvl, ls=':', color='red', alpha=alpha, lw=1.0)
    ax_r.axhline(2430, ls='--', color='tab:red', alpha=0.35,
                 label='Boost 2430 MHz')

    # TFLOPS step
    tf_xs, tf_ys = [], []
    for r in rows:
        tf_xs += [r['t0'], r['t1'], r['t1']]
        tf_ys += [r['tflops'], r['tflops'], float('nan')]
    ax_tf.plot(tf_xs, tf_ys, color='purple', lw=1.5, label='TFLOPS', zorder=4)

    # per-cfg bands + stat labels
    palette = plt.cm.tab20.colors
    bbox_w = dict(boxstyle='round,pad=0.18', facecolor='white',
                  edgecolor='none', alpha=0.82)
    for i, cfg in enumerate(cfg_order):
        bs = cfg_bursts[cfg]
        t0 = bs[0]['t0']; t1 = bs[-1]['t1']
        c = palette[i % len(palette)]
        ax.axvspan(t0, t1, alpha=0.12, color=c, zorder=1)
        mid = t0 + (t1 - t0) / 2
        pw_w = [p['pw'] for p in power if t0 <= p['ts'] <= t1]
        sm_w = [p['sm'] for p in power if t0 <= p['ts'] <= t1]
        mx_w  = max(pw_w) if pw_w else float('nan')
        p10_s = min(sm_w) if sm_w else float('nan')
        mean_tf = sum(b['tflops'] for b in bs) / len(bs)

        # TF ratio vs baseline (BASE/cm0 of same kernel/op/model/M, etc.)
        # appended in parens to the TF label so it reads "366TF (94%)"
        base_tf = bs[0].get('base_tf')
        if base_tf and base_tf > 0:
            ratio_pct = mean_tf / base_tf * 100.0
            tf_label = f'{mean_tf:.0f}TF ({ratio_pct:.0f}%)'
        else:
            tf_label = f'{mean_tf:.0f}TF'

        ax.text(mid, 985, cfg, ha='center', va='top', fontsize=9,
                color='black', fontweight='bold', rotation=90, zorder=11)
        ax.text(mid, 460, f'max={mx_w:.0f}W',
                ha='center', va='center', fontsize=7.5,
                color='darkred', fontweight='bold', rotation=90,
                bbox=bbox_w, zorder=11)
        ax.text(mid, 300, tf_label,
                ha='center', va='center', fontsize=7.5,
                color='purple', fontweight='bold', rotation=90,
                bbox=bbox_w, zorder=11)
        ax.text(mid, 140, f'p10={p10_s:.0f}',
                ha='center', va='center', fontsize=7.5,
                color='tab:red', fontweight='bold', rotation=90,
                bbox=bbox_w, zorder=11)

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

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    h3, l3 = ax_tf.get_legend_handles_labels()
    fig.legend(h1 + h2 + h3, l1 + l2 + l3,
               loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=6, fontsize=9, frameon=False)

    fig.suptitle(f'Mode 10 wall-time timeline — {kernel} ({fmt})', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', default=None)
    ap.add_argument('--power',    default=None)
    ap.add_argument('--out',      default=None)
    args = ap.parse_args()

    in_dir = os.environ.get('IN_DIR', os.getcwd())
    out_dir = os.environ.get('OUT_DIR', in_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.segments:
        seg_path = args.segments
    else:
        for cand in ('segments_4d.csv', 'segments.csv'):
            p = os.path.join(in_dir, cand)
            if os.path.exists(p):
                seg_path = p
                break
        else:
            print(f'[plot] no segments CSV in {in_dir}'); sys.exit(1)

    pwr_path = args.power or os.path.join(in_dir, 'gpu0_power.csv')
    rows, fmt = load_segments(seg_path)
    if not rows:
        print('[plot] empty segments'); sys.exit(1)
    power = load_power(pwr_path) if os.path.exists(pwr_path) else []

    kernels = sorted({r['kernel'] for r in rows})
    for k in kernels:
        out = args.out or os.path.join(out_dir,
            f'timeline_v9_{fmt}_{k}.png')
        plot_one_kernel(rows, power, k, out, fmt)


if __name__ == '__main__':
    main()
