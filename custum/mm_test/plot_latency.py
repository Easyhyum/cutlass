#!/usr/bin/env python3
"""
Compare two latency CSVs (baseline binary vs wave-sleep binary no-prime).

  python plot_latency.py logs/<tag>_baseline.csv logs/<tag>_wsdisabled.csv --tag <tag>
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    rows = list(csv.DictReader(open(path)))
    return np.array([float(r['ms_avg']) for r in rows])


def stats(a, name):
    return {
        'name': name,
        'n':    len(a),
        'mean': a.mean(),
        'median': np.median(a),
        'std':  a.std(),
        'min':  a.min(),
        'max':  a.max(),
        'p10':  np.percentile(a, 10),
        'p90':  np.percentile(a, 90),
    }


def main():
    base_csv = sys.argv[1]
    ws_csv   = sys.argv[2]
    tag = 'latency'
    if '--tag' in sys.argv:
        tag = sys.argv[sys.argv.index('--tag') + 1]

    base = load(base_csv)
    ws   = load(ws_csv)

    s_b = stats(base, 'baseline (no wave-sleep code)')
    s_w = stats(ws,   's100_step0 (wave-sleep code, gate=false)')

    print(f'\n{"":50s}  {"baseline":>14s}  {"ws-disabled":>14s}  {"delta":>10s}')
    print('-' * 95)
    def line(k, label, fmt='{:.6f}'):
        vb, vw = s_b[k], s_w[k]
        d = vw - vb
        dpct = d / vb * 100 if vb != 0 else 0
        print(f'  {label:48s}  {fmt.format(vb):>14s}  {fmt.format(vw):>14s}  '
              f'{fmt.format(d):>10s}  ({dpct:+.2f}%)')
    line('mean',   'mean ms_avg')
    line('median', 'median ms_avg')
    line('p10',    'p10 ms_avg')
    line('p90',    'p90 ms_avg')
    line('min',    'min ms_avg')
    line('max',    'max ms_avg')
    line('std',    'std ms_avg')
    print(f'  {"n (bursts)":48s}  {s_b["n"]:>14d}  {s_w["n"]:>14d}')

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 7))

    # (1) per-burst line plot
    axs[0].plot(base, marker='.', ms=4, label=f"baseline (no code in SASS)\n"
                f"  mean={s_b['mean']:.4f} ms ± {s_b['std']:.4f}",
                color='C0', lw=0.7)
    axs[0].plot(ws, marker='.', ms=4, label=f"wave-sleep code, gate=false\n"
                f"  mean={s_w['mean']:.4f} ms ± {s_w['std']:.4f}",
                color='C3', lw=0.7)
    axs[0].set_xlabel('burst index')
    axs[0].set_ylabel('ms_avg per kernel')
    axs[0].set_title(f'ms_avg per burst   ({tag})')
    axs[0].legend(loc='best')
    axs[0].grid(alpha=0.3)

    # (2) histogram
    lo = min(base.min(), ws.min())
    hi = max(base.max(), ws.max())
    bins = np.linspace(lo, hi, 60)
    axs[1].hist(base, bins=bins, alpha=0.55, label='baseline', color='C0',
                edgecolor='C0')
    axs[1].hist(ws,   bins=bins, alpha=0.55, label='wave-sleep no-prime',
                color='C3', edgecolor='C3')
    axs[1].axvline(s_b['mean'], color='C0', linestyle='--', lw=1)
    axs[1].axvline(s_w['mean'], color='C3', linestyle='--', lw=1)
    delta_pct = (s_w['mean'] - s_b['mean']) / s_b['mean'] * 100
    axs[1].set_xlabel('ms_avg per kernel')
    axs[1].set_ylabel('count')
    axs[1].set_title(f'Distribution comparison   '
                     f'Δmean = {s_w["mean"]-s_b["mean"]:+.4f} ms '
                     f'({delta_pct:+.2f}% vs baseline)')
    axs[1].legend()
    axs[1].grid(alpha=0.3, axis='y')

    fig.tight_layout()
    out_png = os.path.join(os.path.dirname(base_csv), f'{tag}_compare.png')
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f'\nplot: {out_png}')


if __name__ == '__main__':
    main()
