#!/usr/bin/env python3
"""
Scatter plot — per-CTA mainloop duration vs CTA index, one panel per M.

Reads  cta_probe_per_cta_<model>_<kernel>.csv  (made by run_cta_probe.py).

Usage:
  python plot_cta_dur_vs_idx.py                          # default model+kernel
  python plot_cta_dur_vs_idx.py --kernel streamk --model qwen3-8b
  python plot_cta_dur_vs_idx.py --csv path/to/file.csv --out out.png

Color = wave_idx; vertical red dashed lines = wave boundaries (k * n_sm).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',    help='per-CTA CSV path (overrides --model/--kernel)')
    ap.add_argument('--model',  default='qwen3-8b')
    ap.add_argument('--kernel', default='streamk', choices=['streamk', 'basicdp'])
    ap.add_argument('--n_sm',   type=int, default=188,
                    help='SMs per wave used to draw boundary lines (default 188)')
    ap.add_argument('--out',    help='output PNG (default = sibling of CSV)')
    ap.add_argument('--max_wave_lines', type=int, default=12,
                    help='draw red wave-boundary lines only when waves <= this')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(
            here, f'cta_probe_per_cta_{args.model}_{args.kernel}.csv')
    if not os.path.exists(csv_path):
        raise SystemExit(f'CSV not found: {csv_path}')
    df = pd.read_csv(csv_path)

    # Tolerate older column name from earlier runs
    if 'cta_idx' not in df.columns and 'cta_launch_idx' in df.columns:
        df = df.rename(columns={'cta_launch_idx': 'cta_idx'})

    if 'cta_dur_ns' not in df.columns:
        df['cta_dur_ns'] = df['end_ns'].astype('int64') - df['start_ns'].astype('int64')
        df.loc[df['cta_dur_ns'] < 0, 'cta_dur_ns'] = 0

    Ms = sorted(df['M'].unique())
    print(f'[plot] csv={csv_path}  M={Ms}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(Ms), 1,
                             figsize=(13, 2.6 * len(Ms)), squeeze=False)
    for ax, M in zip(axes[:, 0], Ms):
        sub = df[df['M'] == M].sort_values('cta_idx')
        x  = sub['cta_idx'].to_numpy()
        y  = sub['cta_dur_ns'].to_numpy() / 1000.0  # us
        wv = sub['wave_idx'].to_numpy()
        sc = ax.scatter(x, y, c=wv, cmap='viridis', s=6, alpha=0.7,
                        edgecolors='none')
        n_waves = int(wv.max()) + 1
        if n_waves <= args.max_wave_lines:
            for w in range(1, n_waves):
                ax.axvline(w * args.n_sm, color='red', linestyle='--',
                           lw=0.4, alpha=0.5)
        ax.set_title(f'M={M}   {len(sub)} CTAs   {n_waves} waves')
        ax.set_xlabel('CTA index  ((bz*gridDim.y + by)*gridDim.x + bx)')
        ax.set_ylabel('per-CTA duration (us)')
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label='wave_idx')

    fig.suptitle(
        f'CTA mainloop duration vs CTA index  '
        f'({args.model} down_proj, {args.kernel})', y=1.0)
    fig.tight_layout()

    if args.out:
        out = args.out
    else:
        out = os.path.join(
            here, f'cta_probe_dur_vs_idx_{args.model}_{args.kernel}.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'[plot] wrote {out}')


if __name__ == '__main__':
    main()
