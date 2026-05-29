#!/usr/bin/env python3
"""
Side-by-side comparison: streamk vs basicdp.

For each M value, draws two side-by-side scatter panels (streamk | basicdp):
   x = CTA index    y = per-CTA mainloop duration (us)
   color = wave_idx
   red dashed = wave boundaries (k * n_sm)

Also prints a wave-summary table comparing CTAs launched, waves, and
per-CTA duration mean/std for wave 0 + a representative middle/last wave.

Reads:
   cta_probe_per_cta_<model>_streamk.csv
   cta_probe_per_cta_<model>_basicdp.csv
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen3-8b')
    ap.add_argument('--n_sm',  type=int, default=188)
    ap.add_argument('--out',   default=None)
    ap.add_argument('--max_wave_lines', type=int, default=12,
                    help='draw wave boundary lines when num_waves ≤ this')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    paths = {
        'streamk': os.path.join(here, f'cta_probe_per_cta_{args.model}_streamk.csv'),
        'basicdp': os.path.join(here, f'cta_probe_per_cta_{args.model}_basicdp.csv'),
    }
    dfs = {}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise SystemExit(f'missing {p} — run run_cta_probe.py for {k} first.')
        df = pd.read_csv(p)
        if 'cta_idx' not in df.columns and 'cta_launch_idx' in df.columns:
            df = df.rename(columns={'cta_launch_idx': 'cta_idx'})
        if 'cta_dur_ns' not in df.columns:
            df['cta_dur_ns'] = df['end_ns'].astype('int64') - df['start_ns'].astype('int64')
            df.loc[df['cta_dur_ns'] < 0, 'cta_dur_ns'] = 0
        dfs[k] = df

    Ms = sorted(set(dfs['streamk']['M'].unique()) & set(dfs['basicdp']['M'].unique()))

    # ----- summary print -----
    print(f'{"M":>7s}  {"kernel":>8s}  {"n_ctas":>6s}  {"waves":>5s}  '
          f'{"wave0 ctas":>10s}  {"wave0 dur_us μ":>15s}  {"wave0 dur σ":>11s}  '
          f'{"mid wave dur_us μ":>17s}  {"last wave dur_us μ":>18s}')
    print('-' * 130)
    for M in Ms:
        for k in ('streamk', 'basicdp'):
            sub = dfs[k][dfs[k]['M'] == M]
            wv = sub['wave_idx'].to_numpy()
            dur = sub['cta_dur_ns'].to_numpy() / 1000.0
            n_waves = int(wv.max()) + 1
            n_ctas = len(sub)
            d0  = dur[wv == 0]
            mid_w = max(1, n_waves // 2)
            d_mid = dur[wv == mid_w]
            d_lst = dur[wv == n_waves - 1]
            print(f'{M:>7d}  {k:>8s}  {n_ctas:>6d}  {n_waves:>5d}  '
                  f'{len(d0):>10d}  {d0.mean():>15.2f}  {d0.std():>11.2f}  '
                  f'{d_mid.mean() if len(d_mid) else 0:>17.2f}  '
                  f'{d_lst.mean() if len(d_lst) else 0:>18.2f}')
        print()

    # ----- side-by-side scatter -----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(Ms), 2,
                             figsize=(20, 2.6 * len(Ms)), squeeze=False)
    for row, M in enumerate(Ms):
        for col, k in enumerate(('streamk', 'basicdp')):
            ax = axes[row, col]
            sub = dfs[k][dfs[k]['M'] == M].sort_values('cta_idx')
            x = sub['cta_idx'].to_numpy()
            y = sub['cta_dur_ns'].to_numpy() / 1000.0
            wv = sub['wave_idx'].to_numpy()
            sc = ax.scatter(x, y, c=wv, cmap='viridis', s=6, alpha=0.7,
                            edgecolors='none')
            n_waves = int(wv.max()) + 1
            if n_waves <= args.max_wave_lines:
                for w in range(1, n_waves):
                    ax.axvline(w * args.n_sm, color='red', linestyle='--',
                               lw=0.4, alpha=0.5)
            ax.set_title(f'{k}   M={M}   {len(sub)} CTAs   {n_waves} waves',
                         fontsize=10)
            ax.set_xlabel('CTA index')
            ax.set_ylabel('per-CTA dur (us)')
            ax.grid(alpha=0.25)
            fig.colorbar(sc, ax=ax, label='wave_idx', pad=0.02)

    fig.suptitle(
        f'streamk vs basicdp — CTA duration vs index  ({args.model} down_proj)',
        y=1.0)
    fig.tight_layout()
    out = args.out or os.path.join(
        here, f'cta_probe_streamk_vs_basicdp_{args.model}.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print(f'plot: {out}')


if __name__ == '__main__':
    main()
