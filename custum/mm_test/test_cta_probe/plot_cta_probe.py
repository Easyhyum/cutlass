#!/usr/bin/env python3
"""
Per-(kernel, M) scatter plots from the test_cta_probe CSVs:
  - CTA index → SM id, colored by wave_idx
  - CTA duration vs CTA index
  - Side-by-side streamk vs sm80_v3 (same M)

Inputs (looked up in OUT_DIR):
  cta_probe_per_cta_<model>_<op>_streamk.csv
  cta_probe_per_cta_<model>_<op>_sm80_v3.csv

Env:
  MM_MODEL  qwen3-8b (default)
  MM_OP     down_proj (default)
  OUT_DIR   directory containing the per-CTA CSVs (default: this dir)
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_cta_sm(df, out_path, title):
    Ms = sorted(df['M'].unique())
    fig, axes = plt.subplots(len(Ms), 1, figsize=(13, 2.6 * len(Ms)),
                             squeeze=False)
    for ax, M in zip(axes[:, 0], Ms):
        sub = df[df['M'] == M]
        if sub.empty:
            continue
        x  = sub['cta_idx'].to_numpy()
        sm = sub['smid'].to_numpy()
        wv = sub['wave_idx'].to_numpy()
        sc = ax.scatter(x, sm, c=wv, cmap='viridis', s=10, edgecolors='none')
        ax.set_title(f'M={M}   {len(sub)} CTAs, {int(wv.max())+1} waves, '
                     f'{int(np.unique(sm).size)} SMs')
        ax.set_xlabel('CTA index')
        ax.set_ylabel('physical SM id')
        ax.grid(alpha=0.2)
        fig.colorbar(sc, ax=ax, label='wave_idx')
    fig.suptitle(title, y=1.0); fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {out_path}')


def plot_dur_vs_idx(df, out_path, title):
    Ms = sorted(df['M'].unique())
    fig, axes = plt.subplots(len(Ms), 1, figsize=(13, 2.4 * len(Ms)),
                             squeeze=False)
    for ax, M in zip(axes[:, 0], Ms):
        sub = df[df['M'] == M].sort_values('cta_idx')
        x  = sub['cta_idx'].to_numpy()
        d  = sub['cta_dur_ns'].to_numpy() / 1e3   # → µs
        wv = sub['wave_idx'].to_numpy()
        sc = ax.scatter(x, d, c=wv, cmap='viridis', s=8, edgecolors='none')
        ax.set_title(f'M={M}   median dur={np.median(d):.1f} µs')
        ax.set_xlabel('CTA index'); ax.set_ylabel('CTA dur (µs)')
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label='wave_idx')
    fig.suptitle(title, y=1.0); fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {out_path}')


def plot_side_by_side(df_a, df_b, label_a, label_b, out_path, title):
    """One row per M; left = kernel A, right = kernel B; SM dispatch view."""
    Ms = sorted(set(df_a['M'].unique()) | set(df_b['M'].unique()))
    fig, axes = plt.subplots(len(Ms), 2, figsize=(15, 2.6 * len(Ms)),
                             squeeze=False)
    for row_idx, M in enumerate(Ms):
        for col_idx, (df, lbl) in enumerate([(df_a, label_a), (df_b, label_b)]):
            ax = axes[row_idx, col_idx]
            sub = df[df['M'] == M]
            if sub.empty:
                ax.set_title(f'{lbl}  M={M}  (no data)')
                continue
            x  = sub['cta_idx'].to_numpy()
            sm = sub['smid'].to_numpy()
            wv = sub['wave_idx'].to_numpy()
            sc = ax.scatter(x, sm, c=wv, cmap='viridis', s=8, edgecolors='none')
            ax.set_title(f'{lbl}  M={M}   {len(sub)} CTAs, '
                         f'{int(wv.max())+1} waves')
            ax.set_xlabel('CTA index'); ax.set_ylabel('SM id')
            ax.grid(alpha=0.2)
            fig.colorbar(sc, ax=ax, label='wave_idx')
    fig.suptitle(title, y=1.0); fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {out_path}')


def main():
    model = os.environ.get('MM_MODEL', 'qwen3-8b').strip().lower()
    op    = os.environ.get('MM_OP',    'down_proj').strip().lower()
    out_dir = os.environ.get('OUT_DIR', HERE)

    df_by_k = {}
    for kernel in ('streamk', 'sm80_v3'):
        path = os.path.join(out_dir, f'cta_probe_per_cta_{model}_{op}_{kernel}.csv')
        if not os.path.exists(path):
            print(f'[skip] {path} not found')
            continue
        df_by_k[kernel] = pd.read_csv(path)

    if not df_by_k:
        print('[plot] no input CSVs — run eval_cta_probe.py first'); return

    # Per-kernel plots
    for k, df in df_by_k.items():
        plot_cta_sm(
            df,
            os.path.join(out_dir, f'cta_probe_sm_{model}_{op}_{k}.png'),
            f'CTA → SM dispatch  ({k}, {model}/{op})',
        )
        plot_dur_vs_idx(
            df,
            os.path.join(out_dir, f'cta_probe_dur_vs_idx_{model}_{op}_{k}.png'),
            f'CTA duration vs index  ({k}, {model}/{op})',
        )

    # Side-by-side if both present
    if 'streamk' in df_by_k and 'sm80_v3' in df_by_k:
        plot_side_by_side(
            df_by_k['streamk'], df_by_k['sm80_v3'],
            'streamk', 'sm80_v3',
            os.path.join(out_dir, f'cta_probe_streamk_vs_sm80_v3_{model}_{op}.png'),
            f'Stream-K vs sm80_v3 dispatch  ({model}/{op})',
        )


if __name__ == '__main__':
    main()
