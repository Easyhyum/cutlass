#!/usr/bin/env python3
"""
Check whether CTAs belonging to a fixed wave (e.g. wave_idx == 2) occupy
the same index set across different M values.

If the GEMM tile is fixed at 128x128 and the dispatcher fills waves in
launch order, then wave_idx == k should contain CTAs with
cta_idx ∈ [k * n_sm, (k+1) * n_sm)  for every M (as long as M is
large enough that wave k exists). For basicdp where launch_idx tracks the
2D blockIdx linear index L = by*gx + bx, the (bx, by) set should also
match. For streamk the grid is 1D so only bx matters.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
N_SM = 188

TARGET_WAVES = [0, 1, 2, 5, 10]


def summarize(df, kernel, target_wave):
    rows = []
    Ms = sorted(df['M'].unique())
    for M in Ms:
        sub = df[(df['M'] == M) & (df['wave_idx'] == target_wave)]
        if len(sub) == 0:
            rows.append({
                'kernel': kernel, 'M': M, 'wave': target_wave,
                'n_ctas': 0, 'note': 'wave does not exist for this M',
            })
            continue
        rows.append({
            'kernel': kernel, 'M': M, 'wave': target_wave,
            'n_ctas': len(sub),
            'launch_idx_min': int(sub['cta_idx'].min()),
            'launch_idx_max': int(sub['cta_idx'].max()),
            'bx_min': int(sub['bx'].min()),
            'bx_max': int(sub['bx'].max()),
            'by_min': int(sub['by'].min()),
            'by_max': int(sub['by'].max()),
            'bz_min': int(sub['bz'].min()),
            'bz_max': int(sub['bz'].max()),
        })
    return rows


def jaccard(a_set, b_set):
    if not a_set and not b_set:
        return 1.0
    return len(a_set & b_set) / len(a_set | b_set)


def cross_M_overlap(df, kernel, target_wave):
    """For a fixed target_wave, compute pairwise Jaccard overlap of the set
    of cta_idx values across different M, and likewise for (bx,by)."""
    Ms = sorted(df['M'].unique())
    have = []
    for M in Ms:
        sub = df[(df['M'] == M) & (df['wave_idx'] == target_wave)]
        if len(sub) > 0:
            have.append((M,
                         set(sub['cta_idx'].astype(int).tolist()),
                         set(zip(sub['bx'].astype(int), sub['by'].astype(int)))))
    rows = []
    if len(have) < 2:
        return rows
    base_M, base_li, base_xy = have[-1]  # use largest M as reference
    for (M, li, xy) in have:
        rows.append({
            'kernel': kernel, 'wave': target_wave,
            'M':  M, 'ref_M': base_M,
            'n_ctas':       len(li),
            'jaccard(launch_idx, vs ref_M)': round(jaccard(li, base_li), 4),
            'jaccard((bx,by), vs ref_M)':   round(jaccard(xy, base_xy), 4),
        })
    return rows


def main():
    out = {}
    for kernel in ('streamk', 'basicdp'):
        path = os.path.join(HERE, f'cta_probe_per_cta_qwen3-8b_{kernel}.csv')
        df = pd.read_csv(path)
        out[kernel] = df

    # -- A. per-(kernel, M, wave) ranges --
    print('=' * 100)
    print('A) For each wave_idx, the range of cta_idx / bx / by across all M')
    print('=' * 100)
    for kernel in ('streamk', 'basicdp'):
        print(f'\n--- {kernel} ---')
        for w in TARGET_WAVES:
            rows = summarize(out[kernel], kernel, w)
            df_w = pd.DataFrame(rows)
            print(f'\nwave_idx = {w}')
            print(df_w.to_string(index=False))

    # -- B. cross-M Jaccard overlap --
    print('\n' + '=' * 100)
    print('B) Cross-M Jaccard overlap (reference = largest M where the wave exists)')
    print('=' * 100)
    for kernel in ('streamk', 'basicdp'):
        for w in TARGET_WAVES:
            rows = cross_M_overlap(out[kernel], kernel, w)
            if rows:
                print(f'\n[{kernel}]  wave_idx = {w}')
                print(pd.DataFrame(rows).to_string(index=False))

    # -- C. one concrete enumeration: wave 2, basicdp, list (bx,by) sets --
    print('\n' + '=' * 100)
    print('C) Concrete (bx,by) listing for wave_idx == 2 (basicdp), M ≥ 8192')
    print('=' * 100)
    for M in sorted(out['basicdp']['M'].unique()):
        sub = out['basicdp']
        sub = sub[(sub['M'] == M) & (sub['wave_idx'] == 2)]
        if len(sub) == 0: continue
        xy = sorted(zip(sub['bx'].astype(int), sub['by'].astype(int)))
        print(f'\nM={M}  n_ctas={len(xy)}  '
              f'(bx,by) first 5: {xy[:5]}  last 5: {xy[-5:]}')


if __name__ == '__main__':
    main()
