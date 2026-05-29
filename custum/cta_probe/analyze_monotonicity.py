#!/usr/bin/env python3
"""
For each (kernel, M), measure how monotonically wave_idx grows with the
blockIdx-derived linear index  L = (bz * gy + by) * gx + bx.

If "bigger CTA index → later wave" holds, then sorting CTAs by L should
yield a non-decreasing wave_idx sequence; deviations indicate the
threadblock swizzle is reordering.

Outputs:
  cta_probe_monotonicity.csv          per-(kernel, M) summary
  cta_probe_monotonicity_<k>.png      plot of L vs wave_idx, side-by-side
"""
import os, sys, csv
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def kendall_tau(a, b):
    """Stable, vectorized via scipy if available, else fallback."""
    try:
        from scipy.stats import kendalltau
        return kendalltau(a, b, variant='b').statistic
    except Exception:
        pass
    # naive O(n log n) approximation via concordant-pair via mergesort
    n = len(a)
    if n < 2:
        return 1.0
    order = np.argsort(a, kind='stable')
    b_sorted = np.asarray(b)[order]
    # concordant fraction
    concordant = 0
    discordant = 0
    for i in range(0, n - 1):
        d = b_sorted[i + 1:] - b_sorted[i]
        concordant += int((d > 0).sum())
        discordant += int((d < 0).sum())
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def analyze(df, kernel):
    rows = []
    for M, sub in df.groupby('M'):
        sub = sub.sort_values('cta_idx').reset_index(drop=True)
        gx_max = int(sub['bx'].max()) + 1
        gy_max = int(sub['by'].max()) + 1
        gz_max = int(sub['bz'].max()) + 1
        L = (sub['bz'].to_numpy() * gy_max + sub['by'].to_numpy()) * gx_max + \
            sub['bx'].to_numpy()
        wv = sub['wave_idx'].to_numpy()

        # sort by L, then check how often wave does NOT grow
        order = np.argsort(L, kind='stable')
        wv_by_L = wv[order]
        L_sorted = L[order]
        # Two CTAs (i, j) with L_i < L_j   "violate" monotonicity if wv_i > wv_j.
        # Compute fraction of adjacent pairs where wv strictly decreases.
        diffs = np.diff(wv_by_L)
        n_adj = len(diffs)
        viol_strict = int((diffs < 0).sum())
        # Also: among pairs with the same wave_idx, fraction:
        same_wave = int((diffs == 0).sum())
        up = int((diffs > 0).sum())

        # spearman rank-correlation between L and wave_idx
        try:
            from scipy.stats import spearmanr
            rho = spearmanr(L, wv).statistic
        except Exception:
            rho = np.corrcoef(np.argsort(np.argsort(L)),
                              np.argsort(np.argsort(wv)))[0, 1]
        tau = kendall_tau(L, wv)

        # by-axis correlation: how strongly does each of bx,by,bz correlate
        # with wave_idx?
        def corr(x, y):
            try:
                from scipy.stats import spearmanr
                return spearmanr(x, y).statistic
            except Exception:
                return np.corrcoef(np.argsort(np.argsort(x)),
                                   np.argsort(np.argsort(y)))[0, 1]
        rho_bx = corr(sub['bx'], wv)
        rho_by = corr(sub['by'], wv)
        rho_bz = corr(sub['bz'], wv) if gz_max > 1 else float('nan')

        rows.append({
            'kernel': kernel,
            'M': int(M),
            'gx': gx_max, 'gy': gy_max, 'gz': gz_max,
            'n_ctas': len(sub),
            'waves': int(wv.max()) + 1,
            'rho(L,wave)':   round(float(rho), 4),
            'tau(L,wave)':   round(float(tau), 4),
            'rho(bx,wave)':  round(float(rho_bx), 4),
            'rho(by,wave)':  round(float(rho_by), 4),
            'rho(bz,wave)':  round(float(rho_bz), 4) if not np.isnan(rho_bz) else None,
            'adj_pairs':         n_adj,
            'adj_wave_up':       up,
            'adj_wave_same':     same_wave,
            'adj_wave_down':     viol_strict,
            'monotone_frac':     round((up + same_wave) / max(n_adj, 1), 4),
        })
    return rows


def main():
    out_rows = []
    for kernel in ('streamk', 'basicdp'):
        csv_path = os.path.join(HERE, f'cta_probe_per_cta_qwen3-8b_{kernel}.csv')
        if not os.path.exists(csv_path):
            print(f'[skip] {csv_path} not found')
            continue
        df = pd.read_csv(csv_path)
        out_rows += analyze(df, kernel)

    out_csv = os.path.join(HERE, 'cta_probe_monotonicity.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f'[mono] wrote {out_csv}\n')

    # Pretty print
    cols = ['kernel', 'M', 'n_ctas', 'waves',
            'rho(L,wave)', 'tau(L,wave)',
            'rho(bx,wave)', 'rho(by,wave)',
            'monotone_frac', 'adj_wave_down']
    df_out = pd.DataFrame(out_rows)
    print(df_out[cols].to_string(index=False))

    # ---- plot: L vs wave_idx, scatter, one panel per (kernel, M) -------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for kernel in ('streamk', 'basicdp'):
        csv_path = os.path.join(HERE, f'cta_probe_per_cta_qwen3-8b_{kernel}.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        Ms = sorted(df['M'].unique())
        fig, axes = plt.subplots(len(Ms), 1, figsize=(13, 2.0 * len(Ms)),
                                 squeeze=False)
        for ax, M in zip(axes[:, 0], Ms):
            sub = df[df['M'] == M].sort_values('cta_idx')
            gx = int(sub['bx'].max()) + 1
            gy = int(sub['by'].max()) + 1
            L = (sub['bz'].to_numpy() * gy + sub['by'].to_numpy()) * gx + \
                sub['bx'].to_numpy()
            wv = sub['wave_idx'].to_numpy()
            ax.scatter(L, wv, s=4, alpha=0.5)
            ax.set_title(
                f'{kernel}  M={M}   grid {gx}x{gy}   {len(sub)} CTAs, '
                f'{int(wv.max())+1} waves')
            ax.set_xlabel('linear blockIdx  L = (bz*gy + by)*gx + bx')
            ax.set_ylabel('wave_idx')
            ax.grid(alpha=0.3)
        fig.suptitle(f'CTA linear index vs wave  ({kernel})', y=1.0)
        fig.tight_layout()
        png = os.path.join(HERE, f'cta_probe_monotonicity_{kernel}.png')
        fig.savefig(png, dpi=110, bbox_inches='tight')
        print(f'[mono] plot {png}')


if __name__ == '__main__':
    main()
