#!/usr/bin/env python3
"""
For each (kernel, M), measure how monotonically wave_idx grows with the
blockIdx-derived linear index  L = (bz * gy + by) * gx + bx.

If "bigger CTA index → later wave" holds, then sorting CTAs by L should yield
a non-decreasing wave_idx sequence; deviations indicate the threadblock
swizzle is reordering.

Inputs (looked up in OUT_DIR or this dir):
  cta_probe_per_cta_<model>_<op>_streamk.csv
  cta_probe_per_cta_<model>_<op>_sm80_v3.csv

Outputs:
  cta_probe_monotonicity_<model>_<op>.csv      per-(kernel, M) summary
  cta_probe_monotonicity_<model>_<op>_<k>.png  per-kernel L vs wave plot

Env:
  MM_MODEL  qwen3-8b (default)
  MM_OP     down_proj (default)
  OUT_DIR   directory containing the per-CTA CSVs (default: this dir)
"""
import os, sys, csv
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def kendall_tau(a, b):
    try:
        from scipy.stats import kendalltau
        return float(kendalltau(a, b, variant='b').statistic)
    except Exception:
        pass
    n = len(a)
    if n < 2:
        return 1.0
    order = np.argsort(a, kind='stable')
    b_sorted = np.asarray(b)[order]
    c = d = 0
    for i in range(n - 1):
        di = b_sorted[i + 1:] - b_sorted[i]
        c += int((di > 0).sum())
        d += int((di < 0).sum())
    tot = c + d
    return (c - d) / tot if tot else 0.0


def analyze(df, kernel):
    rows = []
    for M, sub in df.groupby('M'):
        sub = sub.sort_values('cta_idx').reset_index(drop=True)
        gx_max = int(sub['bx'].max()) + 1
        gy_max = int(sub['by'].max()) + 1
        gz_max = int(sub['bz'].max()) + 1
        L  = (sub['bz'].to_numpy() * gy_max + sub['by'].to_numpy()) * gx_max + \
             sub['bx'].to_numpy()
        wv = sub['wave_idx'].to_numpy()

        order = np.argsort(L, kind='stable')
        wv_by_L = wv[order]
        diffs = np.diff(wv_by_L)
        n_adj = len(diffs)
        viol_strict = int((diffs < 0).sum())
        same_wave   = int((diffs == 0).sum())
        up          = int((diffs > 0).sum())

        try:
            from scipy.stats import spearmanr
            rho     = float(spearmanr(L,           wv).statistic)
            rho_bx  = float(spearmanr(sub['bx'],   wv).statistic)
            rho_by  = float(spearmanr(sub['by'],   wv).statistic)
            rho_bz  = float(spearmanr(sub['bz'],   wv).statistic) if gz_max > 1 else float('nan')
        except Exception:
            def _rho(x, y):
                return float(np.corrcoef(
                    np.argsort(np.argsort(x)),
                    np.argsort(np.argsort(y)))[0, 1])
            rho     = _rho(L, wv)
            rho_bx  = _rho(sub['bx'], wv)
            rho_by  = _rho(sub['by'], wv)
            rho_bz  = _rho(sub['bz'], wv) if gz_max > 1 else float('nan')
        tau = kendall_tau(L, wv)

        rows.append({
            'kernel': kernel,
            'M': int(M),
            'gx': gx_max, 'gy': gy_max, 'gz': gz_max,
            'n_ctas': len(sub),
            'waves': int(wv.max()) + 1,
            'rho(L,wave)':   round(rho, 4),
            'tau(L,wave)':   round(tau, 4),
            'rho(bx,wave)':  round(rho_bx, 4),
            'rho(by,wave)':  round(rho_by, 4),
            'rho(bz,wave)':  round(rho_bz, 4) if not np.isnan(rho_bz) else None,
            'adj_pairs':         n_adj,
            'adj_wave_up':       up,
            'adj_wave_same':     same_wave,
            'adj_wave_down':     viol_strict,
            'monotone_frac':     round((up + same_wave) / max(n_adj, 1), 4),
        })
    return rows


def main():
    model = os.environ.get('MM_MODEL', 'qwen3-8b').strip().lower()
    op    = os.environ.get('MM_OP',    'down_proj').strip().lower()
    out_dir = os.environ.get('OUT_DIR', HERE)

    all_rows = []
    for kernel in ('streamk', 'sm80_v3'):
        csv_path = os.path.join(out_dir, f'cta_probe_per_cta_{model}_{op}_{kernel}.csv')
        if not os.path.exists(csv_path):
            print(f'[skip] {csv_path} not found')
            continue
        df = pd.read_csv(csv_path)
        all_rows += analyze(df, kernel)

    if not all_rows:
        print('[mono] no input CSVs — run eval_cta_probe.py first')
        return

    out_csv = os.path.join(out_dir, f'cta_probe_monotonicity_{model}_{op}.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f'[mono] wrote {out_csv}\n')

    cols = ['kernel', 'M', 'n_ctas', 'waves',
            'rho(L,wave)', 'tau(L,wave)',
            'rho(bx,wave)', 'rho(by,wave)',
            'monotone_frac', 'adj_wave_down']
    print(pd.DataFrame(all_rows)[cols].to_string(index=False))

    # ─── per-kernel L vs wave_idx scatter ─────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for kernel in ('streamk', 'sm80_v3'):
        csv_path = os.path.join(out_dir, f'cta_probe_per_cta_{model}_{op}_{kernel}.csv')
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
            L  = (sub['bz'].to_numpy() * gy + sub['by'].to_numpy()) * gx + \
                 sub['bx'].to_numpy()
            wv = sub['wave_idx'].to_numpy()
            ax.scatter(L, wv, s=4, alpha=0.5)
            ax.set_title(f'{kernel}  M={M}   grid {gx}x{gy}   '
                         f'{len(sub)} CTAs, {int(wv.max())+1} waves')
            ax.set_xlabel('linear blockIdx  L = (bz*gy + by)*gx + bx')
            ax.set_ylabel('wave_idx')
            ax.grid(alpha=0.3)
        fig.suptitle(f'CTA linear index vs wave  ({kernel}, {model}/{op})', y=1.0)
        fig.tight_layout()
        png = os.path.join(out_dir, f'cta_probe_monotonicity_{model}_{op}_{kernel}.png')
        fig.savefig(png, dpi=110, bbox_inches='tight')
        print(f'[mono] plot {png}')


if __name__ == '__main__':
    main()
