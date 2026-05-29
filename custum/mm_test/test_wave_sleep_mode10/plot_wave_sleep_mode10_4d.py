#!/usr/bin/env python3
"""
Mode 10 4D sweep plots.

For each kernel produces:
  ws10_4d_tflops_<kernel>.png    grid of heatmaps:
                                   outer (first_pct × first_ns) → inner (mid_pct × mid_ns)
  ws10_4d_power_peak_<kernel>.png   same layout, peak W
  ws10_4d_pareto_<kernel>.png       TF vs peak-W scatter, color=mid_pct, marker=first_pct

Env:
  IN_DIR    folder with segments_4d.csv + gpu0_power.csv
  OUT_DIR   output dir (default = IN_DIR)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load_power(path):
    if not os.path.exists(path):
        return None
    pwr = pd.read_csv(path, skipinitialspace=True)
    p_col = next((c for c in pwr.columns if 'power.draw.instant' in c), None)
    ts_col = next((c for c in pwr.columns if 'timestamp' in c.lower()), None)
    if p_col is None or ts_col is None:
        return None
    pwr['ts'] = pd.to_datetime(pwr[ts_col], format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
    pwr['t_ns'] = pwr['ts'].astype('int64')
    pwr['p'] = pwr[p_col].astype(str).str.replace(' W', '').astype(float)
    return pwr.dropna(subset=['ts']).reset_index(drop=True)


def heatmap_cell(ax, arr, xs, ys, title, vmin, vmax, cmap, fmt):
    im = ax.imshow(arr, aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, fontsize=8)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys, fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, fmt.format(arr[i, j]),
                        ha='center', va='center', fontsize=7,
                        color='white' if arr[i, j] < (vmin + (vmax - vmin) * 0.4)
                              else 'black')
    return im


def grid_heatmap(df, value_col, fmt, cmap, title_root, vmin, vmax, out_path):
    first_pcts = sorted(df['first_pct'].unique())
    first_nss  = sorted(df['first_ns'].unique())
    mid_pcts   = sorted(df['mid_pct'].unique())
    mid_nss    = sorted(df['mid_ns'].unique())

    R = len(first_pcts)
    C = len(first_nss)
    fig, axes = plt.subplots(R, C, figsize=(4 * C, 3.6 * R), squeeze=False)
    fig.suptitle(title_root, y=0.99)

    for ri, fp in enumerate(first_pcts):
        for ci, fns in enumerate(first_nss):
            ax = axes[ri, ci]
            cell = df[(df.first_pct == fp) & (df.first_ns == fns)]
            arr = np.full((len(mid_pcts), len(mid_nss)), np.nan)
            for i, mp in enumerate(mid_pcts):
                for j, mn in enumerate(mid_nss):
                    s = cell[(cell.mid_pct == mp) & (cell.mid_ns == mn)][value_col]
                    if not s.empty:
                        arr[i, j] = s.median()
            heatmap_cell(ax, arr, mid_nss, mid_pcts,
                         f'first_pct={fp}, first_ns={fns}', vmin, vmax, cmap, fmt)
            ax.set_xlabel('mid_ns', fontsize=8)
            ax.set_ylabel('mid_pct', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {out_path}')


def pareto_plot(df, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 7))
    # mid_pct → color, first_pct → marker, mid_ns/first_ns visible in label
    mid_pcts = sorted(df['mid_pct'].unique())
    first_pcts = sorted(df['first_pct'].unique())
    markers = ['o', 's', '^', 'D', 'v', 'P']
    cmap = plt.cm.viridis
    for fi, fp in enumerate(first_pcts):
        for mi, mp in enumerate(mid_pcts):
            sub = df[(df.first_pct == fp) & (df.mid_pct == mp)]
            if sub.empty: continue
            color = cmap(mi / max(1, len(mid_pcts) - 1))
            ax.scatter(sub['p_peak'], sub['tf_med'],
                       marker=markers[fi % len(markers)],
                       color=color, s=70, alpha=0.8,
                       edgecolors='black', linewidth=0.4)

    # Baseline
    base = df[(df.first_pct == 0) & (df.mid_pct == 0)]
    if not base.empty:
        ax.scatter(base['p_peak'], base['tf_med'], marker='*',
                   color='red', s=250, label='baseline', edgecolors='black')

    ax.axvline(600, color='red', linestyle=':', alpha=0.5, label='TDP 600 W')
    ax.set_xlabel('peak power (W)')
    ax.set_ylabel('median TFLOPS / burst')
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # Two legends
    from matplotlib.lines import Line2D
    mid_handles = [Line2D([0], [0], marker='o', linestyle='',
                          color=cmap(i / max(1, len(mid_pcts) - 1)),
                          label=f'mid_pct={mp}', markersize=8)
                   for i, mp in enumerate(mid_pcts)]
    first_handles = [Line2D([0], [0], marker=markers[fi % len(markers)],
                            color='gray', linestyle='',
                            label=f'first_pct={fp}', markersize=8)
                     for fi, fp in enumerate(first_pcts)]
    leg1 = ax.legend(handles=mid_handles, loc='lower left', fontsize=9, title='color')
    ax.add_artist(leg1)
    ax.legend(handles=first_handles, loc='upper right', fontsize=9, title='marker')

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {out_path}')


def main():
    in_dir  = os.environ.get('IN_DIR',  HERE)
    out_dir = os.environ.get('OUT_DIR', in_dir)
    os.makedirs(out_dir, exist_ok=True)

    seg_path = os.path.join(in_dir, 'segments_4d.csv')
    if not os.path.exists(seg_path):
        print(f'[plot] missing {seg_path}'); sys.exit(1)
    seg = pd.read_csv(seg_path)

    pwr = load_power(os.path.join(in_dir, 'gpu0_power.csv'))

    # Build per-cfg aggregate with TF + peak/avg power
    rows = []
    for keys, grp in seg.groupby(['kernel', 'first_pct', 'first_ns',
                                   'mid_pct', 'mid_ns']):
        k, fp, fns, mp, mns = keys
        rec = {'kernel': k, 'first_pct': fp, 'first_ns': fns,
               'mid_pct': mp, 'mid_ns': mns,
               'tf_med': grp['tflops_obs'].median()}
        if pwr is not None:
            t0 = grp['t_start_ns'].min()
            t1 = grp['t_end_ns'].max()
            msk = (pwr['t_ns'] >= t0) & (pwr['t_ns'] <= t1)
            if msk.sum() >= 5:
                rec['p_peak'] = pwr[msk]['p'].max()
                rec['p_med']  = pwr[msk]['p'].median()
        rows.append(rec)
    agg = pd.DataFrame(rows)

    for kernel in sorted(agg['kernel'].unique()):
        sub = agg[agg.kernel == kernel]
        non_base = sub[~((sub.first_pct == 0) & (sub.mid_pct == 0))]

        # TFLOPS grid heatmap
        tf_vmin, tf_vmax = non_base['tf_med'].min(), non_base['tf_med'].max()
        grid_heatmap(non_base, 'tf_med', '{:.0f}', 'viridis',
                     f'TFLOPS — {kernel}  (rows=first_pct, cols=first_ns;\n'
                     f'each cell: mid_pct × mid_ns)',
                     tf_vmin, tf_vmax,
                     os.path.join(out_dir, f'ws10_4d_tflops_{kernel}.png'))

        # Power peak grid (if available)
        if 'p_peak' in non_base.columns:
            pp_vmin = max(300, non_base['p_peak'].min())
            pp_vmax = min(750, non_base['p_peak'].max())
            grid_heatmap(non_base, 'p_peak', '{:.0f}', 'plasma',
                         f'peak power (W) — {kernel}  (rows=first_pct, cols=first_ns;\n'
                         f'each cell: mid_pct × mid_ns)',
                         pp_vmin, pp_vmax,
                         os.path.join(out_dir, f'ws10_4d_power_peak_{kernel}.png'))

            pareto_plot(sub, os.path.join(out_dir, f'ws10_4d_pareto_{kernel}.png'),
                        f'Pareto TF vs peak-W — {kernel}')


if __name__ == '__main__':
    main()
