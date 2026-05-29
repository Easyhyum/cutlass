#!/usr/bin/env python3
"""
Chunk-sweep plots — TFLOPS vs chunk_m + power band per cfg.

Outputs (per kernel):
  chunk_tflops_vs_cm_<kernel>.png    median TFLOPS vs chunk_m (line plot)
  chunk_power_vs_cm_<kernel>.png     median peak / avg W vs chunk_m
  chunk_pareto_<kernel>.png          TF vs peak-W scatter for all (chunk_m × extra)

Auto-handles torch-chunked (extra=chunk_gap_us) and kernel-chunked (extra=chunk_idle_us).

Env:
  IN_DIR   directory with segments.csv + gpu0_power.csv (default: CWD)
  OUT_DIR  output dir (default = IN_DIR)
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


def main():
    in_dir  = os.environ.get('IN_DIR',  HERE)
    out_dir = os.environ.get('OUT_DIR', in_dir)
    os.makedirs(out_dir, exist_ok=True)

    seg_path = os.path.join(in_dir, 'segments.csv')
    if not os.path.exists(seg_path):
        print(f'[plot] missing {seg_path}'); sys.exit(1)
    seg = pd.read_csv(seg_path)

    extra_col = ('chunk_gap_us' if 'chunk_gap_us' in seg.columns
                 else 'chunk_idle_us' if 'chunk_idle_us' in seg.columns else None)
    label = 'gap_us' if extra_col == 'chunk_gap_us' else 'idle_us'

    pwr = load_power(os.path.join(in_dir, 'gpu0_power.csv'))

    # aggregate per (kernel, chunk_m, extra)
    rows = []
    grp_keys = ['kernel', 'chunk_m'] + ([extra_col] if extra_col else [])
    for keys, grp in seg.groupby(grp_keys):
        rec = dict(zip(grp_keys, keys if isinstance(keys, tuple) else (keys,)))
        rec['tf_med']     = grp['tflops_obs'].median()
        rec['elapsed_ms'] = grp['elapsed_ms'].median()
        if pwr is not None:
            t0, t1 = grp['t_start_ns'].min(), grp['t_end_ns'].max()
            msk = (pwr['t_ns'] >= t0) & (pwr['t_ns'] <= t1)
            if msk.sum() >= 5:
                rec['p_peak'] = pwr[msk]['p'].max()
                rec['p_med']  = pwr[msk]['p'].median()
        rows.append(rec)
    agg = pd.DataFrame(rows)
    print(agg.round(1).to_string(index=False))

    for kernel in sorted(agg['kernel'].unique()):
        sub = agg[agg.kernel == kernel].sort_values('chunk_m')

        # ── TFLOPS vs chunk_m (line, multi-extra) ────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        if extra_col:
            for ex in sorted(sub[extra_col].unique()):
                s2 = sub[sub[extra_col] == ex]
                ax.plot(s2['chunk_m'], s2['tf_med'],
                        marker='o', label=f'{label}={ex}', linewidth=2)
        else:
            ax.plot(sub['chunk_m'], sub['tf_med'], marker='o', linewidth=2)
        ax.set_xlabel('chunk_m')
        ax.set_ylabel('median TFLOPS / burst')
        ax.set_title(f'TFLOPS vs chunk_m — {kernel}')
        ax.grid(alpha=0.3); ax.legend()
        ax.set_xscale('log', base=2)
        p = os.path.join(out_dir, f'chunk_tflops_vs_cm_{kernel}.png')
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        print(f'[plot] {p}')

        # ── Power vs chunk_m ─────────────────────────────────────────────────
        if 'p_peak' in sub.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            if extra_col:
                for ex in sorted(sub[extra_col].unique()):
                    s2 = sub[sub[extra_col] == ex]
                    ax.plot(s2['chunk_m'], s2['p_peak'],
                            marker='o', label=f'peak {label}={ex}', linewidth=2)
                    ax.plot(s2['chunk_m'], s2['p_med'],
                            marker='s', linestyle='--',
                            label=f'med {label}={ex}', linewidth=1.5, alpha=0.7)
            else:
                ax.plot(sub['chunk_m'], sub['p_peak'], marker='o',
                        label='peak', linewidth=2)
                ax.plot(sub['chunk_m'], sub['p_med'], marker='s',
                        linestyle='--', label='med', linewidth=1.5, alpha=0.7)
            ax.axhline(600, color='red', linestyle=':', alpha=0.5, label='TDP 600W')
            ax.set_xscale('log', base=2)
            ax.set_xlabel('chunk_m'); ax.set_ylabel('W')
            ax.set_title(f'Power vs chunk_m — {kernel}')
            ax.grid(alpha=0.3); ax.legend()
            p = os.path.join(out_dir, f'chunk_power_vs_cm_{kernel}.png')
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            print(f'[plot] {p}')

            # ── Pareto TF vs peak-W ───────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(10, 7))
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(sub['chunk_m'].unique())))
            markers = ['o', 's', '^', 'D', 'v', 'P']
            cms = sorted(sub['chunk_m'].unique())
            extras = sorted(sub[extra_col].unique()) if extra_col else [None]
            for i, cm in enumerate(cms):
                for j, ex in enumerate(extras):
                    s2 = sub[sub.chunk_m == cm]
                    if extra_col: s2 = s2[s2[extra_col] == ex]
                    if s2.empty: continue
                    ax.scatter(s2['p_peak'], s2['tf_med'],
                               color=colors[i], marker=markers[j % len(markers)],
                               s=120, alpha=0.85,
                               edgecolors='black', linewidth=0.5,
                               label=(f'cm={cm}' if j == 0 else None))
            ax.axvline(600, color='red', linestyle=':', alpha=0.5, label='TDP 600W')
            ax.set_xlabel('peak power (W)')
            ax.set_ylabel('median TFLOPS / burst')
            ax.set_title(f'TF vs peak-W — {kernel}')
            ax.grid(alpha=0.3); ax.legend(loc='lower left', fontsize=9, ncol=2)
            p = os.path.join(out_dir, f'chunk_pareto_{kernel}.png')
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            print(f'[plot] {p}')


if __name__ == '__main__':
    main()
