#!/usr/bin/env python3
"""
Mode 10 separate-phase sweep plots.

For each (kernel, phase) combination produces:
  ws10_tflops_heatmap_<kernel>_<phase>.png    median TFLOPS, (pct × ns)
  ws10_power_heatmap_<kernel>_<phase>.png     median peak / avg W (if available)
  ws10_power_timeline_<kernel>_<phase>.png    power(t) grid — one cell per (pct, ns)
                                              shows the 50-burst power waveform
                                              against gpu0_power.csv 50ms samples.

Env:
  IN_DIR    folder with segments.csv + gpu0_power.csv (+ segments_with_power.csv)
  OUT_DIR   where to write PNGs (default = IN_DIR)
"""
import os, sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load_power_csv(path):
    """nvidia-smi --format=csv with units; parse timestamp + power.draw.instant."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, skipinitialspace=True)
    # standard column names from nvidia-smi
    ts_col = next((c for c in df.columns if 'timestamp' in c.lower()), None)
    p_inst = next((c for c in df.columns if 'power.draw.instant' in c), None)
    p_avg  = next((c for c in df.columns if 'power.draw.average' in c), None)
    sm_clk = next((c for c in df.columns if 'clocks.current.sm' in c), None)
    if ts_col is None or (p_inst is None and p_avg is None):
        print(f'[plot] power CSV missing expected columns: {list(df.columns)}')
        return None
    out = pd.DataFrame()
    out['ts'] = pd.to_datetime(df[ts_col].astype(str).str.strip(),
                               format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')
    if p_inst is not None:
        out['p_inst_W'] = pd.to_numeric(
            df[p_inst].astype(str).str.replace(' W', '', regex=False),
            errors='coerce')
    if p_avg is not None:
        out['p_avg_W'] = pd.to_numeric(
            df[p_avg].astype(str).str.replace(' W', '', regex=False),
            errors='coerce')
    if sm_clk is not None:
        out['sm_mhz'] = pd.to_numeric(
            df[sm_clk].astype(str).str.replace(' MHz', '', regex=False),
            errors='coerce')
    out = out.dropna(subset=['ts']).reset_index(drop=True)
    out['t_ns'] = out['ts'].astype('int64')   # ns since epoch (UTC)
    return out


def heatmap(values, pcts, nss, ax, title, vmin=None, vmax=None, cbar_label=''):
    arr = np.full((len(pcts), len(nss)), np.nan)
    for i, p in enumerate(pcts):
        for j, n in enumerate(nss):
            v = values.get((p, n))
            if v is not None and not np.isnan(v):
                arr[i, j] = v
    im = ax.imshow(arr, aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_xticks(range(len(nss))); ax.set_xticklabels(nss)
    ax.set_yticks(range(len(pcts))); ax.set_yticklabels(pcts)
    ax.set_xlabel('sleep ns')
    ax.set_ylabel('sleep pct  (% SMs/CTAs)')
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f'{arr[i, j]:.0f}',
                        ha='center', va='center', fontsize=8,
                        color='white' if arr[i, j] < (np.nanmean(arr)) else 'black')
    plt.colorbar(im, ax=ax, label=cbar_label)


def main():
    in_dir  = os.environ.get('IN_DIR',  HERE)
    out_dir = os.environ.get('OUT_DIR', in_dir)
    os.makedirs(out_dir, exist_ok=True)

    seg_path = os.path.join(in_dir, 'segments.csv')
    if not os.path.exists(seg_path):
        print(f'[plot] no segments.csv at {seg_path}'); sys.exit(1)
    seg = pd.read_csv(seg_path)

    pwr_path = os.path.join(in_dir, 'gpu0_power.csv')
    pwr = load_power_csv(pwr_path)
    if pwr is not None:
        print(f'[plot] power CSV loaded: {len(pwr)} samples')

    enr_path = os.path.join(in_dir, 'segments_with_power.csv')
    enr = pd.read_csv(enr_path) if os.path.exists(enr_path) else None

    kernels = sorted(seg['kernel'].unique())
    phases  = sorted(seg['phase'].unique())

    # ─── TFLOPS heatmap per (kernel, phase) ──────────────────────────────────
    for k in kernels:
        for ph in phases:
            sub = seg[(seg.kernel == k) & (seg.phase == ph) &
                      ~((seg.sleep_pct == 0) & (seg.sleep_ns == 0))]
            if sub.empty:
                continue
            pcts = sorted(sub['sleep_pct'].unique())
            nss  = sorted(sub['sleep_ns'].unique())
            tf_med = {(p, n): sub[(sub.sleep_pct == p) & (sub.sleep_ns == n)]['tflops_obs'].median()
                      for p in pcts for n in nss}
            base = seg[(seg.kernel == k) & (seg.phase == ph) &
                       (seg.sleep_pct == 0) & (seg.sleep_ns == 0)]
            base_tf = base['tflops_obs'].median() if not base.empty else None

            fig, ax = plt.subplots(figsize=(9, 7))
            heatmap(tf_med, pcts, nss, ax,
                    f'median TFLOPS  ({k}, phase={ph})'
                    + (f'   baseline={base_tf:.1f}' if base_tf else ''),
                    cbar_label='TFLOPS')
            p = os.path.join(out_dir, f'ws10_tflops_heatmap_{k}_{ph}.png')
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            print(f'[plot] {p}')

    # ─── Power heatmap per (kernel, phase) using enriched CSV if present ─────
    if enr is not None:
        cand_max = next((c for c in enr.columns if c in
                         ('power_max_W', 'p_max_W', 'max_W')), None)
        cand_avg = next((c for c in enr.columns if c in
                         ('power_avg_W', 'p_avg_W', 'avg_W', 'mean_W')), None)
        if cand_max or cand_avg:
            for k in kernels:
                for ph in phases:
                    sub = enr[(enr.kernel == k) & (enr.phase == ph) &
                              ~((enr.sleep_pct == 0) & (enr.sleep_ns == 0))]
                    if sub.empty:
                        continue
                    pcts = sorted(sub['sleep_pct'].unique())
                    nss  = sorted(sub['sleep_ns'].unique())
                    base = enr[(enr.kernel == k) & (enr.phase == ph) &
                               (enr.sleep_pct == 0) & (enr.sleep_ns == 0)]
                    fig, axes = plt.subplots(1, 2 if (cand_max and cand_avg) else 1,
                                             figsize=(16 if (cand_max and cand_avg) else 9, 7),
                                             squeeze=False)
                    col = 0
                    if cand_max:
                        d = {(p, n): sub[(sub.sleep_pct == p) & (sub.sleep_ns == n)][cand_max].median()
                             for p in pcts for n in nss}
                        b = base[cand_max].median() if not base.empty else None
                        heatmap(d, pcts, nss, axes[0, col],
                                f'peak W  ({k}, {ph})' + (f'  base={b:.0f}' if b else ''),
                                cbar_label='peak W')
                        col += 1
                    if cand_avg:
                        d = {(p, n): sub[(sub.sleep_pct == p) & (sub.sleep_ns == n)][cand_avg].median()
                             for p in pcts for n in nss}
                        b = base[cand_avg].median() if not base.empty else None
                        heatmap(d, pcts, nss, axes[0, col],
                                f'avg W  ({k}, {ph})' + (f'  base={b:.0f}' if b else ''),
                                cbar_label='avg W')
                    p = os.path.join(out_dir, f'ws10_power_heatmap_{k}_{ph}.png')
                    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
                    print(f'[plot] {p}')

    # ─── Power timeline grid per (kernel, phase) ─────────────────────────────
    # Each cell is the power(t) waveform for the 50 bursts of that (pct, ns).
    if pwr is not None:
        for k in kernels:
            for ph in phases:
                sub = seg[(seg.kernel == k) & (seg.phase == ph) &
                          ~((seg.sleep_pct == 0) & (seg.sleep_ns == 0))]
                if sub.empty:
                    continue
                pcts = sorted(sub['sleep_pct'].unique())
                nss  = sorted(sub['sleep_ns'].unique())

                fig, axes = plt.subplots(len(pcts), len(nss),
                                         figsize=(3.2 * len(nss), 1.6 * len(pcts)),
                                         sharex=False, sharey=True, squeeze=False)
                # Use the (kernel, phase) time window to clip the power CSV
                tmin = sub['t_start_ns'].min()
                tmax = sub['t_end_ns'].max()

                for i, p in enumerate(pcts):
                    for j, n in enumerate(nss):
                        ax = axes[i, j]
                        cell = sub[(sub.sleep_pct == p) & (sub.sleep_ns == n)]
                        if cell.empty:
                            ax.set_axis_off(); continue
                        t0 = cell['t_start_ns'].min()
                        t1 = cell['t_end_ns'].max()
                        # power samples within this cell's time window
                        msk = (pwr['t_ns'] >= t0) & (pwr['t_ns'] <= t1)
                        pp = pwr[msk]
                        if not pp.empty and 'p_inst_W' in pp.columns:
                            x = (pp['t_ns'] - t0) / 1e9   # s
                            ax.plot(x, pp['p_inst_W'], linewidth=0.5)
                            ax.axhline(600, color='red', linestyle=':',
                                       alpha=0.4)
                        ax.set_ylim(50, 800)
                        ax.set_title(f'pct={p} ns={n}', fontsize=8)
                        ax.tick_params(labelsize=7)
                        ax.grid(alpha=0.2)
                fig.suptitle(f'Power(t) over 50 bursts — {k}, phase={ph}', y=0.995)
                fig.supxlabel('t (s)  within cfg window')
                fig.supylabel('power instant (W)')
                p = os.path.join(out_dir, f'ws10_power_timeline_{k}_{ph}.png')
                fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
                print(f'[plot] {p}')


if __name__ == '__main__':
    main()
