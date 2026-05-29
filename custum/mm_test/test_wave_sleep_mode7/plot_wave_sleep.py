#!/usr/bin/env python3
"""
Per-(kernel, active_pct) TFLOPS + power plots from segments.csv (+ analyze
output if available).

Output:
  ws_mode7_tflops_vs_pct.png    median TFLOPS vs active_pct, streamk vs sm80_v3
  ws_mode7_power_vs_pct.png     median peak power per burst vs active_pct (if power CSV provided)
  ws_mode7_timeline_<kernel>.png  per-burst tf observed over time (rows = active_pct)

Env:
  IN_DIR    folder with segments.csv (+ optionally segments_with_power.csv)
            default = this file's dir
  OUT_DIR   plot output dir (default = IN_DIR)
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    in_dir  = os.environ.get('IN_DIR',  HERE)
    out_dir = os.environ.get('OUT_DIR', in_dir)
    os.makedirs(out_dir, exist_ok=True)

    seg_path = os.path.join(in_dir, 'segments.csv')
    if not os.path.exists(seg_path):
        print(f'[plot] no segments.csv at {seg_path}'); sys.exit(1)
    seg = pd.read_csv(seg_path)

    pwr_path = os.path.join(in_dir, 'segments_with_power.csv')
    pwr = pd.read_csv(pwr_path) if os.path.exists(pwr_path) else None

    kernels = sorted(seg['kernel'].unique())
    pcts    = sorted(seg['active_pct'].unique())

    # ── median TFLOPS vs active_pct ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for k in kernels:
        med = [seg[(seg.kernel == k) & (seg.active_pct == p)]['tflops_obs'].median()
               for p in pcts]
        ax.plot(pcts, med, marker='o', label=k, linewidth=2)
    ax.set_xlabel('active SM %  (mode 7 SM gating)')
    ax.set_ylabel('median TFLOPS / burst')
    ax.set_title('Wave-sleep mode 7 SM gating — TFLOPS vs active%')
    ax.grid(alpha=0.3); ax.legend()
    p = os.path.join(out_dir, 'ws_mode7_tflops_vs_pct.png')
    fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
    print(f'[plot] {p}')

    # ── median peak / mean power vs active_pct (if available) ────────────────
    if pwr is not None:
        cand_max = next((c for c in pwr.columns if c in
                         ('power_max_W', 'p_max_W', 'max_W')), None)
        cand_avg = next((c for c in pwr.columns if c in
                         ('power_avg_W', 'p_avg_W', 'avg_W', 'mean_W')), None)
        if cand_max or cand_avg:
            fig, ax = plt.subplots(figsize=(9, 5))
            for k in kernels:
                if cand_max:
                    vals = [pwr[(pwr.kernel == k) & (pwr.active_pct == p)][cand_max].median()
                            for p in pcts]
                    ax.plot(pcts, vals, marker='o', label=f'{k} max', linewidth=2)
                if cand_avg:
                    vals = [pwr[(pwr.kernel == k) & (pwr.active_pct == p)][cand_avg].median()
                            for p in pcts]
                    ax.plot(pcts, vals, marker='s', linestyle='--',
                            label=f'{k} avg')
            ax.axhline(600, color='red', linestyle=':', alpha=0.5, label='TDP 600W')
            ax.set_xlabel('active SM %')
            ax.set_ylabel('W (per burst)')
            ax.set_title('Wave-sleep mode 7 SM gating — power vs active%')
            ax.grid(alpha=0.3); ax.legend()
            p = os.path.join(out_dir, 'ws_mode7_power_vs_pct.png')
            fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
            print(f'[plot] {p}')
        else:
            print(f'[plot] {pwr_path} has no power_max_W / power_avg_W columns — '
                  f'cols={list(pwr.columns)}')

    # ── per-burst timeline per kernel (rows = active_pct) ────────────────────
    for k in kernels:
        sub_k = seg[seg.kernel == k]
        fig, axes = plt.subplots(len(pcts), 1, figsize=(13, 1.5 * len(pcts)),
                                 sharex=True, squeeze=False)
        for ax, pct in zip(axes[:, 0], pcts):
            sub = sub_k[sub_k.active_pct == pct].sort_values('burst_idx')
            ax.plot(sub['burst_idx'], sub['tflops_obs'],
                    marker='.', linewidth=1)
            ax.set_ylabel(f'pct={pct}\nTFLOPS')
            ax.grid(alpha=0.25)
            tf_med = sub['tflops_obs'].median()
            ax.axhline(tf_med, color='red', linestyle=':', alpha=0.5)
            ax.text(0.99, 0.92, f'med={tf_med:.1f}', transform=ax.transAxes,
                    ha='right', va='top', fontsize=9)
        axes[-1, 0].set_xlabel('burst idx')
        fig.suptitle(f'{k}  — per-burst TFLOPS across active_pct (mode 7)',
                     y=0.995)
        p = os.path.join(out_dir, f'ws_mode7_timeline_{k}.png')
        fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
        print(f'[plot] {p}')


if __name__ == '__main__':
    main()
