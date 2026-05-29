#!/usr/bin/env python3
"""
Line charts for Qwen3 2-way compare: TFLOPS vs M and per-cycle ms vs M.

Top row    : TFLOPS line per (op, backend), labeled at each point
Bottom row : per-cycle wall time (ms) per (op, backend), labeled at each point
One column per operator (qkv, o, gate_up, down, lm_head).

Inputs: the two segments csv produced by run_qwen3_compare.sh.
Also writes a markdown table covering all (op, M, backend) rows.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_segments(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r['M']      = int(r['M'])
        r['K']      = int(r['K'])
        r['N']      = int(r['N'])
        r['iters']  = int(r['iters'])
        r['ms_avg'] = float(r['ms_avg'])
        r['tflops'] = float(r['tflops'])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments-cublas',  required=True)
    ap.add_argument('--segments-streamk', required=True)
    ap.add_argument('--out',              required=True)
    ap.add_argument('--out-md',           required=True)
    ap.add_argument('--title',            default='')
    args = ap.parse_args()

    data = {
        'cublas':   load_segments(args.segments_cublas),
        'stream_k': load_segments(args.segments_streamk),
    }

    # Preserve operator order from cublas (assumed identical sweep order)
    op_order = []
    for r in data['cublas']:
        if r['operator'] not in op_order:
            op_order.append(r['operator'])
    for r in data['stream_k']:
        if r['operator'] not in op_order:
            op_order.append(r['operator'])

    # Group by (op, backend) -> list of (M, ms_avg, tflops, K, N)
    grouped = defaultdict(list)
    for be in ('cublas', 'stream_k'):
        for r in data[be]:
            grouped[(r['operator'], be)].append(
                (r['M'], r['ms_avg'], r['tflops'], r['K'], r['N']))
    for k in grouped:
        grouped[k].sort(key=lambda x: x[0])

    n_ops = len(op_order)
    fig, axes = plt.subplots(2, n_ops, figsize=(max(16, 4 * n_ops), 9),
                             squeeze=False)

    palette = {'cublas': 'tab:blue', 'stream_k': 'tab:orange'}
    marker  = {'cublas': 'o', 'stream_k': 's'}

    for col, op in enumerate(op_order):
        ax_tf = axes[0, col]
        ax_ms = axes[1, col]

        # gather K, N from first nonempty
        K = N = None
        for be in ('cublas', 'stream_k'):
            if grouped.get((op, be)):
                _, _, _, K, N = grouped[(op, be)][0]
                break

        for be in ('cublas', 'stream_k'):
            rows = grouped.get((op, be), [])
            if not rows:
                continue
            Ms  = [r[0] for r in rows]
            mss = [r[1] for r in rows]
            tfs = [r[2] for r in rows]
            color = palette[be]
            mk    = marker[be]

            ax_tf.plot(Ms, tfs, '-', color=color, marker=mk, markersize=4.5,
                       linewidth=1.6, label=be, alpha=0.92)
            ax_ms.plot(Ms, mss, '-', color=color, marker=mk, markersize=4.5,
                       linewidth=1.6, label=be, alpha=0.92)

            # Point labels (TFLOPS row: 1 decimal, ms row: short)
            for x, tf in zip(Ms, tfs):
                ax_tf.annotate(f'{tf:.0f}', xy=(x, tf), xytext=(0, 5),
                               textcoords='offset points', ha='center',
                               fontsize=6.5, color=color)
            for x, ms in zip(Ms, mss):
                lbl = (f'{ms*1000:.0f}us' if ms < 1.0
                       else f'{ms:.1f}ms' if ms < 100
                       else f'{ms:.0f}ms')
                ax_ms.annotate(lbl, xy=(x, ms), xytext=(0, 5),
                               textcoords='offset points', ha='center',
                               fontsize=6.5, color=color)

        ax_tf.set_xscale('log', base=2)
        ax_ms.set_xscale('log', base=2)
        ax_ms.set_yscale('log')

        ax_tf.set_title(f"{op}\nK={K}, N={N}", fontsize=10)
        ax_tf.set_ylabel('TFLOPS' if col == 0 else '')
        ax_ms.set_ylabel('ms_avg (per call, log)' if col == 0 else '')
        ax_ms.set_xlabel('M (log₂)')

        ax_tf.grid(True, which='both', alpha=0.3)
        ax_ms.grid(True, which='both', alpha=0.3)
        ax_tf.legend(loc='lower right', fontsize=8)
        ax_ms.legend(loc='upper left', fontsize=8)

    if args.title:
        fig.suptitle(args.title, fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f'lines: saved {args.out}')

    # ---- markdown table ----
    md = ['# ' + (args.title or 'Qwen3 sweep'), '']
    md.append('| operator | K | N | M | backend | iters | ms_avg | TFLOPS |')
    md.append('|----------|---:|---:|---:|---------|---:|--------:|------:|')
    by_om = defaultdict(dict)
    for be in ('cublas', 'stream_k'):
        for r in data[be]:
            by_om[(r['operator'], r['M'])][be] = r
    for op in op_order:
        Ms = sorted({M for (o, M) in by_om if o == op})
        for M in Ms:
            for be in ('cublas', 'stream_k'):
                r = by_om[(op, M)].get(be)
                if not r: continue
                md.append(f'| {op} | {r["K"]} | {r["N"]} | {M} | {be} | '
                          f'{r["iters"]} | {r["ms_avg"]:.4f} | {r["tflops"]:.2f} |')
    with open(args.out_md, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f'lines: table {args.out_md}')


if __name__ == '__main__':
    main()
