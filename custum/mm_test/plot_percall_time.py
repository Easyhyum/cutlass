#!/usr/bin/env python3
"""
Per-call MatMul time plot + markdown table.

Reads a segments csv produced by eval_qwen3_timing.py and renders:
  - one grouped bar chart: per-call time (ms) for each (operator, M),
    grouped by backend.
  - a markdown table sorted by (operator, M, backend) with the per-call
    mean (us), std (us), and TFLOPS.
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def split_tag(tag):
    """'cublas:qkv_proj_M1024#3' -> ('cublas', 'qkv_proj', 1024, 3)"""
    if '#' in tag:
        tag, rep = tag.rsplit('#', 1)
        rep = int(rep)
    else:
        rep = 0
    be, cfg = tag.split(':', 1)
    # cfg = "{op}_M{val}" — op itself may contain underscores
    head, _, mtok = cfg.rpartition('_M')
    return be, head, int(mtok), rep


def mean(xs): return sum(xs) / len(xs) if xs else 0
def stdv(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5 if xs else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', required=True)
    ap.add_argument('--out-png',  required=True)
    ap.add_argument('--out-md',   required=True)
    ap.add_argument('--title',    default='per-MatMul time')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.segments)))
    if not rows:
        raise SystemExit('empty segments csv')

    grp = defaultdict(list)         # (op, M, backend) -> list of ms_avg
    op_order = []
    m_set = set()
    be_set = []
    for r in rows:
        be, op, M, _ = split_tag(r['backend'])
        ms = float(r['ms_avg'])
        grp[(op, M, be)].append(ms)
        if op not in op_order:
            op_order.append(op)
        m_set.add(M)
        if be not in be_set:
            be_set.append(be)
    Ms = sorted(m_set)
    bes = be_set                    # preserve insertion order, typically cublas, stream_k

    # ---- per-call stats ----
    stats = {}                      # (op, M, be) -> (mean_ms, std_ms, tflops)
    # Need K, N to compute tflops — pull from any row
    KN = {}                         # op -> (K, N)
    for r in rows:
        _, op, _, _ = split_tag(r['backend'])
        KN.setdefault(op, (int(r['K']), int(r['N'])))

    for (op, M, be), lst in grp.items():
        m = mean(lst); s = stdv(lst)
        K, N = KN[op]
        tf = 2.0 * M * K * N / (m * 1e-3) / 1e12 if m > 0 else 0
        stats[(op, M, be)] = (m, s, tf)

    # ---- plot: grouped bars per (op, M), groups along x, bars within group = backend ----
    labels = []
    for op in op_order:
        for M in Ms:
            if any((op, M, be) in stats for be in bes):
                labels.append((op, M))

    n_labels = len(labels)
    n_be = len(bes)
    bar_w = 0.8 / max(n_be, 1)

    fig, (ax_ms, ax_us) = plt.subplots(2, 1, figsize=(max(11, 0.55 * n_labels), 9),
                                       sharex=True)
    palette = {'cublas': 'tab:blue', 'stream_k': 'tab:orange'}
    for bi, be in enumerate(bes):
        means = []
        stds  = []
        for (op, M) in labels:
            mst = stats.get((op, M, be))
            if mst is None:
                means.append(0); stds.append(0)
            else:
                means.append(mst[0]); stds.append(mst[1])
        xs = [i + (bi - (n_be - 1) / 2) * bar_w for i in range(n_labels)]
        color = palette.get(be, None)
        ax_ms.bar(xs, means, width=bar_w, yerr=stds, capsize=2.5,
                  color=color, edgecolor='black', linewidth=0.4,
                  label=be, alpha=0.85)
        ax_us.bar(xs, [m * 1000.0 for m in means], width=bar_w,
                  yerr=[s * 1000.0 for s in stds], capsize=2.5,
                  color=color, edgecolor='black', linewidth=0.4,
                  label=be, alpha=0.85)

    for ax, ylabel in [(ax_ms, 'per-MatMul time (ms)'),
                       (ax_us, 'per-MatMul time (us)')]:
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

    ax_us.set_xticks(range(n_labels))
    ax_us.set_xticklabels([f'{op}\nM={M}' for (op, M) in labels],
                          rotation=60, fontsize=8, ha='right')
    ax_us.set_xlabel('(operator, M)')

    fig.suptitle(args.title, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out_png, dpi=120)
    print(f'[percall] plot: {args.out_png}')

    # ---- markdown table ----
    md = ['# Per-MatMul timing — ' + args.title, '']
    md.append('| operator | M | K | N | backend | mean (us) | std (us) | mean (ms) | TFLOPS |')
    md.append('|----------|---:|---:|---:|---------|---------:|--------:|---------:|------:|')
    for op in op_order:
        K, N = KN[op]
        for M in Ms:
            for be in bes:
                mst = stats.get((op, M, be))
                if mst is None: continue
                m_ms, s_ms, tf = mst
                md.append(
                    f'| {op} | {M} | {K} | {N} | {be} | '
                    f'{m_ms*1000:.2f} | {s_ms*1000:.2f} | '
                    f'{m_ms:.4f} | {tf:.1f} |')

    with open(args.out_md, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f'[percall] table: {args.out_md}')

    # also print short summary to stdout
    print()
    print(md[2])
    print(md[3])
    for line in md[4:]:
        print(line)


if __name__ == '__main__':
    main()
