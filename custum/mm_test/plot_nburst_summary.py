#!/usr/bin/env python3
"""
Per-config aggregate plot for N-burst RAMP evaluation.

Computes per-config statistics across bursts (mean ± std), shows side-by-side:
  - max_W mean ± std  (target: lower = better)
  - sm_p10 mean       (target: higher = better, towards 2430 MHz boost)
  - TFLOPS mean       (target: maintain baseline)
with horizontal guide lines at:
  - 600W TDP
  - 2430 MHz boost
  - baseline TFLOPS (computed from s100_* configs)
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def split_cfg(tag):
    rep = 0
    if '#' in tag:
        tag, rs = tag.rsplit('#', 1)
        try: rep = int(rs)
        except ValueError: pass
    if ':' in tag:
        be, cfg = tag.split(':', 1)
        return be, cfg, rep
    return tag, 'B', rep


def mean(xs): return sum(xs)/len(xs) if xs else 0
def stdv(xs):
    m = mean(xs)
    return (sum((x-m)**2 for x in xs)/max(len(xs)-1,1))**0.5 if xs else 0


def cfg_key(c):
    # 's70_p5' → (70, 5)
    if c[0] == 's' and '_p' in c:
        s = int(c.split('_')[0][1:]); st = int(c.split('_p')[1])
        return (s, st)
    return (999, 999)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments-with-power', required=True,
                    help='output of analyze_power.py (segments_with_power.csv)')
    ap.add_argument('--segments', required=True,
                    help='original segments.csv (to recover backend tags)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    enr = list(csv.DictReader(open(args.segments_with_power)))
    seg = list(csv.DictReader(open(args.segments)))
    assert len(enr) == len(seg)
    for e, s in zip(enr, seg):
        be, cfg, rep = split_cfg(s['backend'])
        e['cfg'] = cfg; e['rep'] = rep

    groups = defaultdict(list)
    for e in enr:
        groups[e['cfg']].append(e)

    # Baseline: best mean(tflops) among s100_* configs
    base_keys = [k for k in groups if k.startswith('s100')]
    base_key = max(base_keys,
                   key=lambda k: mean([float(r['tflops']) for r in groups[k]])) \
        if base_keys else None
    if base_key:
        b = groups[base_key]
        base_tf = mean([float(r['tflops']) for r in b])
        base_mxw = mean([float(r['max_W']) for r in b])
        base_mxw_std = stdv([float(r['max_W']) for r in b])
        base_p10 = mean([float(r['sm_p10']) for r in b])
    else:
        base_tf = base_mxw = base_p10 = base_mxw_std = 0

    cfgs = sorted(groups.keys(), key=cfg_key)
    n = len(cfgs)
    x = list(range(n))

    mx_mean = [mean([float(r['max_W']) for r in groups[c]]) for c in cfgs]
    mx_std  = [stdv([float(r['max_W']) for r in groups[c]]) for c in cfgs]
    av_mean = [mean([float(r['avg_W']) for r in groups[c]]) for c in cfgs]
    p10_mean = [mean([float(r['sm_p10']) for r in groups[c]]) for c in cfgs]
    tf_mean = [mean([float(r['tflops']) for r in groups[c]]) for c in cfgs]

    # 3 stacked panels
    fig, axes = plt.subplots(3, 1, figsize=(max(14, 0.55 * n), 11), sharex=True)
    ax_p, ax_c, ax_t = axes

    # Panel 1: max_W ± std
    bars = ax_p.bar(x, mx_mean, yerr=mx_std, capsize=3, color='tab:blue',
                    alpha=0.85, edgecolor='navy')
    # Color bars whose mean(max_W) < baseline - 2σ_baseline green
    threshold = base_mxw - 2 * base_mxw_std
    for i, (m, s_) in enumerate(zip(mx_mean, mx_std)):
        if m + s_ < threshold:
            bars[i].set_color('tab:green')
            bars[i].set_alpha(0.85)
    ax_p.axhline(600, ls='--', color='tab:blue', alpha=0.7, lw=1.4,
                 label='TDP 600 W')
    # Spike thresholds at 630/660/690/720W
    for lvl, alpha in [(630, 0.35), (660, 0.50), (690, 0.65), (720, 0.85)]:
        ax_p.axhline(lvl, ls=':', color='red', alpha=alpha, lw=1.0,
                     label=f'spike {lvl} W')
    if base_mxw > 0:
        ax_p.axhline(base_mxw, ls='-', color='gray', alpha=0.5,
                     label=f'baseline {base_mxw:.0f}W ({base_key})')
        ax_p.fill_between([-0.5, n-0.5],
                          base_mxw - base_mxw_std, base_mxw + base_mxw_std,
                          color='gray', alpha=0.15, label='baseline ±1σ')
    ax_p.set_ylabel('max_W per burst (W)\n(mean ± std)')
    ax_p.set_ylim(min(min(mx_mean) - max(mx_std)*1.5, 400), 760)
    ax_p.grid(True, alpha=0.3, axis='y')
    ax_p.legend(loc='lower right', fontsize=7, ncol=2)

    # Panel 2: sm_p10 + avg_W on twin axis
    ax_c.plot(x, p10_mean, 'o-', color='tab:red', markersize=6, label='sm_p10')
    ax_c.axhline(2430, ls='--', color='tab:red', alpha=0.4, label='Boost 2430 MHz')
    if base_p10 > 0:
        ax_c.axhline(base_p10, ls='-', color='gray', alpha=0.5,
                     label=f'baseline {base_p10:.0f} MHz')
    ax_c.set_ylabel('sm_p10 (MHz)', color='tab:red')
    ax_c.tick_params(axis='y', labelcolor='tab:red')
    ax_c.set_ylim(2000, 2500)
    ax_c.grid(True, alpha=0.3, axis='y')

    ax_c_r = ax_c.twinx()
    ax_c_r.plot(x, av_mean, 's--', color='tab:orange', markersize=4,
                alpha=0.7, label='avg_W')
    ax_c_r.set_ylabel('avg_W (W)', color='tab:orange')
    ax_c_r.tick_params(axis='y', labelcolor='tab:orange')
    ax_c_r.set_ylim(min(av_mean)-30, max(av_mean)+30)

    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_c_r.get_legend_handles_labels()
    ax_c.legend(h1+h2, l1+l2, loc='lower right', fontsize=8)

    # Panel 3: TFLOPS
    ax_t.bar(x, tf_mean, color='purple', alpha=0.75, edgecolor='indigo')
    if base_tf > 0:
        ax_t.axhline(base_tf, ls='-', color='gray', alpha=0.5,
                     label=f'baseline {base_tf:.1f} TF ({base_key})')
        ax_t.axhline(base_tf * 0.95, ls=':', color='red', alpha=0.5,
                     label='-5% threshold')
    ax_t.set_ylabel('TFLOPS (burst mean)')
    ax_t.set_ylim(0, max(tf_mean) * 1.05)
    ax_t.grid(True, alpha=0.3, axis='y')
    ax_t.legend(loc='lower right', fontsize=8)

    ax_t.set_xticks(x)
    ax_t.set_xticklabels(cfgs, rotation=70, fontsize=8)
    ax_t.set_xlabel('config (s<start%>_p<step%>)')

    title = args.title or (
        f'RAMP per-config aggregate — N={len(groups[cfgs[0]])} bursts/cfg, '
        f'{n} configs.   Green bars = max_W < baseline-2σ (significant)')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=120)
    print(f'plot: {args.out}')


if __name__ == '__main__':
    main()
