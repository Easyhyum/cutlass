#!/usr/bin/env python3
"""
Method A multi-priority analysis (user's ranking):

  Priority 1 :  reduce first power spike  (max_W ↓)  with perf preserved
  Priority 2 :  preserve SM clock          (sm_p10 ↑)  → downstream layers
  Priority 3 :  energy per token           (tokens/W ↑, energy_J ↓)

Hard floor:  tflops_ratio >= PERF_FLOOR  (default 0.95).  A config that drops
performance below the floor is *not* a candidate for any priority ranking
(because we explicitly do not want to sacrifice throughput).

Input  : segments_with_power.csv  (output of analyze_power.py)
Output :
  - pareto.csv           full per-config metrics with ranks
  - pareto.png           per-group (backend×op×M) Pareto scatter for Priorities
                         1 and 2, with composite recommendation highlighted
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


PERF_FLOOR = 0.95  # tflops_ratio threshold


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r['tflops']  = float(r['tflops'])
            r['avg_W']   = float(r['avg_W'])
            r['max_W']   = float(r['max_W'])
            r['sm_p10']  = float(r['sm_p10'])
            r['sm_p50']  = float(r['sm_p50'])
            r['sm_max']  = float(r['sm_max'])
            r['M']       = int(r['M'])
            r['K']       = int(r['K'])
            r['N']       = int(r['N'])
            r['n_samples'] = int(r['n_samples'])
            rows.append(r)
    return rows


def split_backend(tag):
    """Split 'cutlass_sm80:P-100-8#3' into ('cutlass_sm80', 'P-100-8', 3).
    Backward-compatible: tags without ':' or '#' return ('tag', 'B', 0).
    """
    repeat = 0
    if '#' in tag:
        tag, rep_s = tag.rsplit('#', 1)
        try:
            repeat = int(rep_s)
        except ValueError:
            pass
    if ':' in tag:
        a, b = tag.split(':', 1)
        return a, b, repeat
    return tag, 'B', repeat


def aggregate_repeats(rows):
    """Collapse N repeats per (backend, op, M, cfg) into one row with
    robust spike statistics:
       max_W   -> max  across repeats   (worst observed spike)
       avg_W   -> mean across repeats
       sm_p10  -> min  across repeats   (worst-throttled clock observed)
       sm_p50  -> mean
       tflops  -> mean (+ std)
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        be, cfg, rep = split_backend(r['backend'])
        r['be_base'] = be; r['cfg'] = cfg; r['repeat_idx'] = rep
        key = (be, cfg, r['operator'], r['M'])
        groups[key].append(r)

    agg = []
    for key, group in groups.items():
        be, cfg, op, M = key
        n = len(group)
        max_Ws  = [g['max_W']  for g in group]
        avg_Ws  = [g['avg_W']  for g in group]
        sm_p10s = [g['sm_p10'] for g in group]
        sm_p50s = [g['sm_p50'] for g in group]
        tflopss = [g['tflops'] for g in group]
        # one representative (use the first for K/N/etc.)
        ref = group[0]
        merged = dict(ref)
        merged['n_repeats'] = n
        merged['max_W']         = max(max_Ws)      # ← worst spike across N runs
        merged['max_W_p95']     = sorted(max_Ws)[int(0.95*(n-1))]
        merged['max_W_mean']    = sum(max_Ws)/n
        merged['avg_W']         = sum(avg_Ws)/n    # avg of avg
        merged['avg_W_max']     = max(avg_Ws)
        merged['sm_p10']        = min(sm_p10s)     # ← worst-throttled
        merged['sm_p10_mean']   = sum(sm_p10s)/n
        merged['sm_p50']        = sum(sm_p50s)/n
        merged['tflops']        = sum(tflopss)/n   # mean throughput
        merged['tflops_min']    = min(tflopss)
        merged['tflops_max']    = max(tflopss)
        if n > 1:
            mean_t = merged['tflops']
            merged['tflops_std'] = (sum((t-mean_t)**2 for t in tflopss)/(n-1))**0.5
        else:
            merged['tflops_std'] = 0.0
        merged['backend']       = f'{be}:{cfg}'   # drop the #repeat suffix
        agg.append(merged)
    return agg


def derive_metrics(r):
    """Compute energy + tokens/W metrics."""
    # ms reconstructible from tflops and problem size
    flops = 2.0 * r['M'] * r['K'] * r['N']
    ms = flops / (r['tflops'] * 1e12) * 1e3
    sec = ms / 1000.0

    r['ms_avg']           = ms
    r['energy_J']         = r['avg_W'] * sec                       # per kernel invocation
    # tokens_per_W : LLM batch dim M = tokens processed per invocation
    #   tokens/sec / Watt = M / sec / W  =  M / (avg_W * sec)
    r['tokens_per_W']     = r['M'] / (r['avg_W'] * sec)            # tokens·s⁻¹·W⁻¹
    # energy_per_token (mJ/token)
    r['mJ_per_token']     = (r['energy_J'] / r['M']) * 1000.0      # mJ per token
    # compute-per-watt (architecture-neutral proxy)
    r['tflops_per_W']     = r['tflops'] / r['avg_W']               # TFLOPS/W


def analyse(rows):
    """Attach baseline-relative metrics and rank columns.
    Expects rows already aggregated across repeats (via aggregate_repeats).
    """
    # split backend tag (no longer has #repeat suffix after aggregation)
    for r in rows:
        if 'be_base' not in r:
            be, cfg, _ = split_backend(r['backend'])
            r['be_base'] = be; r['cfg'] = cfg
        derive_metrics(r)

    # group by (be_base, op, M)
    groups = defaultdict(dict)
    for r in rows:
        groups[(r['be_base'], r['operator'], r['M'])][r['cfg']] = r

    enriched = []
    for (be, op, M), cfgs in groups.items():
        base = cfgs.get('B')
        if not base:
            print(f'WARN: no baseline B for {be}/{op}/M={M}, skipping')
            continue
        for cfg, r in cfgs.items():
            r['tflops_baseline']        = base['tflops']
            r['maxW_baseline']          = base['max_W']
            r['avgW_baseline']          = base['avg_W']
            r['smP10_baseline']         = base['sm_p10']
            r['smP50_baseline']         = base['sm_p50']
            r['energy_J_baseline']      = base['energy_J']
            r['tokensPerW_baseline']    = base['tokens_per_W']
            r['mJperTok_baseline']      = base['mJ_per_token']
            r['tflopsPerW_baseline']    = base['tflops_per_W']

            r['tflops_ratio']           = r['tflops']      / base['tflops']
            r['maxW_drop']              = base['max_W']    - r['max_W']        # +good
            r['avgW_drop']              = base['avg_W']    - r['avg_W']        # +good
            r['smP10_gain']             = r['sm_p10']      - base['sm_p10']    # +good
            r['smP50_gain']             = r['sm_p50']      - base['sm_p50']    # +good
            r['energy_J_drop']          = base['energy_J'] - r['energy_J']     # +good
            r['mJperTok_drop']          = base['mJ_per_token'] - r['mJ_per_token']  # +good
            r['tokensPerW_gain']        = r['tokens_per_W']- base['tokens_per_W']   # +good
            r['tflopsPerW_gain']        = r['tflops_per_W']- base['tflops_per_W']   # +good
            r['perf_loss_pct']          = (1 - r['tflops_ratio']) * 100
            r['perf_ok']                = r['tflops_ratio'] >= PERF_FLOOR
            enriched.append(r)

    # ── per-group ranking by user priorities ────────────────────────────────
    for (be, op, M), cfgs in groups.items():
        local = [r for r in enriched
                 if r['be_base']==be and r['operator']==op and r['M']==M]
        # candidates = perf_ok only (PERF_FLOOR hard floor)
        cands = [r for r in local if r['perf_ok'] and r['cfg'] != 'B']

        # Priority 1: minimise max_W (largest maxW_drop)
        for r in local: r['rank_p1'] = None
        for i, r in enumerate(sorted(cands, key=lambda x: -x['maxW_drop'])):
            r['rank_p1'] = i + 1

        # Priority 2: maximise sm_p10 (largest smP10_gain)
        for r in local: r['rank_p2'] = None
        for i, r in enumerate(sorted(cands, key=lambda x: -x['smP10_gain'])):
            r['rank_p2'] = i + 1

        # Priority 3: maximise tokens/W  (== minimise mJ/token)
        for r in local: r['rank_p3'] = None
        for i, r in enumerate(sorted(cands, key=lambda x: -x['tokensPerW_gain'])):
            r['rank_p3'] = i + 1

        # Composite lexicographic: (perf_ok, maxW_drop, smP10_gain, tokensPerW_gain)
        def composite_key(r):
            # perf_ok must be True (else huge penalty);
            # then P1 max_W↓, then P2 sm_p10↑, then P3 tokens/W↑
            return (
                0 if r['perf_ok'] else 1,
                -r['maxW_drop'],
                -r['smP10_gain'],
                -r['tokensPerW_gain'],
            )
        for r in local: r['rank_composite'] = None
        sorted_local = sorted([r for r in local if r['cfg'] != 'B'],
                              key=composite_key)
        for i, r in enumerate(sorted_local):
            r['rank_composite'] = i + 1

    return enriched, groups


def write_csv(enriched, path):
    fields = ['be_base','cfg','operator','M','K','N','n_repeats',
              'tflops','tflops_min','tflops_max','tflops_std',
              'tflops_baseline','tflops_ratio','perf_loss_pct','perf_ok',
              'avg_W','avg_W_max','avgW_baseline','avgW_drop',
              'max_W','max_W_mean','max_W_p95','maxW_baseline','maxW_drop',
              'sm_p10','sm_p10_mean','smP10_baseline','smP10_gain',
              'sm_p50','smP50_baseline','smP50_gain',
              'energy_J','energy_J_baseline','energy_J_drop',
              'mJ_per_token','mJperTok_baseline','mJperTok_drop',
              'tokens_per_W','tokensPerW_baseline','tokensPerW_gain',
              'tflops_per_W','tflopsPerW_baseline','tflopsPerW_gain',
              'rank_p1','rank_p2','rank_p3','rank_composite']
    enriched.sort(key=lambda r: (r['be_base'], r['operator'], r['M'],
                                  r['rank_composite'] or 9999))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in enriched:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f'csv: {path}')


def print_rankings(groups):
    print()
    print('=' * 110)
    print(f'PERF_FLOOR = {PERF_FLOOR}  '
          '(configs below this tflops_ratio are EXCLUDED from rankings)')
    print('=' * 110)
    for (be, op, M), cfgs in groups.items():
        base = cfgs.get('B')
        if not base: continue
        print()
        n_rep = base.get('n_repeats', 1)
        print(f'### {be}  {op}  M={M}   '
              f'(N={n_rep} repeats, max_W = max-of-max)')
        print(f'    baseline: {base["tflops"]:.1f}TF ±{base.get("tflops_std",0):.1f}, '
              f'max_W={base["max_W"]:.0f}W (mean={base.get("max_W_mean", base["max_W"]):.0f}), '
              f'avg_W={base["avg_W"]:.0f}W, '
              f'sm_p10_min={base["sm_p10"]:.0f}MHz, '
              f'tok/W={base["tokens_per_W"]:.1f}, '
              f'mJ/tok={base["mJ_per_token"]:.3f}')

        local = list(cfgs.values())
        cands = [r for r in local if r.get('perf_ok') and r['cfg'] != 'B']
        if not cands:
            print('  (no config passes PERF_FLOOR)')
            continue

        # Priority 1
        p1 = sorted(cands, key=lambda x: -x['maxW_drop'])[:3]
        print(f'  P1 (max_W ↓ with tflops_ratio ≥ {PERF_FLOOR}):')
        for r in p1:
            print(f'    {r["cfg"]:>16s}  tflops={r["tflops"]:6.1f} '
                  f'({r["tflops_ratio"]*100:5.1f}%)  '
                  f'max_W={r["max_W"]:6.1f}  Δ={r["maxW_drop"]:+5.1f}W  '
                  f'sm_p10={r["sm_p10"]:>4.0f}  '
                  f'tok/W={r["tokens_per_W"]:.1f}')

        # Priority 2
        p2 = sorted(cands, key=lambda x: -x['smP10_gain'])[:3]
        print(f'  P2 (sm_p10 ↑ with tflops_ratio ≥ {PERF_FLOOR}):')
        for r in p2:
            print(f'    {r["cfg"]:>16s}  tflops={r["tflops"]:6.1f} '
                  f'({r["tflops_ratio"]*100:5.1f}%)  '
                  f'sm_p10={r["sm_p10"]:>4.0f}  Δ={r["smP10_gain"]:+4.0f}MHz  '
                  f'max_W={r["max_W"]:6.1f}  '
                  f'tok/W={r["tokens_per_W"]:.1f}')

        # Priority 3 (energy / token)
        p3 = sorted(cands, key=lambda x: -x['tokensPerW_gain'])[:3]
        print(f'  P3 (tokens/W ↑  &  mJ/token ↓  with tflops_ratio ≥ {PERF_FLOOR}):')
        for r in p3:
            print(f'    {r["cfg"]:>16s}  tflops={r["tflops"]:6.1f} '
                  f'({r["tflops_ratio"]*100:5.1f}%)  '
                  f'tok/W={r["tokens_per_W"]:7.1f} (Δ{r["tokensPerW_gain"]:+6.1f})  '
                  f'mJ/tok={r["mJ_per_token"]:7.3f} (Δ{r["mJperTok_drop"]:+5.3f})')

        # Composite (1 > 2 > 3 lexicographic, hard perf floor)
        comp = sorted(cands, key=lambda x: x['rank_composite'])[:3]
        print(f'  ⇒ COMPOSITE (P1 > P2 > P3, perf_floor enforced):')
        for r in comp:
            print(f'    [{r["rank_composite"]:2d}] {r["cfg"]:>14s}  '
                  f'tflops={r["tflops"]:6.1f} ({r["tflops_ratio"]*100:5.1f}%)  '
                  f'max_W={r["max_W"]:6.1f}(-{r["maxW_drop"]:.1f})  '
                  f'sm_p10={r["sm_p10"]:>4.0f}({r["smP10_gain"]:+.0f})  '
                  f'tok/W={r["tokens_per_W"]:.1f}  '
                  f'mJ/tok={r["mJ_per_token"]:.3f}')


def plot(enriched, groups, out_path):
    by_group = defaultdict(list)
    for r in enriched:
        by_group[(r['be_base'], r['operator'], r['M'])].append(r)

    n = len(by_group)
    # 3 panels (P1, P2, P3) per group, stacked vertically
    cols = 2
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n * 3, cols,
                              figsize=(cols * 7.5, rows_n * 10.5),
                              squeeze=False)

    def color_for(cfg):
        if cfg == 'B':                return 'black'
        if cfg.startswith('P-'):      return 'tab:blue'
        if cfg.startswith('PR-'):     return 'tab:orange'
        if cfg.startswith('U-'):      return 'tab:green'
        return 'gray'

    def render_panel(ax, recs, x_key, x_label, title, y_floor):
        ax.axhline(PERF_FLOOR, color='red', ls='--', lw=0.9)
        ax.axhline(1.0, color='gray', ls=':', lw=0.7, alpha=0.7)
        ax.axvline(0.0, color='gray', ls=':', lw=0.7, alpha=0.7)
        for r in recs:
            c = color_for(r['cfg'])
            is_comp_best = (r.get('rank_composite') == 1)
            mk, ms_, alpha = ('*', 16, 1.0) if is_comp_best else ('o', 8, 0.85)
            ax.plot(r[x_key], r['tflops_ratio'],
                    marker=mk, color=c, ms=ms_, alpha=alpha,
                    markeredgecolor='red' if is_comp_best else None,
                    markeredgewidth=1.5 if is_comp_best else 0)
            ax.annotate(r['cfg'], (r[x_key], r['tflops_ratio']),
                        textcoords='offset points', xytext=(5, 4),
                        fontsize=6.5, color=c)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(x_label)
        ax.set_ylabel('TFLOPS ratio   ↑')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_floor, 1.15)

    for gi, ((be, op, M), recs) in enumerate(by_group.items()):
        base = next((r for r in recs if r['cfg'] == 'B'), None)
        y_floor = min(0.5, min(r['tflops_ratio'] for r in recs) - 0.05)

        # Top: P1 (max_W drop)
        ax_p1 = axes[(gi // cols) * 3 + 0, gi % cols]
        title1 = f'P1: spike↓ × perf-retention   |   {be}  {op}  M={M}'
        if base:
            title1 += (f'\nbaseline: {base["tflops"]:.1f}TF, '
                       f'max_W={base["max_W"]:.0f}W, '
                       f'sm_p10={base["sm_p10"]:.0f}MHz, '
                       f'tok/W={base["tokens_per_W"]:.1f}')
        render_panel(ax_p1, recs, 'maxW_drop',
                     'max_W drop vs baseline (W)   →',
                     title1, y_floor)

        # Middle: P2 (sm_p10 gain)
        ax_p2 = axes[(gi // cols) * 3 + 1, gi % cols]
        render_panel(ax_p2, recs, 'smP10_gain',
                     'sm_p10 gain vs baseline (MHz)   →',
                     'P2: clock-preservation × perf-retention', y_floor)

        # Bottom: P3 (tokens/W gain)
        ax_p3 = axes[(gi // cols) * 3 + 2, gi % cols]
        render_panel(ax_p3, recs, 'tokensPerW_gain',
                     'tokens/W gain vs baseline   →   (↔ mJ/token ↓)',
                     'P3: energy efficiency × perf-retention', y_floor)

    legend = [
        plt.Line2D([0],[0], marker='s', color='w', mfc='tab:blue',  ms=8, label='P-* (prologue only)'),
        plt.Line2D([0],[0], marker='s', color='w', mfc='tab:orange',ms=8, label='PR-* (prologue+periodic)'),
        plt.Line2D([0],[0], marker='s', color='w', mfc='tab:green', ms=8, label='U-* (uniform throttle)'),
        plt.Line2D([0],[0], marker='*', color='black', ms=14, mec='red',
                   mew=1.5, lw=0, label='composite #1 (P1>P2>P3)'),
        plt.Line2D([0],[0], color='red', ls='--', label=f'perf floor {PERF_FLOOR}'),
    ]
    fig.legend(handles=legend, loc='upper right', ncol=5, fontsize=8,
               bbox_to_anchor=(0.99, 1.0))

    fig.suptitle(
        f'Method A multi-priority analysis  '
        f'(perf hard-floor = {PERF_FLOOR})\n'
        'P1: max_W↓ + perf↑   >   P2: sm_p10↑ + perf↑   >   '
        'P3: tokens/W↑ (mJ/token↓) + perf↑',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    print(f'plot: {out_path}')


def main():
    global PERF_FLOOR
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments-with-power', required=True)
    ap.add_argument('--pareto-csv',          required=True)
    ap.add_argument('--plot',                required=True)
    ap.add_argument('--perf-floor',          type=float, default=PERF_FLOOR,
                    help=f'tflops_ratio hard floor (default {PERF_FLOOR})')
    args = ap.parse_args()
    PERF_FLOOR = args.perf_floor

    rows = load(args.segments_with_power)
    # If 'backend' is missing from enriched csv, recover from original segments.
    with open(args.segments_with_power) as f:
        header = next(csv.reader(f))
    if 'backend' not in header:
        seg_path = args.segments_with_power.replace('_with_power.csv', '.csv')
        # Build by ordered position match (analyze_power preserves order)
        with open(seg_path) as f:
            orig = list(csv.DictReader(f))
        if len(orig) == len(rows):
            for r, o in zip(rows, orig):
                r['backend'] = o['backend']
        else:
            # fallback: best-effort key match
            from collections import defaultdict
            buckets = defaultdict(list)
            for o in orig:
                buckets[(o['operator'], int(o['M']),
                         round(float(o['tflops']), 2))].append(o['backend'])
            for r in rows:
                k = (r['operator'], r['M'], round(r['tflops'], 2))
                if buckets[k]:
                    r['backend'] = buckets[k].pop(0)
                else:
                    r['backend'] = 'unknown:B'

    # ── aggregate across repeats ──
    agg_rows = aggregate_repeats(rows)
    n_in, n_out = len(rows), len(agg_rows)
    print(f'aggregated {n_in} segment rows -> {n_out} (be, cfg, op, M) cells  '
          f'(avg {n_in/max(n_out,1):.1f} repeats/cell)')

    enriched, groups = analyse(agg_rows)
    write_csv(enriched, args.pareto_csv)
    plot(enriched, groups, args.plot)
    print_rankings(groups)


if __name__ == '__main__':
    main()
