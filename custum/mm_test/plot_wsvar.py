#!/usr/bin/env python3
"""
Group + bar plot of the 4 variants (A baseline, B mid-bubble, C all-wave, D quartile).
Reads logs/<tag>_segments[_with_power].csv.

Usage:  python plot_wsvar.py <tag>
"""
import csv, os, sys
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tag = sys.argv[1]
log = 'logs'
seg = list(csv.DictReader(open(f'{log}/{tag}_segments.csv')))
enr = list(csv.DictReader(open(f'{log}/{tag}_segments_with_power.csv')))
assert len(seg) == len(enr)


def cfg_name(b): return b.split(':',1)[1].split('#',1)[0]


# Determine variant group per cfg name
GROUPS = {  # prefix patterns
    's100_step0':  'A',
    'aw_':         'C',
    'q_':          'D',
}
def group_of(c):
    if c.startswith('s100_step0'): return 'A'
    if c.startswith('aw_'):        return 'C'
    if c.startswith('q_'):         return 'D'
    if c.startswith('o_'):         return 'E'   # octile (mode 0)
    if c.startswith('oa_'):        return 'E'   # octile + all-wave (mode 1)
    if c.startswith('h_'):         return 'F'   # 2-step ("half")
    if c.startswith('u_'):         return 'U'   # uniform per-iter (mode 4)
    if c.startswith('p_'):         return 'P'   # every N-th iter (mode 5)
    if c.startswith('n_'):         return 'N'   # every N-th iter no-sync (mode 6)
    if c.startswith('g_'):         return 'G'   # SM gating (mode 7)
    if c.startswith('r_'):         return 'R'   # rotational stagger (mode 8)
    if c.startswith('a_sms'):      return 'V'   # avail_sms (streamk kernel-level)
    if c.startswith('ms_'):        return 'M'   # multi-stream
    if c.startswith('M') and c[1:].isdigit(): return 'X'  # M-sweep single cfgs
    if c.startswith('single_M'):    return 'X'   # M-sweep single
    if c.startswith('chunk_M'):     return 'K'   # M-sweep chunked (torch)
    if c.startswith('tchunk_M'):    return 'K'   # explicit torch-chunked
    if c.startswith('kchunk_M'):    return 'L'   # kernel-chunked
    if c.startswith('ochunk_M'):    return 'O'   # overlapped chunked
    if c.startswith('s') and 'mid' in c: return 'B'
    return '?'

rows = defaultdict(list)
for s, e in zip(seg, enr):
    c = cfg_name(s['backend'])
    rows[c].append({
        'group':  group_of(c),
        'max_W':  float(e['max_W']),
        'avg_W':  float(e['avg_W']),
        'sm_p10': float(e['sm_p10']),
        'tflops': float(e['tflops']),
    })

base = rows.get('s100_step0')
if base is None:
    raise SystemExit('no baseline (s100_step0) in segments')
base_mx_med = np.median([r['max_W']  for r in base])
base_mx_mean= np.mean  ([r['max_W']  for r in base])
base_smp    = np.median([r['sm_p10'] for r in base])
base_tf     = np.median([r['tflops'] for r in base])
print(f'baseline  N={len(base)}  max_W median={base_mx_med:.1f}  '
      f'mean={base_mx_mean:.1f}  sm_p10={base_smp:.0f}  TFLOPS={base_tf:.1f}')

# Build summary per cfg
summary = []
for c, rs in rows.items():
    mx  = np.array([r['max_W']  for r in rs])
    avW = np.array([r['avg_W']  for r in rs])
    smp = np.array([r['sm_p10'] for r in rs])
    tf  = np.array([r['tflops'] for r in rs])
    summary.append({
        'cfg':    c,
        'group':  rs[0]['group'],
        'n':      len(rs),
        'mx_med': np.median(mx),
        'mx_mean':mx.mean(),
        'mx_max': mx.max(),
        'mx_std': mx.std(),
        'sm_med': np.median(smp),
        'tf_med': np.median(tf),
        'avW_med':np.median(avW),
    })

order = ['A', 'B', 'C', 'D', 'E', 'F', 'U', 'P', 'N', 'G', 'R', 'V', 'M', 'X', 'K', 'L', 'O']
gnames = {'A':'A: baseline (no code)',
          'B':'B: wave-0 + mid bubble (sync)',
          'C':'C: ALL-wave staircase',
          'D':'D: quartile-shape staircase',
          'E':'E: octile-shape staircase',
          'F':'F: 2-step (half) staircase',
          'U':'U: uniform per-iter (mode 4)',
          'P':'P: every N-th iter (mode 5, freq-dial)',
          'N':'N: every N-th iter NO-SYNC (mode 6)',
          'G':'G: SM gating, MPS-style (mode 7)',
          'R':'R: ROTATIONAL stagger (mode 8, 100% SM)',
          'V':'V: streamk avail_sms (kernel-level)',
          'M':'M: multi-stream concurrent',
          'X':'X: M-sweep (single, full GEMM)',
          'K':'K: torch-chunked (single stream)',
          'L':'L: kernel-chunked (C++ loop, single stream)',
          'O':'O: overlap-chunked (2 streams alternating)'}
summary.sort(key=lambda r: (order.index(r['group']), r['cfg']))

# print table
print()
print(f'{"grp":>3s} {"cfg":>22s} {"N":>3s}  {"mx_med":>7s} {"mx_mean":>8s} {"mx_max":>7s} '
      f'{"mx_std":>6s}  {"sm_med":>7s}  {"tf_med":>7s}  {"d_W med":>8s}  {"d_TF%":>6s}')
print('-' * 110)
for r in summary:
    d_w = r['mx_med'] - base_mx_med
    d_tf = (r['tf_med']/base_tf - 1)*100
    print(f'{r["group"]:>3s} {r["cfg"]:>22s} {r["n"]:>3d}  '
          f'{r["mx_med"]:>7.1f} {r["mx_mean"]:>8.1f} {r["mx_max"]:>7.1f} '
          f'{r["mx_std"]:>6.1f}  {r["sm_med"]:>7.0f}  {r["tf_med"]:>7.2f}  '
          f'{d_w:>+8.1f}  {d_tf:>+6.2f}')

# CSV summary
out_csv = f'{log}/{tag}_variant_summary.csv'
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['group','cfg','n','mx_med','mx_mean','mx_max','mx_std',
                'sm_med','tf_med','avW_med','d_max_W_med','d_tflops_pct'])
    for r in summary:
        d_w = r['mx_med'] - base_mx_med
        d_tf = (r['tf_med']/base_tf - 1)*100
        w.writerow([r['group'], r['cfg'], r['n'],
                    f'{r["mx_med"]:.2f}', f'{r["mx_mean"]:.2f}', f'{r["mx_max"]:.2f}', f'{r["mx_std"]:.2f}',
                    f'{r["sm_med"]:.0f}', f'{r["tf_med"]:.2f}', f'{r["avW_med"]:.2f}',
                    f'{d_w:+.2f}', f'{d_tf:+.2f}'])
print(f'\nwrote {out_csv}')

# ---- Plot: 3 panels (max_W, sm_p10, TFLOPS) grouped bars w/ colors per variant ----
fig, axs = plt.subplots(3, 1, figsize=(14, 10))
xs       = np.arange(len(summary))
labels   = [r['cfg'] for r in summary]
groups   = [r['group'] for r in summary]
colors   = {'A':'#666', 'B':'#1f77b4', 'C':'#2ca02c', 'D':'#d62728',
            'E':'#ff7f0e', 'F':'#9467bd', 'U':'#17becf', 'P':'#e377c2',
            'N':'#8c564b', 'G':'#bcbd22', 'R':'#1abc9c', 'V':'#f39c12',
            'M':'#34495e', 'X':'#e74c3c', 'K':'#16a085', 'L':'#27ae60',
            'O':'#9b59b6'}
bar_c    = [colors[g] for g in groups]

mx_med  = [r['mx_med']  for r in summary]
mx_max  = [r['mx_max']  for r in summary]
mx_std  = [r['mx_std']  for r in summary]
sm_med  = [r['sm_med']  for r in summary]
tf_med  = [r['tf_med']  for r in summary]

ax = axs[0]
ax.bar(xs, mx_med, color=bar_c, yerr=mx_std, capsize=2,
       label='max_W median ± std')
ax.scatter(xs, mx_max, marker='x', color='black', s=20, label='max_W max (single burst)')
ax.axhline(base_mx_med, color='gray', lw=1, ls='--', label=f'baseline median {base_mx_med:.0f}')
ax.axhline(600, color='blue', lw=0.8, ls=':', label='TDP 600 W')
ax.set_ylabel('max_W (W)')
ax.set_title(f'Wave-sleep variants — max_W per burst    ({tag})')
ax.set_ylim(400, 760)
ax.legend(loc='lower right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

ax = axs[1]
ax.bar(xs, sm_med, color=bar_c)
ax.axhline(base_smp, color='gray', lw=1, ls='--', label=f'baseline {base_smp:.0f} MHz')
ax.axhline(2430, color='red', lw=0.6, ls=':', label='Boost 2430 MHz')
ax.set_ylim(2300, 2440)
ax.set_ylabel('sm_p10 (MHz)')
ax.set_title('SM clock p10')
ax.legend(loc='lower right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

ax = axs[2]
ax.bar(xs, tf_med, color=bar_c)
ax.axhline(base_tf, color='gray', lw=1, ls='--', label=f'baseline {base_tf:.1f} TF')
ax.axhline(base_tf*0.95, color='red', lw=0.6, ls=':', label='−5% threshold')
ax.set_ylabel('TFLOPS')
ax.set_title('TFLOPS')
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=60, fontsize=8)
ax.legend(loc='lower right', fontsize=8)
ax.grid(axis='y', alpha=0.3)

# group legend in figure title
from matplotlib.patches import Patch
handles = [Patch(color=colors[g], label=gnames[g]) for g in order]
fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, 1.02))

fig.tight_layout()
out_png = f'{log}/{tag}_variants.png'
fig.savefig(out_png, dpi=120, bbox_inches='tight')
print(f'plot: {out_png}')
