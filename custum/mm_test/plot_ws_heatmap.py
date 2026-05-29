#!/usr/bin/env python3
"""
Heatmap (S × P) for wave-sleep sweep over the (FIRST_PCT, FIRST_STEP_NS)
grid. Reads <tag>_segments_with_power.csv and emits 3 heatmaps:
   max_W mean   (lower = power spike reduced)
   sm_p10 mean  (higher = freq drop avoided)
   TFLOPS mean  (higher = throughput retained)
plus  Δ vs baseline (s100_p0) for max_W.

Usage:
  python plot_ws_heatmap.py <tag>
e.g.
  python plot_ws_heatmap.py ws_full_050450
"""
import os, sys, csv, re
import numpy as np
from collections import defaultdict

if len(sys.argv) < 2:
    raise SystemExit('usage: plot_ws_heatmap.py <tag>')
tag = sys.argv[1]
log = 'logs'
seg = os.path.join(log, f'{tag}_segments.csv')
enr = os.path.join(log, f'{tag}_segments_with_power.csv')


def split_cfg(b):
    if ':' in b:
        cfg = b.split(':', 1)[1]
    else:
        cfg = b
    if '#' in cfg:
        cfg = cfg.split('#', 1)[0]
    return cfg


# pair seg + enr by row index (analyze_power preserves order)
seg_rows = list(csv.DictReader(open(seg)))
enr_rows = list(csv.DictReader(open(enr)))
assert len(seg_rows) == len(enr_rows), f'mismatch {len(seg_rows)} vs {len(enr_rows)}'

grp = defaultdict(list)
for s, e in zip(seg_rows, enr_rows):
    cfg = split_cfg(s['backend'])
    grp[cfg].append({
        'max_W':  float(e['max_W']),
        'avg_W':  float(e['avg_W']),
        'sm_p10': float(e['sm_p10']),
        'tflops': float(e['tflops']),
    })


def mean(xs): return sum(xs) / len(xs)


# baseline = s100_step0  (no wave-sleep code at all)
base = grp.get('s100_step0', [])
if not base:
    raise SystemExit('no baseline (s100_step0) row in segments')
base_mxW  = mean([r['max_W']  for r in base])
base_smp  = mean([r['sm_p10'] for r in base])
base_avW  = mean([r['avg_W']  for r in base])
base_tf   = mean([r['tflops'] for r in base])
print(f'baseline (s100_step0)  N={len(base)}  '
      f'max_W={base_mxW:.1f}  sm_p10={base_smp:.0f}  '
      f'avg_W={base_avW:.1f}  TFLOPS={base_tf:.1f}')


# Parse "sXX_stepYY" → (S, P)
pat = re.compile(r'^s(\d+)_step(\d+)$')
S_set, P_set = set(), set()
data = {}  # (S, P) → metrics
for cfg, rows in grp.items():
    if cfg == 's100_step0': continue
    m = pat.match(cfg)
    if not m: continue
    s = int(m.group(1)); p = int(m.group(2))
    S_set.add(s); P_set.add(p)
    data[(s, p)] = {
        'max_W':  mean([r['max_W']  for r in rows]),
        'avg_W':  mean([r['avg_W']  for r in rows]),
        'sm_p10': mean([r['sm_p10'] for r in rows]),
        'tflops': mean([r['tflops'] for r in rows]),
        'n':      len(rows),
    }

Ss = sorted(S_set)
Ps = sorted(P_set)
print(f'sweep grid: S={Ss} ({len(Ss)}), P={Ps} ({len(Ps)})')


def matrix(metric):
    M = np.full((len(Ss), len(Ps)), np.nan)
    for i, s in enumerate(Ss):
        for j, p in enumerate(Ps):
            if (s, p) in data:
                M[i, j] = data[(s, p)][metric]
    return M


M_mxW  = matrix('max_W')
M_smp  = matrix('sm_p10')
M_avW  = matrix('avg_W')
M_tf   = matrix('tflops')
M_dmxW = M_mxW - base_mxW
M_dtf  = (M_tf - base_tf) / base_tf * 100.0  # percent change

# Write CSV summary
out_csv = os.path.join(log, f'{tag}_heatmap_summary.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['S_pct', 'P_step_ns', 'max_us_est',
                'max_W', 'sm_p10', 'avg_W', 'tflops',
                'd_max_W', 'd_tflops_pct'])
    for s in Ss:
        thr = (188 * s) // 100
        for p in Ps:
            if (s, p) not in data: continue
            m = data[(s, p)]
            max_us = (188 - thr) * p / 1000.0
            w.writerow([s, p, f'{max_us:.1f}',
                        f'{m["max_W"]:.2f}', f'{m["sm_p10"]:.0f}',
                        f'{m["avg_W"]:.2f}', f'{m["tflops"]:.2f}',
                        f'{m["max_W"]-base_mxW:+.2f}',
                        f'{(m["tflops"]-base_tf)/base_tf*100:+.2f}'])
    # baseline row
    w.writerow([100, 0, 0,
                f'{base_mxW:.2f}', f'{base_smp:.0f}',
                f'{base_avW:.2f}', f'{base_tf:.2f}',
                '0', '0'])
print(f'wrote {out_csv}')


# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(15, 10))


def draw(ax, M, title, cmap, fmt, center=None, vmin=None, vmax=None):
    if vmin is None: vmin = np.nanmin(M)
    if vmax is None: vmax = np.nanmax(M)
    if center is not None:
        # diverging — center=0
        absmax = max(abs(vmin), abs(vmax))
        vmin, vmax = -absmax, absmax
    im = ax.imshow(M, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
                   origin='lower')
    ax.set_xticks(range(len(Ps)))
    ax.set_xticklabels(Ps, rotation=0)
    ax.set_yticks(range(len(Ss)))
    ax.set_yticklabels(Ss)
    ax.set_xlabel('P  (FIRST_STEP_NS)')
    ax.set_ylabel('S  (FIRST_PCT)')
    ax.set_title(title, fontsize=11)
    for i in range(len(Ss)):
        for j in range(len(Ps)):
            v = M[i, j]
            if np.isnan(v): continue
            txt = fmt.format(v)
            # pick text color based on cell luminance heuristic
            ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                    color='white' if (v - vmin) / max(vmax - vmin, 1e-9) > 0.55 else 'black')
    plt.colorbar(im, ax=ax)


draw(axs[0, 0], M_mxW, f'max_W (baseline = {base_mxW:.1f} W)',
     cmap='magma_r', fmt='{:.0f}')
draw(axs[0, 1], M_dmxW, 'Δ max_W vs baseline   (negative = spike reduced)',
     cmap='RdBu', fmt='{:+.0f}', center=0)
draw(axs[1, 0], M_smp, f'sm_p10 MHz (baseline = {base_smp:.0f})',
     cmap='viridis', fmt='{:.0f}')
draw(axs[1, 1], M_dtf, f'Δ TFLOPS %  vs baseline ({base_tf:.1f} TF)',
     cmap='RdBu', fmt='{:+.1f}', center=0)

fig.suptitle(
    f'Wave-sleep sweep ({tag})  —  M=8192 down_proj, N=30 burst, gap=500ms',
    fontsize=12, y=1.00)
fig.tight_layout()
png = os.path.join(log, f'{tag}_heatmap.png')
fig.savefig(png, dpi=120, bbox_inches='tight')
print(f'plot {png}')
