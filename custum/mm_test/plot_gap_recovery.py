#!/usr/bin/env python3
"""Zoom plot for gap recovery sweep — show whether SM clock returns to boost
between bursts for each gap_ms setting."""
import csv
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def parse_ts(s):
    return datetime.strptime(s.strip(), '%Y/%m/%d %H:%M:%S.%f')


def strip_unit(s):
    return float(s.strip().split()[0])


TS = '20260522_042613'
SEG = f'logs/gap_recovery_{TS}_segments.csv'
PWR = f'logs/gap_recovery_{TS}_gpu3_power.csv'

# Load segments grouped by cfg
seg = list(csv.DictReader(open(SEG)))
groups = defaultdict(list)
for s in seg:
    tag = s['backend']
    cfg = tag.split(':',1)[1].split('#',1)[0]
    groups[cfg].append({'t0': parse_ts(s['t_start']), 't1': parse_ts(s['t_end']),
                        'tflops': float(s['tflops'])})

# Load power
power_rows = []
with open(PWR) as f:
    rdr = csv.reader(f)
    h = [c.strip() for c in next(rdr)]
    idx = {x: h.index(x) for x in h}
    for r in rdr:
        if not r: continue
        try:
            ts = parse_ts(r[idx['timestamp']])
            power_rows.append({
                'ts': ts,
                'pw': strip_unit(r[idx['power.draw.instant [W]']]),
                'sm': strip_unit(r[idx['clocks.current.sm [MHz]']]),
            })
        except Exception:
            continue
power_rows.sort(key=lambda r: r['ts'])

# ── Per-cfg zoom (first 6 seconds) ──
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharey=False)

cfg_order = sorted(groups.keys(), key=lambda c: int(c.replace('baseline_g','')))
for ax, cfg in zip(axes, cfg_order):
    ax_r = ax.twinx()
    bursts = groups[cfg]
    t_lo = bursts[0]['t0']
    t_hi = t_lo + timedelta(seconds=6)
    pw = [p for p in power_rows if t_lo <= p['ts'] <= t_hi]
    ts_ = [p['ts'] for p in pw]
    ax.plot(ts_, [p['pw'] for p in pw], 'o-', color='tab:blue', lw=1.2, ms=3,
            label='Power (W)', zorder=3)
    ax_r.plot(ts_, [p['sm'] for p in pw], 's-', color='tab:red', lw=1.4, ms=4,
              alpha=0.85, label='SM clock (MHz)', zorder=4)
    # Guide lines
    ax.axhline(600, ls='--', color='tab:blue', alpha=0.5, label='TDP 600 W')
    for lvl, a in [(630, 0.3), (660, 0.5), (690, 0.65), (720, 0.85)]:
        ax.axhline(lvl, ls=':', color='red', alpha=a, lw=0.8, label=f'spike {lvl}W')
    ax_r.axhline(2430, ls='--', color='tab:red', alpha=0.4, label='Boost 2430 MHz')
    # Shade burst windows
    for b in bursts:
        if not (t_lo <= b['t0'] <= t_hi): continue
        ax.axvspan(b['t0'], b['t1'], alpha=0.10, color='tab:green', zorder=1)

    ax.set_ylim(0, 1000)
    ax.set_ylabel('Power (W)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax_r.set_ylim(1800, 2500)  # zoom on clock region
    ax_r.set_ylabel('SM clock (MHz)', color='tab:red')
    ax_r.tick_params(axis='y', labelcolor='tab:red')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
    gap_ms = int(cfg.replace('baseline_g',''))
    # Statistics on this cfg's clock recovery
    clk_vals = [p['sm'] for p in pw]
    if clk_vals:
        ax.set_title(f'{cfg} (gap = {gap_ms} ms)  '
                     f'clock min={min(clk_vals):.0f}  max={max(clk_vals):.0f}  '
                     f'first 6 sec',
                     loc='left', fontsize=11, fontweight='bold')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='lower right', fontsize=7, ncol=2)

fig.autofmt_xdate(rotation=0)
fig.suptitle('Clock recovery vs burst gap — baseline s100, N=100 bursts each. '
             'Green band = burst (kernel active). Gap = idle.',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('logs/gap_recovery_zoom.png', dpi=130)
print('plot: logs/gap_recovery_zoom.png')

# ── Per-burst clock-recovery analysis ──
# For each burst, measure: what was clock at the LAST sample BEFORE burst start?
# That tells us how much clock recovered during the preceding gap.
print()
print(f'{"cfg":>14s}  {"N":>3s}  {"pre-burst clock μ ± σ":>23s}  '
      f'{"max":>5s}  {"% reached boost (>2400)":>23s}')
print('-' * 95)
for cfg in cfg_order:
    bursts = groups[cfg]
    pre_clocks = []
    for b in bursts:
        # find power sample just BEFORE this burst's t_start (within 100ms before)
        candidates = [p for p in power_rows
                      if (b['t0'] - timedelta(milliseconds=100)) <= p['ts'] < b['t0']]
        if candidates:
            pre_clocks.append(candidates[-1]['sm'])
    if not pre_clocks: continue
    mean = sum(pre_clocks)/len(pre_clocks)
    std = (sum((x-mean)**2 for x in pre_clocks)/max(len(pre_clocks)-1,1))**0.5
    n_boost = sum(1 for c in pre_clocks if c >= 2400)
    print(f'{cfg:>14s}  {len(pre_clocks):>3d}  '
          f'{mean:>15.0f} ± {std:>5.0f}  {max(pre_clocks):>5.0f}  '
          f'{n_boost}/{len(pre_clocks)} ({n_boost/len(pre_clocks)*100:>5.1f}%)')
