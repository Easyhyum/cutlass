#!/usr/bin/env python3
"""
Zoom-in plot of s100 configs only (V9 baseline), wide X axis so the
burst-idle clock toggling is directly visible (not just envelope).
"""
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


def load_power(path, t_lo, t_hi):
    rows = []
    pad = timedelta(seconds=1)
    with open(path) as f:
        reader = csv.reader(f)
        header = [c.strip() for c in next(reader)]
        idx = {h: i for i, h in enumerate(header)}
        ts_idx = idx.get('timestamp')
        sm_idx = idx.get('clocks.current.sm [MHz]')
        pw_idx = idx.get('power.draw.instant [W]')
        for r in reader:
            if not r: continue
            try: ts = parse_ts(r[ts_idx])
            except Exception: continue
            if not ((t_lo - pad) <= ts <= (t_hi + pad)): continue
            rows.append({
                'ts': ts,
                'sm': strip_unit(r[sm_idx]) if sm_idx is not None else float('nan'),
                'pw': strip_unit(r[pw_idx]) if pw_idx is not None else float('nan'),
            })
    rows.sort(key=lambda x: x['ts'])
    return rows


# ── load segments, filter s100 ──
seg = list(csv.DictReader(open('logs/v9_nburst_20260522_025821_segments.csv')))
s100_bursts = defaultdict(list)
for s in seg:
    tag = s['backend']
    cfg = tag.split(':',1)[1].split('#',1)[0] if ':' in tag else tag
    if cfg.startswith('s100'):
        s100_bursts[cfg].append({
            't0': parse_ts(s['t_start']),
            't1': parse_ts(s['t_end']),
            'tflops': float(s['tflops']),
        })

cfg_order = sorted(s100_bursts.keys(), key=lambda c: int(c.split('_step')[1]))
t_lo = min(s100_bursts[cfg_order[0]][0]['t0'], s100_bursts[cfg_order[0]][0]['t0'])
t_hi = s100_bursts[cfg_order[-1]][-1]['t1']
print(f'Time range: {t_lo.strftime("%H:%M:%S")} → {t_hi.strftime("%H:%M:%S")}  '
      f'span={(t_hi-t_lo).total_seconds():.0f}s')

power = load_power('logs/v9_nburst_20260522_025821_gpu0_power.csv', t_lo, t_hi)
print(f'Loaded {len(power)} power samples (50ms each)')

# ── Plot 1: full s100 region wide ──
fig_w = 40   # very wide for visibility
fig, ax = plt.subplots(figsize=(fig_w, 7))
ax_r = ax.twinx()

ts_ = [p['ts'] for p in power]
pw_ = [p['pw'] for p in power]
sm_ = [p['sm'] for p in power]
ax.plot(ts_, pw_, color='tab:blue', lw=0.8, label='Power instant (W)', zorder=3)
ax_r.plot(ts_, sm_, color='tab:red', lw=0.9, label='SM clock (MHz)',
          alpha=0.85, zorder=2)

ax.axhline(600, ls='--', color='tab:blue', alpha=0.5, label='TDP 600 W')
for lvl, alpha in [(630, 0.35), (660, 0.5), (690, 0.65), (720, 0.85)]:
    ax.axhline(lvl, ls=':', color='red', alpha=alpha, lw=0.9,
               label=f'spike {lvl} W')
ax_r.axhline(2430, ls='--', color='tab:red', alpha=0.35, label='Boost 2430 MHz')

palette = plt.cm.tab10.colors
for i, cfg in enumerate(cfg_order):
    bursts = s100_bursts[cfg]
    t0 = bursts[0]['t0']; t1 = bursts[-1]['t1']
    c = palette[i % 10]
    ax.axvspan(t0, t1, alpha=0.10, color=c, zorder=1)
    mid = t0 + (t1 - t0)/2
    ax.text(mid, 990, cfg, ha='center', va='top',
            fontsize=10, color=c, fontweight='bold', zorder=6)

ax.set_ylim(0, 1000)
ax.set_ylabel('Power (W)', color='tab:blue')
ax.tick_params(axis='y', labelcolor='tab:blue')
ax_r.set_ylim(0, 3000)
ax_r.set_ylabel('SM clock (MHz)', color='tab:red')
ax_r.tick_params(axis='y', labelcolor='tab:red')
ax.grid(True, alpha=0.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
fig.autofmt_xdate(rotation=0)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax_r.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc='lower right', fontsize=8, ncol=2)
fig.suptitle('s100 (V9 baseline) zoom-in — 6 configs × 50s each (wide X)',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('logs/s100_zoom_wide.png', dpi=110)
print('plot 1: logs/s100_zoom_wide.png')

# ── Plot 2: extreme zoom — first 5 seconds of s100_step500 ──
cfg = 's100_step500'
bursts = s100_bursts[cfg]
zoom_t0 = bursts[0]['t0']
zoom_t1 = zoom_t0 + timedelta(seconds=5)
pwr_zoom = [p for p in power if zoom_t0 <= p['ts'] <= zoom_t1]
print(f'Zoom: {cfg}, 5s window, {len(pwr_zoom)} samples')

fig2, ax2 = plt.subplots(figsize=(20, 6))
ax2_r = ax2.twinx()
ts_z = [p['ts'] for p in pwr_zoom]
ax2.plot(ts_z, [p['pw'] for p in pwr_zoom], 'o-', color='tab:blue',
         lw=1.5, ms=4, label='Power (W)', zorder=3)
ax2_r.plot(ts_z, [p['sm'] for p in pwr_zoom], 's-', color='tab:red',
           lw=1.5, ms=4, alpha=0.85, label='SM clock (MHz)', zorder=2)
ax2.axhline(600, ls='--', color='tab:blue', alpha=0.5, label='TDP 600 W')
for lvl, a in [(630, 0.35), (660, 0.5), (690, 0.65), (720, 0.85)]:
    ax2.axhline(lvl, ls=':', color='red', alpha=a, label=f'spike {lvl} W')
ax2_r.axhline(2430, ls='--', color='tab:red', alpha=0.35, label='Boost 2430')

# annotate each burst window
for i, b in enumerate(bursts):
    if not (zoom_t0 <= b['t0'] <= zoom_t1): continue
    ax2.axvspan(b['t0'], b['t1'], alpha=0.10, color='tab:green', zorder=1)

ax2.set_ylim(0, 1000)
ax2.set_ylabel('Power (W)', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2_r.set_ylim(0, 3000)
ax2_r.set_ylabel('SM clock (MHz)', color='tab:red')
ax2_r.tick_params(axis='y', labelcolor='tab:red')
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
fig2.autofmt_xdate(rotation=0)
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2_r.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc='lower right', fontsize=9, ncol=2)
fig2.suptitle(f'{cfg} extreme zoom — first 5 seconds (~10 bursts).  '
              'Green band = burst window, gap = idle.   Markers = nvidia-smi 50ms samples.',
              fontsize=12, fontweight='bold')
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig('logs/s100_zoom_5sec.png', dpi=130)
print('plot 2: logs/s100_zoom_5sec.png')
