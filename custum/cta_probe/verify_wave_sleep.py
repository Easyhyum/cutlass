#!/usr/bin/env python3
"""
Within-run verification of the wave-sleep pattern (no cross-run subtraction).

For each M:
 * wave 0:  group by smid → for smid >= threshold,
            start_ns should grow linearly with (smid - threshold).
 * mid waves (1..N-2):  inside each wave,
            ~MID_PCT of CTAs should have a clearly larger start_ns than the
            wave's no-sleep median (= the (100-MID_PCT)% baseline within wave).
 * last wave:  same as mid wave but with no sleeps — entire wave should
               look like the "fast" group of mid waves.

The advantage of this view is that absolute clocks between runs cancel out;
only the within-run pattern matters.
"""
import os, sys, csv
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
N_SM = 188

FIRST_PCT     = float(os.environ.get('FIRST_PCT',     '60'))
FIRST_STEP_NS = int  (os.environ.get('FIRST_STEP_NS', '100'))
MID_PCT       = int  (os.environ.get('MID_PCT',       '30'))
MID_NS        = int  (os.environ.get('MID_NS',        '1000'))


def main():
    csv_path = os.path.join(HERE,
        f'cta_wave_sleep_{os.environ.get("MM_KERNEL", "streamk")}.csv')
    df = pd.read_csv(csv_path)

    thr_smid = int(round(FIRST_PCT / 100.0 * N_SM))
    max_staircase = (N_SM - 1 - thr_smid) * FIRST_STEP_NS
    print(f'expected staircase: smid >= {thr_smid}, step={FIRST_STEP_NS}ns, '
          f'max={max_staircase}ns')
    print(f'expected mid sleep: {MID_PCT}% of CTAs sleep {MID_NS}ns')
    print()

    for M, sub in df.groupby('M'):
        sub = sub.sort_values('cta_launch_idx').reset_index(drop=True)
        n_waves = int(sub['wave_idx'].max()) + 1
        last_w  = n_waves - 1
        s = sub['start_ns_sleep'].astype('int64').to_numpy()

        # --- Wave 0 staircase (within-run): align to min start in wave 0 ---
        w0 = sub[sub['wave_idx'] == 0].copy()
        # one CTA per smid in wave 0 (we hope) — confirm
        # use the lowest-smid CTA's start as reference (smid 0 has no delay)
        w0 = w0.sort_values('smid')
        w0_anchor = int(w0['start_ns_sleep'].min())
        w0['rel'] = w0['start_ns_sleep'].astype('int64') - w0_anchor
        print(f'==================== M={M}  ({n_waves} waves) ====================')
        print(f'\n[wave 0 staircase]  reference: min start in wave 0')
        sample_smids = [0, 30, 60, 100, thr_smid - 1, thr_smid, thr_smid + 1,
                        thr_smid + 10, thr_smid + 30, thr_smid + 50, N_SM - 1]
        for sm in sample_smids:
            row = w0[w0['smid'] == sm]
            if len(row) == 0: continue
            r = int(row.iloc[0]['rel'])
            expect = max(0, (sm - thr_smid)) * FIRST_STEP_NS
            print(f'  smid={sm:>3d}  rel_start={r:>9d} ns   '
                  f'expected_sleep={expect:>6d} ns')
        # spearman rho between smid and rel (only above thr)
        upper = w0[w0['smid'] >= thr_smid]
        if len(upper) > 5:
            from scipy.stats import spearmanr
            rho = spearmanr(upper['smid'], upper['rel']).statistic
            print(f'  spearman(smid, rel_start) for smid >= {thr_smid}: '
                  f'rho = {rho:.4f}  (expect close to +1)')

        # --- Mid wave: bimodal distribution within a single wave ----------
        mid = sub[(sub['wave_idx'] > 0) & (sub['wave_idx'] < last_w)].copy()
        # Per-wave anchor: the FAST mode median within that wave
        anchors = mid.groupby('wave_idx')['start_ns_sleep'].quantile(0.25)
        mid['rel_within_wave'] = (
            mid['start_ns_sleep'].astype('int64')
            - mid['wave_idx'].map(anchors).astype('int64'))
        # CTAs are "slept" if rel_within_wave is much larger than baseline
        # threshold ~ 50% of MID_NS
        slept_threshold = MID_NS // 2
        slept = (mid['rel_within_wave'] >= slept_threshold)
        print(f'\n[mid waves]  thresholding rel_within_wave >= {slept_threshold} ns')
        print(f'  CTAs total: {len(mid)}    slept: {int(slept.sum())}  '
              f'-> {100*slept.sum()/len(mid):.1f}%  (target ≈ {MID_PCT}%)')
        if int(slept.sum()) > 0:
            ds = mid['rel_within_wave'][slept]
            print(f'  delay-when-slept: median={int(ds.median())} ns  '
                  f'p10={int(ds.quantile(0.1))} ns  '
                  f'p90={int(ds.quantile(0.9))} ns  '
                  f'(target ≈ {MID_NS} ns)')
        else:
            print('  no detected sleeps — likely noise floor exceeds MID_NS')

        # Per-wave breakdown (first 5 mid waves)
        print(f'\n  per-wave breakdown (first 5 mid waves):')
        for w in range(1, min(6, last_w)):
            sw = mid[mid['wave_idx'] == w]
            if len(sw) == 0: continue
            slept_w = (sw['rel_within_wave'] >= slept_threshold).sum()
            print(f'    wave {w:>3d}  n={len(sw):>4d}  slept={slept_w:>4d}  '
                  f'-> {100*slept_w/len(sw):.0f}%   '
                  f'median delay-when-slept = '
                  f'{int(sw["rel_within_wave"][sw["rel_within_wave"]>=slept_threshold].median()) if slept_w>0 else 0} ns')

        # --- Last wave: should have ZERO additional sleeps -----------------
        lastw = sub[sub['wave_idx'] == last_w].copy()
        anchor_l = int(lastw['start_ns_sleep'].quantile(0.25))
        lastw['rel'] = lastw['start_ns_sleep'].astype('int64') - anchor_l
        slept_last = (lastw['rel'] >= slept_threshold).sum()
        print(f'\n[last wave {last_w}]   n={len(lastw)}  '
              f'rel>={slept_threshold}ns: {int(slept_last)} '
              f'({100*slept_last/max(len(lastw),1):.1f}%)   '
              f'(target 0%)')
        print()


if __name__ == '__main__':
    main()
