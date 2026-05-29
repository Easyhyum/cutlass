#!/usr/bin/env python3
"""
Apply wave-aware entry-delay nanosleep to a CUTLASS streamk GEMM and use
the existing probe machinery to verify the resulting per-CTA start times.

Pattern (only when num_waves ≥ 3):
  wave 0       : staircase by smid — first 60% of SMs no delay, top 40%
                 sleep (smid - thr) * FIRST_STEP_NS
  wave 1..N-2  : MID_PCT % of CTAs sleep MID_NS ns (deterministic hash)
  wave N-1     : no sleep — drain as fast as possible

Env:
  MM_MODEL    qwen3-8b (default) | qwen3-32b
  MM_KERNEL   streamk (default) | basicdp
  MM_MS       M list (default: 8192,65536,131072 — must yield ≥3 waves)
  FIRST_PCT   percent of SMs with NO delay in wave 0  (default 60)
  FIRST_STEP_NS  ns/smid-step beyond threshold        (default 100)
  MID_PCT     percent of mid-wave CTAs that sleep     (default 30)
  MID_NS      ns sleep for selected mid-wave CTAs     (default 1000)
  SEED        hash seed for mid-wave selection        (default 3235871249)
  OUT_DIR     output dir
"""
import os, sys, csv
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import probe_streamk as ext


MODELS = {
    'qwen3-8b':  {'INTER': 12288, 'H': 4096},
    'qwen3-32b': {'INTER': 25600, 'H': 5120},
}

TB_M, TB_N = 128, 128


def grid_for(M, K, N):
    gy = (M + TB_M - 1) // TB_M
    gx = (N + TB_N - 1) // TB_N
    return gx, gy, 1


def expected_launched_streamk(tiles, n_sm):
    """For streamk on this GPU: launched = n_sm * floor(tiles / n_sm) when
    tiles >= n_sm. (For tiles < n_sm the K-split kicks in.)"""
    if tiles >= n_sm:
        return n_sm * (tiles // n_sm)
    return n_sm  # rough fallback — exact value depends on K-split factor


def assign_waves(smid, start):
    n = len(smid)
    wave = np.full(n, -1, dtype=np.int32)
    for s in np.unique(smid):
        if s < 0: continue
        idx = np.where(smid == s)[0]
        order = idx[np.argsort(start[idx], kind='stable')]
        for w, i in enumerate(order):
            wave[i] = w
    return wave


def probe_with_sleep(M, K, N, kernel, sleep_cfg, n_sm):
    """Run one GEMM with wave-sleep configured + probe enabled."""
    gx, gy, gz = grid_for(M, K, N)
    tiles = gx * gy * gz
    if kernel == 'streamk':
        launched = expected_launched_streamk(tiles, n_sm)
    else:
        launched = tiles
    num_waves = max(1, (launched + n_sm - 1) // n_sm)

    cap = max(launched, n_sm * 4) * 2 + 4096
    dev = torch.device('cuda:0')
    smid    = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    start_t = torch.zeros(cap, dtype=torch.int64, device=dev)
    bx      = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    by      = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    bz      = torch.full((cap,), -1, dtype=torch.int32, device=dev)

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)

    # Choose what we tell the device about num_waves. The device gates on
    # num_waves ≥ 3, so we just pass the estimate here.
    ext.configure_wave_sleep(
        num_waves          = num_waves,
        n_sm               = n_sm,
        first_wave_smid_thr= sleep_cfg['first_thr'],
        first_wave_step_ns = sleep_cfg['first_step_ns'],
        mid_wave_pct       = sleep_cfg['mid_pct'],
        mid_wave_ns        = sleep_cfg['mid_ns'],
        hash_seed          = sleep_cfg['seed'],
    )
    ext.set_probe_buffers(smid, start_t, bx, by, bz, cap)

    if kernel == 'streamk':
        gemm = lambda: ext.gemm_streamk_probe(A, B, 1, -1)
    else:
        gemm = lambda: ext.gemm_basicdp_probe(A, B)

    # warmup once with sleep on (so kernel cache is warm with this config)
    gemm()
    torch.cuda.synchronize()

    smid.fill_(-1); start_t.zero_()
    ext.set_probe_buffers(smid, start_t, bx, by, bz, cap)
    ext.configure_wave_sleep(
        num_waves          = num_waves,
        n_sm               = n_sm,
        first_wave_smid_thr= sleep_cfg['first_thr'],
        first_wave_step_ns = sleep_cfg['first_step_ns'],
        mid_wave_pct       = sleep_cfg['mid_pct'],
        mid_wave_ns        = sleep_cfg['mid_ns'],
        hash_seed          = sleep_cfg['seed'],
    )
    gemm()
    torch.cuda.synchronize()
    ext.clear_probe_buffers()
    ext.clear_wave_sleep()

    smid_np  = smid.cpu().numpy()
    start_np = start_t.cpu().numpy().astype(np.uint64)
    bx_np    = bx.cpu().numpy()
    by_np    = by.cpu().numpy()
    bz_np    = bz.cpu().numpy()
    valid = np.where(smid_np >= 0)[0]
    return {
        'M': M, 'K': K, 'N': N, 'gx': gx, 'gy': gy, 'gz': gz,
        'tiles': tiles, 'launched_expected': launched, 'num_waves_expected': num_waves,
        'linear_idx': valid,
        'smid':  smid_np[valid],
        'start': start_np[valid],
        'bx':    bx_np[valid],
        'by':    by_np[valid],
        'bz':    bz_np[valid],
    }


def probe_baseline(M, K, N, kernel, n_sm):
    """Same call but with wave-sleep disabled — for direct comparison."""
    gx, gy, gz = grid_for(M, K, N)
    tiles = gx * gy * gz
    launched = expected_launched_streamk(tiles, n_sm) if kernel == 'streamk' else tiles
    num_waves = max(1, (launched + n_sm - 1) // n_sm)

    cap = max(launched, n_sm * 4) * 2 + 4096
    dev = torch.device('cuda:0')
    smid    = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    start_t = torch.zeros(cap, dtype=torch.int64, device=dev)
    bx      = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    by      = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    bz      = torch.full((cap,), -1, dtype=torch.int32, device=dev)

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)

    ext.clear_wave_sleep()
    ext.set_probe_buffers(smid, start_t, bx, by, bz, cap)
    if kernel == 'streamk':
        gemm = lambda: ext.gemm_streamk_probe(A, B, 1, -1)
    else:
        gemm = lambda: ext.gemm_basicdp_probe(A, B)
    gemm(); torch.cuda.synchronize()
    smid.fill_(-1); start_t.zero_()
    ext.set_probe_buffers(smid, start_t, bx, by, bz, cap)
    gemm(); torch.cuda.synchronize()
    ext.clear_probe_buffers()

    smid_np  = smid.cpu().numpy()
    start_np = start_t.cpu().numpy().astype(np.uint64)
    bx_np    = bx.cpu().numpy()
    by_np    = by.cpu().numpy()
    bz_np    = bz.cpu().numpy()
    valid = np.where(smid_np >= 0)[0]
    return {
        'M': M, 'num_waves_expected': num_waves,
        'linear_idx': valid,
        'smid':  smid_np[valid],
        'start': start_np[valid],
    }


def main():
    model_name = os.environ.get('MM_MODEL', 'qwen3-8b').lower()
    cfg = MODELS[model_name]
    K, N = cfg['INTER'], cfg['H']

    kernel = os.environ.get('MM_KERNEL', 'streamk').lower()
    Ms = [int(x) for x in os.environ.get('MM_MS', '8192,65536,131072').split(',')]

    n_sm = torch.cuda.get_device_properties(0).multi_processor_count

    sleep_cfg = {
        'first_thr':     int(round(float(os.environ.get('FIRST_PCT', '60')) / 100.0 * n_sm)),
        'first_step_ns': int(os.environ.get('FIRST_STEP_NS', '100')),
        'mid_pct':       int(os.environ.get('MID_PCT', '30')),
        'mid_ns':        int(os.environ.get('MID_NS',  '1000')),
        'seed':          int(os.environ.get('SEED', str(0xC0FFEE11))),
    }

    out_dir = os.environ.get('OUT_DIR', HERE)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[wsleep] device n_sm={n_sm}  model={model_name}  kernel={kernel}')
    print(f'[wsleep] sleep cfg = {sleep_cfg}')
    print(f'[wsleep] M list   = {Ms}')
    print()

    all_rows = []
    by_M = {}

    for M in Ms:
        # baseline (no sleep) first, then with sleep
        rec_base = probe_baseline(M, K, N, kernel, n_sm)
        rec_slp  = probe_with_sleep(M, K, N, kernel, sleep_cfg, n_sm)

        wave_base = assign_waves(rec_base['smid'], rec_base['start'])
        wave_slp  = assign_waves(rec_slp['smid'],  rec_slp['start'])

        # Normalize start times to start of wave 0 (== min start in that run)
        # — both runs use globaltimer in ns so subtraction yields ns offsets.
        t0_base = int(rec_base['start'].min())
        t0_slp  = int(rec_slp['start'].min())

        print(f'M={M}  num_waves_expected(streamk)={rec_slp["num_waves_expected"]}')
        # Summary: avg start offset per wave_idx (base vs sleep)
        for w in sorted(np.unique(wave_slp).tolist()):
            mask_b = (wave_base == w) if w in np.unique(wave_base) else np.zeros_like(wave_base, bool)
            mask_s = (wave_slp  == w)
            n_b = int(mask_b.sum()); n_s = int(mask_s.sum())
            if n_s == 0: continue
            mean_b = (rec_base['start'][mask_b].astype(np.int64) - t0_base).mean() if n_b > 0 else float('nan')
            mean_s = (rec_slp ['start'][mask_s].astype(np.int64) - t0_slp ).mean()
            std_s  = (rec_slp ['start'][mask_s].astype(np.int64) - t0_slp ).std()
            print(f'  wave {w:>3d}  n={n_s:>4d}  base mean={mean_b:>9.0f} ns   '
                  f'sleep mean={mean_s:>9.0f} ns   std={std_s:>9.0f} ns')

        # collect rows
        for i in range(len(rec_slp['smid'])):
            all_rows.append({
                'M': M, 'kernel': kernel,
                'cta_launch_idx':   int(rec_slp['linear_idx'][i]),
                'bx': int(rec_slp['bx'][i]),
                'by': int(rec_slp['by'][i]),
                'bz': int(rec_slp['bz'][i]),
                'smid':              int(rec_slp['smid'][i]),
                'wave_idx':          int(wave_slp[i]),
                'start_ns_baseline': None,
                'start_ns_sleep':    int(rec_slp['start'][i]) - t0_slp,
            })
        # zip baseline start_ns onto rows (by cta_launch_idx)
        base_by_idx = {int(rec_base['linear_idx'][i]):
                       int(rec_base['start'][i]) - t0_base
                       for i in range(len(rec_base['smid']))}
        for r in all_rows:
            if r['M'] == M and r['start_ns_baseline'] is None:
                r['start_ns_baseline'] = base_by_idx.get(r['cta_launch_idx'])
        by_M[M] = (rec_base, rec_slp, wave_base, wave_slp, t0_base, t0_slp)
        print()

    # ---- write CSV -----------------------------------------------------------
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'cta_wave_sleep_{kernel}_{ts}.csv')
    csv_latest = os.path.join(out_dir, f'cta_wave_sleep_{kernel}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    with open(csv_latest, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f'[wsleep] CSV: {csv_path}')

    # ---- plot ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Two-row plot per M:
    #   row 1: smid vs (start_ns_sleep) for wave 0 only — should be staircase
    #   row 2: cta_launch_idx vs (start_ns_sleep - start_ns_baseline)
    #          → highlights mid-wave random delays and last-wave zero delay
    for M, (rec_b, rec_s, wb, ws, t0_b, t0_s) in by_M.items():
        sm = rec_s['smid']
        st_s = rec_s['start'].astype(np.int64) - t0_s
        st_b = rec_b['start'].astype(np.int64) - t0_b
        lidx = rec_s['linear_idx']

        # build a launch_idx → baseline start map
        base_map = {int(rec_b['linear_idx'][i]): int(st_b[i])
                    for i in range(len(rec_b['smid']))}
        st_b_aligned = np.array([base_map.get(int(li), 0) for li in lidx],
                                dtype=np.int64)

        fig, axs = plt.subplots(3, 1, figsize=(13, 9))

        # (1) first wave: smid vs start_ns_sleep   — staircase shape
        mask0 = ws == 0
        axs[0].scatter(sm[mask0], st_s[mask0], s=10, alpha=0.7, label='with sleep')
        # overlay baseline first-wave start
        mask0_b = wb == 0
        axs[0].scatter(rec_b['smid'][mask0_b], st_b[mask0_b],
                       s=8, alpha=0.4, c='gray', label='baseline')
        axs[0].set_title(f'M={M}   wave 0:  start_ns vs smid  '
                         f'(staircase ≥ smid {by_M[M][3].size and "thr"})')
        axs[0].set_xlabel('smid')
        axs[0].set_ylabel('start_ns (relative)')
        axs[0].axvline(int(round(float(os.environ.get("FIRST_PCT","60"))/100.0*188)),
                       color='red', linestyle='--', alpha=0.5, label='smid_thr')
        axs[0].legend()
        axs[0].grid(alpha=0.3)

        # (2) all CTAs: launch_idx vs (sleep - baseline) — extra delay
        delta = st_s - st_b_aligned
        # Color by wave
        sc = axs[1].scatter(lidx, delta, c=ws, s=6, cmap='viridis', alpha=0.6)
        axs[1].set_title(f'M={M}   extra delay per CTA   '
                         f'(start_ns_sleep − start_ns_baseline)')
        axs[1].set_xlabel('cta_launch_idx')
        axs[1].set_ylabel('delta ns')
        axs[1].grid(alpha=0.3)
        plt.colorbar(sc, ax=axs[1], label='wave_idx')

        # (3) histogram of per-wave delta
        unique_waves = np.unique(ws)
        if len(unique_waves) <= 20:
            data = [delta[ws == w] for w in unique_waves]
            axs[2].boxplot(data, positions=unique_waves, widths=0.6,
                           showfliers=False)
            axs[2].set_xlabel('wave_idx')
            axs[2].set_ylabel('extra delay (ns)')
            axs[2].set_title(f'M={M}   per-wave extra-delay distribution')
            axs[2].grid(alpha=0.3, axis='y')
        else:
            axs[2].set_visible(False)

        fig.tight_layout()
        png = os.path.join(out_dir,
                           f'cta_wave_sleep_{kernel}_M{M}.png')
        fig.savefig(png, dpi=110, bbox_inches='tight')
        print(f'[wsleep] plot: {png}')


if __name__ == '__main__':
    main()
