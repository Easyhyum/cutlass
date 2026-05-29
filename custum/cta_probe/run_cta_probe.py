#!/usr/bin/env python3
"""
Qwen3 MLP down_proj M-sweep — CTA → SM / wave dispatch observation.

Calls the real CUTLASS Stream-K GEMM (built with -DCUTLASS_CTA_PROBE_ENABLED),
which makes MmaMultistage::operator() record (smid, globaltimer, blockIdx)
for each CTA and then RETURN before the mainloop. The grid CUTLASS launches —
and therefore the streamk swizzle's effect on it — is preserved exactly.

Wave index per CTA is reconstructed as: for each smid, sort CTAs by start
time and use that rank — so wave 0 = the first CTA the dispatcher gave to
each SM, wave 1 = the second CTA, etc.

Env:
  MM_MODEL    qwen3-8b (default) | qwen3-32b
  MM_MS       comma list of M (default: 32,256,1024,8192,65536)
  MM_KERNEL   streamk (default) | basicdp
  OUT_DIR     output dir (default = this dir)
"""
import os, sys, csv
from datetime import datetime

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import probe_streamk as ext  # built with -DCUTLASS_CTA_PROBE_ENABLED


MODELS = {
    'qwen3-8b':  {'INTER': 12288, 'H': 4096},
    'qwen3-32b': {'INTER': 25600, 'H': 5120},
}

# Streamk kernel ThreadblockShape (must match probe_streamk.cu / production)
TB_M, TB_N = 128, 128


def grid_for_down_proj(M, K, N):
    """down_proj: (M x K) @ (K x N) -> (M x N), 128x128 tiling."""
    gy = (M + TB_M - 1) // TB_M
    gx = (N + TB_N - 1) // TB_N
    return gx, gy, 1


def probe_one(M, K, N, kernel='streamk'):
    """Run one probed GEMM. Output buffers must be sized to the *production*
    grid the swizzle would produce. For ThreadblockSwizzleStreamK that grid
    is 1D and not equal to gx*gy — we allocate generously (max(gx*gy, n_sm*8))
    to be safe, and rely on linear_idx for indexing."""
    gx, gy, gz = grid_for_down_proj(M, K, N)
    # CUTLASS streamk swizzle launches a 1D-ish grid; size depends on SMs and
    # tile count. We overprovision to (gx*gy) * 2 + 4096 to never under-allocate.
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    cap = max(gx * gy * gz, n_sm * 8) * 2 + 4096

    dev = torch.device('cuda:0')
    smid    = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    start_t = torch.zeros(cap, dtype=torch.int64, device=dev)
    end_t   = torch.zeros(cap, dtype=torch.int64, device=dev)
    bx      = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    by      = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    bz      = torch.full((cap,), -1, dtype=torch.int32, device=dev)

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16)
    A.normal_(0, 0.02)
    B.normal_(0, 0.02)

    ext.set_probe_buffers(smid, start_t, end_t, bx, by, bz, cap)

    # 1 warmup (doesn't really matter — no mainloop), then measurement
    if kernel == 'streamk':
        gemm = lambda: ext.gemm_streamk_probe(A, B, 1, -1)
    elif kernel == 'basicdp':
        gemm = lambda: ext.gemm_basicdp_probe(A, B)
    else:
        raise SystemExit(f'unknown MM_KERNEL={kernel}')

    gemm()
    torch.cuda.synchronize()

    # measurement — clear buffers, re-install, then run
    smid.fill_(-1)
    start_t.zero_()
    end_t.zero_()
    ext.set_probe_buffers(smid, start_t, end_t, bx, by, bz, cap)
    gemm()
    torch.cuda.synchronize()

    # detach probe so subsequent unrelated kernels can't accidentally clobber
    ext.clear_probe_buffers()

    # crop to the CTAs that actually recorded (smid >= 0)
    smid_np  = smid.cpu().numpy()
    start_np = start_t.cpu().numpy().astype(np.uint64)
    end_np   = end_t.cpu().numpy().astype(np.uint64)
    bx_np    = bx.cpu().numpy()
    by_np    = by.cpu().numpy()
    bz_np    = bz.cpu().numpy()
    valid = np.where(smid_np >= 0)[0]
    return {
        'M': M, 'K': K, 'N': N,
        'gx': gx, 'gy': gy, 'gz': gz,
        'tile_total': gx * gy * gz,
        'linear_idx': valid,
        'smid':       smid_np[valid],
        'start':      start_np[valid],
        'end':        end_np[valid],
        'bx':         bx_np[valid],
        'by':         by_np[valid],
        'bz':         bz_np[valid],
    }


def assign_waves(smid, start):
    """For each smid, rank its CTAs by start time → that rank is wave_idx."""
    n = len(smid)
    wave = np.full(n, -1, dtype=np.int32)
    for s in np.unique(smid):
        if s < 0:
            continue
        idx = np.where(smid == s)[0]
        order = idx[np.argsort(start[idx], kind='stable')]
        for w, i in enumerate(order):
            wave[i] = w
    return wave


def main():
    model_name = os.environ.get('MM_MODEL', 'qwen3-8b').lower()
    if model_name not in MODELS:
        raise SystemExit(f'unknown MM_MODEL={model_name}')
    cfg = MODELS[model_name]
    K, N = cfg['INTER'], cfg['H']

    Ms = [int(x) for x in os.environ.get(
        'MM_MS', '32,256,1024,4096,8192,65536,131072').split(',')]
    kernel = os.environ.get('MM_KERNEL', 'streamk').lower()
    out_dir = os.environ.get('OUT_DIR', HERE)
    os.makedirs(out_dir, exist_ok=True)

    props = torch.cuda.get_device_properties(0)
    n_sm = props.multi_processor_count
    print(f'[probe] device={props.name}  sm={props.major}.{props.minor}  n_sm={n_sm}')
    print(f'[probe] model={model_name}  down_proj K={K} N={N}  tile={TB_M}x{TB_N}')
    print(f'[probe] kernel={kernel}  M list={Ms}')
    print()

    per_cta_rows = []
    summary_rows = []
    by_M = {}

    print(f'{"M":>7s} {"grid_xy":>10s} {"tiles":>6s} {"CTAs_launched":>14s} '
          f'{"SMs_used":>9s} {"waves":>6s} {"wave0":>6s} '
          f'{"ctas/wave_avg":>14s} {"util_first":>11s}')
    print('-' * 90)

    for M in Ms:
        rec = probe_one(M, K, N, kernel=kernel)
        wave = assign_waves(rec['smid'], rec['start'])

        n_ctas       = len(rec['smid'])
        n_waves      = (int(wave.max()) + 1) if n_ctas > 0 else 0
        sms_used     = int(np.unique(rec['smid']).size)
        ctas_per_w   = np.bincount(wave, minlength=max(1, n_waves))
        first_wave   = int(ctas_per_w[0]) if n_waves > 0 else 0
        avg_per_wave = n_ctas / max(n_waves, 1)
        util_first   = first_wave / n_sm

        print(f'{M:>7d} {rec["gx"]:>4d}x{rec["gy"]:<5d} {rec["tile_total"]:>6d} '
              f'{n_ctas:>14d} {sms_used:>9d} {n_waves:>6d} {first_wave:>6d} '
              f'{avg_per_wave:>14.2f} {util_first:>11.3f}')

        sub_rows = []
        for i in range(n_ctas):
            s_ns = int(rec['start'][i])
            e_ns = int(rec['end'][i])
            row = {
                'M': M, 'K': K, 'N': N,
                'cta_idx': int(rec['linear_idx'][i]),
                'bx': int(rec['bx'][i]),
                'by': int(rec['by'][i]),
                'bz': int(rec['bz'][i]),
                'smid':     int(rec['smid'][i]),
                'wave_idx': int(wave[i]),
                'start_ns': s_ns,
                'end_ns':   e_ns,
                'cta_dur_ns': max(0, e_ns - s_ns),
            }
            sub_rows.append(row)
            per_cta_rows.append(row)
        by_M[M] = sub_rows

        # per-wave timing summary
        for w in range(n_waves):
            mask = wave == w
            if not mask.any():
                continue
            ws_start = rec['start'][mask].astype(np.int64)
            we_end   = rec['end'][mask].astype(np.int64)
            cta_durs = np.maximum(0, we_end - ws_start)
            wave_start = int(ws_start.min())
            wave_end   = int(we_end.max())
            wave_dur   = wave_end - wave_start
            summary_rows.append({
                'M': M, 'kernel': kernel,
                'wave_idx': w,
                'n_ctas':         int(ctas_per_w[w]),
                'wave_dur_ns':    wave_dur,
                'cta_dur_mean_ns':  int(cta_durs.mean()),
                'cta_dur_median_ns':int(np.median(cta_durs)),
                'cta_dur_min_ns':   int(cta_durs.min()),
                'cta_dur_max_ns':   int(cta_durs.max()),
                'cta_dur_std_ns':   int(cta_durs.std()),
                'wave_start_ns':  wave_start,
                'wave_end_ns':    wave_end,
                'n_ctas_total':   n_ctas,
                'sms_used_total': sms_used,
                'n_sm':           n_sm,
            })

    # ---- write CSVs ----------------------------------------------------------
    suffix = f'_{model_name}_{kernel}'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    per_cta_path   = os.path.join(out_dir, f'cta_probe_per_cta{suffix}_{ts}.csv')
    summary_path   = os.path.join(out_dir, f'cta_probe_summary{suffix}_{ts}.csv')
    per_cta_latest = os.path.join(out_dir, f'cta_probe_per_cta{suffix}.csv')
    summary_latest = os.path.join(out_dir, f'cta_probe_summary{suffix}.csv')

    def write_csv(path, rows):
        if not rows:
            return
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    for p in (per_cta_path, per_cta_latest):
        write_csv(p, per_cta_rows)
    for p in (summary_path, summary_latest):
        write_csv(p, summary_rows)

    print()
    print(f'[probe] per-CTA CSV:  {per_cta_path}')
    print(f'[probe] summary CSV:  {summary_path}')

    # ---- plots ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[probe] matplotlib not available — skipping plot')
        return

    # Plot 1: cta_idx vs SM id, colored by wave_idx
    fig, axes = plt.subplots(len(Ms), 1, figsize=(13, 2.8 * len(Ms)), squeeze=False)
    for ax, M in zip(axes[:, 0], Ms):
        sub = by_M[M]
        if not sub:
            ax.set_title(f'M={M}  (no CTAs recorded)')
            continue
        x  = np.array([r['cta_idx'] for r in sub])
        sm = np.array([r['smid']           for r in sub])
        wv = np.array([r['wave_idx']       for r in sub])
        sc = ax.scatter(x, sm, c=wv, cmap='viridis', s=12, edgecolors='none')
        n_wv = int(wv.max()) + 1
        ax.set_title(f'M={M}   {len(sub)} CTAs,  {n_wv} waves '
                     f'(SMs used: {len(np.unique(sm))} / {props.multi_processor_count})')
        ax.set_xlabel('CTA index  ((bz*gy + by)*gx + bx)')
        ax.set_ylabel('physical SM id')
        ax.grid(alpha=0.2)
        fig.colorbar(sc, ax=ax, label='wave_idx')

    fig.suptitle(
        f'CTA → SM dispatch  ({model_name} down_proj, K={K}, N={N})  '
        f'kernel={kernel}  n_sm={props.multi_processor_count}, tile={TB_M}x{TB_N}',
        fontsize=12, y=1.0)
    fig.tight_layout()
    png_path = os.path.join(out_dir, f'cta_probe_sm{suffix}.png')
    fig.savefig(png_path, dpi=110, bbox_inches='tight')
    print(f'[probe] plot:         {png_path}')

    # Plot 2: cta_idx vs wave_idx
    fig2, axes2 = plt.subplots(len(Ms), 1, figsize=(13, 2.3 * len(Ms)), squeeze=False)
    for ax, M in zip(axes2[:, 0], Ms):
        sub = by_M[M]
        if not sub:
            continue
        x  = np.array([r['cta_idx'] for r in sub])
        wv = np.array([r['wave_idx']       for r in sub])
        ax.plot(x, wv, marker='.', ms=3, lw=0.4)
        ax.set_title(f'M={M}   launch_idx → wave_idx  ({len(sub)} CTAs)')
        ax.set_xlabel('CTA index')
        ax.set_ylabel('wave_idx')
        yt = list(range(0, int(wv.max()) + 1))
        if len(yt) <= 16:
            ax.set_yticks(yt)
        ax.grid(alpha=0.3)

    fig2.suptitle(f'CTA index vs wave  ({model_name} down_proj, kernel={kernel})',
                  fontsize=12, y=1.0)
    fig2.tight_layout()
    png2 = os.path.join(out_dir, f'cta_probe_waves{suffix}.png')
    fig2.savefig(png2, dpi=110, bbox_inches='tight')
    print(f'[probe] plot:         {png2}')

    # Plot 3: wave duration / CTA mainloop time per M
    fig3, axes3 = plt.subplots(len(Ms), 1, figsize=(13, 2.4 * len(Ms)),
                               squeeze=False)
    for ax, M in zip(axes3[:, 0], Ms):
        sub = by_M[M]
        if not sub:
            continue
        wv  = np.array([r['wave_idx']   for r in sub])
        dur = np.array([r['cta_dur_ns'] for r in sub])
        unique_w = np.arange(int(wv.max()) + 1)
        if len(unique_w) <= 20:
            box_data = [dur[wv == w] / 1000.0 for w in unique_w]
            ax.boxplot(box_data, positions=unique_w, widths=0.6,
                       showfliers=True)
            ax.set_xticks(unique_w[::max(1, len(unique_w)//12)])
        else:
            # large wave count — overlay scatter mean only
            wmean = np.array([dur[wv == w].mean() for w in unique_w])
            ax.plot(unique_w, wmean / 1000.0, marker='.', ms=2, lw=0.6)
        ax.set_title(
            f'M={M}  per-CTA mainloop duration  '
            f'({len(sub)} CTAs, {int(wv.max())+1} waves)')
        ax.set_xlabel('wave_idx')
        ax.set_ylabel('per-CTA dur (us)')
        ax.grid(alpha=0.3)
    fig3.suptitle(f'CTA mainloop duration per wave ({model_name}, {kernel})',
                  fontsize=12, y=1.0)
    fig3.tight_layout()
    png3 = os.path.join(out_dir, f'cta_probe_durations{suffix}.png')
    fig3.savefig(png3, dpi=110, bbox_inches='tight')
    print(f'[probe] plot:         {png3}')


if __name__ == '__main__':
    main()
