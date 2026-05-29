#!/usr/bin/env python3
"""
CTA → SM / wave dispatch probe for streamk and sm80_v3 (per-kernel run).

Loads ONE probe binary (streamk OR sm80_v3 — never both, to avoid the cross-TU
"Error Internal" we hit before when both extensions are loaded into the same
process) and sweeps M for the Qwen3-8B down_proj op, writing per-CTA records
to CSV.

Env:
  MM_KERNEL   streamk | sm80_v3        (REQUIRED)
  MM_MODEL    qwen3-8b (default) | qwen3-32b
  MM_OP       down_proj (default) | qkv_proj | o_proj | up_proj | lm_head
  MM_MS       comma list of M (default: 32,256,1024,8192,65536)
  OUT_DIR     output dir (default = this dir)
  MM_GPU      cuda device index (default 0)
"""
import os, sys, csv
from datetime import datetime
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# (op, model) → (K, N)
OPS = {
    ('qkv_proj',   'qwen3-8b'):  ( 4096,   6144),
    ('o_proj',     'qwen3-8b'):  ( 4096,   4096),
    ('up_proj',    'qwen3-8b'):  ( 4096,  12288),
    ('down_proj',  'qwen3-8b'):  (12288,   4096),
    ('lm_head',    'qwen3-8b'):  ( 4096, 151936),
    ('qkv_proj',   'qwen3-32b'): ( 5120,  10240),
    ('o_proj',     'qwen3-32b'): ( 8192,   5120),
    ('up_proj',    'qwen3-32b'): ( 5120,  25600),
    ('down_proj',  'qwen3-32b'): (25600,   5120),
    ('lm_head',    'qwen3-32b'): ( 5120, 151936),
}

# Both probe kernels use 128×128 M×N tile (same as production builds)
TB_M, TB_N = 128, 128


def load_kernel(kernel):
    if kernel == 'streamk':
        sys.path.insert(0, os.path.join(HERE, 'build_streamk_probe'))
        import probe_streamk as ext
        fn = lambda A, B: ext.gemm_streamk_probe(A, B, 1, -1)
        return ext, fn
    elif kernel == 'sm80_v3':
        sys.path.insert(0, os.path.join(HERE, 'build_sm80_v3_probe'))
        import probe_sm80_v3 as ext
        fn = lambda A, B: ext.gemm_sm80_v3_probe(A, B)
        return ext, fn
    else:
        raise SystemExit(f'unknown MM_KERNEL={kernel}')


def grid_for(M, N):
    gy = (M + TB_M - 1) // TB_M
    gx = (N + TB_N - 1) // TB_N
    return gx, gy, 1


def probe_one(ext, fn, M, K, N, dev):
    gx, gy, gz = grid_for(M, N)
    n_sm = torch.cuda.get_device_properties(dev.index).multi_processor_count
    # streamk swizzle launches a 1D grid not equal to gx*gy; overprovision
    cap = max(gx * gy * gz, n_sm * 8) * 2 + 4096

    smid    = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    start_t = torch.zeros(cap, dtype=torch.int64, device=dev)
    end_t   = torch.zeros(cap, dtype=torch.int64, device=dev)
    bx      = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    by      = torch.full((cap,), -1, dtype=torch.int32, device=dev)
    bz      = torch.full((cap,), -1, dtype=torch.int32, device=dev)

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)

    ext.set_probe_buffers(smid, start_t, end_t, bx, by, bz, cap)
    fn(A, B)                      # warmup
    torch.cuda.synchronize()

    # measurement run
    smid.fill_(-1)
    start_t.zero_()
    end_t.zero_()
    ext.set_probe_buffers(smid, start_t, end_t, bx, by, bz, cap)
    fn(A, B)
    torch.cuda.synchronize()
    ext.clear_probe_buffers()

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
    kernel = os.environ.get('MM_KERNEL', '').strip().lower()
    if kernel not in ('streamk', 'sm80_v3'):
        raise SystemExit('MM_KERNEL must be one of: streamk | sm80_v3')

    model = os.environ.get('MM_MODEL', 'qwen3-8b').strip().lower()
    op    = os.environ.get('MM_OP',    'down_proj').strip().lower()
    if (op, model) not in OPS:
        raise SystemExit(f'unknown (op={op}, model={model})')
    K, N = OPS[(op, model)]

    Ms = [int(x) for x in os.environ.get(
        'MM_MS', '32,256,1024,4096,8192,65536,131072').split(',')]

    out_dir = os.environ.get('OUT_DIR', HERE)
    os.makedirs(out_dir, exist_ok=True)

    cuda_idx = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(cuda_idx)
    dev = torch.device(f'cuda:{cuda_idx}')

    props = torch.cuda.get_device_properties(cuda_idx)
    n_sm = props.multi_processor_count

    print(f'[probe] device={props.name}  sm={props.major}.{props.minor}  n_sm={n_sm}')
    print(f'[probe] kernel={kernel}  model={model}  op={op}  K={K} N={N}')
    print(f'[probe] M list={Ms}  out_dir={out_dir}')

    ext, fn = load_kernel(kernel)

    per_cta_rows, summary_rows, by_M = [], [], {}

    print(f'\n{"M":>7s} {"grid_xy":>10s} {"tiles":>6s} {"CTAs":>6s} '
          f'{"SMs":>4s} {"waves":>6s} {"wave0_CTAs":>10s} '
          f'{"util_first":>11s}')
    print('-' * 80)

    for M in Ms:
        rec  = probe_one(ext, fn, M, K, N, dev)
        wave = assign_waves(rec['smid'], rec['start'])

        n_ctas     = len(rec['smid'])
        n_waves    = (int(wave.max()) + 1) if n_ctas > 0 else 0
        sms_used   = int(np.unique(rec['smid']).size)
        ctas_per_w = np.bincount(wave, minlength=max(1, n_waves))
        wave0_ctas = int(ctas_per_w[0]) if n_waves > 0 else 0
        util_first = wave0_ctas / n_sm

        print(f'{M:>7d} {rec["gx"]:>4d}x{rec["gy"]:<5d} '
              f'{rec["tile_total"]:>6d} {n_ctas:>6d} {sms_used:>4d} '
              f'{n_waves:>6d} {wave0_ctas:>10d} {util_first:>11.3f}')

        sub = []
        for i in range(n_ctas):
            row = {
                'M': M, 'K': K, 'N': N,
                'cta_idx': int(rec['linear_idx'][i]),
                'bx': int(rec['bx'][i]),
                'by': int(rec['by'][i]),
                'bz': int(rec['bz'][i]),
                'smid':     int(rec['smid'][i]),
                'wave_idx': int(wave[i]),
                'start_ns': int(rec['start'][i]),
                'end_ns':   int(rec['end'][i]),
                'cta_dur_ns': max(0, int(rec['end'][i]) - int(rec['start'][i])),
            }
            sub.append(row); per_cta_rows.append(row)
        by_M[M] = sub

        for w in range(n_waves):
            mask = wave == w
            if not mask.any():
                continue
            ws = rec['start'][mask].astype(np.int64)
            we = rec['end'][mask].astype(np.int64)
            durs = np.maximum(0, we - ws)
            summary_rows.append({
                'M': M, 'kernel': kernel, 'op': op, 'model': model,
                'wave_idx': w, 'n_ctas': int(ctas_per_w[w]),
                'wave_dur_ns':       int(we.max() - ws.min()),
                'cta_dur_mean_ns':   int(durs.mean()),
                'cta_dur_median_ns': int(np.median(durs)),
                'cta_dur_min_ns':    int(durs.min()),
                'cta_dur_max_ns':    int(durs.max()),
                'cta_dur_std_ns':    int(durs.std()),
                'wave_start_ns':     int(ws.min()),
                'wave_end_ns':       int(we.max()),
                'n_ctas_total':      n_ctas,
                'sms_used_total':    sms_used,
                'n_sm':              n_sm,
            })

    # ─── CSV outputs (timestamped + "latest" symlink-like alias) ──────────────
    suffix = f'_{model}_{op}_{kernel}'
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    paths = {
        'per_cta_ts':     os.path.join(out_dir, f'cta_probe_per_cta{suffix}_{ts}.csv'),
        'summary_ts':     os.path.join(out_dir, f'cta_probe_summary{suffix}_{ts}.csv'),
        'per_cta_latest': os.path.join(out_dir, f'cta_probe_per_cta{suffix}.csv'),
        'summary_latest': os.path.join(out_dir, f'cta_probe_summary{suffix}.csv'),
    }

    def write_csv(path, rows):
        if not rows:
            return
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    write_csv(paths['per_cta_ts'],     per_cta_rows)
    write_csv(paths['per_cta_latest'], per_cta_rows)
    write_csv(paths['summary_ts'],     summary_rows)
    write_csv(paths['summary_latest'], summary_rows)

    print(f'\n[probe] per-CTA CSV: {paths["per_cta_latest"]}')
    print(f'[probe] summary CSV: {paths["summary_latest"]}')


if __name__ == '__main__':
    main()
