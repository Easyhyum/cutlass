#!/usr/bin/env python3
"""
Compare 4 wave-sleep variants against the pristine-baseline binary:
  A: baseline   — no wave-sleep code in SASS         (binary = baseline)
  B: wave-0 + small mid bubble  (mode=0, mid params, sync pattern)
  C: ALL-wave staircase         (mode=1, shape=0)
  D: wave-0 + quartile staircase shape (mode=0, shape=1)

I/O shape == eval_wave_sleep_n_burst.py.

Tag format kept compatible with analyze_power.py:
   stream_k:<short_cfg>#<burst>
"""
import csv, os, sys, time
from datetime import datetime
import torch

H, INTER, VOCAB = 4096, 12288, 151936
TB_M, TB_N = 128, 128


def fmt_ts(t):
    dt = datetime.fromtimestamp(t)
    return dt.strftime('%Y/%m/%d %H:%M:%S.') + f'{int(dt.microsecond/1000):03d}'


def expected_num_waves(M, K, N, n_sm):
    tiles = ((M + TB_M - 1) // TB_M) * ((N + TB_N - 1) // TB_N)
    if tiles < n_sm:
        return 1
    return tiles // n_sm


# avail_sms sweep — kernel-level SM limit via streamk's `avail_sms` argument.
# No wave-sleep code involved; uses the BASELINE binary. Streamk swizzle
# self-limits the launched grid to `avail_sms` SMs (idle ones not used).
# 9th tuple field = avail_sms value (-1 = full = 188 SMs).
VARIANTS = [
    # ('grp', 'cfg', S, P, mid_pct, mid_ns, mode_dev, shape, avail_sms)
    ('V', 'a_sms188',  0, 0, 0, 0, 0, 0, 188),
    ('V', 'a_sms170',  0, 0, 0, 0, 0, 0, 170),
    ('V', 'a_sms160',  0, 0, 0, 0, 0, 0, 160),
    ('V', 'a_sms150',  0, 0, 0, 0, 0, 0, 150),
    ('V', 'a_sms140',  0, 0, 0, 0, 0, 0, 140),
    ('V', 'a_sms130',  0, 0, 0, 0, 0, 0, 130),   # ~70% (MPS-equiv)
    ('V', 'a_sms120',  0, 0, 0, 0, 0, 0, 120),
    ('V', 'a_sms110',  0, 0, 0, 0, 0, 0, 110),
    ('V', 'a_sms100',  0, 0, 0, 0, 0, 0, 100),
    ('V', 'a_sms90',   0, 0, 0, 0, 0, 0,  90),
    ('V', 'a_sms80',   0, 0, 0, 0, 0, 0,  80),
]


@torch.no_grad()
def main():
    gpu = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(gpu)
    dev = torch.device(f'cuda:{gpu}')

    op_name = 'down_proj'
    M = int(os.environ.get('MM_M', '8192'))
    K, N = INTER, H

    n_bursts   = int(os.environ.get('MM_N_BURSTS', '30'))
    m_kernels  = int(os.environ.get('MM_M_KERNELS', '150'))
    burst_gap  = int(os.environ.get('MM_BURST_GAP_MS', '500'))
    cfg_gap    = int(os.environ.get('MM_CFG_GAP_MS', '1000'))
    warmup_ms  = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
    seg_path   = os.environ['MM_SEGMENTS']
    mode       = os.environ.get('MM_MODE', 'all').lower()
    if mode not in ('baseline', 'ws', 'all'):
        raise SystemExit(f'unknown MM_MODE={mode}')

    props = torch.cuda.get_device_properties(gpu)
    n_sms = props.multi_processor_count

    sys.path.insert(0, '/workspace/custum')
    ext_ws   = None
    ext_base = None
    if mode in ('ws', 'all'):
        import bf16_gemm_sm80_streamk          as _ws;   ext_ws   = _ws
    if mode in ('baseline', 'all'):
        import bf16_gemm_sm80_streamk_baseline as _base; ext_base = _base

    num_waves_expected = expected_num_waves(M, K, N, n_sms)
    print(f'[var] device={gpu}  n_sms={n_sms}  mode={mode}')
    print(f'[var] op={op_name}  M={M} K={K} N={N}   num_waves={num_waves_expected}')
    print(f'[var] N_BURSTS={n_bursts}  K/burst={m_kernels}  gap={burst_gap}ms')

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)

    warm = ext_base if mode == 'baseline' else ext_ws
    if warm is None: warm = ext_ws if ext_ws else ext_base
    if warmup_ms > 0:
        t0 = time.time()
        while (time.time() - t0)*1000.0 < warmup_ms:
            warm.gemm_streamk(A, B)
        torch.cuda.synchronize()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for _ in range(5): warm.gemm_streamk(A, B)
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(50): warm.gemm_streamk(A, B)
    ev1.record(); torch.cuda.synchronize()
    base_ms = ev0.elapsed_time(ev1)/50
    print(f'[var] reference ms={base_ms:.4f}  TFLOPS={2.0*M*K*N/(base_ms*1e-3)/1e12:.1f}')

    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    csv_exists = os.path.exists(seg_path)
    seg_f = open(seg_path, 'a' if csv_exists else 'w', newline='')
    w = csv.writer(seg_f)
    if not csv_exists:
        w.writerow(['backend','operator','M','K','N','iters',
                    't_start','t_end','ms_avg','tflops'])
        seg_f.flush()

    runs = []
    if mode in ('baseline', 'all'):
        runs.append({'group': 'A', 'cfg': 's100_step0',
                     'use_ws': False, 'S': 100, 'P': 0,
                     'mid_pct': 0, 'mid_ns': 0, 'mode_dev': 0, 'shape': 0})
    if mode in ('ws', 'all'):
        for vt in VARIANTS:
            # back-compat: 8-tuple = wave-sleep, 9-tuple = avail_sms (V group)
            if len(vt) == 8:
                grp, cfg, S, P, mid_pct, mid_ns, m_dev, shape = vt
                avail_sms = -1
            else:
                grp, cfg, S, P, mid_pct, mid_ns, m_dev, shape, avail_sms = vt
            if grp == 'V':
                # V variants use BASELINE binary (no wave-sleep code) and the
                # streamk `avail_sms` argument. Skip during MM_MODE=ws phase;
                # we'll inject them into the baseline phase instead.
                continue
            thr = (n_sms * S) // 100
            runs.append({'group': grp, 'cfg': cfg,
                         'use_ws': True, 'S': S, 'P': P, 'thr': thr,
                         'mid_pct': mid_pct, 'mid_ns': mid_ns,
                         'mode_dev': m_dev, 'shape': shape,
                         'avail_sms': -1})

    if mode in ('baseline', 'all'):
        # also handle V variants here (they use baseline binary)
        for vt in VARIANTS:
            if len(vt) < 9: continue
            grp, cfg, S, P, mid_pct, mid_ns, m_dev, shape, avail_sms = vt
            if grp != 'V': continue
            runs.append({'group': grp, 'cfg': cfg,
                         'use_ws': False, 'S': 0, 'P': 0, 'thr': 0,
                         'mid_pct': 0, 'mid_ns': 0,
                         'mode_dev': 0, 'shape': 0,
                         'avail_sms': avail_sms})

    print()
    print(f'{"grp":>3s} {"cfg":>20s}  {"S":>3s} {"P":>5s}  {"mid":>5s} {"mid_ns":>6s}  '
          f'{"mode":>4s} {"shape":>5s}  progress')
    print('-' * 95)

    for r in runs:
        time.sleep(cfg_gap / 1000.0)
        ext = ext_base if not r['use_ws'] else ext_ws
        if ext is None:
            print(f'  SKIP {r["cfg"]} (binary not loaded for mode={mode})')
            continue
        for burst_i in range(n_bursts):
            torch.cuda.synchronize()
            t0 = time.time()
            ev0.record()
            _avail = r.get('avail_sms', -1)
            for _ in range(m_kernels):
                if r['use_ws']:
                    ext.prime_wave_sleep(num_waves_expected, n_sms,
                                         r['thr'], r['P'],
                                         r['mid_pct'], r['mid_ns'],
                                         r['mode_dev'], r['shape'])
                ext.gemm_streamk(A, B, 1, _avail)
            ev1.record(); torch.cuda.synchronize()
            t1 = time.time()
            ms_avg = ev0.elapsed_time(ev1)/m_kernels
            tflops = 2.0*M*K*N/(ms_avg*1e-3)/1e12
            ts_, te_ = fmt_ts(t0), fmt_ts(t1)
            tag = f'stream_k:{r["cfg"]}#{burst_i}'
            w.writerow([tag, op_name, M, K, N, m_kernels,
                        ts_, te_, f'{ms_avg:.6f}', f'{tflops:.4f}'])
            seg_f.flush()
            time.sleep(burst_gap / 1000.0)
        print(f'{r["group"]:>3s} {r["cfg"]:>20s}  {r["S"]:>3d} {r["P"]:>5d}  '
              f'{r["mid_pct"]:>5d} {r["mid_ns"]:>6d}  {r["mode_dev"]:>4d} '
              f'{r["shape"]:>5d}  {n_bursts} done',
              flush=True)

    seg_f.close()
    print('[var] DONE')


if __name__ == '__main__':
    main()
