#!/usr/bin/env python3
"""
V9 spatial SM ramp evaluation — N independent bursts per config.

Sweeps:
  start_pct ∈ {70, 80, 90, 100}     → smid_threshold = pct * n_sms / 100
  step_ns   ∈ {500, 2000, 5000, 10000, 20000, 50000}

n_sms: read from device (RTX PRO 6000 Blackwell = 188).

For each (start_pct, step_ns):
  - N bursts × M kernels, ramp applied to every kernel (burst mode for
    power measurement convenience). One-shot deployment cost is N times
    smaller (the v9 ramp adds only ~max_delay per kernel call, which is
    negligible if applied once per session).
"""
import csv
import os
import sys
import time
from datetime import datetime

import torch

H, INTER, VOCAB = 4096, 12288, 151936
OPS = {
    'qkv_proj':     (H,     6144),
    'o_proj':       (4096,  H),
    'gate_up_proj': (H,     24576),
    'down_proj':    (INTER, H),
    'lm_head':      (H,     VOCAB),
}


def fmt_ts(t):
    dt = datetime.fromtimestamp(t)
    return dt.strftime('%Y/%m/%d %H:%M:%S.') + f'{int(dt.microsecond/1000):03d}'


@torch.no_grad()
def main():
    gpu = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(gpu)
    dev = torch.device(f'cuda:{gpu}')

    op_name = os.environ.get('MM_OP', 'down_proj')
    M       = int(os.environ.get('MM_M', '8192'))
    K, N    = OPS[op_name]

    n_bursts   = int(os.environ.get('MM_N_BURSTS', '100'))
    m_kernels  = int(os.environ.get('MM_M_KERNELS', '150'))
    burst_gap_ms = int(os.environ.get('MM_BURST_GAP_MS', '200'))
    cfg_gap_ms = int(os.environ.get('MM_CFG_GAP_MS', '500'))
    warmup_ms  = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
    seg_path   = os.environ['MM_SEGMENTS']

    starts = [int(x) for x in
              os.environ.get('MM_V9_STARTS', '70,80,90,100').split(',')]
    steps  = [int(x) for x in
              os.environ.get('MM_V9_STEPS_NS',
                             '500,2000,5000,10000,20000,50000').split(',')]

    # n_sms from device
    props = torch.cuda.get_device_properties(gpu)
    n_sms = props.multi_processor_count
    print(f'[eval] device={gpu} ({torch.cuda.get_device_name(gpu)})  n_sms={n_sms}')

    ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ext_dir not in sys.path:
        sys.path.insert(0, ext_dir)
    import bf16_gemm_sm80_streamk as ext

    print(f'[eval] {op_name}  M={M} K={K} N={N}')
    print(f'[eval] N_BURSTS={n_bursts}  M_KERNELS/burst={m_kernels}')
    print(f'[eval] starts={starts}  steps_ns={steps}')

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)

    if warmup_ms > 0:
        print(f'[eval] global warmup {warmup_ms:.0f}ms...')
        t0 = time.time()
        while (time.time() - t0)*1000.0 < warmup_ms:
            ext.gemm_streamk(A, B)
        torch.cuda.synchronize()

    # baseline TFLOPS
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for _ in range(5): ext.gemm_streamk(A, B)
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(50): ext.gemm_streamk(A, B)
    ev1.record(); torch.cuda.synchronize()
    baseline_ms = ev0.elapsed_time(ev1)/50
    baseline_tf = 2.0*M*K*N/(baseline_ms*1e-3)/1e12
    print(f'[eval] baseline ms_avg={baseline_ms:.4f}  TFLOPS={baseline_tf:.1f}')

    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    seg_f = open(seg_path, 'w', newline='')
    w = csv.writer(seg_f)
    w.writerow(['backend','operator','M','K','N','iters',
                't_start','t_end','ms_avg','tflops'])
    seg_f.flush()

    rows = []
    for s in starts:
        threshold = (n_sms * s) // 100
        for stp in steps:
            ramp_sms = n_sms - threshold
            max_delay_us = ramp_sms * stp / 1000.0
            cfg = f's{s}_step{stp}'
            rows.append((cfg, s, stp, threshold, max_delay_us))

    n_total = len(rows) * n_bursts
    est_per_burst_s = m_kernels * 2.2 / 1000.0
    est_total_s = n_total * (est_per_burst_s + burst_gap_ms/1000.0)
    print(f'[eval] total bursts={n_total}  estimated time={est_total_s/60:.1f} min')
    print()
    print(f'{"cfg":>14s}  {"start%":>6s} {"step_ns":>7s} {"thr":>4s} '
          f'{"max_delay":>10s}  progress')
    print('-' * 70)

    for cfg, s, stp, thr, mxus in rows:
        time.sleep(cfg_gap_ms / 1000.0)
        for burst_i in range(n_bursts):
            torch.cuda.synchronize()
            t0 = time.time()
            ev0.record()
            for _ in range(m_kernels):
                # v9 ramp applied to every kernel (burst mode)
                ext.gemm_streamk(A, B, 1, -1, 0, 1, 0, 1, 100, 100, 0,
                                 v9_smid_threshold=thr, v9_step_ns=stp)
            ev1.record(); torch.cuda.synchronize()
            t1 = time.time()
            ms_avg = ev0.elapsed_time(ev1)/m_kernels
            tflops = 2.0*M*K*N/(ms_avg*1e-3)/1e12
            ts_, te_ = fmt_ts(t0), fmt_ts(t1)
            tag = f'stream_k:{cfg}#{burst_i}'
            w.writerow([tag, op_name, M, K, N, m_kernels,
                        ts_, te_, f'{ms_avg:.6f}', f'{tflops:.4f}'])
            seg_f.flush()
            time.sleep(burst_gap_ms / 1000.0)
        print(f'{cfg:>14s}  {s:>6d} {stp:>7d} {thr:>4d} '
              f'{mxus:>8.1f}us  {n_bursts} done', flush=True)

    seg_f.close()
    print('[eval] DONE')


if __name__ == '__main__':
    main()
