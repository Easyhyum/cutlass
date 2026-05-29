#!/usr/bin/env python3
"""
Baseline s100 (V9 inactive) with sweep over burst_gap_ms.

Goal: does SM clock fully recover to boost between bursts? Larger gap should
allow more recovery. If gap is too short, next burst starts before clock
returns to boost → 'persistent throttle' state.

Sweeps gap_ms ∈ MM_GAPS_MS (default 200,400,600).
Each gap value = one config segment (N bursts × M kernels).
"""
import csv
import os
import sys
import time
from datetime import datetime

import torch

H, INTER = 4096, 12288


def fmt_ts(t):
    dt = datetime.fromtimestamp(t)
    return dt.strftime('%Y/%m/%d %H:%M:%S.') + f'{int(dt.microsecond/1000):03d}'


@torch.no_grad()
def main():
    gpu = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(gpu)
    dev = torch.device(f'cuda:{gpu}')

    M = int(os.environ.get('MM_M', '8192'))
    K, N = INTER, H  # down_proj
    n_bursts   = int(os.environ.get('MM_N_BURSTS', '100'))
    m_kernels  = int(os.environ.get('MM_M_KERNELS', '150'))
    cfg_gap_ms = int(os.environ.get('MM_CFG_GAP_MS', '2000'))
    warmup_ms  = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
    seg_path   = os.environ['MM_SEGMENTS']
    gaps = [int(x) for x in os.environ.get('MM_GAPS_MS', '200,400,600').split(',')]

    print(f'[eval] device={gpu} ({torch.cuda.get_device_name(gpu)})')
    print(f'[eval] down_proj M={M} K={K} N={N}')
    print(f'[eval] N_BURSTS={n_bursts}  M_KERNELS={m_kernels}  '
          f'gap_ms_sweep={gaps}  cfg_gap_ms={cfg_gap_ms}')

    ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ext_dir not in sys.path:
        sys.path.insert(0, ext_dir)
    import bf16_gemm_sm80_streamk as ext

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)

    if warmup_ms > 0:
        print(f'[eval] global warmup {warmup_ms:.0f}ms...')
        t0 = time.time()
        while (time.time() - t0)*1000.0 < warmup_ms:
            ext.gemm_streamk(A, B)
        torch.cuda.synchronize()

    # baseline measure
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for _ in range(5): ext.gemm_streamk(A, B)
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(50): ext.gemm_streamk(A, B)
    ev1.record(); torch.cuda.synchronize()
    baseline_ms = ev0.elapsed_time(ev1)/50
    baseline_tf = 2.0*M*K*N/(baseline_ms*1e-3)/1e12
    print(f'[eval] baseline kernel ms={baseline_ms:.4f}  TFLOPS={baseline_tf:.1f}')
    print()

    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    seg_f = open(seg_path, 'w', newline='')
    w = csv.writer(seg_f)
    w.writerow(['backend','operator','M','K','N','iters',
                't_start','t_end','ms_avg','tflops'])
    seg_f.flush()

    print(f'{"cfg":>10s} {"gap_ms":>6s} {"N":>4s}  burst progress')
    print('-' * 60)

    for gap_ms in gaps:
        # extra cfg gap so clock fully relaxes between sweep cfgs
        time.sleep(cfg_gap_ms / 1000.0)
        cfg = f'baseline_g{gap_ms}'
        for burst_i in range(n_bursts):
            torch.cuda.synchronize()
            t0 = time.time()
            ev0.record()
            for _ in range(m_kernels):
                ext.gemm_streamk(A, B)   # baseline: no V9
            ev1.record(); torch.cuda.synchronize()
            t1 = time.time()
            ms_avg = ev0.elapsed_time(ev1)/m_kernels
            tflops = 2.0*M*K*N/(ms_avg*1e-3)/1e12
            ts_, te_ = fmt_ts(t0), fmt_ts(t1)
            tag = f'stream_k:{cfg}#{burst_i}'
            w.writerow([tag, 'down_proj', M, K, N, m_kernels,
                        ts_, te_, f'{ms_avg:.6f}', f'{tflops:.4f}'])
            seg_f.flush()
            time.sleep(gap_ms / 1000.0)
        print(f'{cfg:>10s} {gap_ms:>6d} {n_bursts:>4d}  done', flush=True)

    seg_f.close()
    print('[eval] DONE')


if __name__ == '__main__':
    main()
