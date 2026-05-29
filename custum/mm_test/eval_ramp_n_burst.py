#!/usr/bin/env python3
"""
RAMP power-spike statistical evaluation.

For each (start_pct, step_pct) config:
  - Run N_BURSTS independent bursts, each of M_KERNELS kernels.
  - Ramp is applied to EVERY kernel in the burst (so per-burst power profile
    is consistent and the per-burst max_W is a meaningful sample).
  - Idle gap between bursts so GPU/DVFS state can briefly relax.
  - Each burst recorded as a separate segment row → analyze_power.py
    extracts per-burst max_W, avg_W, sm_p10.

Post-processing (computed inline at end):
  per config, aggregate across N bursts:
    max_W_mean / max_W_std / max_W_min / max_W_max
    avg_W_mean
    sm_p10_mean
    burst_tflops_mean
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

    n_bursts   = int(os.environ.get('MM_N_BURSTS', '20'))
    m_kernels  = int(os.environ.get('MM_M_KERNELS', '200'))
    burst_gap_ms = int(os.environ.get('MM_BURST_GAP_MS', '200'))
    cfg_gap_ms = int(os.environ.get('MM_CFG_GAP_MS', '500'))
    warmup_ms  = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
    seg_path   = os.environ['MM_SEGMENTS']

    starts = [int(x) for x in
              os.environ.get('MM_RAMP_START_PCTS',
                              '70,75,80,85,90,95,100').split(',')]
    steps  = [int(x) for x in
              os.environ.get('MM_RAMP_STEP_PCTS', '1,2,5,10').split(',')]

    ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ext_dir not in sys.path:
        sys.path.insert(0, ext_dir)
    import bf16_gemm_sm80_streamk as ext

    print(f'[eval] device={gpu} ({torch.cuda.get_device_name(gpu)})')
    print(f'[eval] {op_name}  M={M} K={K} N={N}')
    print(f'[eval] N_BURSTS={n_bursts}  M_KERNELS/burst={m_kernels}  '
          f'burst_gap={burst_gap_ms}ms  cfg_gap={cfg_gap_ms}ms')
    print(f'[eval] starts={starts}  steps={steps}')
    n_cfg = len(starts) * len(steps)
    burst_dur_s = m_kernels * 2.2 / 1000.0  # ~2.2ms/kernel for down_proj M=8192
    est_per_cfg_s = n_bursts * (burst_dur_s + burst_gap_ms/1000.0) + cfg_gap_ms/1000.0
    print(f'[eval] estimated total time: {n_cfg * est_per_cfg_s / 60:.1f} min')

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)

    if warmup_ms > 0:
        print(f'[eval] global warmup {warmup_ms:.0f}ms...')
        t0 = time.time()
        while (time.time() - t0)*1000.0 < warmup_ms:
            ext.gemm_streamk(A, B)
        torch.cuda.synchronize()

    # iter_time estimate
    for _ in range(5): ext.gemm_streamk(A, B)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(50): ext.gemm_streamk(A, B)
    ev1.record(); torch.cuda.synchronize()
    baseline_ms = ev0.elapsed_time(ev1)/50
    baseline_tf = 2.0*M*K*N/(baseline_ms*1e-3)/1e12
    BLOCK_K = 32
    iter_time_ns = int((baseline_ms * 1e6) / (K // BLOCK_K))
    print(f'[eval] baseline ms_avg={baseline_ms:.4f}  TFLOPS={baseline_tf:.1f}  '
          f'iter_time≈{iter_time_ns}ns')

    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    seg_f = open(seg_path, 'w', newline='')
    w = csv.writer(seg_f)
    w.writerow(['backend','operator','M','K','N','iters',
                't_start','t_end','ms_avg','tflops'])
    seg_f.flush()

    rows = []
    for s in starts:
        for st in steps:
            r_iters = 0 if s >= 100 else (100 - s + st - 1) // st
            rows.append((f's{s}_p{st}', s, st, r_iters))

    print()
    print(f'{"cfg":>10s}  {"start%":>6s} {"step%":>5s}  '
          f'{"bursts":>6s} progress')
    print('-' * 60)
    for cfg, s, st, r_iters in rows:
        time.sleep(cfg_gap_ms / 1000.0)
        for burst_i in range(n_bursts):
            torch.cuda.synchronize()
            t0 = time.time()
            ev0.record()
            for _ in range(m_kernels):
                ext.gemm_streamk(A, B, 1, -1, 0, 1, 0, 1,
                                 ramp_start_pct=s, ramp_step_pct=st,
                                 ramp_iter_time_ns=iter_time_ns)
            ev1.record(); torch.cuda.synchronize()
            t1 = time.time()
            ms_avg = ev0.elapsed_time(ev1)/m_kernels
            tflops = 2.0*M*K*N/(ms_avg*1e-3)/1e12
            ts_, te_ = fmt_ts(t0), fmt_ts(t1)
            # tag: cfg#burstN — so post-processing can group by cfg
            tag = f'stream_k:{cfg}#{burst_i}'
            w.writerow([tag, op_name, M, K, N, m_kernels,
                        ts_, te_, f'{ms_avg:.6f}', f'{tflops:.4f}'])
            seg_f.flush()
            time.sleep(burst_gap_ms / 1000.0)
        print(f'{cfg:>10s}  {s:>6d} {st:>5d}  {n_bursts:>6d} done', flush=True)

    seg_f.close()
    print('[eval] DONE')


if __name__ == '__main__':
    main()
