#!/usr/bin/env python3
"""
RAMP power-spike measurement.

Strategy: in a one-shot deployment, only 1 kernel per session has ramp →
nvidia-smi 50ms sampling cannot resolve a single 2ms kernel. For *power
measurement* purposes only, we apply the ramp to ALL kernels in a burst
(N=500 kernels per config) so the cumulative window is long enough for
nvidia-smi to capture representative samples. Throughput cost of applying
ramp to every kernel is small (~0.3%) for the configs we sweep, since each
ramp is only a few outer iters out of 384.

For each (start_pct, step_pct):
  - Per-kernel: pass ramp args directly (not prime_ramp) so every call has ramp
  - Burst: N kernels back-to-back, timed as one segment
  - Output: segments csv → analyze_power.py extracts max_W, avg_W, sm_p10, etc.

If start_pct=100 → ramp inactive → that segment = pristine baseline.
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

    n_burst    = int(os.environ.get('MM_BURST_KERNELS', '500'))
    gap_ms     = int(os.environ.get('MM_BURST_GAP_MS', '300'))
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
    print(f'[eval] {op_name}  M={M} K={K} N={N}  '
          f'n_burst={n_burst}  gap_ms={gap_ms}')
    print(f'[eval] starts={starts}  steps={steps}')

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)

    # Global warmup
    if warmup_ms > 0:
        print(f'[eval] global warmup {warmup_ms:.0f}ms...')
        t0 = time.time()
        while (time.time() - t0)*1000.0 < warmup_ms:
            ext.gemm_streamk(A, B)
        torch.cuda.synchronize()

    # Estimate iter_time
    for _ in range(5): ext.gemm_streamk(A, B)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(50): ext.gemm_streamk(A, B)
    ev1.record(); torch.cuda.synchronize()
    baseline_ms = ev0.elapsed_time(ev1)/50
    baseline_tflops = 2.0*M*K*N/(baseline_ms*1e-3)/1e12
    BLOCK_K = 32
    outer_iters_per_kernel = K // BLOCK_K
    iter_time_ns = int((baseline_ms * 1e6) / outer_iters_per_kernel)
    print(f'[eval] baseline ms_avg={baseline_ms:.4f}  TFLOPS={baseline_tflops:.1f}')
    print(f'[eval] outer_iters={outer_iters_per_kernel}  iter_time≈{iter_time_ns} ns')

    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    seg_f = open(seg_path, 'w', newline='')
    w = csv.writer(seg_f)
    w.writerow(['backend','operator','M','K','N','iters',
                't_start','t_end','ms_avg','tflops'])
    seg_f.flush()

    print()
    hdr = (f'{"cfg":>14s}  {"start%":>6s} {"step%":>5s} {"r_iters":>7s}  '
           f'{"burst_TF":>8s} {"%base":>6s}')
    print(hdr); print('-' * len(hdr))

    rows = []
    for s in starts:
        for st in steps:
            r_iters = 0 if s >= 100 else (100 - s + st - 1) // st  # ceil
            cfg = f's{s}_p{st}'
            rows.append((cfg, s, st, r_iters))

    for cfg, s, st, r_iters in rows:
        # idle gap so DVFS state can settle (also gives nvidia-smi sample boundary)
        time.sleep(gap_ms / 1000.0)

        # Burst with ramp applied to EVERY kernel (via direct args, not prime)
        torch.cuda.synchronize()
        t0 = time.time()
        ev0.record()
        for _ in range(n_burst):
            ext.gemm_streamk(A, B, 1, -1, 0, 1, 0, 1,
                             ramp_start_pct=s, ramp_step_pct=st,
                             ramp_iter_time_ns=iter_time_ns)
        ev1.record(); torch.cuda.synchronize()
        t1 = time.time()

        burst_ms = ev0.elapsed_time(ev1) / n_burst
        burst_tflops = 2.0*M*K*N/(burst_ms*1e-3)/1e12
        pct = burst_tflops / baseline_tflops * 100.0

        print(f'{cfg:>14s}  {s:>6d} {st:>5d} {r_iters:>7d}  '
              f'{burst_tflops:>8.1f} {pct:>5.2f}%')
        sys.stdout.flush()

        ts, te = fmt_ts(t0), fmt_ts(t1)
        w.writerow([f'stream_k:{cfg}', op_name, M, K, N, n_burst,
                    ts, te, f'{burst_ms:.6f}', f'{burst_tflops:.4f}'])
        seg_f.flush()

    seg_f.close()
    print('[eval] DONE')


if __name__ == '__main__':
    main()
