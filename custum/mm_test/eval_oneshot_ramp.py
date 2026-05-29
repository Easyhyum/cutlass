#!/usr/bin/env python3
"""
One-shot RAMP evaluation — v8 (A) linear-activity model.

For each combination of (start_pct, step_pct):
  activity[k] = min(100, start_pct + k * step_pct)
  At iter k where activity < 100, the kernel sleeps:
     sleep_ns = iter_time_ns * (100 - activity[k]) / activity[k]
  Once activity hits 100, no more sleep for the rest of THAT kernel.
  Ramp is ONE-SHOT — applied only to the next kernel after prime_ramp().
  Subsequent sustained kernels run baseline (no sleep).

Sweeps:
  start_pct  ∈ MM_RAMP_START_PCTS (default: 70,75,80,85,90,95,100)
  step_pct   ∈ MM_RAMP_STEP_PCTS  (default: 1,2,5,10)

For each (start, step):
  - prime_ramp(start, step, iter_time_ns)
  - First kernel: timed individually (ramp applies here)
  - Then 200 sustained kernels: timed (ramp NOT applied)
  - Compare sustained TFLOPS against pristine baseline.

If start_pct=100, ramp is no-op → first kernel == sustained == baseline.
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

    n_sustained = int(os.environ.get('MM_SUSTAINED_KERNELS', '200'))
    warmup_ms   = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
    seg_path    = os.environ['MM_SEGMENTS']

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
    print(f'[eval] {op_name}  M={M} K={K} N={N}  sustained_kernels={n_sustained}')
    print(f'[eval] start_pcts={starts}  step_pcts={steps}')

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)

    # Global warmup
    if warmup_ms > 0:
        print(f'[eval] global warmup {warmup_ms:.0f}ms...')
        t0 = time.time()
        while (time.time() - t0)*1000.0 < warmup_ms:
            ext.gemm_streamk(A, B)
        torch.cuda.synchronize()

    # Estimate per-outer-iter wall time
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
    print(f'[eval] baseline sustained: ms_avg={baseline_ms:.4f}  TFLOPS={baseline_tflops:.1f}')
    print(f'[eval] outer_iters/kernel={outer_iters_per_kernel}  iter_time≈{iter_time_ns} ns')

    # CSV
    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    seg_f = open(seg_path, 'w', newline='')
    w = csv.writer(seg_f)
    w.writerow(['backend','operator','M','K','N','iters',
                't_start','t_end','ms_avg','tflops'])
    seg_f.flush()

    print()
    hdr = (f'{"cfg":>14s}  {"start%":>6s} {"step%":>5s} {"ramp_iters":>10s}  '
           f'{"first_ms":>9s} {"first_TF":>8s}  '
           f'{"sust_ms":>9s} {"sust_TF":>8s}  {"sust_%base":>10s}')
    print(hdr); print('-' * len(hdr))

    rows_to_test = []
    for s in starts:
        for st in steps:
            ramp_iters = 0 if s >= 100 else (100 - s + st - 1) // st  # ceil
            cfg = f's{s}_p{st}'
            rows_to_test.append((cfg, s, st, ramp_iters))

    for cfg, s, st, r_iters in rows_to_test:
        # prime one-shot ramp
        if s < 100:
            ext.prime_ramp(s, st, iter_time_ns)

        # First kernel: ramp applies
        torch.cuda.synchronize()
        t_first_start = time.time()
        ev0.record()
        ext.gemm_streamk(A, B)
        ev1.record(); torch.cuda.synchronize()
        first_ms = ev0.elapsed_time(ev1)
        first_tflops = 2.0*M*K*N/(first_ms*1e-3)/1e12

        # Sustained: no priming, no ramp
        ev0.record()
        for _ in range(n_sustained):
            ext.gemm_streamk(A, B)
        ev1.record(); torch.cuda.synchronize()
        t_sust_end = time.time()
        sust_ms = ev0.elapsed_time(ev1)/n_sustained
        sust_tflops = 2.0*M*K*N/(sust_ms*1e-3)/1e12
        sust_pct = sust_tflops / baseline_tflops * 100.0

        print(f'{cfg:>14s}  {s:>6d} {st:>5d} {r_iters:>10d}  '
              f'{first_ms:>9.3f} {first_tflops:>8.1f}  '
              f'{sust_ms:>9.4f} {sust_tflops:>8.1f}  {sust_pct:>9.2f}%')
        sys.stdout.flush()

        ts = fmt_ts(t_first_start); te = fmt_ts(t_sust_end)
        w.writerow([f'stream_k:{cfg}', op_name, M, K, N, n_sustained+1,
                    ts, te, f'{sust_ms:.6f}', f'{sust_tflops:.4f}'])
        seg_f.flush()

    # final baseline reference
    ev0.record()
    for _ in range(n_sustained): ext.gemm_streamk(A, B)
    ev1.record(); torch.cuda.synchronize()
    bms = ev0.elapsed_time(ev1)/n_sustained
    btf = 2.0*M*K*N/(bms*1e-3)/1e12
    print(f'{"BASELINE_END":>14s}  {"-":>6s} {"-":>5s} {"-":>10s}  '
          f'{"-":>9s} {"-":>8s}  {bms:>9.4f} {btf:>8.1f}  '
          f'{btf/baseline_tflops*100:>9.2f}%')

    seg_f.close()
    print('[eval] DONE')


if __name__ == '__main__':
    main()
