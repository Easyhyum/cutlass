#!/usr/bin/env python3
"""Probe NVML power sampling rate while running an inline GEMM workload.
Single-threaded: each poll interval iterates and runs a GEMM in between."""
import argparse
import os
import sys
import time

import pynvml
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=3,
                    help='NVML physical GPU index')
    ap.add_argument('--seconds', type=float, default=2.0)
    args = ap.parse_args()

    # Map to torch via CUDA_VISIBLE_DEVICES already set externally
    torch.cuda.set_device(0)
    dev = torch.device('cuda:0')

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes): name = name.decode()
    print(f'GPU{args.gpu}  {name}', flush=True)

    # Setup workload (down_proj M=8192)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import bf16_gemm_sm80_streamk as ext
    M, K, N = 8192, 12288, 4096
    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)

    # Warmup
    print('warmup 2s...', flush=True)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        ext.gemm_streamk(A, B)
    torch.cuda.synchronize()

    print(f'\nProbing each interval for {args.seconds}s '
          f'(workload runs continuously between polls):', flush=True)
    print(f'{"interval(ms)":>14s} {"n":>5s} {"n_uniq":>7s} {"uniq%":>6s} '
          f'{"min_dt(ms)":>11s} {"avg_dt(ms)":>11s}  {"range(W)":>14s}',
          flush=True)

    for interval_ms in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 50]:
        samples = []; ts = []
        t_end = time.time() + args.seconds
        # Run workload between polls (one kernel per poll cycle)
        while time.time() < t_end:
            ext.gemm_streamk(A, B)  # one kernel, no sync (queued)
            now = time.time()
            samples.append(pynvml.nvmlDeviceGetPowerUsage(h))
            ts.append(now)
            time.sleep(interval_ms / 1000.0)
        torch.cuda.synchronize()

        n = len(samples)
        dts = []
        last_t = ts[0]; last_v = samples[0]
        for t_, v in zip(ts[1:], samples[1:]):
            if v != last_v:
                dts.append((t_ - last_t)*1000.0)
                last_t = t_; last_v = v
        n_uniq = len(dts) + 1
        min_dt = min(dts) if dts else float('nan')
        avg_dt = sum(dts)/len(dts) if dts else float('nan')
        w_lo = min(samples)/1000.0; w_hi = max(samples)/1000.0
        print(f'{interval_ms:>14d} {n:>5d} {n_uniq:>7d} '
              f'{n_uniq/n*100:>5.1f}% {min_dt:>9.2f}  {avg_dt:>9.2f}  '
              f'{w_lo:>6.1f}-{w_hi:>6.1f}', flush=True)

    pynvml.nvmlShutdown()


if __name__ == '__main__':
    main()
