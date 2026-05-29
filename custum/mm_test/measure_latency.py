#!/usr/bin/env python3
"""
Measure pure ms_avg per burst for ONE binary.

Two modes (one process each, picked via MEAS_MODE env):
  baseline    → bf16_gemm_sm80_streamk_baseline (-DCUTLASS_WAVE_SLEEP_ENABLED off)
  wsdisabled  → bf16_gemm_sm80_streamk           (compiled-in but never primed)

If the wave-sleep binary is primed with num_waves<3 (or not primed at all),
the device-side gate `if (kWaveSleepNumWaves >= 3 && ...)` evaluates false
on every CTA — but the comparison itself is still executed, plus the host
side runs `cudaMemcpyToSymbol(kWaveSleepNumWaves, &zero, sizeof(int))` once
per gemm_streamk call (else-branch in production code). The latency delta
between the two modes is the cost of "wave-sleep code present but inactive".
"""
import csv
import os
import sys
import time

import numpy as np
import torch

MODE   = os.environ.get('MEAS_MODE')
N      = int(os.environ.get('MEAS_N', '50'))
K      = int(os.environ.get('MEAS_K', '300'))
GAP_MS = int(os.environ.get('MEAS_GAP_MS', '500'))
WARMUP_MS = int(os.environ.get('MEAS_WARMUP_MS', '3000'))
M      = int(os.environ.get('MM_M', '8192'))
K_DIM  = 12288
N_DIM  = 4096

if MODE not in ('baseline', 'wsdisabled'):
    raise SystemExit(f'usage: MEAS_MODE in {{baseline, wsdisabled}}  (got {MODE!r})')

dev = torch.device('cuda:0')
torch.cuda.set_device(0)

sys.path.insert(0, '/workspace/custum')
if MODE == 'baseline':
    import bf16_gemm_sm80_streamk_baseline as ext
    label = 'default (no wave-sleep code in SASS)'
else:
    import bf16_gemm_sm80_streamk as ext
    label = 's100_step0 / no-prime (wave-sleep code present, gate=false)'

A = torch.empty(M, K_DIM, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
B = torch.empty(K_DIM, N_DIM, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)

print(f'[lat] mode={MODE}  ({label})')
print(f'[lat] N_BURSTS={N}  K_per_burst={K}  M={M}  K={K_DIM}  N_DIM={N_DIM}  '
      f'gap_ms={GAP_MS}  warmup_ms={WARMUP_MS}')

# Global warmup on the same binary
if WARMUP_MS > 0:
    t0 = time.time()
    while (time.time() - t0)*1000.0 < WARMUP_MS:
        ext.gemm_streamk(A, B)
    torch.cuda.synchronize()

ev0 = torch.cuda.Event(enable_timing=True)
ev1 = torch.cuda.Event(enable_timing=True)
ms_list = []
for i in range(N):
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(K):
        ext.gemm_streamk(A, B)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / K
    ms_list.append(ms)
    if (i+1) % 10 == 0:
        print(f'  burst {i+1:>3d}/{N}  ms={ms:.6f}')
    time.sleep(GAP_MS / 1000.0)

out_file = os.environ.get('MEAS_OUT',
            os.path.join(os.path.dirname(__file__),
                         f'logs/latency_{MODE}.csv'))
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['burst_idx', 'ms_avg', 'tflops'])
    for i, ms in enumerate(ms_list):
        tf = 2.0 * M * K_DIM * N_DIM / (ms * 1e-3) / 1e12
        w.writerow([i, f'{ms:.6f}', f'{tf:.4f}'])

a = np.array(ms_list)
tf_arr = 2.0 * M * K_DIM * N_DIM / (a * 1e-3) / 1e12
print()
print(f'  mean   ms = {a.mean():.6f}   TFLOPS = {tf_arr.mean():.2f}')
print(f'  median ms = {np.median(a):.6f}')
print(f'  std    ms = {a.std():.6f}  ({a.std()/a.mean()*100:.2f}%)')
print(f'  min    ms = {a.min():.6f}')
print(f'  max    ms = {a.max():.6f}')
print(f'  p10/p50/p90 = {np.percentile(a, 10):.6f} / {np.percentile(a, 50):.6f} / '
      f'{np.percentile(a, 90):.6f}')
print(f'  -> {out_file}')
