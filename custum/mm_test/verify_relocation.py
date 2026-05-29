#!/usr/bin/env python3
"""
Quick verification that the relocated wave-sleep blocks each have the
expected effect on ms_avg. Use process-isolation when comparing the two
binaries.

mode=baseline  : bf16_gemm_sm80_streamk_baseline (no code)
mode=ws_*      : bf16_gemm_sm80_streamk  with the indicated prime params

Run via run_verify_relocation.sh.
"""
import csv, os, sys, time
import numpy as np
import torch

mode = os.environ.get('VER_MODE')
N    = int(os.environ.get('VER_N', '40'))
K    = int(os.environ.get('VER_K', '150'))
GAP  = int(os.environ.get('VER_GAP_MS', '500'))
WARM = int(os.environ.get('VER_WARMUP_MS', '2500'))
M    = 8192
K_DIM, N_DIM = 12288, 4096
N_SMS_FIXED = 188

dev = torch.device('cuda:0'); torch.cuda.set_device(0)
sys.path.insert(0, '/workspace/custum')

if mode == 'baseline':
    import bf16_gemm_sm80_streamk_baseline as ext
    label = 'baseline binary (no code)'
    def prime(): pass
else:
    import bf16_gemm_sm80_streamk as ext
    if mode == 'noprime':
        label = 'ws bin, NO prime (gate=false)'
        def prime(): pass
    elif mode == 'stair_only':
        label = 'staircase only (post-prologue): S=80, P=2000, mid=off'
        def prime():
            ext.prime_wave_sleep(num_waves=10, n_sm=188,
                                 first_smid_thr=(188*80)//100,
                                 first_step_ns=2000,
                                 mid_pct=0, mid_ns=0)
    elif mode == 'bubble_only':
        label = 'bubble only (mac_loop_iter start): mid=30%, 5us, no staircase'
        def prime():
            ext.prime_wave_sleep(num_waves=10, n_sm=188,
                                 first_smid_thr=188,         # thr == n_sm → no SM gets staircase
                                 first_step_ns=0,
                                 mid_pct=30, mid_ns=5000)
    elif mode == 'both':
        label = 'staircase + bubble combined'
        def prime():
            ext.prime_wave_sleep(num_waves=10, n_sm=188,
                                 first_smid_thr=(188*80)//100,
                                 first_step_ns=2000,
                                 mid_pct=30, mid_ns=5000)
    else:
        raise SystemExit(f'unknown VER_MODE={mode}')

print(f'[ver] mode={mode}  ({label})  N={N}  K={K}  gap={GAP}ms')

A = torch.empty(M, K_DIM, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
B = torch.empty(K_DIM, N_DIM, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)

# warmup with same prime behavior (so kernel cache + steady state matches)
t0 = time.time()
while (time.time() - t0)*1000.0 < WARM:
    prime(); ext.gemm_streamk(A, B)
torch.cuda.synchronize()

ev0 = torch.cuda.Event(enable_timing=True)
ev1 = torch.cuda.Event(enable_timing=True)
ms_list = []
for i in range(N):
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(K):
        prime(); ext.gemm_streamk(A, B)
    ev1.record(); torch.cuda.synchronize()
    ms_list.append(ev0.elapsed_time(ev1)/K)
    time.sleep(GAP/1000.0)

a = np.array(ms_list)
tf = 2.0*M*K_DIM*N_DIM/(a*1e-3)/1e12
print(f'  ms mean={a.mean():.4f}  med={np.median(a):.4f}  std={a.std():.4f}  '
      f'TFLOPS_mean={tf.mean():.1f}')

# append to /tmp result file
out = '/tmp/verify_relocation.tsv'
new = not os.path.exists(out)
with open(out, 'a') as f:
    if new: f.write('mode\tlabel\tms_mean\tms_med\tms_std\ttflops_mean\n')
    f.write(f'{mode}\t{label}\t{a.mean():.6f}\t{np.median(a):.6f}\t{a.std():.6f}\t{tf.mean():.4f}\n')
print(f'  -> appended to {out}')
