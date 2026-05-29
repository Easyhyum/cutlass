#!/usr/bin/env python3
"""
M-size sweep — baseline streamk GEMM at various M, single stream.
Goal: find the M values that give the lowest power spike.

Each cfg has its own M_KERNELS chosen so 1 burst ≥ 500 ms regardless of M.
Uses the BASELINE binary (no wave-sleep code in SASS).
"""
import csv, os, sys, time
from datetime import datetime
import torch

sys.path.insert(0, '/workspace/custum')
import bf16_gemm_sm80_streamk_baseline as ext

K, N_DIM = 12288, 4096

n_bursts   = int(os.environ.get('MM_N_BURSTS', '50'))
burst_gap  = int(os.environ.get('MM_BURST_GAP_MS', '500'))
cfg_gap    = int(os.environ.get('MM_CFG_GAP_MS', '1500'))
warmup_ms  = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
seg_path   = os.environ['MM_SEGMENTS']

torch.cuda.set_device(0)
dev = torch.device('cuda:0')

# (M, M_KERNELS)  — M_KERNELS is set so burst_time ≈ M_KERNELS × ms/kernel ≥ 500 ms.
# Approximate ms/kernel: 1024≈0.27, 2048≈0.55, 4096≈1.05, 8192≈2.09,
# 16384≈4.1, 32768≈8.0, 65536≈16, 131072≈32, 262144≈64, 524288≈130.
M_CFGS = [
    (1024,    2000),   # ~540 ms
    (2048,    1000),   # ~550 ms
    (4096,     500),   # ~525 ms
    (8192,     300),   # ~625 ms (matches earlier runs)
    (16384,    150),   # ~615 ms
    (32768,     75),   # ~600 ms
    (65536,     35),   # ~560 ms
    (131072,    17),   # ~544 ms
    (262144,     9),   # ~576 ms
    (524288,     5),   # ~650 ms
]

print('[M-sweep] allocating tensors per M ...')
print(f'  K={K}  N={N_DIM}  N_BURSTS={n_bursts}  burst_gap={burst_gap}ms')

def fmt_ts(t):
    dt = datetime.fromtimestamp(t)
    return dt.strftime('%Y/%m/%d %H:%M:%S.') + f'{int(dt.microsecond/1000):03d}'

os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
csv_exists = os.path.exists(seg_path)
seg_f = open(seg_path, 'a' if csv_exists else 'w', newline='')
w = csv.writer(seg_f)
if not csv_exists:
    w.writerow(['backend','operator','M','K','N','iters',
                't_start','t_end','ms_avg','tflops'])
    seg_f.flush()

ev0 = torch.cuda.Event(enable_timing=True)
ev1 = torch.cuda.Event(enable_timing=True)

# global warmup with smallest M
warm_A = torch.empty(1024, K, device=dev, dtype=torch.bfloat16); warm_A.normal_(0,0.02)
warm_B = torch.empty(K, N_DIM, device=dev, dtype=torch.bfloat16); warm_B.normal_(0,0.02)
if warmup_ms > 0:
    t0 = time.time()
    while (time.time() - t0)*1000.0 < warmup_ms:
        ext.gemm_streamk(warm_A, warm_B)
    torch.cuda.synchronize()
del warm_A, warm_B
torch.cuda.empty_cache()

print()
print(f'{"cfg":>14s} {"M":>7s} {"M_K":>5s} {"size_GB":>7s}  progress')
print('-' * 60)

for m_val, m_kernels in M_CFGS:
    # Allocate tensors for this M
    A = torch.empty(m_val, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N_DIM, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)
    size_gb = (m_val*K + K*N_DIM) * 2 / (1024**3)
    cfg_name = f'M{m_val}'
    # Treat M=8192 as the "baseline" group for plot scripts
    if m_val == 8192:
        cfg_name = 's100_step0'

    time.sleep(cfg_gap / 1000.0)
    for burst in range(n_bursts):
        torch.cuda.synchronize()
        t0_wall = time.time()
        ev0.record()
        for _ in range(m_kernels):
            ext.gemm_streamk(A, B)
        ev1.record()
        torch.cuda.synchronize()
        t1_wall = time.time()
        ms_avg = ev0.elapsed_time(ev1) / m_kernels
        tflops = 2.0 * m_val * K * N_DIM / (ms_avg * 1e-3) / 1e12
        w.writerow([f'stream_k:{cfg_name}#{burst}',
                    'down_proj', m_val, K, N_DIM, m_kernels,
                    fmt_ts(t0_wall), fmt_ts(t1_wall),
                    f'{ms_avg:.6f}', f'{tflops:.4f}'])
        seg_f.flush()
        time.sleep(burst_gap / 1000.0)
    del A, B
    torch.cuda.empty_cache()
    print(f'{cfg_name:>14s} {m_val:>7d} {m_kernels:>5d} {size_gb:>7.2f}  {n_bursts} done',
          flush=True)

seg_f.close()
print('[M-sweep] DONE')
