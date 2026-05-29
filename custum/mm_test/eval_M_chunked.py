#!/usr/bin/env python3
"""
Torch-level chunked GEMM eval.

Compare:
  single_M{M}    — one streamk call with full A (M × K)
  chunk_M{M}     — split A into chunks of CHUNK_M rows, sequential streamk calls

Goal: verify that chunking M into CHUNK_M (default 2048) keeps the per-call
power around the M=2048 sweet-spot (~617 W observed) instead of spiking to
700 W+ for large M.

Uses the BASELINE binary (no wave-sleep code in SASS).
"""
import csv, os, sys, time
from datetime import datetime
import torch

sys.path.insert(0, '/workspace/custum')
import bf16_gemm_sm80_streamk_baseline as ext

K, N_DIM = 12288, 4096
CHUNK_M = int(os.environ.get('CHUNK_M', '2048'))

n_bursts   = int(os.environ.get('MM_N_BURSTS', '50'))
burst_gap  = int(os.environ.get('MM_BURST_GAP_MS', '500'))
cfg_gap    = int(os.environ.get('MM_CFG_GAP_MS', '1500'))
warmup_ms  = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
seg_path   = os.environ['MM_SEGMENTS']

torch.cuda.set_device(0)
dev = torch.device('cuda:0')

# (M, M_KERNELS) — M_KERNELS chosen so single-call burst time ≥ 500 ms.
M_CFGS = [
    (2048,    1000),   # baseline 600 W cfg
    (4096,     500),
    (8192,     300),
    (16384,    150),
    (32768,     75),
    (65536,     35),
    (131072,    17),
    (262144,     9),
]

print(f'[chunked] CHUNK_M={CHUNK_M}  N_BURSTS={n_bursts}  burst_gap={burst_gap}ms')

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

# global warmup
warm_A = torch.empty(CHUNK_M, K, device=dev, dtype=torch.bfloat16); warm_A.normal_(0,0.02)
warm_B = torch.empty(K, N_DIM, device=dev, dtype=torch.bfloat16); warm_B.normal_(0,0.02)
if warmup_ms > 0:
    t0 = time.time()
    while (time.time() - t0)*1000.0 < warmup_ms:
        ext.gemm_streamk(warm_A, warm_B)
    torch.cuda.synchronize()
del warm_A, warm_B
torch.cuda.empty_cache()


def burst_single(A, B, m_kernels):
    for _ in range(m_kernels):
        ext.gemm_streamk(A, B)


def burst_chunked(A, B, m_kernels, chunk_m):
    M = A.size(0)
    n_chunks = (M + chunk_m - 1) // chunk_m
    for _ in range(m_kernels):
        for c in range(n_chunks):
            start = c * chunk_m
            end = min(start + chunk_m, M)
            A_chunk = A[start:end, :]
            ext.gemm_streamk(A_chunk, B)


print()
print(f'{"cfg":>22s}  {"M":>7s}  {"chunks":>6s}  {"M_K":>5s}  progress')
print('-' * 65)

for m_val, m_kernels in M_CFGS:
    A = torch.empty(m_val, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N_DIM, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)
    # Each M gets two cfgs: single + chunked (when M > CHUNK_M)
    cfg_list = [(f'single_M{m_val}', 1)]
    if m_val > CHUNK_M:
        cfg_list.append((f'chunk_M{m_val}', (m_val + CHUNK_M - 1) // CHUNK_M))
    # M=8192 single doubles as the s100_step0 "baseline" for plot scripts
    if m_val == 8192:
        cfg_list = [('s100_step0', 1),
                    (f'chunk_M{m_val}', m_val // CHUNK_M)]

    for cfg_name, n_chunks in cfg_list:
        time.sleep(cfg_gap / 1000.0)
        for burst in range(n_bursts):
            torch.cuda.synchronize()
            t0_wall = time.time()
            ev0.record()
            if n_chunks == 1:
                burst_single(A, B, m_kernels)
            else:
                burst_chunked(A, B, m_kernels, CHUNK_M)
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
        print(f'{cfg_name:>22s}  {m_val:>7d}  {n_chunks:>6d}  {m_kernels:>5d}  {n_bursts} done',
              flush=True)

    del A, B
    torch.cuda.empty_cache()

seg_f.close()
print('[chunked] DONE')
