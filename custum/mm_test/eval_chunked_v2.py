#!/usr/bin/env python3
"""
Chunked GEMM eval v2 — focus on M=8192:
  single
  tchunk         — torch loop, single stream
  kchunk_idle{X} — kernel-chunked with X us inter-chunk idle (X ∈ 0,5,10,30,100,300)
  ochunk         — torch 2 streams alternating (wall-time measured)

All use baseline binary (no wave-sleep code).
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

streams = [torch.cuda.Stream() for _ in range(2)]

# Per M, M_KERNELS so 1 burst ≥ 500 ms.
M_CFGS = [
    (8192,    300),
    (16384,   150),
    (32768,    75),
    (65536,    35),
    (131072,   17),
]

IDLE_VARIANTS = [0, 5, 10, 30, 100, 300]

print(f'[v2] CHUNK_M={CHUNK_M}  N_BURSTS={n_bursts}  burst_gap={burst_gap}ms')


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


def burst_single(A, B, m_kernels):
    for _ in range(m_kernels):
        ext.gemm_streamk(A, B)


def burst_torch_chunked(A, B, m_kernels, chunk_m):
    M = A.size(0)
    n_chunks = (M + chunk_m - 1) // chunk_m
    for _ in range(m_kernels):
        for c in range(n_chunks):
            start = c * chunk_m
            end = min(start + chunk_m, M)
            ext.gemm_streamk(A[start:end, :], B)


def burst_kernel_chunked(A, B, m_kernels, chunk_m, idle_us):
    for _ in range(m_kernels):
        ext.gemm_streamk_chunked(A, B, chunk_m, 1, -1, idle_us)


def burst_overlap_chunked(A, B, m_kernels, chunk_m, n_streams=2):
    M = A.size(0)
    n_chunks = (M + chunk_m - 1) // chunk_m
    for _ in range(m_kernels):
        for c in range(n_chunks):
            start = c * chunk_m
            end = min(start + chunk_m, M)
            with torch.cuda.stream(streams[c % n_streams]):
                ext.gemm_streamk(A[start:end, :], B)
    # ensure all streams finished before next burst boundary
    for s in streams[:n_streams]:
        s.synchronize()


# warmup
warm_A = torch.empty(CHUNK_M, K, device=dev, dtype=torch.bfloat16); warm_A.normal_(0,0.02)
warm_B = torch.empty(K, N_DIM, device=dev, dtype=torch.bfloat16); warm_B.normal_(0,0.02)
if warmup_ms > 0:
    t0 = time.time()
    while (time.time() - t0)*1000.0 < warmup_ms:
        ext.gemm_streamk(warm_A, warm_B)
    torch.cuda.synchronize()
del warm_A, warm_B
torch.cuda.empty_cache()


def time_burst(fn):
    """Wall-time measurement — works for multi-stream too because we sync
    inside the burst function (overlap variant) or because everything runs
    on default stream (other variants)."""
    torch.cuda.synchronize()
    t0_wall = time.time()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    t1_wall = time.time()
    return (t1 - t0) * 1000.0, t0_wall, t1_wall


print()
print(f'{"cfg":>26s} {"M":>7s} {"type":>14s}  {"M_K":>4s}  progress')
print('-' * 70)

for m_val, m_kernels in M_CFGS:
    A = torch.empty(m_val, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N_DIM, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)

    cfgs = []
    if m_val == 8192:
        cfgs.append(('s100_step0', 'single', lambda: burst_single(A, B, m_kernels)))
    else:
        cfgs.append((f'single_M{m_val}', 'single',
                     lambda: burst_single(A, B, m_kernels)))
    cfgs.append((f'tchunk_M{m_val}', 'torch',
                 lambda: burst_torch_chunked(A, B, m_kernels, CHUNK_M)))
    for idle_us in IDLE_VARIANTS:
        cfgs.append((f'kchunk_M{m_val}_id{idle_us}', f'kernel_id{idle_us}',
                     (lambda i=idle_us: burst_kernel_chunked(A, B, m_kernels, CHUNK_M, i))))
    cfgs.append((f'ochunk_M{m_val}', 'overlap_wt',
                 lambda: burst_overlap_chunked(A, B, m_kernels, CHUNK_M, 2)))

    for cfg_name, kind, burst_fn in cfgs:
        time.sleep(cfg_gap / 1000.0)
        for burst in range(n_bursts):
            ms_total, t0_wall, t1_wall = time_burst(burst_fn)
            ms_avg = ms_total / m_kernels
            tflops = 2.0 * m_val * K * N_DIM / (ms_avg * 1e-3) / 1e12
            w.writerow([f'stream_k:{cfg_name}#{burst}',
                        'down_proj', m_val, K, N_DIM, m_kernels,
                        fmt_ts(t0_wall), fmt_ts(t1_wall),
                        f'{ms_avg:.6f}', f'{tflops:.4f}'])
            seg_f.flush()
            time.sleep(burst_gap / 1000.0)
        print(f'{cfg_name:>26s} {m_val:>7d} {kind:>14s}  {m_kernels:>4d}  {n_bursts} done',
              flush=True)

    del A, B
    torch.cuda.empty_cache()

seg_f.close()
print('[v2] DONE')
