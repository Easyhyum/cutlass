#!/usr/bin/env python3
"""
Multi-stream + M-sweep eval — compare concurrent streams across M values.

Each cfg specifies (M, n_concurrent). For each M, M_KERNELS is chosen so
that one burst takes ≥ 500ms regardless of M.

Uses the BASELINE binary (no wave-sleep code) — pure host-side scheduling.
"""
import csv, os, sys, time
from datetime import datetime
import torch

sys.path.insert(0, '/workspace/custum')
import bf16_gemm_sm80_streamk_baseline as ext

K, N_DIM = 12288, 4096

n_bursts   = int(os.environ.get('MM_N_BURSTS', '80'))
burst_gap  = int(os.environ.get('MM_BURST_GAP_MS', '500'))
cfg_gap    = int(os.environ.get('MM_CFG_GAP_MS', '1500'))
warmup_ms  = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
seg_path   = os.environ['MM_SEGMENTS']

torch.cuda.set_device(0)
dev = torch.device('cuda:0')

# (M : (M_KERNELS, [N_streams ...]))  — each M's M_KERNELS chosen so
# burst_time ≈ M_KERNELS × ms_per_kernel ≈ 600 ms.
M_CFGS = {
    8192:   (300, [1, 2, 3, 4, 6, 8]),
    16384:  (150, [1, 2, 4, 8]),
    32768:  (75,  [1, 2, 4]),
    65536:  (35,  [1, 2]),
    131072: (17,  [1]),
    262144: (8,   [1]),
}

# Pre-allocate tensor pools — one A_pool[M] holds up to max(N) tensors of (M,K).
print('[ms-eval] allocating tensors...')
A_pool = {}
B_pool = {}
for m_val, (mk, n_list) in M_CFGS.items():
    n_max = max(n_list)
    A_pool[m_val] = [torch.empty(m_val, K, device=dev, dtype=torch.bfloat16)
                     for _ in range(n_max)]
    B_pool[m_val] = [torch.empty(K, N_DIM, device=dev, dtype=torch.bfloat16)
                     for _ in range(n_max)]
    for a in A_pool[m_val]: a.normal_(0, 0.02)
    for b in B_pool[m_val]: b.normal_(0, 0.02)
    print(f'  M={m_val}: {n_max} tensor pairs allocated (M_KERNELS={mk})')

streams = [torch.cuda.Stream() for _ in range(8)]

# warmup on M=8192 default
warm_A = A_pool[8192][0]; warm_B = B_pool[8192][0]
if warmup_ms > 0:
    t0 = time.time()
    while (time.time() - t0)*1000.0 < warmup_ms:
        ext.gemm_streamk(warm_A, warm_B)
    torch.cuda.synchronize()

ev0 = torch.cuda.Event(enable_timing=True)
ev1 = torch.cuda.Event(enable_timing=True)


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


def burst_n(m_val, n_concurrent, m_kernels):
    A_ = A_pool[m_val]; B_ = B_pool[m_val]
    if n_concurrent == 1:
        for _ in range(m_kernels):
            ext.gemm_streamk(A_[0], B_[0])
    else:
        for _ in range(m_kernels):
            for s_idx in range(n_concurrent):
                with torch.cuda.stream(streams[s_idx]):
                    ext.gemm_streamk(A_[s_idx], B_[s_idx])


cfgs = []
# Insert s100_step0 as M=8192 N=1 baseline
for m_val, (mk, n_list) in M_CFGS.items():
    for n_c in n_list:
        if m_val == 8192 and n_c == 1:
            cfgs.append(('s100_step0', m_val, n_c, mk))
        else:
            cfgs.append((f'ms_M{m_val}_N{n_c}', m_val, n_c, mk))

print(f'[ms-eval] {len(cfgs)} configs, N_BURSTS={n_bursts}, burst_gap={burst_gap}ms')
print()

for cfg_name, m_val, n_conc, m_kernels in cfgs:
    time.sleep(cfg_gap / 1000.0)
    for burst in range(n_bursts):
        torch.cuda.synchronize()
        t0_wall = time.time()
        ev0.record()
        burst_n(m_val, n_conc, m_kernels)
        ev1.record()
        torch.cuda.synchronize()
        t1_wall = time.time()
        ms_avg_kernel = ev0.elapsed_time(ev1) / m_kernels
        tflops_sum = 2.0 * m_val * K * N_DIM * n_conc / (ms_avg_kernel * 1e-3) / 1e12
        w.writerow([f'stream_k:{cfg_name}#{burst}',
                    'down_proj', m_val, K, N_DIM, m_kernels,
                    fmt_ts(t0_wall), fmt_ts(t1_wall),
                    f'{ms_avg_kernel:.6f}', f'{tflops_sum:.4f}'])
        seg_f.flush()
        time.sleep(burst_gap / 1000.0)
    print(f'{cfg_name:>20s}  M={m_val:>6d}  N={n_conc}  M_K={m_kernels:>4d}  '
          f'{n_bursts} done', flush=True)

seg_f.close()
print('[ms-eval] DONE')
