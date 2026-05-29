#!/usr/bin/env python3
"""
M-size sweep across multiple GEMM kernels (cublas, streamk, ...).

For each (kernel, M) pair:
  - Allocate (M × K) A and (K × N) B in bfloat16.
  - Run M_KERNELS calls per burst so each burst ≥ 500 ms.
  - Repeat N_BURSTS bursts (sample size).
  - Record one row per burst in segments CSV.

cfg name encodes (kernel, M, wave_count) so the plot script can render
one row per kernel and label wave count.
   backend tag = '<kernel>:M<M>_w<W>#<burst_idx>'
   wave count is set per kernel — NaN (literally 'NaN') for kernels we
   can't introspect (e.g. cublas).

Output:
  $MM_SEGMENTS  — segments CSV in the standard analyze_power-compatible
                  format.

Env knobs:
  MM_N_BURSTS         (50)
  MM_BURST_GAP_MS     (500)
  MM_CFG_GAP_MS       (1500)
  MM_GLOBAL_WARMUP_MS (3000)
"""
import csv
import os
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, '/workspace/custum')
import bf16_gemm_sm80_streamk_baseline as ext
try:
    import cutlass_sm80_v3 as ext_v3   # pure CUTLASS sm80 128x128k64, ALL sleep off
except ImportError:
    ext_v3 = None

TB_M, TB_N = 128, 128

# Honor MM_GPU (CUDA device index within whatever CUDA_VISIBLE_DEVICES exposes).
_cuda_idx = int(os.environ.get('MM_GPU', '0'))
torch.cuda.set_device(_cuda_idx)
dev = torch.device(f'cuda:{_cuda_idx}')
N_SMS = torch.cuda.get_device_properties(_cuda_idx).multi_processor_count   # 188

n_bursts   = int(os.environ.get('MM_N_BURSTS', '50'))
burst_gap  = int(os.environ.get('MM_BURST_GAP_MS', '500'))
cfg_gap    = int(os.environ.get('MM_CFG_GAP_MS', '1500'))
warmup_ms  = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
seg_path   = os.environ['MM_SEGMENTS']

# ── Operations from Qwen3-8B / 32B ──────────────────────────────────────────
# Comment out any (model, op) line to skip; uncomment to include.
# Tuple format: (model_tag, op_name, K, N)
#
# qwen3-8b   constants: H=4096,  INTER=12288, Q_DIM=4096, KV_DIM=1024,  VOCAB=151936
#   qkv_proj      :  H → Q + 2·KV   = 4096 →  4096+2*1024 = 6144
#   o_proj        :  Q → H          = 4096 → 4096
#   gate_up_proj  :  H → 2·INTER    = 4096 → 24576
#   down_proj     :  INTER → H      = 12288 → 4096
#   lm_head       :  H → VOCAB      = 4096 → 151936
#
# qwen3-32b  constants: H=5120,  INTER=25600, Q_DIM=8192, KV_DIM=1024,  VOCAB=151936
#   qkv_proj      :  H → Q + 2·KV   = 5120 → 8192+2*1024 = 10240
#   o_proj        :  Q → H          = 8192 → 5120
#   gate_up_proj  :  H → 2·INTER    = 5120 → 51200
#   down_proj     :  INTER → H      = 25600 → 5120
#   lm_head       :  H → VOCAB      = 5120 → 151936
OPS = [
    # ---- qwen3-8b ----
    ('qwen3-8b',  'qkv_proj',      4096,   6144),
    ('qwen3-8b',  'o_proj',        4096,   4096),
    ('qwen3-8b',  'up_proj',       4096,  12288),
    # ('qwen3-8b',  'gate_up_proj',  4096,  24576),
    ('qwen3-8b',  'down_proj',    12288,   4096),    # original default
    ('qwen3-8b',  'lm_head',       4096, 151936),
    # ---- qwen3-32b ----
    ('qwen3-32b', 'qkv_proj',      5120,  10240),
    ('qwen3-32b', 'o_proj',        8192,   5120),
    ('qwen3-32b', 'up_proj',       5120,  25600),
    # ('qwen3-32b', 'gate_up_proj',  5120,  51200),
    ('qwen3-32b', 'down_proj',    25600,   5120),
    ('qwen3-32b', 'lm_head',       5120, 151936),
]

# Optional env-var filter: e.g. MM_OPS=down_proj,o_proj,qkv_proj
_op_filter = os.environ.get('MM_OPS', '').strip()
if _op_filter:
    keep_ops = [s.strip() for s in _op_filter.split(',')]
    OPS = [op for op in OPS if op[1] in keep_ops]

# Optional env-var filter for model: e.g. MM_MODEL=qwen3-8b
_mo_filter = os.environ.get('MM_MODEL', '').strip()
if _mo_filter:
    OPS = [op for op in OPS if op[0] == _mo_filter]

# Short tags for the cfg name (kept short so plot labels fit).
MODEL_SHORT = {'qwen3-8b': '8b', 'qwen3-32b': '32b'}
OP_SHORT    = {
    'qkv_proj':     'qkv',
    'o_proj':       'o',
    'gate_up_proj': 'gu',
    'down_proj':    'down',
    'lm_head':      'lm',
}


def _tiles(M, N):
    return ((M + TB_M - 1) // TB_M) * ((N + TB_N - 1) // TB_N)


def waves_streamk(M, K, N):
    """Stream-K launches n_sm * floor(tiles/n_sm) CTAs when tiles ≥ n_sm.
    Number of waves = floor(tiles/n_sm) (integer)."""
    t = _tiles(M, N)
    if t < N_SMS: return 1
    return t // N_SMS


def waves_basicdp(M, K, N):
    """Basic DP wave count = tiles / n_sm (fractional, occupancy=1 assumed)."""
    return _tiles(M, N) / N_SMS


def waves_sm80_v3(M, K, N):
    """sm80_v3 wave count (occupancy=1 assumed — real occupancy may be 2
    based on 96 KB SMEM/CTA, see README)."""
    return _tiles(M, N) / N_SMS


def waves_cublas(M, K, N):
    """cuBLAS picks its own tile shape & grid — we don't introspect it."""
    return None   # → 'NaN' in cfg name


# M values — comment lines out to skip. M_KERNELS is auto-computed per
# (M, K, N) so 1 burst ≥ MM_TARGET_BURST_MS regardless of op shape.
M_LIST = [
    1024,
    1536,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
]
_m_env = os.environ.get('MM_M_LIST', '').strip()
if _m_env:
    M_LIST = [int(x) for x in _m_env.split(',')]

# Auto M_KERNELS targets:
TARGET_BURST_MS = float(os.environ.get('MM_TARGET_BURST_MS', '500'))
PEAK_TFLOPS     = float(os.environ.get('MM_PEAK_TFLOPS', '400'))
M_KERNELS_MIN   = int(os.environ.get('MM_M_KERNELS_MIN', '5'))
M_KERNELS_MAX   = int(os.environ.get('MM_M_KERNELS_MAX', '5000'))
# Memory budget for the (A, B, C) tensors of a single GEMM.
MEM_BUDGET_GB   = float(os.environ.get('MM_MEM_BUDGET_GB', '40.0'))


def m_kernels_for(M, K, N):
    """Compute M_KERNELS so M_KERNELS × ms/kernel ≈ TARGET_BURST_MS."""
    ms_per = 2.0 * M * K * N / (PEAK_TFLOPS * 1e12) * 1000.0
    n = int(round(TARGET_BURST_MS / max(ms_per, 1e-3)))
    return max(M_KERNELS_MIN, min(M_KERNELS_MAX, n))


def estimate_gb(M, K, N):
    """Estimate (A, B, C) BF16 tensor footprint in GB (A+B+C all BF16)."""
    return (M*K + K*N + M*N) * 2 / (1024**3)


def fits_memory(M, K, N):
    """Pre-check: A+B+C BF16 must fit within MEM_BUDGET_GB."""
    return estimate_gb(M, K, N) < MEM_BUDGET_GB


def is_oom(exc):
    """True if exception is a CUDA out-of-memory error (PyTorch + CUDA)."""
    s = str(exc).lower()
    return ('out of memory' in s) or ('cuda_error_out_of_memory' in s) or \
           isinstance(exc, torch.cuda.OutOfMemoryError)

# (kernel_name_used_in_tag, callable, waves-fn)
# Plot row order: cublas → basicdp → stream_k (set in plot_kernel_timeline.py).
KERNELS_ALL = [
    ('cublas',   lambda A, B: torch.matmul(A, B),       waves_cublas),
    # ('basicdp',  lambda A, B: ext.gemm_basicdp(A, B),   waves_basicdp),
    ('stream_k', lambda A, B: ext.gemm_streamk(A, B),   waves_streamk),
]
if ext_v3 is not None:
    KERNELS_ALL.append(
        ('sm80_v3', lambda A, B: ext_v3.gemm_sm80_v3(A, B), waves_sm80_v3))

# Allow env-var to pick subset (CSV of kernel names). Default = all.
_sel = os.environ.get('MM_KERNELS', '').strip()
if _sel:
    keep = [k.strip() for k in _sel.split(',')]
    KERNELS = [k for k in KERNELS_ALL if k[0] in keep]
else:
    KERNELS = KERNELS_ALL


def fmt_ts(t):
    dt = datetime.fromtimestamp(t)
    return dt.strftime('%Y/%m/%d %H:%M:%S.') + f'{int(dt.microsecond/1000):03d}'


def fmt_waves(w):
    if w is None:                       return 'NaN'
    if isinstance(w, float):            return f'{w:.2f}'
    return f'{w}'


def main():
    print(f'[mksweep] N_SMS={N_SMS}  N_BURSTS={n_bursts}  burst_gap={burst_gap}ms  '
          f'target_burst={TARGET_BURST_MS:.0f}ms  mem_budget={MEM_BUDGET_GB:.0f}GB')
    print(f'[mksweep] OPS: {len(OPS)} ops, {[(o[0],o[1]) for o in OPS]}')
    print(f'[mksweep] M_LIST: {M_LIST}')
    print(f'[mksweep] kernels: {[k for k, _, _ in KERNELS]}')

    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    csv_exists = os.path.exists(seg_path)
    seg_f = open(seg_path, 'a' if csv_exists else 'w', newline='')
    w = csv.writer(seg_f)
    if not csv_exists:
        w.writerow(['backend','operator','M','K','N','iters',
                    't_start','t_end','ms_avg','tflops'])
        seg_f.flush()

    # Global warmup on a moderate (M,K,N)
    _, warm_fn, _ = KERNELS[0]
    wA = torch.empty(1024, 4096, device=dev, dtype=torch.bfloat16); wA.normal_(0,0.02)
    wB = torch.empty(4096, 4096, device=dev, dtype=torch.bfloat16); wB.normal_(0,0.02)
    if warmup_ms > 0:
        print(f'[mksweep] global warmup {warmup_ms:.0f}ms...')
        t0 = time.time()
        while (time.time() - t0)*1000.0 < warmup_ms:
            warm_fn(wA, wB)
        torch.cuda.synchronize()
    del wA, wB
    torch.cuda.empty_cache()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)

    print()
    print(f'{"kernel":>10s}  {"cfg":>32s}  {"M":>7s}  {"K":>5s}  {"N":>6s}  '
          f'{"waves":>7s}  {"M_K":>5s}  progress')
    print('-' * 100)

    for model, op_name, K, N in OPS:
        op_short = OP_SHORT.get(op_name, op_name)
        mdl_short = MODEL_SHORT.get(model, model)
        for m_val in M_LIST:
            est = estimate_gb(m_val, K, N)
            # Pre-check 1: static estimate vs budget
            if not fits_memory(m_val, K, N):
                print(f'  SKIP (pre-budget)  M={m_val:>7d}  op={op_name:>14s}  '
                      f'model={model}  est={est:.1f}GB > {MEM_BUDGET_GB:.0f}GB',
                      flush=True)
                continue
            # Pre-check 2: current GPU free-mem (handles fragmentation/other procs)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024**3)
            except Exception:
                free_gb = float('inf')
            # We need A+B+C plus ~3x slack for workspace/temp + caching allocator.
            need_gb = est * 3.0 + 0.5
            if need_gb > free_gb:
                print(f'  SKIP (live-free)  M={m_val:>7d}  op={op_name:>14s}  '
                      f'model={model}  need~{need_gb:.1f}GB free={free_gb:.1f}GB',
                      flush=True)
                continue

            m_kernels = m_kernels_for(m_val, K, N)

            # ── Allocate A,B once per (M,K,N) — try/except OOM ────────────
            try:
                A = torch.empty(m_val, K, device=dev, dtype=torch.bfloat16)
                B = torch.empty(K, N, device=dev, dtype=torch.bfloat16)
                A.normal_(0, 0.02); B.normal_(0, 0.02)
            except Exception as e:
                if is_oom(e):
                    print(f'  SKIP (alloc OOM)  M={m_val:>7d}  op={op_name:>14s}  '
                          f'model={model}  est={est:.1f}GB  err: {type(e).__name__}',
                          flush=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            for kernel_name, fn, waves_fn in KERNELS:
                w_str = fmt_waves(waves_fn(m_val, K, N))
                cfg_name = f'{op_short}{mdl_short}_M{m_val}_w{w_str}'

                time.sleep(cfg_gap / 1000.0)
                oom_in_burst = False
                for burst in range(n_bursts):
                    try:
                        torch.cuda.synchronize()
                        t0_wall = time.time()
                        ev0.record()
                        for _ in range(m_kernels):
                            fn(A, B)
                        ev1.record()
                        torch.cuda.synchronize()
                        t1_wall = time.time()
                    except Exception as e:
                        if is_oom(e):
                            print(f'  SKIP (runtime OOM)  cfg={cfg_name}  '
                                  f'kernel={kernel_name}  burst={burst}  '
                                  f'err: {type(e).__name__}', flush=True)
                            torch.cuda.empty_cache()
                            oom_in_burst = True
                            break
                        raise
                    ms_avg = ev0.elapsed_time(ev1) / m_kernels
                    tflops = 2.0 * m_val * K * N / (ms_avg * 1e-3) / 1e12
                    w.writerow([f'{kernel_name}:{cfg_name}#{burst}',
                                op_name, m_val, K, N, m_kernels,
                                fmt_ts(t0_wall), fmt_ts(t1_wall),
                                f'{ms_avg:.6f}', f'{tflops:.4f}'])
                    seg_f.flush()
                    time.sleep(burst_gap / 1000.0)

                if oom_in_burst:
                    # don't try further kernels with this (M,K,N) — they would
                    # only need more temporaries; bail out of the kernel loop
                    print(f'  bail kernel loop for cfg={cfg_name}', flush=True)
                    break

                print(f'{kernel_name:>10s}  {cfg_name:>32s}  {m_val:>7d}  '
                      f'{K:>5d}  {N:>6d}  {w_str:>7s}  {m_kernels:>5d}  '
                      f'{n_bursts} done', flush=True)

            del A, B
            torch.cuda.empty_cache()

    seg_f.close()
    print('[mksweep] DONE')


if __name__ == '__main__':
    main()
