#!/usr/bin/env python3
"""
Qwen3-8B GEMM power/clock sweep.

Walks every dense matmul in a Qwen3-8B transformer block (plus lm_head),
sweeping M = batch*seq_len from 1 up to 524288. Uses torch.matmul on
bf16 CUDA tensors which dispatches to cublasGemmEx (CUDA_R_16BF inputs,
CUBLAS_COMPUTE_32F accumulation) internally.

For each (operator, M):
  - allocates A(MxK), B(KxN), C(MxN) bf16
  - warmup, probe to estimate ms/iter, then auto-scales iters so the
    measured run is >= MM_MIN_MS milliseconds (default 2500 ms)
  - records wall-clock t_start/t_end in nvidia-smi-compatible format
  - writes a row to MM_SEGMENTS csv
  - inserts an idle gap (MM_GAP_MS, default 1500 ms) so the power/clock
    trace dips between segments

Env vars
  MM_GPU       device index (default 0; CUDA_VISIBLE_DEVICES is honored)
  MM_BACKEND   cublas | cutlass_sm80 | stream_k (default cublas)
                 cublas       — torch.matmul -> cublasGemmEx
                 cutlass_sm80 — bf16_gemm_sm80.gemm_sm80_v3
                                CUTLASS 2.x SM80 HMMA path, 128x128x64
                                block tile, 3-stage, compiled for sm_120
                 stream_k     — bf16_gemm_sm80_streamk.gemm_streamk
                                CUTLASS 2.x Stream-K decomposition,
                                128x128x32 tile, 4-stage, sm_120
  MM_WARMUP    warmup iters (default 3)
  MM_MIN_MS    minimum measured run duration per (op,M) (default 2500)
  MM_GAP_MS    host sleep between segments (default 1500)
  MM_SEGMENTS  output csv path (required from run_qwen3.sh)
  MM_OPS       comma-separated subset of operators to run (default: all)
  MM_M         comma-separated M values to use (default: built-in sweep)
  MM_REVERSE   1 = walk M values large -> small (default 0). Useful for
               studying temperature / leakage / DVFS effects: large M
               heats the GPU first, then smaller M is measured while the
               die is already hot.
  MM_MEM_FRAC  fraction of GPU memory to budget (default 0.85)
"""

import csv
import gc
import os
import sys
import time
from datetime import datetime

import torch


# ---- Qwen3-8B dims -----------------------------------------------------------
H        = 4096    # hidden
INTER    = 12288   # FFN intermediate
N_HEADS  = 32      # Q heads
N_KVH    = 8       # KV heads (GQA)
HEAD_DIM = 128
VOCAB    = 151936
KV_DIM   = N_KVH * HEAD_DIM      # 1024
Q_DIM    = N_HEADS * HEAD_DIM    # 4096
QKV_DIM  = Q_DIM + 2 * KV_DIM    # 6144 — fused QKV output
GU_DIM   = 2 * INTER             # 24576 — fused gate+up output

# Optimized inference (vLLM / TRT-LLM style) fuses QKV into one GEMM and
# fuses gate/up of the SwiGLU FFN into one GEMM. Set MM_OPS=q_proj,k_proj,v_proj
# to fall back to HF-style separated projections.
#
# (name, K, N) for X @ W where X is (M, K) and W is (K, N)
OPS_FUSED = [
    ('qkv_proj',     H,     QKV_DIM),  # (M,4096) @ (4096, 6144)
    ('o_proj',       Q_DIM, H),        # (M,4096) @ (4096, 4096)
    ('gate_up_proj', H,     GU_DIM),   # (M,4096) @ (4096, 24576)
    ('down_proj',    INTER, H),        # (M,12288)@ (12288,4096)
    ('lm_head',      H,     VOCAB),    # (M,4096) @ (4096, 151936)
]

OPS_SEPARATE = [
    ('q_proj',    H,     Q_DIM),
    ('k_proj',    H,     KV_DIM),
    ('v_proj',    H,     KV_DIM),
    ('o_proj',    Q_DIM, H),
    ('gate_proj', H,     INTER),
    ('up_proj',   H,     INTER),
    ('down_proj', INTER, H),
    ('lm_head',   H,     VOCAB),
]

# All known ops (used when MM_OPS filters by name)
OPS_ALL = OPS_FUSED + OPS_SEPARATE
OPS = OPS_FUSED

M_VALUES_DEFAULT = [
    1, 32, 128, 512,
    1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072,
    262144, 524288,
]

BF16 = 2  # bytes


def mem_required(M, K, N):
    return BF16 * (M*K + K*N + M*N)


def fmt_ts(t_epoch):
    """nvidia-smi style: '2026/05/19 09:00:16.847' (local time, ms precision)."""
    dt = datetime.fromtimestamp(t_epoch)
    return dt.strftime('%Y/%m/%d %H:%M:%S.') + f'{int(dt.microsecond / 1000):03d}'


@torch.no_grad()
def main():
    gpu = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(gpu)
    dev = torch.device(f'cuda:{gpu}')
    name = torch.cuda.get_device_name(gpu)
    props = torch.cuda.get_device_properties(gpu)
    total_mem = props.total_memory
    print(f'[qwen3] torch={torch.__version__}  cuda={torch.version.cuda}')
    print(f'[qwen3] device={gpu}  name={name}  '
          f'SM={props.major}.{props.minor}  mem={total_mem/(1024**3):.1f} GB')

    mem_frac = float(os.environ.get('MM_MEM_FRAC', '0.85'))
    mem_budget = int(total_mem * mem_frac)

    warmup       = int(os.environ.get('MM_WARMUP', '3'))
    min_ms       = float(os.environ.get('MM_MIN_MS', '2500'))
    gap_ms       = int(os.environ.get('MM_GAP_MS', '1500'))
    segments_path = os.environ.get('MM_SEGMENTS')
    if not segments_path:
        print('ERROR: MM_SEGMENTS env not set', file=sys.stderr)
        sys.exit(1)

    ops_filter = os.environ.get('MM_OPS')
    if ops_filter:
        wanted = [s.strip() for s in ops_filter.split(',')]
        # Resolve in order of MM_OPS list, looking up in OPS_ALL
        op_index = {n: (n, k, nn) for (n, k, nn) in OPS_ALL}
        ops = []
        for name in wanted:
            if name not in op_index:
                print(f'[qwen3] unknown op "{name}", available: '
                      f'{sorted(op_index.keys())}', file=sys.stderr)
                sys.exit(1)
            ops.append(op_index[name])
    else:
        ops = OPS_FUSED  # fused QKV + gate_up (production inference layout)

    if os.environ.get('MM_M'):
        m_values = [int(x) for x in os.environ['MM_M'].split(',')]
    else:
        m_values = M_VALUES_DEFAULT

    reverse = os.environ.get('MM_REVERSE', '0').strip().lower() in ('1', 'true', 'yes')
    if reverse:
        m_values = list(reversed(m_values))

    # ---- backend selection ----
    backend = os.environ.get('MM_BACKEND', 'cublas').strip().lower()
    if backend == 'cublas':
        def matmul_fn(A, B):
            return torch.matmul(A, B)
    elif backend == 'cutlass_sm80':
        # bf16_gemm_sm80.cpython-*.so lives in the parent dir
        ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        import bf16_gemm_sm80  # noqa: F401
        # gemm_sm80_v3 = 128x128x64 block tile, 3 stages, sm_80 HMMA
        gemm_v3 = bf16_gemm_sm80.gemm_sm80_v3

        def matmul_fn(A, B):
            return gemm_v3(A, B, 0, 1)  # sleep_ns=0, sleep_freq=1
    elif backend == 'stream_k':
        # bf16_gemm_sm80_streamk.cpython-*.so lives in the parent dir
        ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        import bf16_gemm_sm80_streamk  # noqa: F401
        # gemm_streamk = 128x128x32 block tile, 4 stages, Stream-K swizzle, sm_80 HMMA
        gemm_sk = bf16_gemm_sm80_streamk.gemm_streamk

        def matmul_fn(A, B):
            return gemm_sk(A, B, 1, -1)  # split_k_factor=1 (basic Stream-K), avail_sms=auto
    else:
        print(f'[qwen3] unknown MM_BACKEND="{backend}" '
              f'(expected cublas | cutlass_sm80 | stream_k)', file=sys.stderr)
        sys.exit(1)

    print(f'[qwen3] backend={backend}')
    print(f'[qwen3] mem_budget={mem_budget/(1024**3):.1f} GB  '
          f'warmup={warmup}  min_ms={min_ms:.0f}  gap_ms={gap_ms}')
    print(f'[qwen3] segments={segments_path}')
    print(f'[qwen3] M sweep ({len(m_values)} values, '
          f'order={"reverse (large->small)" if reverse else "forward (small->large)"}): '
          f'{m_values}')
    print(f'[qwen3] operators: {[o[0] for o in ops]}')

    os.makedirs(os.path.dirname(segments_path) or '.', exist_ok=True)
    seg_file = open(segments_path, 'w', newline='')
    writer = csv.writer(seg_file)
    writer.writerow(['backend', 'operator', 'M', 'K', 'N', 'iters',
                     't_start', 't_end', 'ms_avg', 'tflops'])
    seg_file.flush()

    hdr = (f'{"op":12s} {"M":>7s} {"K":>6s} {"N":>7s} '
           f'{"iters":>6s} {"ms_avg":>10s} {"TFLOPS":>8s}  '
           f'{"t_start":24s}  {"t_end":24s}')
    print()
    print(hdr)
    print('-' * len(hdr))
    sys.stdout.flush()

    for op_name, K, N in ops:
        # Allocate B once per operator (shape is M-independent)
        try:
            B = torch.empty(K, N, device=dev, dtype=torch.bfloat16)
            B.normal_(0.0, 0.02)
        except torch.cuda.OutOfMemoryError:
            print(f'{op_name:12s}  cannot allocate B({K}x{N})  SKIP')
            torch.cuda.empty_cache()
            continue

        for M in m_values:
            req = mem_required(M, K, N)
            if req > mem_budget:
                print(f'{op_name:12s} {M:>7d} {K:>6d} {N:>7d}  '
                      f'SKIP (needs {req/(1024**3):.1f} GB)')
                sys.stdout.flush()
                continue

            try:
                A = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
                A.normal_(0.0, 0.02)
            except torch.cuda.OutOfMemoryError:
                print(f'{op_name:12s} {M:>7d} {K:>6d} {N:>7d}  OOM')
                torch.cuda.empty_cache()
                sys.stdout.flush()
                continue

            # warmup. Both backends allocate output per call: torch.matmul
            # without out= goes through the caching allocator, and the
            # cutlass extension allocates internally.
            try:
                for _ in range(warmup):
                    C = matmul_fn(A, B)
                torch.cuda.synchronize()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f'{op_name:12s} {M:>7d} {K:>6d} {N:>7d}  '
                      f'matmul failed ({backend}): {type(e).__name__}')
                del A
                torch.cuda.empty_cache()
                sys.stdout.flush()
                continue

            # probe: estimate ms per iter
            probe = 3
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for _ in range(probe):
                C = matmul_fn(A, B)
            ev1.record()
            torch.cuda.synchronize()
            probe_ms = ev0.elapsed_time(ev1)
            ms_per_iter = probe_ms / probe

            iters = max(2, int(min_ms / max(ms_per_iter, 1e-3)) + 1)

            # measured run with wall-clock bracketing
            t_start_wall = time.time()
            ev0.record()
            for _ in range(iters):
                C = matmul_fn(A, B)
            ev1.record()
            torch.cuda.synchronize()
            t_end_wall = time.time()

            ms_total = ev0.elapsed_time(ev1)
            ms_avg = ms_total / iters
            tflops = 2.0 * M * K * N / (ms_avg * 1e-3) / 1e12

            ts = fmt_ts(t_start_wall)
            te = fmt_ts(t_end_wall)
            print(f'{op_name:12s} {M:>7d} {K:>6d} {N:>7d} '
                  f'{iters:>6d} {ms_avg:>10.4f} {tflops:>8.2f}  '
                  f'{ts:24s}  {te:24s}')
            sys.stdout.flush()

            writer.writerow([backend, op_name, M, K, N, iters, ts, te,
                             f'{ms_avg:.6f}', f'{tflops:.4f}'])
            seg_file.flush()

            del A, C
            torch.cuda.empty_cache()

            if gap_ms > 0:
                time.sleep(gap_ms / 1000.0)

        del B
        torch.cuda.empty_cache()
        gc.collect()
        print(f'--- end {op_name} ---')
        sys.stdout.flush()

    seg_file.close()
    print('[qwen3] DONE')


if __name__ == '__main__':
    main()
