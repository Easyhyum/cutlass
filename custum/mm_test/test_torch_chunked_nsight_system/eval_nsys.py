#!/usr/bin/env python3
"""
Nsight Systems profiling for torch-level chunked GEMM.

For each (kernel, M):
  • warmup with the smaller workload
  • per-shape warmup with the actual A,B
  • run ONCE per cfg (BASE + seq/pipe × chunk_list) with NVTX markers so the
    nsys timeline cleanly labels each phase
  • few profile iterations per cfg (default 3) for clean timeline

NVTX hierarchy:
  <kernel>_M<M>
    <kernel>_M<M>_BASE                 (single fn call)
    <kernel>_M<M>_cm<cm>_seq           (chunked, sequential)
    <kernel>_M<M>_cm<cm>_pipe          (chunked, MMA on s_mma + copy on s_copy)

Open the resulting .nsys-rep in Nsight Systems UI to verify:
  • s_mma stream shows MMAs strictly sequential
  • s_copy stream shows copies in parallel with next MMA
  • Default stream is used by BASE / seq paths
  • For cublas (use_out=True): no s_copy activity (direct write to C)

Env:
  MM_KERNELS    default: cublas,sm80_v3,streamk
  MM_M_LIST     default: 1024,2048,4096,8192,16384,32768,65536,131072,262144
  MM_CHUNK_LIST default: 1024,1280,2048
  MM_K, MM_N    default: 25600, 5120  (qwen3-32b down_proj)
  MM_OP_NAME    default: down_proj
  MM_MODEL      default: qwen3-32b
  N_WARMUP      default: 30
  N_PROFILE     default: 3
  MM_MEM_BUDGET_GB default 40
"""
import os, sys, time
import torch
import torch.cuda.nvtx as nvtx

sys.path.insert(0, '/workspace/custum')

# ─── Config ──────────────────────────────────────────────────────────────────
KERNELS    = os.environ.get('MM_KERNELS', 'cublas,sm80_v3,streamk').split(',')
M_LIST     = [int(x) for x in os.environ.get(
    'MM_M_LIST', '1024,2048,4096,8192,16384,32768,65536,131072,262144').split(',')]
CHUNK_LIST = [int(x) for x in os.environ.get('MM_CHUNK_LIST', '1024,2048').split(',')]
K          = int(os.environ.get('MM_K', '25600'))
N          = int(os.environ.get('MM_N', '5120'))
OP_LABEL   = os.environ.get('MM_OP_NAME', 'down_proj')
MODEL_LABEL= os.environ.get('MM_MODEL',   'qwen3-32b')
N_WARMUP   = int(os.environ.get('N_WARMUP',  '30'))
N_PROFILE  = int(os.environ.get('N_PROFILE', '3'))
MEM_BUDGET = float(os.environ.get('MM_MEM_BUDGET_GB', '40'))

cuda_idx = int(os.environ.get('MM_GPU', '0'))
torch.cuda.set_device(cuda_idx)
dev = torch.device(f'cuda:{cuda_idx}')

# ─── Module-level streams for pipe mode (single allocation, reused) ─────────
_PIPE_S_MMA = None
_PIPE_S_COPY = None
def _pipe_streams():
    global _PIPE_S_MMA, _PIPE_S_COPY
    if _PIPE_S_MMA is None:
        _PIPE_S_MMA  = torch.cuda.Stream()
        _PIPE_S_COPY = torch.cuda.Stream()
    return _PIPE_S_MMA, _PIPE_S_COPY


def load_kernel(kernel):
    """Return (ext, fn, use_out).  Same as test_torch_chunked."""
    if kernel == 'cublas':
        return None, torch.matmul, True
    elif kernel == 'streamk':
        import bf16_gemm_sm80_streamk_baseline as ext
        return ext, (lambda A, B: ext.gemm_streamk(A, B)), False
    elif kernel == 'sm80_v3':
        import cutlass_sm80_v3 as ext
        return ext, (lambda A, B: ext.gemm_sm80_v3(A, B)), False
    raise ValueError(f'unknown kernel {kernel}')


def chunked_matmul(fn, A, B, C, chunk_m, use_out, mode):
    """Same logic as test_torch_chunked.eval_torch_chunked.chunked_matmul."""
    M = A.size(0)
    chunk_m = min(chunk_m, M)

    if mode == 'pipe':
        s_mma, s_copy = _pipe_streams()
        prev_out = prev_slice = prev_event = None
        for start in range(0, M, chunk_m):
            rows = min(chunk_m, M - start)
            A_slice = A[start:start + rows].contiguous()
            with torch.cuda.stream(s_mma):
                if use_out:
                    fn(A_slice, B, out=C[start:start + rows])
                else:
                    out = fn(A_slice, B)
            e_mma = torch.cuda.Event(); e_mma.record(s_mma)
            if not use_out:
                if prev_out is not None:
                    with torch.cuda.stream(s_copy):
                        s_copy.wait_event(prev_event)
                        prev_slice.copy_(prev_out)
                prev_out, prev_slice, prev_event = out, C[start:start + rows], e_mma
        if not use_out and prev_out is not None:
            with torch.cuda.stream(s_copy):
                s_copy.wait_event(prev_event)
                prev_slice.copy_(prev_out)
        cur = torch.cuda.current_stream()
        cur.wait_stream(s_mma)
        cur.wait_stream(s_copy)
        return

    # seq
    start = 0
    while start < M:
        rows = min(chunk_m, M - start)
        A_slice = A[start:start + rows].contiguous()
        if use_out:
            fn(A_slice, B, out=C[start:start + rows])
        else:
            C[start:start + rows].copy_(fn(A_slice, B))
        start += rows


# ─── Run ─────────────────────────────────────────────────────────────────────
props = torch.cuda.get_device_properties(cuda_idx)
print(f'[nsys] device={props.name}  n_sm={props.multi_processor_count}')
print(f'[nsys] kernels={KERNELS}  M_list={M_LIST}  chunk_list={CHUNK_LIST}')
print(f'[nsys] K={K} N={N}  op={OP_LABEL}  model={MODEL_LABEL}')
print(f'[nsys] N_WARMUP={N_WARMUP}  N_PROFILE={N_PROFILE} per cfg')

for kernel in KERNELS:
    kernel = kernel.strip()
    print(f'\n[nsys] === kernel={kernel} ===')
    ext, fn, use_out = load_kernel(kernel)

    # Generic warmup on small workload
    Aw = torch.empty(4096, K, device=dev, dtype=torch.bfloat16); Aw.normal_(0, 0.02)
    Bw = torch.empty(K, N, device=dev, dtype=torch.bfloat16);    Bw.normal_(0, 0.02)
    Cw = torch.empty(4096, N, device=dev, dtype=torch.bfloat16)
    nvtx.range_push(f'WARMUP_{kernel}')
    for _ in range(N_WARMUP):
        if use_out: fn(Aw, Bw, out=Cw)
        else:       Cw.copy_(fn(Aw, Bw))
    torch.cuda.synchronize()
    nvtx.range_pop()
    del Aw, Bw, Cw
    torch.cuda.empty_cache()
    time.sleep(0.5)

    for M in M_LIST:
        est_gb = (M * K + K * N + M * N) * 2 / 1024**3
        if est_gb > MEM_BUDGET:
            print(f'[nsys]   skip M={M:>7}  (est={est_gb:.1f}GB > {MEM_BUDGET}GB)')
            continue

        try:
            A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
            B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)
            C = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        except torch.cuda.OutOfMemoryError:
            print(f'[nsys]   skip M={M:>7}  OOM')
            torch.cuda.empty_cache()
            continue

        nvtx.range_push(f'{kernel}_M{M}')

        # Per-shape warmup (cublas heuristic stabilization)
        for _ in range(5):
            if use_out: fn(A, B, out=C)
            else:       C.copy_(fn(A, B))
        torch.cuda.synchronize()

        # BASE
        nvtx.range_push(f'{kernel}_M{M}_BASE')
        for _ in range(N_PROFILE):
            if use_out: fn(A, B, out=C)
            else:       C.copy_(fn(A, B))
        torch.cuda.synchronize()
        nvtx.range_pop()

        # Chunked: cm <= M only (skip cm > M as that's equivalent to cm==M)
        for cm in CHUNK_LIST:
            if cm > M:
                continue
            for mode in ('seq', 'pipe'):
                tag = f'{kernel}_M{M}_cm{cm}_{mode}'
                nvtx.range_push(tag)
                for _ in range(N_PROFILE):
                    chunked_matmul(fn, A, B, C, cm, use_out, mode)
                torch.cuda.synchronize()
                nvtx.range_pop()

        nvtx.range_pop()   # close M range

        del A, B, C
        torch.cuda.empty_cache()
        print(f'[nsys]   M={M:>7}  done')

print('\n[nsys] DONE — open .nsys-rep in Nsight Systems UI')
