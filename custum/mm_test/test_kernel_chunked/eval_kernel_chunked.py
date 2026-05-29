#!/usr/bin/env python3
"""
Kernel-level M-chunking sweep — all-ops × M × chunk × kernel.

For each (kernel, op, model, M, chunk_m, idle_us):
  • streamk / sm80_v3 : ONE call to gemm_*_chunked(A, B, chunk_m, idle_us) —
                        C++ side issues N=ceil(M/chunk_m) sequential GEMM
                        launches on the same CUDA stream.
  • cublas            : Python loop on torch.matmul slices (no C++ chunked
                        binding exists for cublas; same as torch_chunked,
                        included for fair comparison).

Imports the local build_*_chunked .so:
  bf16_gemm_sm80_streamk_chunked  (gemm_streamk + gemm_streamk_chunked)
  bf16_gemm_sm80_v3_chunked       (gemm_sm80_v3 + gemm_sm80_v3_chunked)

Env:
  MM_KERNEL          cublas | sm80_v3 | streamk    (REQUIRED)
  MM_M_LIST          comma list of full-M values
  MM_CHUNK_LIST      comma list of chunk_m sizes
  MM_IDLE_US_LIST    comma list of chunk_idle_us
  MM_OPS             comma list of "op:model" to enable (default: uncommented OPS)
  MM_N_BURSTS        default 50
  MM_BURST_MS        default 500
  MM_BURST_GAP_MS    default 500
  MM_PEAK_TFLOPS     default 400
  MM_MEM_BUDGET_GB   default 40
  MM_GPU             default 0
"""
import os, sys, csv, time, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# ─── OPS catalog. Comment-out to disable. ────────────────────────────────────
OPS = [
    # ('qkv_proj',   'qwen3-8b',    4096,   6144),
    # ('o_proj',     'qwen3-8b',    4096,   4096),
    # ('up_proj',    'qwen3-8b',    4096,  12288),
    ('down_proj',  'qwen3-8b',   12288,   4096),
    # ('lm_head',    'qwen3-8b',    4096, 151936),
    # ('qkv_proj',   'qwen3-32b',   5120,  10240),
    # ('o_proj',     'qwen3-32b',   8192,   5120),
    # ('up_proj',    'qwen3-32b',   5120,  25600),
    ('down_proj',  'qwen3-32b',  25600,   5120),
    # ('lm_head',    'qwen3-32b',   5120, 151936),
]


def load_kernel(kernel):
    """Return (ext, fn_chunked, fn_baseline).
       fn_baseline calls the kernel directly with NO chunking code path."""
    if kernel == 'cublas':
        def fn(A, B, cm, idle):
            M = A.size(0)
            if cm >= M:
                return torch.matmul(A, B)
            outs = []
            start = 0
            while start < M:
                rows = min(cm, M - start)
                outs.append(torch.matmul(A[start:start + rows].contiguous(), B))
                start += rows
                if idle > 0 and start < M:
                    torch.cuda.synchronize()
                    time.sleep(idle / 1e6)
            return torch.cat(outs, dim=0)
        return None, fn, (lambda A, B: torch.matmul(A, B))
    if kernel == 'streamk':
        sys.path.insert(0, os.path.join(HERE, 'build_streamk_chunked'))
        import bf16_gemm_sm80_streamk_chunked as ext
        return (ext,
                lambda A, B, cm, idle: ext.gemm_streamk_chunked(A, B, cm, 1, -1, idle),
                lambda A, B: ext.gemm_streamk(A, B))
    if kernel == 'sm80_v3':
        sys.path.insert(0, os.path.join(HERE, 'build_sm80_v3_chunked'))
        import bf16_gemm_sm80_v3_chunked as ext
        return (ext,
                lambda A, B, cm, idle: ext.gemm_sm80_v3_chunked(A, B, cm, idle),
                lambda A, B: ext.gemm_sm80_v3(A, B))
    raise SystemExit(f'unknown MM_KERNEL={kernel}')


def n_kernels_for_burst(M, K, N, burst_ms, peak_tflops):
    flops = 2.0 * M * K * N
    per_call_ms = (flops / (peak_tflops * 1e12)) * 1e3
    return max(1, int(round(burst_ms / per_call_ms))), per_call_ms


def parse_ops_filter(env_val):
    if not env_val.strip():
        return None
    keys = set()
    for tok in env_val.split(','):
        if ':' not in tok: continue
        op, model = tok.strip().split(':', 1)
        keys.add((op.strip(), model.strip()))
    return keys


def main():
    kernel = os.environ.get('MM_KERNEL', '').strip().lower()
    if kernel not in ('cublas', 'sm80_v3', 'streamk'):
        raise SystemExit('MM_KERNEL must be: cublas | sm80_v3 | streamk')

    M_list     = [int(x) for x in os.environ.get(
        'MM_M_LIST',     '1024,2048,4096,8192,16384,32768,65536,131072,262144').split(',')]
    chunk_list = [int(x) for x in os.environ.get(
        'MM_CHUNK_LIST', '1024,1280,2048').split(',')]
    idle_list  = [int(x) for x in os.environ.get(
        'MM_IDLE_US_LIST', '0').split(',')]
    op_filter  = parse_ops_filter(os.environ.get('MM_OPS', ''))

    n_bursts     = int(os.environ.get('MM_N_BURSTS',     '50'))
    burst_ms     = float(os.environ.get('MM_BURST_MS',     '500'))
    burst_gap_ms = float(os.environ.get('MM_BURST_GAP_MS', '500'))
    peak_tflops  = float(os.environ.get('MM_PEAK_TFLOPS',  '400'))
    mem_budget   = float(os.environ.get('MM_MEM_BUDGET_GB','40'))

    seg_path = os.environ.get('MM_SEGMENTS', os.path.join(HERE, 'segments.csv'))

    cuda_idx = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(cuda_idx)
    dev = torch.device(f'cuda:{cuda_idx}')
    props = torch.cuda.get_device_properties(cuda_idx)
    n_sm  = props.multi_processor_count

    ext, fn, fn_base = load_kernel(kernel)

    print(f'[kern-chunk] device={props.name}  n_sm={n_sm}')
    print(f'[kern-chunk] kernel={kernel}')
    print(f'[kern-chunk] M_list={M_list}  chunk_list={chunk_list}  idle_list={idle_list}')
    print(f'[kern-chunk] ops={[f"{o[0]}@{o[1]}" for o in OPS]}')

    new_file = not os.path.exists(seg_path)
    seg_f = open(seg_path, 'a', newline='')
    seg_w = csv.writer(seg_f)
    if new_file:
        seg_w.writerow([
            'kernel', 'op', 'model', 'M', 'K', 'N',
            'chunk_m', 'n_chunks', 'chunk_idle_us',
            'cfg_name',
            'burst_idx', 't_start_ns', 't_end_ns',
            'elapsed_ms', 'n_kernels', 'tflops_obs',
        ])

    # Warmup
    A0 = torch.empty(1024, 1024, device=dev, dtype=torch.bfloat16); A0.normal_(0, 0.02)
    B0 = torch.empty(1024, 1024, device=dev, dtype=torch.bfloat16); B0.normal_(0, 0.02)
    for _ in range(20):
        fn_base(A0, B0)
    torch.cuda.synchronize()
    del A0, B0; torch.cuda.empty_cache()

    # ── Sweep ────────────────────────────────────────────────────────────────
    for op_name, model, K, N in OPS:
        if op_filter is not None and (op_name, model) not in op_filter:
            continue
        for M in M_list:
            est_gb = (M * K + K * N + M * N) * 2 / 1024**3
            if est_gb > mem_budget:
                print(f'[skip mem ] {op_name} {model} M={M:>7} est={est_gb:.2f}GB > {mem_budget}GB')
                continue
            try:
                A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
                B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)
            except torch.cuda.OutOfMemoryError:
                print(f'[skip oom ] {op_name} {model} M={M:>7}'); torch.cuda.empty_cache()
                continue

            n_kernels, per_call_ms = n_kernels_for_burst(M, K, N, burst_ms, peak_tflops)
            flops_per_call = 2.0 * M * K * N

            for _ in range(3):
                fn_base(A, B)
            torch.cuda.synchronize()
            time.sleep(0.3)

            # cfg list: BASE first (direct kernel call, no chunked path),
            #           then chunked cfgs with chunk_m <= M (cm==M kept to
            #           isolate chunked-code-path overhead vs BASE).
            cfg_runs = [('BASE', 0, 0)]
            for cm in chunk_list:
                if cm <= M:
                    for iu in idle_list:
                        cfg_runs.append(('chunk', cm, iu))
                else:
                    print(f'  [skip] chunk_m={cm} > M={M}')

            for tag, chunk_m, idle_us in cfg_runs:
                if tag == 'BASE':
                    n_chunks = 1
                    cfg_name = f'{kernel}_{op_name}_{model}_M{M}_BASE'
                else:
                    n_chunks = (M + chunk_m - 1) // chunk_m
                    cfg_name = f'{kernel}_{op_name}_{model}_M{M}_cm{chunk_m}_iu{idle_us}'
                print(f'\n[kern-chunk] {cfg_name}  '
                      f'(n_chunks={n_chunks}, per_call≈{per_call_ms:.1f}ms, '
                      f'n_kernels={n_kernels})')
                time.sleep(0.5)

                for b in range(n_bursts):
                    torch.cuda.synchronize()
                    t0_ns = time.time_ns()
                    for _ in range(n_kernels):
                        if tag == 'BASE':
                            fn_base(A, B)
                        else:
                            fn(A, B, chunk_m, idle_us)
                    torch.cuda.synchronize()
                    t1_ns = time.time_ns()

                    elapsed_ms = (t1_ns - t0_ns) / 1e6
                    tf = (flops_per_call * n_kernels / (elapsed_ms / 1e3)) / 1e12
                    seg_w.writerow([
                        kernel, op_name, model, M, K, N,
                        chunk_m, n_chunks, idle_us,
                        cfg_name, b, t0_ns, t1_ns,
                        f'{elapsed_ms:.3f}', n_kernels, f'{tf:.2f}',
                    ])
                    seg_f.flush()
                    if b == 0 or (b + 1) % 25 == 0:
                        print(f'  burst {b+1}/{n_bursts}  elapsed={elapsed_ms:.1f}ms  tf={tf:.1f}')
                    time.sleep(burst_gap_ms / 1000.0)

            del A, B
            torch.cuda.empty_cache()

    seg_f.close()
    print(f'\n[kern-chunk] DONE — segments → {seg_path}')


if __name__ == '__main__':
    main()
