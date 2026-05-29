#!/usr/bin/env python3
"""
Torch-level M-chunking sweep — all-ops × M × chunk × kernel.

For each (kernel, op, model, M, chunk_m):
  • slice A along M-axis into ceil(M / chunk_m) chunks
  • Python loop: for each slice → call gemm, write to C[start:start+rows]
  • 50 bursts per cfg, each burst = N "virtual" full-M chunked-GEMMs
  • optional MM_CHUNK_GAP_US adds time.sleep between chunks

Imports the production baselines (already at /workspace/custum):
  cublas    → torch.matmul (slices)
  streamk   → bf16_gemm_sm80_streamk_baseline (no wave-sleep)
  sm80_v3   → cutlass_sm80_v3 (pristine)
No new C++ build needed.

Env:
  MM_KERNEL          cublas | sm80_v3 | streamk   (REQUIRED)
  MM_M_LIST          comma list of full-M values
  MM_CHUNK_LIST      comma list of chunk_m sizes
  MM_OPS             comma list of "op:model" to enable (default: all uncommented in OPS)
  MM_CHUNK_GAP_US    Python time.sleep between chunks (default 0)
  MM_N_BURSTS        default 50
  MM_BURST_MS        default 500
  MM_BURST_GAP_MS    default 500
  MM_PEAK_TFLOPS     default 400
  MM_MEM_BUDGET_GB   default 40 (skip cfg above this)
  MM_GPU             default 0
"""
import os, sys, csv, time, math
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

# ─── Module-level streams for pipe mode (OVERLAP variant) ────────────────────
# This _ol variant intentionally lets chunk N+1 MMA start while chunk N is
# still finishing its tail wave.  CUDA's hardware scheduler only overlaps
# when there are free SMs, so for chunks large enough to fill all SMs in
# their first wave, the overlap is *only* at the tail (small).
#
# Implementation: 2 MMA streams alternating per chunk.  Both streams are
# module-level cached so within each stream MMAs are still serialized.
_PIPE_S_MMA0 = None
_PIPE_S_MMA1 = None
_PIPE_S_COPY = None


def _pipe_streams():
    global _PIPE_S_MMA0, _PIPE_S_MMA1, _PIPE_S_COPY
    if _PIPE_S_MMA0 is None:
        _PIPE_S_MMA0 = torch.cuda.Stream()
        _PIPE_S_MMA1 = torch.cuda.Stream()
        _PIPE_S_COPY = torch.cuda.Stream()
    return _PIPE_S_MMA0, _PIPE_S_MMA1, _PIPE_S_COPY

# ─── OPS catalog (Qwen3 8B / 32B). Comment-out lines to disable. ─────────────
OPS = [
    # (op, model, K, N)
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
    """Return (ext, fn, use_out).
       cublas supports out= directly via torch.matmul.
       streamk/sm80_v3 bindings don't accept out= → use_out=False, caller
       must do C[slice].copy_(fn(A, B)) for merge.
    """
    sys.path.insert(0, '/workspace/custum')
    if kernel == 'cublas':
        return None, torch.matmul, True
    elif kernel == 'streamk':
        import bf16_gemm_sm80_streamk_baseline as ext
        return ext, (lambda A, B: ext.gemm_streamk(A, B)), False
    elif kernel == 'sm80_v3':
        import cutlass_sm80_v3 as ext
        return ext, (lambda A, B: ext.gemm_sm80_v3(A, B)), False
    raise SystemExit(f'unknown MM_KERNEL={kernel}')


def n_kernels_for_burst(M, K, N, burst_ms, peak_tflops):
    flops = 2.0 * M * K * N
    per_call_ms = (flops / (peak_tflops * 1e12)) * 1e3
    return max(1, int(round(burst_ms / per_call_ms))), per_call_ms


def chunked_matmul(fn, A, B, C, chunk_m, chunk_gap_us=0,
                   use_out=False, mode='seq'):
    """Slice A by M-rows, call fn per slice, write into pre-allocated C.

    Modes:
      mode='seq'  : single-stream sequential — copy in-line after each MMA.
                    Production-realistic worst-case (no pipelining).
      mode='pipe' : 2-stream pipeline — MMA on s_mma, copy on s_copy.
                    chunk N's copy overlaps chunk N+1's MMA.  GPU memory
                    controller is full-duplex so load/store happens
                    concurrently → copy effectively hidden behind MMA.

    `use_out=True` (cublas only): write directly into C[slice] via out=
    kwarg — no intermediate tensor, no merge step → pipe mode is identical
    to seq.  CUTLASS extensions don't accept out= → pipe gives true benefit.
    """
    M = A.size(0)
    chunk_m = min(chunk_m, M)   # cap so cm == M produces exactly 1 chunk
                                # (still goes through seq / pipe code path
                                #  so wrapper/stream overhead is measured)

    if mode == 'pipe':
        # OVERLAP variant — 2 alternating MMA streams.
        # chunk i goes on streams[i % 2].  Both streams are MODULE-LEVEL
        # cached so per-stream MMAs are still serialized, BUT across streams
        # the CUDA scheduler can run chunk N (stream0) tail concurrently
        # with chunk N+1 (stream1) head when SMs become free.  For 1024+
        # tile chunks this yields *tail* overlap only — chunks fill SMs in
        # their first wave then free SMs progressively at the last wave,
        # and that's the only window where the other stream can start.
        s_mma0, s_mma1, s_copy = _pipe_streams()
        streams = (s_mma0, s_mma1)
        prev_out_per = [None, None]
        prev_slice_per = [None, None]
        prev_event_per = [None, None]

        for i, start in enumerate(range(0, M, chunk_m)):
            rows = min(chunk_m, M - start)
            A_slice = A[start:start + rows].contiguous()
            s_cur = streams[i % 2]

            # MMA on alternating stream — CUDA scheduler runs streams[0] and
            # streams[1] concurrently only when SMs are free → tail-overlap.
            with torch.cuda.stream(s_cur):
                if use_out:
                    # cublas: writes directly into C, no separate copy.
                    fn(A_slice, B, out=C[start:start + rows])
                else:
                    out = fn(A_slice, B)
            e_mma = torch.cuda.Event(); e_mma.record(s_cur)

            if not use_out:
                # Schedule PREVIOUS chunk's copy from the SAME stream as
                # *its* MMA (so memory ordering is preserved per-stream).
                # That copy runs on s_copy in parallel with the CURRENT MMA
                # (which is on the OTHER MMA stream) → no MMA stalls behind
                # the copy.  Each stream keeps its own prev_* state slot.
                slot = i % 2
                if prev_out_per[slot] is not None:
                    with torch.cuda.stream(s_copy):
                        s_copy.wait_event(prev_event_per[slot])
                        prev_slice_per[slot].copy_(prev_out_per[slot])
                prev_out_per[slot]   = out
                prev_slice_per[slot] = C[start:start + rows]
                prev_event_per[slot] = e_mma

        # tail copies for any remaining per-stream prev_out (CUTLASS only)
        if not use_out:
            for slot in (0, 1):
                if prev_out_per[slot] is not None:
                    with torch.cuda.stream(s_copy):
                        s_copy.wait_event(prev_event_per[slot])
                        prev_slice_per[slot].copy_(prev_out_per[slot])

        cur = torch.cuda.current_stream()
        cur.wait_stream(s_mma0)
        cur.wait_stream(s_mma1)
        cur.wait_stream(s_copy)
        return

    # seq mode (single-stream)
    start = 0
    while start < M:
        rows = min(chunk_m, M - start)
        A_slice = A[start:start + rows].contiguous()
        if use_out:
            fn(A_slice, B, out=C[start:start + rows])
        else:
            C[start:start + rows].copy_(fn(A_slice, B))
        start += rows
        if chunk_gap_us > 0 and start < M:
            time.sleep(chunk_gap_us / 1e6)


def parse_ops_filter(env_val):
    """Parse 'op:model,op:model' into a set; empty → use all uncommented OPS."""
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
    op_filter  = parse_ops_filter(os.environ.get('MM_OPS', ''))
    chunk_gap_us = int(os.environ.get('MM_CHUNK_GAP_US', '0'))

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

    ext, fn, use_out = load_kernel(kernel)

    print(f'[torch-chunk] device={props.name}  n_sm={n_sm}')
    print(f'[torch-chunk] kernel={kernel}')
    print(f'[torch-chunk] M_list={M_list}  chunk_list={chunk_list}  gap_us={chunk_gap_us}')
    print(f'[torch-chunk] ops={[f"{o[0]}@{o[1]}" for o in OPS]}')

    new_file = not os.path.exists(seg_path)
    seg_f = open(seg_path, 'a', newline='')
    seg_w = csv.writer(seg_f)
    if new_file:
        seg_w.writerow([
            'kernel', 'op', 'model', 'M', 'K', 'N',
            'chunk_m', 'n_chunks', 'chunk_gap_us', 'mode',
            'cfg_name',
            'burst_idx', 't_start_ns', 't_end_ns',
            'elapsed_ms', 'n_kernels', 'tflops_obs',
        ])

    # Warmup (small problem)
    A0 = torch.empty(1024, 1024, device=dev, dtype=torch.bfloat16); A0.normal_(0, 0.02)
    B0 = torch.empty(1024, 1024, device=dev, dtype=torch.bfloat16); B0.normal_(0, 0.02)
    for _ in range(20):
        fn(A0, B0)
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
                C = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            except torch.cuda.OutOfMemoryError:
                print(f'[skip oom ] {op_name} {model} M={M:>7}'); torch.cuda.empty_cache()
                continue

            n_kernels, per_call_ms = n_kernels_for_burst(M, K, N, burst_ms, peak_tflops)
            flops_per_call = 2.0 * M * K * N

            # Per-shape warmup
            for _ in range(3):
                fn(A, B)
            torch.cuda.synchronize()
            time.sleep(0.3)

            # ── Build cfg list ────────────────────────────────────────────────
            # BASE: direct fn(A,B), production baseline (no chunked-path).
            # For each chunk_m (cm <= M): seq first, then pipe.  Both run even
            # at cm == M — single-chunk case still exercises seq/pipe code
            # path, so wrapper/stream overhead is captured consistently.
            cfg_runs = [('BASE', 0, 'seq')]
            for cm in chunk_list:
                if cm <= M:
                    cfg_runs.append(('chunk', cm, 'seq'))
                    cfg_runs.append(('chunk', cm, 'pipe'))
                else:
                    print(f'  [skip] chunk_m={cm} > M={M}')

            for tag, chunk_m, mode in cfg_runs:
                if tag == 'BASE':
                    n_chunks = 1
                    cfg_name = f'{kernel}_{op_name}_{model}_M{M}_BASE'
                else:
                    n_chunks = (M + chunk_m - 1) // chunk_m
                    cfg_name = (f'{kernel}_{op_name}_{model}_M{M}'
                                f'_cm{chunk_m}_{mode}_ng{chunk_gap_us}')
                print(f'\n[torch-chunk] {cfg_name}  '
                      f'(mode={mode}, n_chunks={n_chunks}, '
                      f'per_call≈{per_call_ms:.1f}ms, n_kernels={n_kernels})')
                time.sleep(0.5)

                for b in range(n_bursts):
                    torch.cuda.synchronize()
                    t0_ns = time.time_ns()
                    for _ in range(n_kernels):
                        if tag == 'BASE':
                            if use_out:
                                fn(A, B, out=C)
                            else:
                                C.copy_(fn(A, B))
                        else:
                            chunked_matmul(fn, A, B, C, chunk_m,
                                           chunk_gap_us, use_out=use_out,
                                           mode=mode)
                    torch.cuda.synchronize()
                    t1_ns = time.time_ns()

                    elapsed_ms = (t1_ns - t0_ns) / 1e6
                    tf = (flops_per_call * n_kernels / (elapsed_ms / 1e3)) / 1e12
                    seg_w.writerow([
                        kernel, op_name, model, M, K, N,
                        chunk_m, n_chunks, chunk_gap_us, mode,
                        cfg_name, b, t0_ns, t1_ns,
                        f'{elapsed_ms:.3f}', n_kernels, f'{tf:.2f}',
                    ])
                    seg_f.flush()
                    if b == 0 or (b + 1) % 25 == 0:
                        print(f'  burst {b+1}/{n_bursts}  elapsed={elapsed_ms:.1f}ms  tf={tf:.1f}')
                    time.sleep(burst_gap_ms / 1000.0)

            del A, B, C
            torch.cuda.empty_cache()

    seg_f.close()
    print(f'\n[torch-chunk] DONE — segments → {seg_path}')


if __name__ == '__main__':
    main()
