#!/usr/bin/env python3
"""
Qwen3 M-sweep eval for power+clock+TFLOPS comparison plot.

For ONE backend (chosen via MM_BACKEND in {cublas, stream_k}), runs every
(op, M) pair in the M sweep. Each (op, M) is a single timed cycle of
roughly 320 ms of GEMM work followed by a 500 ms rest gap so the SM clock
recovers towards the 2430 MHz boost target before the next cycle.

Env
  MM_MODEL       qwen3-8b | qwen3-32b                       (default qwen3-8b)
  MM_BACKEND     cublas | stream_k                          (default cublas)
  MM_GPU         device index                               (default 0)
  MM_SEGMENTS    output segments csv                        (required)
  MM_MS          comma-separated M values (overrides model default)
  MM_OPS         comma-separated op subset (default = all 5)
  MM_CYCLE_MS    target work duration per cycle (ms)        (default 320)
  MM_REST_MS     rest gap after each cycle (ms)             (default 500)
  MM_GLOBAL_WARMUP_MS                                       (default 2000)
  MM_MEM_FRAC                                               (default 0.85)
"""
import csv
import os
import sys
import time
from datetime import datetime

import torch


MODELS = {
    'qwen3-8b': {
        'H':     4096,
        'INTER': 12288,
        'Q_DIM': 4096,
        'KV_DIM': 1024,
        'VOCAB': 151936,
    },
    'qwen3-32b': {
        'H':     5120,
        'INTER': 25600,
        'Q_DIM': 8192,
        'KV_DIM': 1024,
        'VOCAB': 151936,
    },
}

M_VALUES_DEFAULT = [
    1, 16, 32, 64, 128, 256, 512, 1024,
    2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288,
]


def make_ops(cfg):
    H, I = cfg['H'], cfg['INTER']
    Q, KV = cfg['Q_DIM'], cfg['KV_DIM']
    QKV = Q + 2 * KV
    GU  = 2 * I
    return [
        ('qkv_proj',     H, QKV),
        ('o_proj',       Q, H),
        ('gate_up_proj', H, GU),
        ('down_proj',    I, H),
        ('lm_head',      H, cfg['VOCAB']),
    ]


def fmt_ts(t):
    dt = datetime.fromtimestamp(t)
    return dt.strftime('%Y/%m/%d %H:%M:%S.') + f'{int(dt.microsecond/1000):03d}'


def make_backend(name):
    if name == 'cublas':
        def fn(A, B):
            return torch.matmul(A, B)
        return fn
    if name == 'stream_k':
        ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        import bf16_gemm_sm80_streamk as ext
        def fn(A, B):
            return ext.gemm_streamk(A, B, 1, -1)
        return fn
    raise SystemExit(f'unknown backend: {name}')


@torch.no_grad()
def main():
    gpu = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(gpu)
    dev = torch.device(f'cuda:{gpu}')

    model = os.environ.get('MM_MODEL', 'qwen3-8b').strip().lower()
    if model not in MODELS:
        raise SystemExit(f'unknown MM_MODEL={model}; expected one of {list(MODELS)}')
    backend = os.environ.get('MM_BACKEND', 'cublas').strip().lower()

    cfg = MODELS[model]
    ops_all = make_ops(cfg)
    ops_filter = os.environ.get('MM_OPS', '').strip()
    if ops_filter:
        wanted = [s.strip() for s in ops_filter.split(',')]
        ops = [o for o in ops_all if o[0] in wanted]
    else:
        ops = ops_all

    if os.environ.get('MM_MS', '').strip():
        Ms = [int(x) for x in os.environ['MM_MS'].split(',')]
    else:
        Ms = M_VALUES_DEFAULT

    cycle_ms  = float(os.environ.get('MM_CYCLE_MS', '320'))
    rest_ms   = float(os.environ.get('MM_REST_MS',  '500'))
    warmup_ms = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '2000'))
    mem_frac  = float(os.environ.get('MM_MEM_FRAC', '0.85'))
    seg_path  = os.environ['MM_SEGMENTS']

    props = torch.cuda.get_device_properties(gpu)
    total_mem = props.total_memory
    mem_budget = int(total_mem * mem_frac)
    print(f'[compare] device={gpu} ({torch.cuda.get_device_name(gpu)})  '
          f'SM={props.major}.{props.minor}  mem={total_mem/(1024**3):.1f} GB  '
          f'budget={mem_budget/(1024**3):.1f} GB  n_sms={props.multi_processor_count}')
    print(f'[compare] model={model}  backend={backend}')
    print(f'[compare] ops={[o[0] for o in ops]}')
    print(f'[compare] M list={Ms}')
    print(f'[compare] cycle_ms={cycle_ms}  rest_ms={rest_ms}  '
          f'warmup_ms={warmup_ms}')

    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    seg_f = open(seg_path, 'w', newline='')
    w = csv.writer(seg_f)
    w.writerow(['backend', 'operator', 'M', 'K', 'N', 'iters',
                't_start', 't_end', 'ms_avg', 'tflops'])
    seg_f.flush()

    BF16 = 2

    fn = make_backend(backend)

    # one-time global warmup (use a generic GEMM)
    A_w = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    B_w = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    if warmup_ms > 0:
        print(f'[compare] global warmup {warmup_ms:.0f}ms ...')
        t0 = time.time()
        while (time.time() - t0) * 1000.0 < warmup_ms:
            fn(A_w, B_w)
        torch.cuda.synchronize()
    del A_w, B_w
    torch.cuda.empty_cache()

    print()
    print(f'{"op":13s} {"M":>7s} {"K":>6s} {"N":>7s} '
          f'{"iters":>6s} {"ms_avg":>10s} {"TFLOPS":>8s}  '
          f'{"t_start":24s}  {"t_end":24s}')
    print('-' * 110)
    sys.stdout.flush()

    for op_name, K, N in ops:
        # Allocate B once per op
        try:
            B = torch.empty(K, N, device=dev, dtype=torch.bfloat16)
            B.normal_(0, 0.02)
        except torch.cuda.OutOfMemoryError:
            print(f'{op_name:13s}  cannot allocate B({K}x{N})  SKIP')
            torch.cuda.empty_cache()
            continue

        for M in Ms:
            req = BF16 * (M * K + K * N + M * N)
            if req > mem_budget:
                print(f'{op_name:13s} {M:>7d} {K:>6d} {N:>7d}  '
                      f'SKIP (needs {req/(1024**3):.1f} GB > '
                      f'{mem_budget/(1024**3):.1f} GB)')
                sys.stdout.flush()
                continue

            try:
                A = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
                A.normal_(0, 0.02)
            except torch.cuda.OutOfMemoryError:
                print(f'{op_name:13s} {M:>7d} {K:>6d} {N:>7d}  OOM A')
                torch.cuda.empty_cache()
                continue

            try:
                for _ in range(3):
                    fn(A, B)
                torch.cuda.synchronize()
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f'{op_name:13s} {M:>7d} {K:>6d} {N:>7d}  '
                      f'warmup failed: {type(e).__name__}')
                del A
                torch.cuda.empty_cache()
                continue

            # Probe ms_per_iter to size the 320 ms cycle
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            probe = 3
            ev0.record()
            for _ in range(probe):
                fn(A, B)
            ev1.record()
            torch.cuda.synchronize()
            probe_ms = ev0.elapsed_time(ev1)
            ms_per_iter = probe_ms / probe

            iters = max(1, int(cycle_ms / max(ms_per_iter, 1e-3)))

            # Measured cycle
            t_start_wall = time.time()
            ev0.record()
            for _ in range(iters):
                fn(A, B)
            ev1.record()
            torch.cuda.synchronize()
            t_end_wall = time.time()

            ms_total = ev0.elapsed_time(ev1)
            ms_avg = ms_total / iters
            tflops = 2.0 * M * K * N / (ms_avg * 1e-3) / 1e12

            ts = fmt_ts(t_start_wall)
            te = fmt_ts(t_end_wall)
            print(f'{op_name:13s} {M:>7d} {K:>6d} {N:>7d} '
                  f'{iters:>6d} {ms_avg:>10.4f} {tflops:>8.2f}  '
                  f'{ts:24s}  {te:24s}', flush=True)
            w.writerow([backend, op_name, M, K, N, iters, ts, te,
                        f'{ms_avg:.6f}', f'{tflops:.4f}'])
            seg_f.flush()

            del A
            torch.cuda.empty_cache()

            if rest_ms > 0:
                time.sleep(rest_ms / 1000.0)

        del B
        torch.cuda.empty_cache()
        print(f'--- end {op_name} ---')
        sys.stdout.flush()

    seg_f.close()
    print('[compare] DONE')


if __name__ == '__main__':
    main()
