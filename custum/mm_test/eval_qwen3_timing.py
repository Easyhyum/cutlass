#!/usr/bin/env python3
"""
Qwen3 per-matmul timing eval (no V9 ramp, no power profile).

For each (op, M) and each backend in {cublas, stream_k}:
  - allocate A(MxK), B(KxN) bf16 on GPU
  - global warmup
  - per-burst Event timing: N inner kernels, ms_avg over the burst
  - repeat for MM_N_BURSTS bursts
  - record per-burst rows to MM_SEGMENTS csv (backend tag = "{be}:{op}_M{M}#{burst}")

Env:
  MM_MODEL      qwen3-8b | qwen3-32b           (default qwen3-8b)
  MM_GPU        device index                    (default 0)
  MM_SEGMENTS   output csv                      (required)
  MM_N_BURSTS   bursts per (op, M, backend)     (default 50)
  MM_M_KERNELS  inner kernels per burst         (default 50)
  MM_BURST_GAP_MS  sleep between bursts         (default 100)
  MM_CFG_GAP_MS    sleep between cfgs           (default 300)
  MM_GLOBAL_WARMUP_MS                           (default 2000)
  MM_OPS        comma-list (default = all 5 ops)
  MM_MS         comma-list (default = built-in per-model)
  MM_BACKENDS   comma-list (default = "cublas,stream_k")
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
        'Q_DIM': 4096,    # 32 heads x 128
        'KV_DIM': 1024,   # 8 heads  x 128 (GQA)
        'VOCAB': 151936,
        'M_DEFAULT': [1024, 2048, 4096, 8192],
    },
    'qwen3-32b': {
        'H':     5120,
        'INTER': 25600,
        'Q_DIM': 8192,    # 64 heads x 128
        'KV_DIM': 1024,   # 8 heads  x 128 (GQA)
        'VOCAB': 151936,
        'M_DEFAULT': [512, 1024, 2048, 4096],
    },
}


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
            # Basic Stream-K: no V9 ramp args, no sleeps, just the GEMM.
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
        Ms = cfg['M_DEFAULT']

    backends = [s.strip() for s in
                os.environ.get('MM_BACKENDS', 'cublas,stream_k').split(',') if s.strip()]

    n_bursts   = int(os.environ.get('MM_N_BURSTS', '50'))
    m_kernels  = int(os.environ.get('MM_M_KERNELS', '50'))
    burst_gap_ms = int(os.environ.get('MM_BURST_GAP_MS', '100'))
    cfg_gap_ms   = int(os.environ.get('MM_CFG_GAP_MS', '300'))
    warmup_ms    = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '2000'))
    seg_path     = os.environ['MM_SEGMENTS']

    props = torch.cuda.get_device_properties(gpu)
    print(f'[timing] device={gpu} ({torch.cuda.get_device_name(gpu)})  '
          f'SM={props.major}.{props.minor}  '
          f'mem={props.total_memory/(1024**3):.1f} GB  n_sms={props.multi_processor_count}')
    print(f'[timing] model={model}  ops={[o[0] for o in ops]}')
    print(f'[timing] M list={Ms}  backends={backends}')
    print(f'[timing] bursts={n_bursts}  kernels/burst={m_kernels}  '
          f'burst_gap={burst_gap_ms}ms  cfg_gap={cfg_gap_ms}ms')

    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    seg_f = open(seg_path, 'w', newline='')
    w = csv.writer(seg_f)
    w.writerow(['backend', 'operator', 'M', 'K', 'N', 'iters',
                't_start', 't_end', 'ms_avg', 'tflops'])
    seg_f.flush()

    # Total configs
    total_cfgs = len(ops) * len(Ms) * len(backends)
    cfg_idx = 0

    # One-time global warmup using a generic GEMM
    A_w = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    B_w = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    if warmup_ms > 0:
        print(f'[timing] global warmup {warmup_ms:.0f}ms ...')
        t0 = time.time()
        while (time.time() - t0)*1000.0 < warmup_ms:
            torch.matmul(A_w, B_w)
        torch.cuda.synchronize()
    del A_w, B_w
    torch.cuda.empty_cache()

    print()
    print(f'{"cfg":>30s}  {"backend":>10s}  {"bursts":>6s}  '
          f'{"ms_avg":>10s}  {"TFLOPS":>8s}')
    print('-' * 80)

    for op_name, K, N in ops:
        for M in Ms:
            try:
                A = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
                A.normal_(0, 0.02)
                B = torch.empty(K, N, device=dev, dtype=torch.bfloat16)
                B.normal_(0, 0.02)
            except torch.cuda.OutOfMemoryError:
                print(f'  [skip OOM] {op_name} M={M} K={K} N={N}')
                torch.cuda.empty_cache()
                continue

            for be_name in backends:
                cfg_idx += 1
                fn = make_backend(be_name)

                # backend-specific warmup so JITs / first-call slack don't bias the first burst
                try:
                    for _ in range(3):
                        fn(A, B)
                    torch.cuda.synchronize()
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    print(f'  [skip {be_name}] {op_name} M={M} '
                          f'{type(e).__name__}: {e}')
                    continue

                time.sleep(cfg_gap_ms / 1000.0)

                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)

                tag_base = f'{be_name}:{op_name}_M{M}'
                burst_msavg_sum = 0.0
                for burst_i in range(n_bursts):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    ev0.record()
                    for _ in range(m_kernels):
                        fn(A, B)
                    ev1.record()
                    torch.cuda.synchronize()
                    t1 = time.time()

                    ms_avg = ev0.elapsed_time(ev1) / m_kernels
                    tflops = 2.0 * M * K * N / (ms_avg * 1e-3) / 1e12
                    ts_, te_ = fmt_ts(t0), fmt_ts(t1)
                    w.writerow([f'{tag_base}#{burst_i}', op_name, M, K, N,
                                m_kernels, ts_, te_,
                                f'{ms_avg:.6f}', f'{tflops:.4f}'])
                    seg_f.flush()
                    burst_msavg_sum += ms_avg
                    if burst_gap_ms > 0:
                        time.sleep(burst_gap_ms / 1000.0)

                mean_ms = burst_msavg_sum / n_bursts
                mean_tf = 2.0 * M * K * N / (mean_ms * 1e-3) / 1e12
                cfg_str = f'{op_name}_M{M}'
                print(f'  [{cfg_idx:>2d}/{total_cfgs}] '
                      f'{cfg_str:>30s}  {be_name:>10s}  '
                      f'{n_bursts:>6d}  '
                      f'{mean_ms:>10.4f}  {mean_tf:>8.2f}',
                      flush=True)

            del A, B
            torch.cuda.empty_cache()

    seg_f.close()
    print('[timing] DONE')


if __name__ == '__main__':
    main()
