#!/usr/bin/env python3
"""
Method A (SM-staggered nanosleep) parameter sweep with TFLOPS + power measurement.

Goal: find (sleep_ns, sleep_freq, stagger_ns, stagger_mod) configurations that
**reduce power spike (max_W) while preserving throughput (TFLOPS retention)**.

Backends swept:
  - cutlass_sm80  : bf16_gemm_sm80.gemm_sm80_v3
  - stream_k      : bf16_gemm_sm80_streamk.gemm_streamk

Output rows are written incrementally to MM_SEGMENTS so that the parallel
nvidia-smi power sampler logs and the analyze_power.py post-processor work
end-to-end (same protocol as qwen3_8b_sweep.py).

Env vars (besides MM_GPU, MM_SEGMENTS):
  MM_OPS              ops to sweep (default: qkv_proj,o_proj,down_proj)
  MM_M                M values (default: 1024,8192)
  MM_MIN_MS           per-config min measured duration ms (default: 2500)
  MM_GAP_MS           idle gap between configs ms (default: 400)
  MM_REPEATS          repeat each (backend, op, M, cfg) cell N times — needed
                      because nvidia-smi @ 100ms misses spikes shorter than its
                      sampling interval. max_W is then aggregated as max-of-max
                      across N repeats, which converges to the true peak as N
                      grows. (default: 10; use 100 for high-confidence spike
                      capture but reduce MM_MIN_MS proportionally.)
  MM_REPEAT_GAP_MS    idle gap between repeats of the same config (default: 200)
  MM_GLOBAL_WARMUP_MS one-shot dummy kernel warmup at the start (default: 3000).
  MM_WARMUP           per-iter warmup count (default: 2) - L1/L2 priming only.
  MM_BACKENDS         backends (default: cutlass_sm80,stream_k)
"""
import csv
import gc
import os
import sys
import time
from datetime import datetime

import torch

# ── ops ──────────────────────────────────────────────────────────────────────
H, INTER, VOCAB = 4096, 12288, 151936
QKV_DIM = 6144; Q_DIM = 4096; GU_DIM = 24576
OPS_ALL = {
    'qkv_proj':     (H,     QKV_DIM),
    'o_proj':       (Q_DIM, H),
    'gate_up_proj': (H,     GU_DIM),
    'down_proj':    (INTER, H),
    'lm_head':      (H,     VOCAB),
}

BF16 = 2

# ── compact Method A sweep ───────────────────────────────────────────────────
# Config = (label, sleep_ns, sleep_freq, stagger_ns, stagger_mod)
# Three families:
#   B   = baseline (sleep 모두 0)               -> control
#   P   = prologue-only stagger (sleep_ns=0)    -> ideal: throughput 거의 보존
#   PR  = periodic SM-staggered (sleep_ns>0)    -> 더 강한 spike 억제
DEFAULT_CONFIGS = [
    # baseline
    ('B',           0, 1,    0,  1),
    # ── prologue-only stagger sweep (성능 보존 위주) ──
    ('P-50-4',      0, 1,   50,  4),
    ('P-50-8',      0, 1,   50,  8),
    ('P-100-8',     0, 1,  100,  8),
    ('P-100-16',    0, 1,  100, 16),
    ('P-200-8',     0, 1,  200,  8),
    ('P-200-16',    0, 1,  200, 16),
    ('P-500-8',     0, 1,  500,  8),
    ('P-500-16',    0, 1,  500, 16),
    ('P-1000-16',   0, 1, 1000, 16),
    ('P-2000-32',   0, 1, 2000, 32),
    # ── periodic + stagger (조금의 throughput 손실로 강한 spike 억제) ──
    ('PR-50-4-100-8',    50, 4,  100,  8),
    ('PR-50-8-100-8',    50, 8,  100,  8),
    ('PR-100-8-200-8',  100, 8,  200,  8),
    ('PR-100-16-500-8', 100,16,  500,  8),
    # ── periodic-only (uniform throttle, stagger 없음) - control ──
    ('U-50-4',     50, 4,    0,  1),
    ('U-100-4',   100, 4,    0,  1),
    ('U-100-8',   100, 8,    0,  1),
]


def fmt_ts(t):
    dt = datetime.fromtimestamp(t)
    return dt.strftime('%Y/%m/%d %H:%M:%S.') + f'{int(dt.microsecond/1000):03d}'


def pick_backend(name):
    ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ext_dir not in sys.path:
        sys.path.insert(0, ext_dir)
    if name == 'cutlass_sm80':
        import bf16_gemm_sm80 as ext
        def call(A, B, sleep_ns, sleep_freq, stagger_ns, stagger_mod):
            return ext.gemm_sm80_v3(A, B, sleep_ns, sleep_freq,
                                    stagger_ns, stagger_mod)
        return call
    if name == 'stream_k':
        import bf16_gemm_sm80_streamk as ext
        def call(A, B, sleep_ns, sleep_freq, stagger_ns, stagger_mod):
            return ext.gemm_streamk(A, B, 1, -1,
                                    sleep_ns, sleep_freq,
                                    stagger_ns, stagger_mod)
        return call
    raise ValueError(f'unknown backend {name}')


@torch.no_grad()
def main():
    gpu = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(gpu)
    dev = torch.device(f'cuda:{gpu}')
    props = torch.cuda.get_device_properties(gpu)
    print(f'[method_a] device={gpu} name={torch.cuda.get_device_name(gpu)} '
          f'SM={props.major}.{props.minor}')

    warmup           = int(os.environ.get('MM_WARMUP', '2'))
    global_warmup_ms = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
    min_ms           = float(os.environ.get('MM_MIN_MS', '2500'))
    gap_ms           = int(os.environ.get('MM_GAP_MS', '400'))
    repeats          = int(os.environ.get('MM_REPEATS', '10'))
    repeat_gap_ms    = int(os.environ.get('MM_REPEAT_GAP_MS', '200'))
    seg_path         = os.environ['MM_SEGMENTS']
    op_names         = os.environ.get('MM_OPS', 'qkv_proj,o_proj,down_proj').split(',')
    m_values         = [int(x) for x in os.environ.get('MM_M', '1024,8192').split(',')]
    backends         = os.environ.get('MM_BACKENDS', 'cutlass_sm80,stream_k').split(',')

    configs = DEFAULT_CONFIGS

    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    seg_f = open(seg_path, 'w', newline='')
    w = csv.writer(seg_f)
    # NOTE: 'backend' column is overloaded as "<backend>:<config_label>".
    # Each (backend, op, M, cfg) is repeated MM_REPEATS times; repeat_idx 0..N-1
    # is encoded into the backend tag as "<backend>:<cfg>#<repeat>" so the
    # existing analyze_power.py treats each repeat as a distinct segment, while
    # the downstream pareto script can groupby (be_base, cfg) across repeats.
    w.writerow(['backend','operator','M','K','N','iters',
                't_start','t_end','ms_avg','tflops'])
    seg_f.flush()

    print(f'[method_a] backends={backends} ops={op_names} M={m_values}  '
          f'configs={len(configs)} repeats={repeats}')
    print(f'[method_a] total measurements = '
          f'{len(backends)*len(op_names)*len(m_values)*len(configs)*repeats}  '
          f'(~{len(backends)*len(op_names)*len(m_values)*len(configs)*repeats*(min_ms+repeat_gap_ms)/60000:.1f} min @ min_ms={min_ms})')
    print(f'[method_a] segments={seg_path}')
    print(f'[method_a] global_warmup_ms={global_warmup_ms} '
          f'gap_ms={gap_ms} min_ms={min_ms} repeat_gap_ms={repeat_gap_ms}')

    # ── ONE-SHOT global warmup ──────────────────────────────────────────────
    # Use the largest (op, M) to push the GPU to boost clock and warm leakage.
    if global_warmup_ms > 0:
        warm_op = max(op_names, key=lambda n: OPS_ALL[n][1] * OPS_ALL[n][0])
        K_w, N_w = OPS_ALL[warm_op]
        M_w = max(m_values)
        Aw = torch.empty(M_w, K_w, device=dev, dtype=torch.bfloat16); Aw.normal_(0, 0.02)
        Bw = torch.empty(K_w, N_w, device=dev, dtype=torch.bfloat16); Bw.normal_(0, 0.02)
        # use the first backend's call (any will do — purpose is just to heat)
        warm_call = pick_backend(backends[0])
        print(f'[method_a] global warmup: {warm_op} M={M_w} for {global_warmup_ms:.0f}ms ...')
        t0 = time.time()
        while (time.time() - t0) * 1000.0 < global_warmup_ms:
            warm_call(Aw, Bw, 0, 1, 0, 1)
        torch.cuda.synchronize()
        print(f'[method_a] global warmup done in {(time.time()-t0)*1000:.0f}ms')
        del Aw, Bw
        torch.cuda.empty_cache()
    print()
    hdr = (f'{"backend:cfg":28s} {"op":12s} {"M":>6s} {"K":>6s} {"N":>7s} '
           f'{"iters":>5s} {"ms":>9s} {"TFLOPS":>7s}  '
           f'{"t_start":24s}  {"t_end":24s}')
    print(hdr); print('-' * len(hdr))
    sys.stdout.flush()

    for be_name in backends:
        be_call = pick_backend(be_name)
        for op_name in op_names:
            K, N = OPS_ALL[op_name]
            for M in m_values:
                try:
                    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16)
                    A.normal_(0.0, 0.02)
                    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16)
                    B.normal_(0.0, 0.02)
                except torch.cuda.OutOfMemoryError:
                    print(f'{be_name:14s} M={M} OOM, skip')
                    torch.cuda.empty_cache(); continue

                for (label, sl_ns, sl_fr, st_ns, st_md) in configs:
                    # per-iter warmup just to prime caches with this PTX path.
                    base_tag = f'{be_name}:{label}'
                    try:
                        for _ in range(warmup):
                            be_call(A, B, sl_ns, sl_fr, st_ns, st_md)
                        torch.cuda.synchronize()
                    except RuntimeError as e:
                        print(f'{base_tag:30s} {op_name:12s} {M:>6d} FAILED: {type(e).__name__}')
                        continue

                    # probe ms/iter (once per config, reused across repeats)
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    for _ in range(3):
                        be_call(A, B, sl_ns, sl_fr, st_ns, st_md)
                    ev1.record(); torch.cuda.synchronize()
                    ms_per = ev0.elapsed_time(ev1) / 3
                    iters  = max(2, int(min_ms / max(ms_per, 1e-3)) + 1)

                    # ── REPEAT measurements N times to oversample spike ────
                    for rep in range(repeats):
                        tag = f'{base_tag}#{rep}'
                        t0 = time.time()
                        ev0.record()
                        for _ in range(iters):
                            be_call(A, B, sl_ns, sl_fr, st_ns, st_md)
                        ev1.record(); torch.cuda.synchronize()
                        t1 = time.time()
                        ms_avg = ev0.elapsed_time(ev1) / iters
                        tflops = 2.0 * M * K * N / (ms_avg * 1e-3) / 1e12

                        ts, te = fmt_ts(t0), fmt_ts(t1)
                        # compact print for high-repeat runs
                        if rep == 0 or rep == repeats - 1:
                            print(f'{tag:30s} {op_name:12s} {M:>6d} {K:>6d} {N:>7d} '
                                  f'{iters:>5d} {ms_avg:>9.4f} {tflops:>7.2f}  '
                                  f'{ts:24s}  {te:24s}')
                            sys.stdout.flush()

                        w.writerow([tag, op_name, M, K, N, iters, ts, te,
                                    f'{ms_avg:.6f}', f'{tflops:.4f}'])
                        seg_f.flush()

                        if repeat_gap_ms > 0:
                            time.sleep(repeat_gap_ms / 1000.0)

                    if gap_ms > 0:
                        time.sleep(gap_ms / 1000.0)

                del A, B
                torch.cuda.empty_cache()
                gc.collect()
            print(f'--- end {be_name} {op_name} ---')
            sys.stdout.flush()

    seg_f.close()
    print('[method_a] DONE')


if __name__ == '__main__':
    main()
