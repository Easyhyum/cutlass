#!/usr/bin/env python3
"""
Method A focused experiment — single (backend, op, M, cfg) sustained run.

Goal: visualize *where in the time domain* a config's power spikes occur, and
whether Method A actually moves/reduces them. Designed for iterative kernel
optimization: change kernel code → rerun this → compare power timeline.

Env vars:
  MM_GPU              GPU index (default 0)
  MM_SEGMENTS         segments csv path
  MM_OP               single op name (default down_proj)
  MM_M                single M (default 8192)
  MM_BACKEND          single backend (cutlass_sm80 | stream_k, default stream_k)
  MM_CONFIGS          comma-separated cfg labels (default: B,P-200-8,P-500-16,P-1000-16,P-2000-32)
  MM_DURATION_S       per-config sustained run duration sec (default 8.0)
  MM_GAP_S            idle gap between configs (default 3.0 — long enough for
                      DVFS to fully relax so each config starts from same state)
  MM_GLOBAL_WARMUP_MS one-shot warmup (default 3000)

Output (in MM_SEGMENTS path):
  - one segment row per config covering the full sustained window
  - downstream plot_power.py can render the timeline
"""
import csv
import gc
import os
import sys
import time
from datetime import datetime

import torch

H, INTER, VOCAB = 4096, 12288, 151936
QKV_DIM = 6144; Q_DIM = 4096; GU_DIM = 24576
OPS = {
    'qkv_proj':     (H,     QKV_DIM),
    'o_proj':       (Q_DIM, H),
    'gate_up_proj': (H,     GU_DIM),
    'down_proj':    (INTER, H),
    'lm_head':      (H,     VOCAB),
}

def _cfg(sleep_ns=0, sleep_freq=1, stagger_ns=0, stagger_mod=1,
         ramp_peak_ns=0, ramp_iters=1):
    return (sleep_ns, sleep_freq, stagger_ns, stagger_mod,
            ramp_peak_ns, ramp_iters)

CONFIG_DEFS = {
    'B':            _cfg(),
    # BM-* (v7 ring rotation, requires METHOD_A=1 build)
    'BM-25-2':      _cfg(stagger_ns=25,  stagger_mod=2),
    'BM-50-2':     _cfg(stagger_ns=50,  stagger_mod=2),
    'BM-100-2':    _cfg(stagger_ns=100, stagger_mod=2),
    'BM-200-2':    _cfg(stagger_ns=200, stagger_mod=2),
    # ── RAMP-* (v8 soft-launch, requires METHOD_A_RAMP=1 build) ──────────────
    # Format: RAMP-<peak_ns>-i<iters>
    # peak_ns = initial sleep at outer_iter=0 (linearly decreasing to 0 at iter=iters)
    # Activity ratio @ iter 0 = iter_time / (iter_time + peak_ns)
    # down_proj M=8192 K=12288: ~384 outer iters, ~12ms total → outer_iter ~31us
    #   peak_ns=  1000  →  31/(31+1)   = 97% activity at start
    #   peak_ns=  3000  →  31/(31+3)   = 91% activity at start
    #   peak_ns=  6000  →  31/(31+6)   = 84% activity at start
    #   peak_ns= 10000  →  31/(31+10)  = 76% activity at start  ← ~70-80% target
    #   peak_ns= 13000  →  31/(31+13)  = 70% activity at start  ← user's spec
    #   peak_ns= 20000  →  31/(31+20)  = 61% activity at start
    # iters: ramp duration. Larger = smoother DVFS adaptation, more total overhead.
    #   i30  : ramp finishes ~7.8% into kernel
    #   i50  : ~13%
    #   i100 : ~26%
    'RAMP-3000-i30':   _cfg(ramp_peak_ns=3000,  ramp_iters=30),   # 91% start, fast
    'RAMP-6000-i30':   _cfg(ramp_peak_ns=6000,  ramp_iters=30),   # 84% start, fast
    'RAMP-10000-i30':  _cfg(ramp_peak_ns=10000, ramp_iters=30),   # 76% start, fast
    'RAMP-13000-i30':  _cfg(ramp_peak_ns=13000, ramp_iters=30),   # 70% start, fast
    'RAMP-6000-i50':   _cfg(ramp_peak_ns=6000,  ramp_iters=50),   # 84% start, medium
    'RAMP-10000-i50':  _cfg(ramp_peak_ns=10000, ramp_iters=50),   # 76% start, medium
    'RAMP-13000-i50':  _cfg(ramp_peak_ns=13000, ramp_iters=50),   # 70% start, medium ★
    'RAMP-20000-i50':  _cfg(ramp_peak_ns=20000, ramp_iters=50),   # 61% start, medium
    'RAMP-10000-i100': _cfg(ramp_peak_ns=10000, ramp_iters=100),  # 76% start, slow
    'RAMP-13000-i100': _cfg(ramp_peak_ns=13000, ramp_iters=100),  # 70% start, slow
}


def fmt_ts(t):
    dt = datetime.fromtimestamp(t)
    return dt.strftime('%Y/%m/%d %H:%M:%S.') + f'{int(dt.microsecond/1000):03d}'


def pick_backend(name):
    ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ext_dir not in sys.path:
        sys.path.insert(0, ext_dir)
    if name == 'cutlass_sm80':
        import bf16_gemm_sm80 as ext
        def call(A, B, sl_ns, sl_fr, st_ns, st_md, rp_ns=0, rp_it=1):
            # sm80 path: ramp params not exposed in pybind yet → ignore
            del rp_ns, rp_it
            return ext.gemm_sm80_v3(A, B, sl_ns, sl_fr, st_ns, st_md)
        return call
    if name == 'stream_k':
        import bf16_gemm_sm80_streamk as ext
        def call(A, B, sl_ns, sl_fr, st_ns, st_md, rp_ns=0, rp_it=1):
            return ext.gemm_streamk(A, B, 1, -1, sl_ns, sl_fr,
                                    st_ns, st_md, rp_ns, rp_it)
        return call
    raise ValueError(name)


@torch.no_grad()
def main():
    gpu = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(gpu)
    dev = torch.device(f'cuda:{gpu}')

    op_name     = os.environ.get('MM_OP', 'down_proj')
    M           = int(os.environ.get('MM_M', '8192'))
    be_name     = os.environ.get('MM_BACKEND', 'stream_k')
    cfg_names   = os.environ.get('MM_CONFIGS',
                  'B,P-200-8,P-500-16,P-1000-16,P-2000-32').split(',')
    duration_s  = float(os.environ.get('MM_DURATION_S', '8.0'))
    gap_s       = float(os.environ.get('MM_GAP_S', '3.0'))
    warmup_ms   = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
    seg_path    = os.environ['MM_SEGMENTS']

    K, N = OPS[op_name]
    be_call = pick_backend(be_name)

    print(f'[focus] device={gpu} ({torch.cuda.get_device_name(gpu)})')
    print(f'[focus] {be_name}  {op_name}  M={M} K={K} N={N}')
    print(f'[focus] configs={cfg_names}  duration={duration_s}s  gap={gap_s}s')

    # global warmup
    if warmup_ms > 0:
        A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
        B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)
        print(f'[focus] global warmup {warmup_ms:.0f}ms...')
        t0 = time.time()
        while (time.time() - t0) * 1000.0 < warmup_ms:
            be_call(A, B, 0, 1, 0, 1, 0, 1)
        torch.cuda.synchronize()

    # Allocate tensors once
    if 'A' not in locals():
        A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
        B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)

    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    seg_f = open(seg_path, 'w', newline='')
    w = csv.writer(seg_f)
    w.writerow(['backend','operator','M','K','N','iters',
                't_start','t_end','ms_avg','tflops'])
    seg_f.flush()

    print()
    print(f'{"cfg":>14s} {"iters":>7s} {"ms_avg":>9s} {"TFLOPS":>7s}  '
          f'{"t_start":24s} {"t_end":24s}')
    print('-' * 100)

    for cfg in cfg_names:
        if cfg not in CONFIG_DEFS:
            print(f'WARN unknown cfg "{cfg}", skip'); continue
        sl_ns, sl_fr, st_ns, st_md, rp_ns, rp_it = CONFIG_DEFS[cfg]
        tag = f'{be_name}:{cfg}'

        # per-cfg warmup (small, just to prime PTX + caches)
        for _ in range(3):
            be_call(A, B, sl_ns, sl_fr, st_ns, st_md, rp_ns, rp_it)
        torch.cuda.synchronize()

        # sustained run with wall-clock bracketing + cuda event timing
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        t0 = time.time()
        ev0.record()
        iters = 0
        while (time.time() - t0) < duration_s:
            be_call(A, B, sl_ns, sl_fr, st_ns, st_md, rp_ns, rp_it)
            iters += 1
        ev1.record()
        torch.cuda.synchronize()
        t1 = time.time()

        ms_avg = ev0.elapsed_time(ev1) / iters
        tflops = 2.0 * M * K * N / (ms_avg * 1e-3) / 1e12

        ts, te = fmt_ts(t0), fmt_ts(t1)
        print(f'{cfg:>14s} {iters:>7d} {ms_avg:>9.4f} {tflops:>7.2f}  {ts}  {te}')
        sys.stdout.flush()
        w.writerow([tag, op_name, M, K, N, iters, ts, te,
                    f'{ms_avg:.6f}', f'{tflops:.4f}'])
        seg_f.flush()

        time.sleep(gap_s)

    seg_f.close()
    print('[focus] DONE')


if __name__ == '__main__':
    main()
