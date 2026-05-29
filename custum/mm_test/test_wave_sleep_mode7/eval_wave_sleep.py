#!/usr/bin/env python3
"""
Wave-sleep mode 7 (SM gating) sweep — qwen3-32b down_proj, M=8192.

For each (kernel, active_pct):
  • prime mode=7 with first_smid_thr = n_sm * active_pct / 100
  • run N_BURSTS bursts, each ~MM_BURST_MS ms of kernel launches
  • idle MM_BURST_GAP_MS between bursts
  • record per-burst (start, end, kernels, elapsed) into segments.csv
  • nvidia-smi sampler (started by run.sh) captures 50ms power.

Designed to be invoked once per kernel (separate process) — eval_wave_sleep.py
loads ONE binary based on MM_KERNEL env var to avoid cross-TU __constant__
collisions.

Env:
  MM_KERNEL          streamk_ws | sm80_v3_ws   (REQUIRED)
  MM_ACTIVE_PCTS     comma list of active_pct  (default: 100,90,80,70,60,50,40)
  MM_M               default 8192
  MM_K, MM_N         default 25600 / 5120 (qwen3-32b down_proj)
  MM_OP_NAME         label used in CSV (default down_proj)
  MM_MODEL           label used in CSV (default qwen3-32b)
  MM_N_BURSTS        bursts per config       (default 50)
  MM_BURST_MS        per-burst target time   (default 500)
  MM_BURST_GAP_MS    idle gap between bursts (default 500)
  MM_PEAK_TFLOPS     used to size kernels/burst   (default 400)
  MM_SEGMENTS        path to segments.csv   (default segments.csv in cwd)
  MM_GPU             cuda device index (default 0)
"""
import os, sys, csv, time
from datetime import datetime
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_kernel(kernel):
    if kernel == 'streamk_ws':
        sys.path.insert(0, os.path.join(HERE, 'build_streamk_ws'))
        import bf16_gemm_sm80_streamk_ws as ext
        return ext, (lambda A, B: ext.gemm_streamk(A, B))
    elif kernel == 'sm80_v3_ws':
        sys.path.insert(0, os.path.join(HERE, 'build_sm80_v3_ws'))
        import bf16_gemm_sm80_v3_ws as ext
        return ext, (lambda A, B: ext.gemm_sm80_v3_ws(A, B))
    raise SystemExit(f'unknown MM_KERNEL={kernel}')


def n_kernels_for_burst(M, K, N, burst_ms, peak_tflops):
    flops = 2.0 * M * K * N                              # per call
    per_call_ms = (flops / (peak_tflops * 1e12)) * 1e3   # at PEAK
    n = max(1, int(round(burst_ms / per_call_ms)))
    return n, per_call_ms


def wave_count(M, N, n_sm, kernel):
    """Same wave-count formula used in test_M_kernel_sweep."""
    import math
    tiles = math.ceil(M / 128) * math.ceil(N / 128)
    if kernel == 'streamk_ws':
        return max(1, tiles // n_sm)
    else:
        # sm80_v3 / DP: ceil(tiles / n_sm) (last wave partial → still counts)
        return max(1, math.ceil(tiles / n_sm))


def main():
    kernel = os.environ.get('MM_KERNEL', '').strip().lower()
    if kernel not in ('streamk_ws', 'sm80_v3_ws'):
        raise SystemExit('MM_KERNEL must be: streamk_ws | sm80_v3_ws')

    M = int(os.environ.get('MM_M', '8192'))
    K = int(os.environ.get('MM_K', '25600'))     # qwen3-32b down_proj
    N = int(os.environ.get('MM_N', '5120'))
    op_label    = os.environ.get('MM_OP_NAME', 'down_proj')
    model_label = os.environ.get('MM_MODEL',   'qwen3-32b')

    pcts = [int(x) for x in os.environ.get(
        'MM_ACTIVE_PCTS', '100,90,80,70,60,50,40').split(',')]

    n_bursts     = int(os.environ.get('MM_N_BURSTS',     '50'))
    burst_ms     = float(os.environ.get('MM_BURST_MS',     '500'))
    burst_gap_ms = float(os.environ.get('MM_BURST_GAP_MS', '500'))
    peak_tflops  = float(os.environ.get('MM_PEAK_TFLOPS',  '400'))

    seg_path = os.environ.get('MM_SEGMENTS', os.path.join(HERE, 'segments.csv'))

    cuda_idx = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(cuda_idx)
    dev = torch.device(f'cuda:{cuda_idx}')

    props = torch.cuda.get_device_properties(cuda_idx)
    n_sm  = props.multi_processor_count

    ext, fn = load_kernel(kernel)

    n_kernels, per_call_ms = n_kernels_for_burst(M, K, N, burst_ms, peak_tflops)
    waves = wave_count(M, N, n_sm, kernel)

    print(f'[ws] device={props.name}  n_sm={n_sm}')
    print(f'[ws] kernel={kernel}  M={M} K={K} N={N}  op={op_label}  model={model_label}')
    print(f'[ws] active_pcts={pcts}  bursts/cfg={n_bursts}  burst_ms={burst_ms}  '
          f'gap_ms={burst_gap_ms}')
    print(f'[ws] per-call ≈ {per_call_ms:.2f} ms @ {peak_tflops:.0f} TFLOPS  '
          f'→ {n_kernels} kernels / burst   waves={waves}')

    # ── Allocate A,B once — same shape across the whole eval ─────────────────
    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)

    # ── Warmup (no sleep) ────────────────────────────────────────────────────
    for _ in range(20):
        fn(A, B)
    torch.cuda.synchronize()
    time.sleep(0.5)

    # ── Segments CSV header (append) ─────────────────────────────────────────
    new_file = not os.path.exists(seg_path)
    seg_f = open(seg_path, 'a', newline='')
    seg_w = csv.writer(seg_f)
    if new_file:
        seg_w.writerow([
            'kernel', 'op', 'model', 'M', 'K', 'N',
            'active_pct', 'first_smid_thr', 'waves',
            'cfg_name', 'burst_idx', 't_start_ns', 't_end_ns', 'elapsed_ms',
            'n_kernels', 'tflops_obs',
        ])

    # ── Sweep ────────────────────────────────────────────────────────────────
    flops_per_call = 2.0 * M * K * N
    for pct in pcts:
        first_smid_thr = max(0, min(n_sm, n_sm * pct // 100))
        cfg = f'{kernel}_pct{pct}_thr{first_smid_thr}'
        print(f'\n[ws] === {cfg}  '
              f'(active SMs 0..{first_smid_thr-1}, '
              f'gated SMs {first_smid_thr}..{n_sm-1})')

        # cfg gap before first burst — let GPU return to idle
        time.sleep(1.0)

        for b in range(n_bursts):
            # Re-prime once per burst (one-shot consumed by first call,
            # subsequent calls in the same burst still see the kWaveSleep*
            # constants because no other kernel resets them — but we re-prime
            # at the start of every burst to be safe).
            if pct < 100:
                ext.prime_wave_sleep(
                    int(waves), int(n_sm), int(first_smid_thr),
                    0,          # first_step_ns — unused by mode 7
                    0, 0,       # mid_pct, mid_ns — unused by mode 7
                    7, 0,       # mode=7 (SM gating), shape=0
                    0xC0FFEE11,
                )

            torch.cuda.synchronize()
            t0_ns = time.time_ns()
            for _ in range(n_kernels):
                fn(A, B)
            torch.cuda.synchronize()
            t1_ns = time.time_ns()

            elapsed_ms = (t1_ns - t0_ns) / 1e6
            tf_obs = (flops_per_call * n_kernels / (elapsed_ms / 1e3)) / 1e12
            seg_w.writerow([
                kernel, op_label, model_label, M, K, N,
                pct, first_smid_thr, waves,
                cfg, b, t0_ns, t1_ns, f'{elapsed_ms:.3f}',
                n_kernels, f'{tf_obs:.2f}',
            ])
            seg_f.flush()

            if b == 0 or (b + 1) % 10 == 0:
                print(f'  burst {b+1}/{n_bursts}  '
                      f'elapsed={elapsed_ms:.1f}ms  tf={tf_obs:.1f}')

            time.sleep(burst_gap_ms / 1000.0)

    seg_f.close()
    print(f'\n[ws] DONE — segments → {seg_path}')


if __name__ == '__main__':
    main()
