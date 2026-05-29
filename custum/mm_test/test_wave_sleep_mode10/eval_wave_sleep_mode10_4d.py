#!/usr/bin/env python3
"""
Wave-sleep mode 10 — 4D cross-product sweep:
  (first_pct × first_ns × mid_pct × mid_ns)

Each cfg primes BOTH the first-wave staircase AND the mid-wave bubble in one
prime_wave_sleep() call, then runs 50 bursts.  Plus one baseline (no sleep)
per kernel.

Output:  segments.csv  (sleep_pct → first_pct, sleep_ns → first_ns, plus
                        new columns mid_pct_sweep, mid_ns_sweep).

Env:
  MM_KERNEL          streamk_ws | sm80_v3_ws   (REQUIRED)
  MM_FIRST_PCTS      default: 60,80,100
  MM_FIRST_NS_LIST   default: 500,1000
  MM_MID_PCTS        default: 60,80,100
  MM_MID_NS_LIST     default: 500,1000

Problem & burst profile (same as 2D eval):
  MM_M=8192  MM_K=25600  MM_N=5120
  MM_N_BURSTS=50  MM_BURST_MS=500  MM_BURST_GAP_MS=500  MM_PEAK_TFLOPS=400
"""
import os, sys, csv, time, math
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
    flops = 2.0 * M * K * N
    per_call_ms = (flops / (peak_tflops * 1e12)) * 1e3
    return max(1, int(round(burst_ms / per_call_ms))), per_call_ms


def wave_count(M, N, n_sm, kernel):
    tiles = math.ceil(M / 128) * math.ceil(N / 128)
    if kernel == 'streamk_ws':
        return max(1, tiles // n_sm)
    else:
        return max(1, math.ceil(tiles / n_sm))


def main():
    kernel = os.environ.get('MM_KERNEL', '').strip().lower()
    if kernel not in ('streamk_ws', 'sm80_v3_ws'):
        raise SystemExit('MM_KERNEL must be: streamk_ws | sm80_v3_ws')

    M = int(os.environ.get('MM_M', '8192'))
    K = int(os.environ.get('MM_K', '25600'))
    N = int(os.environ.get('MM_N', '5120'))
    op_label    = os.environ.get('MM_OP_NAME', 'down_proj')
    model_label = os.environ.get('MM_MODEL',   'qwen3-32b')

    first_pcts = [int(x) for x in os.environ.get('MM_FIRST_PCTS',    '60,80,100').split(',')]
    first_nss  = [int(x) for x in os.environ.get('MM_FIRST_NS_LIST', '500,1000').split(',')]
    mid_pcts   = [int(x) for x in os.environ.get('MM_MID_PCTS',      '60,80,100').split(',')]
    mid_nss    = [int(x) for x in os.environ.get('MM_MID_NS_LIST',   '500,1000').split(',')]

    n_bursts     = int(os.environ.get('MM_N_BURSTS',     '50'))
    burst_ms     = float(os.environ.get('MM_BURST_MS',     '500'))
    burst_gap_ms = float(os.environ.get('MM_BURST_GAP_MS', '500'))
    peak_tflops  = float(os.environ.get('MM_PEAK_TFLOPS',  '400'))

    seg_path = os.environ.get('MM_SEGMENTS', os.path.join(HERE, 'segments_4d.csv'))

    cuda_idx = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(cuda_idx)
    dev = torch.device(f'cuda:{cuda_idx}')
    props = torch.cuda.get_device_properties(cuda_idx)
    n_sm  = props.multi_processor_count

    ext, fn = load_kernel(kernel)
    n_kernels, per_call_ms = n_kernels_for_burst(M, K, N, burst_ms, peak_tflops)
    num_waves = wave_count(M, N, n_sm, kernel)

    print(f'[ws10-4d] device={props.name}  n_sm={n_sm}')
    print(f'[ws10-4d] kernel={kernel}  M={M} K={K} N={N}')
    print(f'[ws10-4d] first_pcts={first_pcts}  first_nss={first_nss}')
    print(f'[ws10-4d] mid_pcts  ={mid_pcts}    mid_nss  ={mid_nss}')
    print(f'[ws10-4d] bursts/cfg={n_bursts}  burst_ms={burst_ms}  gap_ms={burst_gap_ms}')
    print(f'[ws10-4d] per-call ≈ {per_call_ms:.2f} ms @ {peak_tflops:.0f} TFLOPS  '
          f'→ {n_kernels} kernels/burst   waves={num_waves}')

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)

    for _ in range(20):
        fn(A, B)
    torch.cuda.synchronize()
    time.sleep(0.5)

    new_file = not os.path.exists(seg_path)
    seg_f = open(seg_path, 'a', newline='')
    seg_w = csv.writer(seg_f)
    if new_file:
        seg_w.writerow([
            'kernel', 'op', 'model', 'M', 'K', 'N',
            'first_pct', 'first_ns', 'mid_pct', 'mid_ns',
            'first_smid_thr', 'num_waves', 'cfg_name',
            'burst_idx', 't_start_ns', 't_end_ns',
            'elapsed_ms', 'n_kernels', 'tflops_obs',
        ])

    flops_per_call = 2.0 * M * K * N

    # cfg_list: baseline first, then all 4D combinations
    cfg_list = [('baseline', 0, 0, 0, 0)]
    for fp in first_pcts:
        for fn_ in first_nss:
            for mp in mid_pcts:
                for mn in mid_nss:
                    cfg_list.append(('combined', fp, fn_, mp, mn))

    print(f'\n[ws10-4d] total configs = {len(cfg_list)}  '
          f'({len(first_pcts)}×{len(first_nss)}×{len(mid_pcts)}×{len(mid_nss)} + 1 baseline)')

    for cfg_idx, (tag, fp, fns, mp, mns) in enumerate(cfg_list):
        if tag == 'baseline':
            cfg_name = f'{kernel}_BASELINE'
            ext.clear_wave_sleep()  # explicit disable for clean baseline
            first_smid_thr = n_sm
        else:
            first_smid_thr = max(0, min(n_sm, n_sm * (100 - fp) // 100))
            cfg_name = f'{kernel}_f{fp}_{fns}_m{mp}_{mns}'
            ext.prime_wave_sleep(
                int(num_waves), int(n_sm),
                int(first_smid_thr), int(fns),     # first wave staircase
                int(mp), int(mns),                 # mid wave bubble
                10, 0,                              # mode=10, shape=0
                0xC0FFEE11,
            )

        print(f'\n[ws10-4d] [{cfg_idx+1}/{len(cfg_list)}] {cfg_name}')

        time.sleep(1.0)

        for b in range(n_bursts):
            torch.cuda.synchronize()
            t0_ns = time.time_ns()
            for _ in range(n_kernels):
                fn(A, B)
            torch.cuda.synchronize()
            t1_ns = time.time_ns()

            elapsed_ms = (t1_ns - t0_ns) / 1e6
            tf = (flops_per_call * n_kernels / (elapsed_ms / 1e3)) / 1e12
            seg_w.writerow([
                kernel, op_label, model_label, M, K, N,
                fp, fns, mp, mns,
                first_smid_thr, num_waves, cfg_name,
                b, t0_ns, t1_ns,
                f'{elapsed_ms:.3f}', n_kernels, f'{tf:.2f}',
            ])
            seg_f.flush()
            if b == 0 or (b + 1) % 25 == 0:
                print(f'  burst {b+1}/{n_bursts}  elapsed={elapsed_ms:.1f}ms  tf={tf:.1f}')
            time.sleep(burst_gap_ms / 1000.0)

    seg_f.close()
    print(f'\n[ws10-4d] DONE — segments → {seg_path}')


if __name__ == '__main__':
    main()
