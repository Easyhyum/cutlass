#!/usr/bin/env python3
"""
Wave-sleep mode 10 (unified 3-phase, single-shot per CTA) sweep.

Two PHASES — each invocation runs ONE phase:
  • phase="first" : first-wave staircase only (mid disabled)
      vary (first_sleep_pct × first_ns)
  • phase="mid"   : mid-wave bubble only (first wave disabled)
      vary (mid_sleep_pct × mid_ns)

Always preceded by ONE baseline burst (no sleep) per phase so the rest of the
plot can be normalized.

Semantics of *sleep_pct* (per user spec):
  pct = % of SMs/CTAs that SLEEP in the affected wave-group.
  • first wave staircase  : first_smid_thr = n_sm * (100 - pct) / 100
                            → SMs with smid >= thr sleep with staircase delay
  • mid wave bubble       : mid_pct (kernel constant) = pct
                            → hash-selected `pct %` of mid-wave CTAs sleep mid_ns

Env vars (REQUIRED):
  MM_KERNEL          streamk_ws | sm80_v3_ws
  MM_PHASE           first | mid

Sweep:
  MM_PCT_LIST        default: 60,65,70,75,80,85,90,95,100
  MM_NS_LIST         default: 250,500,750,1000,5000

Problem (Qwen3-32B down_proj):
  MM_M=8192  MM_K=25600  MM_N=5120

Burst profile (same as mode 7 test):
  MM_N_BURSTS=50  MM_BURST_MS=500  MM_BURST_GAP_MS=500  MM_PEAK_TFLOPS=400

Output: appends per-burst rows to $MM_SEGMENTS (default: ./segments.csv).
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


def prime_for(ext, phase, pct, ns, num_waves, n_sm):
    """Translate (phase, pct, ns) into the prime_wave_sleep call."""
    if phase == 'first':
        first_smid_thr = max(0, min(n_sm, n_sm * (100 - pct) // 100))
        ext.prime_wave_sleep(
            int(num_waves), int(n_sm),
            int(first_smid_thr), int(ns),    # first_step_ns
            0, 0,                            # mid disabled
            10, 0,                           # mode=10, shape=0 (linear)
            0xC0FFEE11,
        )
        return first_smid_thr, 0             # log fields
    elif phase == 'mid':
        ext.prime_wave_sleep(
            int(num_waves), int(n_sm),
            int(n_sm), 0,                    # first disabled (thr=n_sm → no SM sleeps)
            int(pct), int(ns),               # mid_pct, mid_ns
            10, 0,
            0xC0FFEE11,
        )
        return n_sm, pct                     # log fields
    else:
        raise SystemExit(f'unknown MM_PHASE={phase}')


def main():
    kernel = os.environ.get('MM_KERNEL', '').strip().lower()
    phase  = os.environ.get('MM_PHASE',  '').strip().lower()
    if kernel not in ('streamk_ws', 'sm80_v3_ws'):
        raise SystemExit('MM_KERNEL must be: streamk_ws | sm80_v3_ws')
    if phase not in ('first', 'mid'):
        raise SystemExit('MM_PHASE must be: first | mid')

    M = int(os.environ.get('MM_M', '8192'))
    K = int(os.environ.get('MM_K', '25600'))
    N = int(os.environ.get('MM_N', '5120'))
    op_label    = os.environ.get('MM_OP_NAME', 'down_proj')
    model_label = os.environ.get('MM_MODEL',   'qwen3-32b')

    pct_list = [int(x) for x in os.environ.get(
        'MM_PCT_LIST', '60,65,70,75,80,85,90,95,100').split(',')]
    ns_list  = [int(x) for x in os.environ.get(
        'MM_NS_LIST',  '250,500,750,1000,5000').split(',')]

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
    num_waves = wave_count(M, N, n_sm, kernel)

    print(f'[ws10] device={props.name}  n_sm={n_sm}')
    print(f'[ws10] kernel={kernel}  phase={phase}  M={M} K={K} N={N}')
    print(f'[ws10] pct_list={pct_list}')
    print(f'[ws10] ns_list ={ns_list}')
    print(f'[ws10] bursts/cfg={n_bursts}  burst_ms={burst_ms}  gap_ms={burst_gap_ms}')
    print(f'[ws10] per-call ≈ {per_call_ms:.2f} ms @ {peak_tflops:.0f} TFLOPS  '
          f'→ {n_kernels} kernels/burst   waves={num_waves}')

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)

    # Warmup (no sleep)
    for _ in range(20):
        fn(A, B)
    torch.cuda.synchronize()
    time.sleep(0.5)

    # CSV
    new_file = not os.path.exists(seg_path)
    seg_f = open(seg_path, 'a', newline='')
    seg_w = csv.writer(seg_f)
    if new_file:
        seg_w.writerow([
            'kernel', 'phase', 'op', 'model', 'M', 'K', 'N',
            'sleep_pct', 'sleep_ns',
            'first_smid_thr', 'mid_pct_kernel',
            'num_waves', 'cfg_name',
            'burst_idx', 't_start_ns', 't_end_ns',
            'elapsed_ms', 'n_kernels', 'tflops_obs',
        ])

    flops_per_call = 2.0 * M * K * N

    # Build sweep order:  pct outer, ns inner  (so each pct row in the heatmap
    # is contiguous in time → easier thermal/clock interpretation).
    cfg_list = [(0, 0)]   # baseline first
    for pct in pct_list:
        for ns in ns_list:
            cfg_list.append((pct, ns))

    print(f'\n[ws10] total configs = {len(cfg_list)}  '
          f'(1 baseline + {len(pct_list)} pct × {len(ns_list)} ns)')

    for cfg_idx, (pct, ns) in enumerate(cfg_list):
        if pct == 0 and ns == 0:
            cfg_name = f'{kernel}_{phase}_BASELINE'
            first_thr, mid_pct_k = n_sm, 0
            do_prime = False
        else:
            cfg_name = f'{kernel}_{phase}_pct{pct}_ns{ns}'
            do_prime = True

        print(f'\n[ws10] [{cfg_idx+1}/{len(cfg_list)}] {cfg_name}')

        time.sleep(1.0)   # cfg gap

        for b in range(n_bursts):
            if do_prime:
                first_thr, mid_pct_k = prime_for(
                    ext, phase, pct, ns, num_waves, n_sm)

            torch.cuda.synchronize()
            t0_ns = time.time_ns()
            for _ in range(n_kernels):
                fn(A, B)
            torch.cuda.synchronize()
            t1_ns = time.time_ns()

            elapsed_ms = (t1_ns - t0_ns) / 1e6
            tf = (flops_per_call * n_kernels / (elapsed_ms / 1e3)) / 1e12
            seg_w.writerow([
                kernel, phase, op_label, model_label, M, K, N,
                pct, ns,
                first_thr, mid_pct_k,
                num_waves, cfg_name,
                b, t0_ns, t1_ns,
                f'{elapsed_ms:.3f}', n_kernels, f'{tf:.2f}',
            ])
            seg_f.flush()
            if b == 0 or (b + 1) % 25 == 0:
                print(f'  burst {b+1}/{n_bursts}  elapsed={elapsed_ms:.1f}ms  tf={tf:.1f}')
            time.sleep(burst_gap_ms / 1000.0)

    seg_f.close()
    print(f'\n[ws10] DONE — segments → {seg_path}')


if __name__ == '__main__':
    main()
