#!/usr/bin/env python3
"""
Wave-aware sleep N-burst evaluation with TRUE baseline.

Two backends are loaded:
  * bf16_gemm_sm80_streamk_baseline  — built WITHOUT -DCUTLASS_WAVE_SLEEP_ENABLED
                                       (no wave-sleep code at all in SASS)
  * bf16_gemm_sm80_streamk           — built WITH    -DCUTLASS_WAVE_SLEEP_ENABLED,
                                       activated via prime_wave_sleep()

Same I/O shape as eval_v9_n_burst.py → drop-in for make_plots.sh.

Configuration tagging (post _step→_p rename in make_plots.sh):
  baseline run  → 'stream_k:s100_step0#i'      (mirrors V9 baseline labeling)
  wave-sleep    → 'stream_k:s<first_pct>_step<step_ns>#i'   (mid OFF in this run)

Env knobs:
  MM_WS_FIRST_PCT      CSV first-wave % (default '70,80')
  MM_WS_FIRST_STEP_NS  CSV first-wave step (default '500,2000,5000')  ← user spec
  MM_N_BURSTS          default 30
  MM_M_KERNELS         default 150
  MM_M                 default 8192
  MM_OP                default down_proj
"""
import csv
import os
import sys
import time
from datetime import datetime

import torch

H, INTER, VOCAB = 4096, 12288, 151936
OPS = {
    'qkv_proj':     (H,     6144),
    'o_proj':       (4096,  H),
    'gate_up_proj': (H,     24576),
    'down_proj':    (INTER, H),
    'lm_head':      (H,     VOCAB),
}

TB_M, TB_N = 128, 128


def fmt_ts(t):
    dt = datetime.fromtimestamp(t)
    return dt.strftime('%Y/%m/%d %H:%M:%S.') + f'{int(dt.microsecond/1000):03d}'


def expected_num_waves(M, K, N, n_sm):
    tiles = ((M + TB_M - 1) // TB_M) * ((N + TB_N - 1) // TB_N)
    if tiles < n_sm:
        return 1
    return tiles // n_sm


@torch.no_grad()
def main():
    gpu = int(os.environ.get('MM_GPU', '0'))
    torch.cuda.set_device(gpu)
    dev = torch.device(f'cuda:{gpu}')

    op_name = os.environ.get('MM_OP', 'down_proj')
    M       = int(os.environ.get('MM_M', '8192'))
    K, N    = OPS[op_name]

    n_bursts   = int(os.environ.get('MM_N_BURSTS', '30'))
    m_kernels  = int(os.environ.get('MM_M_KERNELS', '150'))
    burst_gap_ms = int(os.environ.get('MM_BURST_GAP_MS', '200'))
    cfg_gap_ms = int(os.environ.get('MM_CFG_GAP_MS', '500'))
    warmup_ms  = float(os.environ.get('MM_GLOBAL_WARMUP_MS', '3000'))
    seg_path   = os.environ['MM_SEGMENTS']

    first_pcts  = [int(x) for x in
                   os.environ.get('MM_WS_FIRST_PCT', '70,80').split(',')]
    first_steps = [int(x) for x in
                   os.environ.get('MM_WS_FIRST_STEP_NS',
                                  '500,2000,5000').split(',')]

    props = torch.cuda.get_device_properties(gpu)
    n_sms = props.multi_processor_count

    # MODE=baseline → only the pristine baseline binary is imported / measured.
    # MODE=ws       → only the wave-sleep binary is imported / sweep configs.
    # Both modes write into MM_SEGMENTS as a CSV append (shell script glues them).
    mode = os.environ.get('MM_MODE', 'all').lower()
    if mode not in ('baseline', 'ws', 'all'):
        raise SystemExit(f'unknown MM_MODE={mode}')

    ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ext_dir not in sys.path:
        sys.path.insert(0, ext_dir)
    ext_ws   = None
    ext_base = None
    if mode in ('ws', 'all'):
        import bf16_gemm_sm80_streamk          as _ws;   ext_ws   = _ws
    if mode in ('baseline', 'all'):
        import bf16_gemm_sm80_streamk_baseline as _base; ext_base = _base

    print(f'[ws-eval] device={gpu} ({torch.cuda.get_device_name(gpu)})  n_sms={n_sms}  mode={mode}')
    print(f'[ws-eval] op={op_name}  M={M} K={K} N={N}')
    num_waves_expected = expected_num_waves(M, K, N, n_sms)
    print(f'[ws-eval] expected num_waves = {num_waves_expected}')
    print(f'[ws-eval] N_BURSTS={n_bursts}  M_KERNELS/burst={m_kernels}')
    print(f'[ws-eval] sweep  first_pct={first_pcts}  step_ns={first_steps}')

    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0,0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0,0.02)

    # Pick the binary to warm up with — match the mode's main work.
    warm_ext = ext_base if mode == 'baseline' else ext_ws
    if warm_ext is None:
        warm_ext = ext_ws if ext_ws is not None else ext_base
    if warmup_ms > 0:
        print(f'[ws-eval] global warmup {warmup_ms:.0f}ms...')
        t0 = time.time()
        while (time.time() - t0)*1000.0 < warmup_ms:
            warm_ext.gemm_streamk(A, B)
        torch.cuda.synchronize()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    for _ in range(5): warm_ext.gemm_streamk(A, B)
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(50): warm_ext.gemm_streamk(A, B)
    ev1.record(); torch.cuda.synchronize()
    base_ms = ev0.elapsed_time(ev1)/50
    base_tf = 2.0*M*K*N/(base_ms*1e-3)/1e12
    print(f'[ws-eval] reference ms_avg={base_ms:.4f}  TFLOPS={base_tf:.1f}  '
          f'(binary used for warmup = {"baseline" if warm_ext is ext_base else "ws"})')

    # CSV: in mode != all we APPEND so shell can concatenate runs.
    os.makedirs(os.path.dirname(seg_path) or '.', exist_ok=True)
    csv_exists = os.path.exists(seg_path)
    seg_f = open(seg_path, 'a' if csv_exists else 'w', newline='')
    w = csv.writer(seg_f)
    if not csv_exists:
        w.writerow(['backend','operator','M','K','N','iters',
                    't_start','t_end','ms_avg','tflops'])
        seg_f.flush()

    # ─────────────────────────────────────────────────────────────────────
    # Run config list:
    #   "s100_step0"  → baseline binary (no wave-sleep code in SASS)
    #   "s<P>_step<S>" → wave-sleep binary, primed each kernel
    # Same N_BURSTS for both; cfg_gap_ms between runs.
    # ─────────────────────────────────────────────────────────────────────
    runs = []
    if mode in ('baseline', 'all'):
        runs.append({'cfg': 's100_step0', 'use_ws': False,
                     'thr': n_sms, 'step_ns': 0, 'max_us': 0.0,
                     'first_pct': 100})
    if mode in ('ws', 'all'):
        for fp in first_pcts:
            thr = (n_sms * fp) // 100
            for fs in first_steps:
                max_us = (n_sms - thr) * fs / 1000.0
                runs.append({'cfg': f's{fp}_step{fs}', 'use_ws': True,
                             'thr': thr, 'step_ns': fs, 'max_us': max_us,
                             'first_pct': fp})

    est_per_burst_s = m_kernels * base_ms / 1000.0
    est_total_s = len(runs) * n_bursts * (est_per_burst_s + burst_gap_ms/1000.0)
    print(f'[ws-eval] total configs={len(runs)}  bursts={len(runs)*n_bursts}  '
          f'est_time={est_total_s/60:.1f} min')
    print()
    print(f'{"cfg":>14s} {"binary":>10s} {"first%":>6s} {"step":>5s} '
          f'{"thr":>4s} {"max_us":>7s}  progress')
    print('-' * 78)

    for r in runs:
        time.sleep(cfg_gap_ms / 1000.0)
        ext = ext_base if not r['use_ws'] else ext_ws
        if ext is None:
            print(f'  SKIP {r["cfg"]}  (binary not loaded for mode={mode})')
            continue
        # Per-burst timing
        for burst_i in range(n_bursts):
            torch.cuda.synchronize()
            t0 = time.time()
            ev0.record()
            for _ in range(m_kernels):
                if r['use_ws']:
                    ext.prime_wave_sleep(num_waves_expected, n_sms,
                                         r['thr'], r['step_ns'], 0, 0)
                ext.gemm_streamk(A, B)
            ev1.record(); torch.cuda.synchronize()
            t1 = time.time()
            ms_avg = ev0.elapsed_time(ev1)/m_kernels
            tflops = 2.0*M*K*N/(ms_avg*1e-3)/1e12
            ts_, te_ = fmt_ts(t0), fmt_ts(t1)
            tag = f'stream_k:{r["cfg"]}#{burst_i}'
            w.writerow([tag, op_name, M, K, N, m_kernels,
                        ts_, te_, f'{ms_avg:.6f}', f'{tflops:.4f}'])
            seg_f.flush()
            time.sleep(burst_gap_ms / 1000.0)
        binary = 'baseline' if not r['use_ws'] else 'ws'
        print(f'{r["cfg"]:>14s} {binary:>10s} {r["first_pct"]:>6d} '
              f'{r["step_ns"]:>5d} {r["thr"]:>4d} {r["max_us"]:>5.1f}us  '
              f'{n_bursts} done', flush=True)

    seg_f.close()
    print('[ws-eval] DONE')


if __name__ == '__main__':
    main()
