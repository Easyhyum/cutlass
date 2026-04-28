#!/usr/bin/env python3
"""
warp_time_check.py
==================
wmma_gemm_sleep_kernel_time_check 커널을 사용해 warp별 절대 start/end 시각을
여러 iteration에 걸쳐 측정하고 통계·성능·CSV를 출력합니다.

타이머: PTX globaltimer (1 ns 해상도, GPU 전역 동기화)
  - SM 간 절대 시각 비교 가능, 별도 클럭 변환 불필요

행렬 크기:
    (BS×SEQ, 12288) @ (12288, 4096)   →  M = BS×SEQ, K = 12288, N = 4096

사용법:
    python warp_time_check.py \\
        --BS 1 --SEQ 512 \\
        --iters 10 --sleep-ns 0 \\
        --device 0 --csv warp_timing.csv

CSV 구조:
    warp_id, tile_row, tile_col,
    iter0_start_ns, iter0_end_ns, iter0_elapsed_ns,
    iter1_start_ns, iter1_end_ns, iter1_elapsed_ns, ...

  첫 행 (warp_id=-1): 각 iteration의 커널 전체 시각
      start = min(모든 warp의 start), end = max(모든 warp의 end)
"""

import argparse
import csv
import sys

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 6 (hiperf) 기준 성능 측정 — CUDA Event 사용
# ─────────────────────────────────────────────────────────────────────────────
def run_hiperf_benchmark(
    M: int, N: int, K: int,
    sleep_ns: int,
    iters: int,
    device: torch.device,
    sw,
) -> dict:
    """
    gemm_hiperf (Kernel 6: BLOCK_K=64, float4, double-buf smem) 를 CUDA Event 로 측정.
    globaltimer 오버헤드 없이 정확한 커널 wall time / TFLOPS 를 반환.
    """
    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(K, N, dtype=torch.float16, device=device)
    flops = 2.0 * M * N * K

    # 워밍업 3회
    for _ in range(3):
        sw.gemm_hiperf(A, B, sleep_ns=0, sleep_freq=1)
    torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        sw.gemm_hiperf(A, B, sleep_ns=sleep_ns, sleep_freq=1)
        t1.record()
        torch.cuda.synchronize(device)
        times_ms.append(t0.elapsed_time(t1))   # ms

    t_t = torch.tensor(times_ms, dtype=torch.float64)
    tfl = [flops / (ms * 1e-3) / 1e12 for ms in times_ms]
    tfl_t = torch.tensor(tfl, dtype=torch.float64)

    return {
        "times_ms": times_ms,
        "tflops":   tfl,
        "stats": {
            "min_ms":     float(t_t.min()),
            "max_ms":     float(t_t.max()),
            "mean_ms":    float(t_t.mean()),
            "min_tflops": float(tfl_t.min()),
            "max_tflops": float(tfl_t.max()),
            "mean_tflops":float(tfl_t.mean()),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 측정 함수 (Kernel 1b — per-warp globaltimer)
# ─────────────────────────────────────────────────────────────────────────────
def run_time_check(
    M: int, N: int, K: int,
    sleep_ns: int,
    sleep_freq: int,
    iters: int,
    device: torch.device,
) -> dict:
    """
    wmma_gemm_sleep_time_check (Kernel 1b) 를 iters 회 호출하고 결과를 반환.

    반환 dict:
        starts      : list[Tensor[num_warps] int64]  – iteration별 warp start ns
        ends        : list[Tensor[num_warps] int64]  – iteration별 warp end ns
        elapsed     : list[Tensor[num_warps] int64]  – iteration별 warp elapsed ns
        kernel_ns   : list[float]                    – iteration별 커널 전체 소요 ns
        tflops      : list[float]                    – iteration별 TFLOPS
        warp_stats  : dict                           – 전체 warp elapsed 통계 (ns)
        kernel_stats: dict                           – kernel_ns 통계
    """
    import sleep_wmma as sw   # main() 에서 이미 확인됨

    A = torch.randn(M, K, dtype=torch.float16, device=device)
    B = torch.randn(K, N, dtype=torch.float16, device=device)

    flops = 2.0 * M * N * K   # GEMM FLOPs

    # 워밍업 (1회)
    sw.gemm_time_check(A, B, sleep_ns=0, sleep_freq=1)
    torch.cuda.synchronize(device)

    starts_list:  list[torch.Tensor] = []
    ends_list:    list[torch.Tensor] = []
    elapsed_list: list[torch.Tensor] = []
    kernel_ns_list: list[float]      = []
    tflops_list:    list[float]      = []

    for _ in range(iters):
        _, start_buf, end_buf = sw.gemm_time_check(
            A, B, sleep_ns=sleep_ns, sleep_freq=sleep_freq
        )
        torch.cuda.synchronize(device)

        start_cpu   = start_buf.cpu()   # int64 ns (절대 시각)
        end_cpu     = end_buf.cpu()     # int64 ns (절대 시각)
        elapsed_cpu = end_cpu - start_cpu

        # 커널 전체 시간: 가장 빠른 start → 가장 늦은 end
        k_start = int(start_cpu.min())
        k_end   = int(end_cpu.max())
        k_ns    = k_end - k_start

        starts_list.append(start_cpu)
        ends_list.append(end_cpu)
        elapsed_list.append(elapsed_cpu)
        kernel_ns_list.append(float(k_ns))
        tflops_list.append(flops / (k_ns * 1e-9) / 1e12)

    # 전체 warp elapsed 통계 (모든 iter 합산)
    all_elapsed = torch.cat(elapsed_list).double()
    q = torch.tensor([0.5, 0.95, 0.99], dtype=torch.float64)
    qv = torch.quantile(all_elapsed, q)

    warp_stats = {
        "num_warps": int(elapsed_list[0].numel()),
        "min_ns":    float(all_elapsed.min()),
        "max_ns":    float(all_elapsed.max()),
        "mean_ns":   float(all_elapsed.mean()),
        "std_ns":    float(all_elapsed.std()),
        "p50_ns":    float(qv[0]),
        "p95_ns":    float(qv[1]),
        "p99_ns":    float(qv[2]),
    }

    kns_t = torch.tensor(kernel_ns_list, dtype=torch.float64)
    kernel_stats = {
        "min_ns":  float(kns_t.min()),
        "max_ns":  float(kns_t.max()),
        "mean_ns": float(kns_t.mean()),
        "std_ns":  float(kns_t.std()) if iters > 1 else 0.0,
    }

    return {
        "starts":       starts_list,
        "ends":         ends_list,
        "elapsed":      elapsed_list,
        "kernel_ns":    kernel_ns_list,
        "tflops":       tflops_list,
        "warp_stats":   warp_stats,
        "kernel_stats": kernel_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 성능 출력
# ─────────────────────────────────────────────────────────────────────────────
def print_hiperf_report(hp: dict, M: int, N: int, K: int, sleep_ns: int):
    """Kernel 6 (hiperf) CUDA Event 측정 결과 출력."""
    s = hp["stats"]
    iters = len(hp["times_ms"])
    print("=" * 70)
    print("  [Kernel 6 — hiperf (BLOCK_K=64, float4, double-buf smem)]")
    print(f"  ({M}, {K}) @ ({K}, {N})   sleep_ns={sleep_ns}  iters={iters}")
    print(f"  {'iter':>6}  {'wall_ms':>10}  {'TFLOPS':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}")
    for i, (ms, tfl) in enumerate(zip(hp["times_ms"], hp["tflops"])):
        print(f"  {i:>6}  {ms:>10.4f}  {tfl:>10.2f}")
    print()
    print(f"  wall time  min={s['min_ms']:.4f} ms  max={s['max_ms']:.4f} ms"
          f"  mean={s['mean_ms']:.4f} ms")
    print(f"  TFLOPS     min={s['min_tflops']:.2f}  max={s['max_tflops']:.2f}"
          f"  mean={s['mean_tflops']:.2f}")
    print("=" * 70)
    print()


def print_report(result: dict, M: int, N: int, K: int, sleep_ns: int):
    ws  = result["warp_stats"]
    ks  = result["kernel_stats"]
    iters = len(result["kernel_ns"])

    print("=" * 70)
    print("  [Kernel 1 — simple global-mem WMMA (per-warp globaltimer)]")
    print(f"  NOTE: smem/double-buf 없음 → TFLOPS 낮은 것이 정상 (profiling 전용)")
    print(f"  ({M}, {K}) @ ({K}, {N})   sleep_ns={sleep_ns}  iters={iters}")
    print(f"  num_warps/iter : {ws['num_warps']}")
    print("=" * 70)

    # ── 커널 전체 성능 (iteration별) ─────────────────────────────────────
    print(f"  {'[Kernel Wall Time / TFLOPS per iteration]'}")
    print(f"  {'iter':>6}  {'wall_ns':>12}  {'wall_us':>10}  {'TFLOPS':>10}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}")
    for i, (kns, tfl) in enumerate(zip(result["kernel_ns"], result["tflops"])):
        print(f"  {i:>6}  {kns:>12.1f}  {kns/1e3:>10.2f}  {tfl:>10.4f}")

    print()
    tfl_t = torch.tensor(result["tflops"], dtype=torch.float64)
    print(f"  kernel wall time  min={ks['min_ns']:.1f} ns  max={ks['max_ns']:.1f} ns"
          f"  mean={ks['mean_ns']:.1f} ns  std={ks['std_ns']:.1f} ns")
    print(f"  TFLOPS            min={float(tfl_t.min()):.4f}  max={float(tfl_t.max()):.4f}"
          f"  mean={float(tfl_t.mean()):.4f}  std={float(tfl_t.std()) if iters>1 else 0.0:.4f}")

    # ── 전체 warp elapsed 통계 ─────────────────────────────────────────
    print()
    print("  [Warp Elapsed Time  (all iters aggregated)]")
    print(f"  {'min':<8}: {ws['min_ns']:>12.1f} ns")
    print(f"  {'max':<8}: {ws['max_ns']:>12.1f} ns")
    print(f"  {'mean':<8}: {ws['mean_ns']:>12.1f} ns")
    print(f"  {'std':<8}: {ws['std_ns']:>12.1f} ns")
    print(f"  {'p50':<8}: {ws['p50_ns']:>12.1f} ns")
    print(f"  {'p95':<8}: {ws['p95_ns']:>12.1f} ns")
    print(f"  {'p99':<8}: {ws['p99_ns']:>12.1f} ns")

    # ── 첫 번째 iteration 기준 top/bottom warp ─────────────────────────
    tilesN = N // 16
    first_elapsed = result["elapsed"][0].double()
    print()
    topk_val, topk_idx = torch.topk(first_elapsed, k=min(5, len(first_elapsed)))
    print("  [top-5 slowest warps  (iter 0)]")
    for rank, (idx, v) in enumerate(zip(topk_idx.tolist(), topk_val.tolist()), 1):
        print(f"    #{rank}  warp={idx:>6}  tile=({idx//tilesN:>4},{idx%tilesN:>4})"
              f"  {v:>10.1f} ns")

    botk_val, botk_idx = torch.topk(first_elapsed, k=min(5, len(first_elapsed)), largest=False)
    print("  [top-5 fastest warps  (iter 0)]")
    for rank, (idx, v) in enumerate(zip(botk_idx.tolist(), botk_val.tolist()), 1):
        print(f"    #{rank}  warp={idx:>6}  tile=({idx//tilesN:>4},{idx%tilesN:>4})"
              f"  {v:>10.1f} ns")

    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CSV 저장
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(result: dict, N_tiles: int, path: str):
    """
    CSV 컬럼:
        warp_id, tile_row, tile_col,
        iter0_start_ns, iter0_end_ns, iter0_elapsed_ns,
        iter1_start_ns, iter1_end_ns, iter1_elapsed_ns, ...

    첫 행 (warp_id=-1):
        각 iteration의 커널 전체 시각
        start = min(모든 warp start), end = max(모든 warp end), elapsed = end-start
    """
    iters       = len(result["starts"])
    starts_list = result["starts"]
    ends_list   = result["ends"]
    elapsed_list= result["elapsed"]
    num_warps   = int(starts_list[0].numel())

    # ── 헤더 ──────────────────────────────────────────────────────────
    header = ["warp_id", "tile_row", "tile_col"]
    for i in range(iters):
        header += [f"iter{i}_start_ns", f"iter{i}_end_ns", f"iter{i}_elapsed_ns"]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # ── warp_id=-1: 커널 전체 요약 행 (맨 위) ─────────────────────
        row_summary = [-1, "-", "-"]
        for i in range(iters):
            k_start = int(starts_list[i].min())
            k_end   = int(ends_list[i].max())
            row_summary += [k_start, k_end, k_end - k_start]
        writer.writerow(row_summary)

        # ── warp별 행 ──────────────────────────────────────────────────
        for wid in range(num_warps):
            tile_row = wid // N_tiles
            tile_col = wid % N_tiles
            row = [wid, tile_row, tile_col]
            for i in range(iters):
                s = int(starts_list[i][wid])
                e = int(ends_list[i][wid])
                row += [s, e, e - s]
            writer.writerow(row)

    print(f"  CSV saved → {path}  ({num_warps} warps × {iters} iters)")


# ─────────────────────────────────────────────────────────────────────────────
# 히스토그램 (optional)
# ─────────────────────────────────────────────────────────────────────────────
def save_histogram(elapsed_ns: torch.Tensor, path: str, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns = elapsed_ns.double()
        data = ns.numpy()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.axvline(float(ns.mean()), color="red", linestyle="--",
                   label=f"mean={float(ns.mean()):.0f} ns")
        ax.axvline(float(torch.quantile(ns, 0.99)), color="orange",
                   linestyle=":", label=f"p99={float(torch.quantile(ns, 0.99)):.0f} ns")
        ax.set_xlabel("elapsed time (ns)  [globaltimer]")
        ax.set_ylabel("warp count")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  histogram saved → {path}")
    except ImportError:
        print("  [skip] matplotlib 없음 – 히스토그램 저장 생략")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="warp-level WMMA timing check (globaltimer)\n"
                    "행렬: (BS×SEQ, 12288) @ (12288, 4096)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--BS",  type=int, default=1,   help="배치 크기 (default: 1)")
    p.add_argument("--SEQ", type=int, default=512, help="시퀀스 길이 (default: 512)")
    p.add_argument("--iters", type=int, default=10,
                   help="측정 반복 횟수 (default: 10, 워밍업 1회 별도)")
    p.add_argument("--sleep-ns",   type=int, default=0,
                   help="__nanosleep duration (ns)")
    p.add_argument("--sleep-freq", type=int, default=1,
                   help="sleep every N mma ops (unused in kernel 1)")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--csv",  type=str, default="warp_timing.csv",
                   help="CSV 저장 경로 (기본: warp_timing.csv). 빈 문자열이면 저장 안 함")
    p.add_argument("--hist", type=str, default="",
                   help="히스토그램 저장 경로 (e.g. hist.png). 비워두면 저장 안 함")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.device}")

    # (BS×SEQ, 12288) @ (12288, 4096)
    M = args.BS * args.SEQ
    K = 12288
    N = 4096

    print(f"  shape : ({args.BS} × {args.SEQ} = {M}, {K}) @ ({K}, {N})")
    print()

    try:
        import sleep_wmma as sw
    except ImportError:
        sys.exit(
            "[error] sleep_wmma 모듈을 찾을 수 없습니다.\n"
            "  python setup_wmma_sleep.py build_ext --inplace 을 먼저 실행하세요."
        )

    # ── Kernel 6 (hiperf) 기준 성능 먼저 출력 ──────────────────────────
    hp = run_hiperf_benchmark(
        M=M, N=N, K=K,
        sleep_ns=args.sleep_ns,
        iters=args.iters,
        device=device,
        sw=sw,
    )
    print_hiperf_report(hp, M, N, K, args.sleep_ns)

    # ── Kernel 1b (per-warp globaltimer) ───────────────────────────────
    result = run_time_check(
        M=M, N=N, K=K,
        sleep_ns=args.sleep_ns,
        sleep_freq=args.sleep_freq,
        iters=args.iters,
        device=device,
    )

    print_report(result, M, N, K, args.sleep_ns)

    if args.csv:
        save_csv(result, N_tiles=N // 16, path=args.csv)

    if args.hist:
        all_elapsed = torch.cat(result["elapsed"])
        title = (f"WMMA warp elapsed  (BS={args.BS} SEQ={args.SEQ}) M={M} K={K} N={N}"
                 f"  sleep={args.sleep_ns}ns  iters={args.iters}")
        save_histogram(all_elapsed, args.hist, title)


if __name__ == "__main__":
    main()
