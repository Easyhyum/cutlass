"""
Stream-K BF16 vs cuBLAS BF16 GEMM benchmark
- Transformer layer shapes × varying batch (M)
- CSV 출력 + 레이어별 그래프(TFLOPS / Speedup) 생성

사용:
  CUDA_VISIBLE_DEVICES=1 python bench_streamk_layers_csv.py
"""
import csv
import gc
import os
import torch
import bf16_gemm_sm80_streamk as streamk_ext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 모델 상수 ────────────────────────────────────────────────────────────────
H        = 4096
QKV_DIM  = 6144
Q_DIM    = 4096
GU_DIM   = 24576
INTER    = 12288
VOCAB    = 151936

LAYERS = [
    ('qkv_proj',     H,     QKV_DIM),
    ('o_proj',       Q_DIM, H),
    ('gate_up_proj', H,     GU_DIM),
    ('down_proj',    INTER, H),
    ('lm_head',      H,     VOCAB),
]

M_LIST = [1, 32, 128, 512, 1024, 2048, 4096, 8192,
          16384, 32768, 65536, 131072, 262144, 524288]

OUT_DIR = "/workspace/custum/bench_results"
CSV_PATH = os.path.join(OUT_DIR, "streamk_vs_cublas.csv")
os.makedirs(OUT_DIR, exist_ok=True)


def tflops(ms, M, N, K):
    if ms is None or ms <= 0: return None
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def bench(fn, *args, iters, warmup=5):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn(*args)
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / iters


def pick_iters(M, N, K):
    flops = 2.0 * M * N * K
    if flops < 1e10:  return 200
    if flops < 1e11:  return 50
    if flops < 1e12:  return 20
    if flops < 1e13:  return 10
    return 5


def free():
    gc.collect()
    torch.cuda.empty_cache()


def measure(layer, M, K, N):
    """returns dict with results, including 'status'"""
    rec = dict(layer=layer, M=M, K=K, N=N,
               ms_streamk=None, ms_basicdp=None, ms_cublas=None,
               tf_streamk=None, tf_basicdp=None, tf_cublas=None,
               max_err=None, status="ok",
               need_gb=None, free_gb=None)

    bytes_needed = (M * K + K * N + M * N) * 2
    free_mem, _ = torch.cuda.mem_get_info()
    rec["need_gb"] = bytes_needed / 1e9
    rec["free_gb"] = free_mem / 1e9
    if bytes_needed > free_mem * 0.85:
        rec["status"] = "skip_oom_pre"
        return rec

    try:
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    except torch.cuda.OutOfMemoryError:
        free()
        rec["status"] = "oom_alloc"
        return rec

    try:
        iters = pick_iters(M, N, K)

        try:
            Cs = streamk_ext.gemm_streamk(A, B)
            Ct = torch.matmul(A, B)
            rec["max_err"] = (Cs.float() - Ct.float()).abs().max().item()
            del Cs, Ct
        except Exception:
            pass

        try:
            rec["ms_streamk"] = bench(streamk_ext.gemm_streamk, A, B, iters=iters)
        except torch.cuda.OutOfMemoryError:
            rec["status"] = "oom_streamk"; free()
        try:
            rec["ms_basicdp"] = bench(streamk_ext.gemm_basicdp, A, B, iters=iters)
        except torch.cuda.OutOfMemoryError:
            rec["status"] = "oom_basicdp"; free()
        try:
            rec["ms_cublas"] = bench(torch.matmul, A, B, iters=iters)
        except torch.cuda.OutOfMemoryError:
            rec["status"] = "oom_cublas"; free()
    finally:
        del A, B
        free()

    rec["tf_streamk"] = tflops(rec["ms_streamk"], M, N, K)
    rec["tf_basicdp"] = tflops(rec["ms_basicdp"], M, N, K)
    rec["tf_cublas"]  = tflops(rec["ms_cublas"],  M, N, K)
    return rec


def run_all():
    results = []
    for name, K, N in LAYERS:
        for M in M_LIST:
            rec = measure(name, M, K, N)
            ratio = (rec["ms_cublas"]/rec["ms_streamk"]) \
                    if (rec["ms_cublas"] and rec["ms_streamk"]) else None
            ratio_s = f"{ratio:.2f}" if ratio else " -- "

            sk = f"{rec['tf_streamk']:6.1f}TF" if rec['tf_streamk'] else "  fail "
            dp = f"{rec['tf_basicdp']:6.1f}TF" if rec['tf_basicdp'] else "  fail "
            cb = f"{rec['tf_cublas']:6.1f}TF"  if rec['tf_cublas']  else "  fail "
            print(f"{name:14s} M={M:>7d}  SK {sk}  DP {dp}  cuBLAS {cb}  "
                  f"SK/cB={ratio_s}  status={rec['status']}")
            results.append(rec)
        print()
    return results


def write_csv(results, path):
    fields = ["layer", "M", "K", "N",
              "ms_streamk", "ms_basicdp", "ms_cublas",
              "tf_streamk", "tf_basicdp", "tf_cublas",
              "speedup_sk_over_cublas",
              "speedup_sk_over_basicdp",
              "max_err",
              "status", "need_gb", "free_gb"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = dict(r)
            sk = r["ms_streamk"]; cb = r["ms_cublas"]; dp = r["ms_basicdp"]
            row["speedup_sk_over_cublas"] = (cb / sk) if (sk and cb) else None
            row["speedup_sk_over_basicdp"] = (dp / sk) if (sk and dp) else None
            w.writerow(row)
    print(f"CSV 저장: {path}")


def plot_layer(layer, recs, out_dir):
    M  = [r["M"] for r in recs]
    sk = [r["tf_streamk"] for r in recs]
    dp = [r["tf_basicdp"] for r in recs]
    cb = [r["tf_cublas"]  for r in recs]

    K = recs[0]["K"]; N = recs[0]["N"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── (1) TFLOPS ─────────────────────────────────────────────
    def filt(xs, ys):
        return zip(*[(x, y) for x, y in zip(xs, ys) if y is not None]) \
               if any(y is not None for y in ys) else ([], [])

    xs, ys = filt(M, sk);  ax1.plot(xs, ys, "o-", label="Stream-K",  color="C0", lw=2)
    xs, ys = filt(M, dp);  ax1.plot(xs, ys, "s--", label="Basic-DP", color="C2", lw=1.5, alpha=0.8)
    xs, ys = filt(M, cb);  ax1.plot(xs, ys, "^-", label="cuBLAS",    color="C3", lw=2)

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("M (batch · seq)")
    ax1.set_ylabel("TFLOPS")
    ax1.set_title(f"{layer}  (K={K}, N={N})  BF16 throughput")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # ── (2) Speedup (SK / cuBLAS) ──────────────────────────────
    speed = []
    Ms_ok = []
    for r in recs:
        if r["ms_streamk"] and r["ms_cublas"]:
            speed.append(r["ms_cublas"] / r["ms_streamk"])
            Ms_ok.append(r["M"])
    ax2.plot(Ms_ok, speed, "o-", color="C0", lw=2, label="Stream-K / cuBLAS")
    ax2.axhline(1.0, color="black", ls=":", lw=1, label="parity")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("M")
    ax2.set_ylabel("Speedup vs cuBLAS")
    ax2.set_title(f"{layer}  speedup")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    # 색칠
    for x, y in zip(Ms_ok, speed):
        if y >= 1.0:
            ax2.plot(x, y, "o", color="C0", ms=8)
        else:
            ax2.plot(x, y, "o", color="C3", ms=8)

    fig.tight_layout()
    path = os.path.join(out_dir, f"layer_{layer}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"plot 저장: {path}")


def plot_summary(results, out_dir):
    """모든 레이어의 speedup 한 장에 비교"""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    by_layer = {}
    for r in results:
        by_layer.setdefault(r["layer"], []).append(r)

    for i, (layer, recs) in enumerate(by_layer.items()):
        Ms, speed = [], []
        for r in recs:
            if r["ms_streamk"] and r["ms_cublas"]:
                Ms.append(r["M"])
                speed.append(r["ms_cublas"] / r["ms_streamk"])
        ax.plot(Ms, speed, "o-", label=layer, lw=2)

    ax.axhline(1.0, color="black", ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("M (batch · seq)")
    ax.set_ylabel("Speedup (Stream-K / cuBLAS)")
    ax.set_title("Stream-K vs cuBLAS  —  BF16 GEMM, RTX PRO 6000 Blackwell")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_speedup.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"summary plot 저장: {path}")


def plot_tflops_summary(results, out_dir):
    """모든 레이어의 Stream-K vs cuBLAS TFLOPS 비교 (multi-panel)"""
    by_layer = {}
    for r in results:
        by_layer.setdefault(r["layer"], []).append(r)

    n = len(by_layer)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4.0))
    axes = axes.flatten()

    for ax, (layer, recs) in zip(axes, by_layer.items()):
        M = [r["M"] for r in recs]
        sk = [r["tf_streamk"] for r in recs]
        cb = [r["tf_cublas"]  for r in recs]
        dp = [r["tf_basicdp"] for r in recs]

        def filt(xs, ys):
            pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
            return zip(*pairs) if pairs else ([], [])

        xs, ys = filt(M, sk);  ax.plot(xs, ys, "o-", label="Stream-K", color="C0", lw=2)
        xs, ys = filt(M, dp);  ax.plot(xs, ys, "s--", label="Basic-DP", color="C2", lw=1.2, alpha=0.7)
        xs, ys = filt(M, cb);  ax.plot(xs, ys, "^-", label="cuBLAS",   color="C3", lw=2)

        K = recs[0]["K"]; N = recs[0]["N"]
        ax.set_xscale("log", base=2)
        ax.set_title(f"{layer}  K={K}, N={N}")
        ax.set_xlabel("M")
        ax.set_ylabel("TFLOPS")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(by_layer):]:
        ax.set_visible(False)

    fig.suptitle("BF16 GEMM throughput  —  RTX PRO 6000 Blackwell (SM120, HMMA path)")
    fig.tight_layout()
    path = os.path.join(out_dir, "summary_tflops.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"summary tflops plot 저장: {path}")


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    fm, tm = torch.cuda.mem_get_info()
    print(f"VRAM free / total: {fm/1e9:.1f} / {tm/1e9:.1f} GB\n")

    results = run_all()
    write_csv(results, CSV_PATH)

    by_layer = {}
    for r in results:
        by_layer.setdefault(r["layer"], []).append(r)
    for layer, recs in by_layer.items():
        plot_layer(layer, recs, OUT_DIR)

    plot_summary(results, OUT_DIR)
    plot_tflops_summary(results, OUT_DIR)


if __name__ == "__main__":
    main()
