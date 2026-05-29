"""
Stream-K BF16 vs cuBLAS BF16 GEMM benchmark
Transformer layer shapes × varying batch (M)
"""
import gc
import time
import torch
import bf16_gemm_sm80_streamk as streamk_ext


# ── 모델 상수 ────────────────────────────────────────────────────────────────
H        = 4096
QKV_DIM  = 6144
Q_DIM    = 4096
GU_DIM   = 24576
INTER    = 12288
VOCAB    = 151936

LAYERS = [
    ('qkv_proj',     H,     QKV_DIM),   # (M, 4096)  @ (4096, 6144)
    ('o_proj',       Q_DIM, H),         # (M, 4096)  @ (4096, 4096)
    ('gate_up_proj', H,     GU_DIM),    # (M, 4096)  @ (4096, 24576)
    ('down_proj',    INTER, H),         # (M, 12288) @ (12288, 4096)
    ('lm_head',      H,     VOCAB),     # (M, 4096)  @ (4096, 151936)
]

M_LIST = [1, 32, 128, 512, 1024, 2048, 4096, 8192,
          16384, 32768, 65536, 131072, 262144, 524288]


def tflops(ms, M, N, K):
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
    """크기별 반복 횟수 자동 조절 (총 측정 시간 ~ 100 ms 목표)"""
    flops = 2.0 * M * N * K
    if flops < 1e10:  return 200
    if flops < 1e11:  return 50
    if flops < 1e12:  return 20
    if flops < 1e13:  return 10
    return 5


def free():
    gc.collect()
    torch.cuda.empty_cache()


def run_one(name, M, K, N):
    """returns (status, ms_sk, ms_dp, ms_cb, max_err_sk)"""
    bytes_needed = (M * K + K * N + M * N) * 2  # BF16
    # 80% safety margin
    free_mem, _ = torch.cuda.mem_get_info()
    if bytes_needed > free_mem * 0.85:
        return ('skip_oom_pre', None, None, None, None,
                bytes_needed / 1e9, free_mem / 1e9)

    try:
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    except torch.cuda.OutOfMemoryError as e:
        free()
        return ('oom_alloc', None, None, None, None,
                bytes_needed / 1e9, None)

    try:
        iters = pick_iters(M, N, K)

        # 정확도 한 번 (Stream-K only)
        try:
            Cs = streamk_ext.gemm_streamk(A, B)
            Ct = torch.matmul(A, B)
            max_err = (Cs.float() - Ct.float()).abs().max().item()
            del Cs, Ct
        except Exception:
            max_err = float('nan')

        try:
            ms_sk = bench(streamk_ext.gemm_streamk, A, B, iters=iters)
        except Exception as e:
            ms_sk = None
        try:
            ms_dp = bench(streamk_ext.gemm_basicdp, A, B, iters=iters)
        except Exception:
            ms_dp = None
        try:
            ms_cb = bench(torch.matmul, A, B, iters=iters)
        except Exception:
            ms_cb = None

        return ('ok', ms_sk, ms_dp, ms_cb, max_err, None, None)
    except torch.cuda.OutOfMemoryError:
        return ('oom_run', None, None, None, None, bytes_needed/1e9, None)
    finally:
        del A, B
        free()


def fmt(ms, M, N, K):
    if ms is None: return "    fail   "
    return f"{ms:8.3f}ms {tflops(ms,M,N,K):6.1f}TF"


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    fm, tm = torch.cuda.mem_get_info()
    print(f"VRAM free / total: {fm/1e9:.1f} / {tm/1e9:.1f} GB\n")

    print(f"{'layer':14s} {'M':>7s} {'K':>6s} {'N':>7s} | "
          f"{'Stream-K':>17s} | {'Basic-DP':>17s} | {'cuBLAS':>17s} | "
          f"SK/cB  err")
    print("-" * 130)

    for name, K, N in LAYERS:
        for M in M_LIST:
            status, ms_sk, ms_dp, ms_cb, err, need_gb, free_gb = run_one(name, M, K, N)
            if status.startswith('skip') or status.startswith('oom'):
                msg = f"{status}"
                if need_gb is not None:
                    msg += f" need={need_gb:.1f}GB"
                if free_gb is not None:
                    msg += f" free={free_gb:.1f}GB"
                print(f"{name:14s} {M:7d} {K:6d} {N:7d} |   {msg}")
                continue

            ratio = (ms_cb / ms_sk) if (ms_sk and ms_cb) else float('nan')
            err_s = f"{err:.1e}" if err is not None else "  ---"
            print(f"{name:14s} {M:7d} {K:6d} {N:7d} | "
                  f"{fmt(ms_sk,M,N,K)} | {fmt(ms_dp,M,N,K)} | {fmt(ms_cb,M,N,K)} | "
                  f"{ratio:5.2f} {err_s}")
        print()


if __name__ == "__main__":
    main()
