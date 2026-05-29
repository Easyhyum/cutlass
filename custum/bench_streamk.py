"""
Stream-K BF16 vs cuBLAS BF16 GEMM benchmark
"""
import time
import torch
import bf16_gemm_sm80_streamk as streamk_ext


def bench(fn, *args, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        out = fn(*args)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / iters
    return ms, out


def tflops(ms, M, N, K):
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def run(M, N, K, iters=50):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1

    # 정확도 비교 (작은 alpha,beta)
    Cs = streamk_ext.gemm_streamk(A, B)
    Cb = streamk_ext.gemm_basicdp(A, B)
    Ct = torch.matmul(A, B)  # cuBLAS-backed
    max_err_sk = (Cs.float() - Ct.float()).abs().max().item()
    max_err_dp = (Cb.float() - Ct.float()).abs().max().item()

    ms_sk, _ = bench(streamk_ext.gemm_streamk, A, B, iters=iters)
    ms_dp, _ = bench(streamk_ext.gemm_basicdp, A, B, iters=iters)
    ms_tb, _ = bench(torch.matmul, A, B, iters=iters)

    print(f"\n[M={M}, N={N}, K={K}]")
    print(f"  Stream-K  : {ms_sk:.4f} ms  {tflops(ms_sk,M,N,K):8.2f} TFLOPS"
          f"   max_err={max_err_sk:.2e}")
    print(f"  Basic DP  : {ms_dp:.4f} ms  {tflops(ms_dp,M,N,K):8.2f} TFLOPS"
          f"   max_err={max_err_dp:.2e}")
    print(f"  cuBLAS    : {ms_tb:.4f} ms  {tflops(ms_tb,M,N,K):8.2f} TFLOPS")
    print(f"  StreamK/cuBLAS speed = {ms_tb/ms_sk:.3f}x  "
          f"({(tflops(ms_sk,M,N,K)/tflops(ms_tb,M,N,K)*100):.1f}%)")


if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(0))
    # 타일 정렬 사이즈
    run(2048, 2048, 2048)
    run(4096, 4096, 4096)
    run(8192, 8192, 8192)
    # 비-타일-정렬 사이즈 (Stream-K 의 강점)
    run(2048, 2048, 2176)
    run(4096, 4096, 4160)
    run(1792, 4096, 4096)
