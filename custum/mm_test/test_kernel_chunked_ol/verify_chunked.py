#!/usr/bin/env python3
"""
Correctness check for chunked GEMM — compares C from chunked path against
non-chunked baseline at several (M, K, N, chunk_m) configs.

Tolerance: bit-exact equality (chunking on M-axis doesn't change reduction
order for any element).

Usage:
  cd /workspace/custum/mm_test/test_kernel_chunked
  python3 verify_chunked.py
"""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'build_streamk_chunked'))
sys.path.insert(0, os.path.join(HERE, 'build_sm80_v3_chunked'))

import bf16_gemm_sm80_streamk_chunked as ext_sk
import bf16_gemm_sm80_v3_chunked      as ext_v3

torch.cuda.set_device(0)
dev = torch.device('cuda:0')

cases = [
    # (op, M, K, N, chunk_list)
    ('down_proj_8B',   1024, 12288,  4096, [1024]),
    ('down_proj_8B',   2048, 12288,  4096, [1024, 1280, 2048]),
    ('down_proj_8B',   8192, 12288,  4096, [1024, 1280, 2048]),
    ('down_proj_32B',  2048, 25600,  5120, [1024, 1280, 2048]),
    ('down_proj_32B', 16384, 25600,  5120, [1024, 1280, 2048]),
]

print(f"{'op':12s} {'M':>6s} {'cm':>5s}  {'streamk':>10s}  {'sm80_v3':>10s}  cublas")
print('-' * 80)

for op, M, K, N, chunks in cases:
    A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
    B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)

    # Baselines
    C_cublas_base = torch.matmul(A, B)
    C_sk_base = ext_sk.gemm_streamk(A, B)
    C_v3_base = ext_v3.gemm_sm80_v3(A, B)

    for cm in chunks:
        # Stream-K chunked
        C_sk_chk = ext_sk.gemm_streamk_chunked(A, B, cm, 1, -1, 0)
        # sm80_v3 chunked
        C_v3_chk = ext_v3.gemm_sm80_v3_chunked(A, B, cm, 0)
        # torch-level chunked (manual concat)
        outs = []
        for s in range(0, M, cm):
            r = min(cm, M - s)
            outs.append(torch.matmul(A[s:s+r], B))
        C_cb_chk = torch.cat(outs, dim=0)

        # Compare against baselines
        sk_exact = torch.equal(C_sk_chk, C_sk_base)
        v3_exact = torch.equal(C_v3_chk, C_v3_base)
        cb_exact = torch.equal(C_cb_chk, C_cublas_base)
        sk_mxabs = (C_sk_chk.float() - C_sk_base.float()).abs().max().item()
        v3_mxabs = (C_v3_chk.float() - C_v3_base.float()).abs().max().item()
        cb_mxabs = (C_cb_chk.float() - C_cublas_base.float()).abs().max().item()

        def tag(ok, mx): return f"OK" if ok else f"mx={mx:.2e}"
        print(f"{op:12s} {M:>6d} {cm:>5d}  "
              f"{tag(sk_exact, sk_mxabs):>10s}  "
              f"{tag(v3_exact, v3_mxabs):>10s}  "
              f"{tag(cb_exact, cb_mxabs):>10s}")

    del A, B, C_cublas_base, C_sk_base, C_v3_base
    torch.cuda.empty_cache()
