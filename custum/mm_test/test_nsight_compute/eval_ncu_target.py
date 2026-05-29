#!/usr/bin/env python3
"""
ncu target — sweeps over (kernel × op × model × M) inside ONE Python process.

Why: spinning up `ncu python3 ...` per config is slow (ncu init + cuda init +
extension load each time).  ncu can capture multiple kernels in one run, so we
push the loop into Python and use `cudaProfilerStart/Stop` (via
`torch.cuda.profiler.start/stop`) to mark the profile region, plus NVTX ranges
to label each (kernel, op, M) launch in the report.

Run with:
   ncu --profile-from-start no --nvtx --export <out> python3 eval_ncu_target.py

Env vars (set by run.sh):
  MM_KERNELS         comma list of kernels (default: cublas,stream_k,sm80_v3)
  MM_OPS             comma list of op names (default: all)
  MM_MODEL           single model filter (default: both)
  MM_M_LIST          comma list of M values (default: 1024..262144)
  MM_MEM_BUDGET_GB   skip cfg whose (A+B+C) bf16 estimate exceeds this (default 40)
  NCU_INITIAL_WARMUP one-time warmup launches per kernel (default 20)
  NCU_WARMUP_PER_SHAPE  per-shape warmup before profile (default 3)
  NCU_PROFILE        profile launches per (kernel, op, M) (default 1)
  MM_GPU             cuda device index (default 0)
"""
import os
import sys
import torch
import torch.cuda.profiler as cprof

# ── Catalogue ─────────────────────────────────────────────────────────────
# (op, model, K, N) — must match test_M_kernel_sweep
OPS = [
    ("qkv_proj",   "qwen3-8b",    4096,   6144),
    ("o_proj",     "qwen3-8b",    4096,   4096),
    ("up_proj",    "qwen3-8b",    4096,  12288),
    ("down_proj",  "qwen3-8b",   12288,   4096),
    ("lm_head",    "qwen3-8b",    4096, 151936),
    ("qkv_proj",   "qwen3-32b",   5120,  10240),
    ("o_proj",     "qwen3-32b",   8192,   5120),
    ("up_proj",    "qwen3-32b",   5120,  25600),
    ("down_proj",  "qwen3-32b",  25600,   5120),
    ("lm_head",    "qwen3-32b",   5120, 151936),
]
M_LIST_DEFAULT = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
OP_SHORT    = {"qkv_proj": "qkv", "o_proj": "o", "up_proj": "up",
               "down_proj": "down", "lm_head": "lm"}
MODEL_SHORT = {"qwen3-8b": "8b", "qwen3-32b": "32b"}

# ── Env-var filters ───────────────────────────────────────────────────────
KERNELS      = [k for k in os.environ.get('MM_KERNELS', 'cublas,stream_k,sm80_v3').split(',') if k]
OP_FILTER    = [x for x in os.environ.get('MM_OPS', '').split(',') if x]
MODEL_FILTER = os.environ.get('MM_MODEL', '').strip()
_m_env       = os.environ.get('MM_M_LIST', '').strip()
M_LIST       = [int(x) for x in _m_env.split(',') if x] if _m_env else M_LIST_DEFAULT
MEM_BUDGET   = float(os.environ.get('MM_MEM_BUDGET_GB', '40'))

INITIAL_WARMUP   = int(os.environ.get('NCU_INITIAL_WARMUP',   '20'))
WARMUP_PER_SHAPE = int(os.environ.get('NCU_WARMUP_PER_SHAPE', '3'))
PROFILE_LAUNCHES = int(os.environ.get('NCU_PROFILE',          '1'))

cuda_idx = int(os.environ.get('MM_GPU', '0'))
torch.cuda.set_device(cuda_idx)
dev = torch.device(f'cuda:{cuda_idx}')

# ── Load kernels (once) ───────────────────────────────────────────────────
sys.path.insert(0, '/workspace/custum')
kernel_fns = {}
for k in KERNELS:
    if k == 'cublas':
        kernel_fns[k] = lambda A, B: torch.matmul(A, B)
    elif k == 'stream_k':
        import bf16_gemm_sm80_streamk_baseline as ext_sk
        kernel_fns[k] = lambda A, B: ext_sk.gemm_streamk(A, B)
    elif k == 'sm80_v3':
        import cutlass_sm80_v3 as ext_v3
        kernel_fns[k] = lambda A, B: ext_v3.gemm_sm80_v3(A, B)
    else:
        raise SystemExit(f'unknown kernel: {k}')

print(f"[ncu] kernels={list(kernel_fns)}  ops_filter={OP_FILTER or '(all)'}  "
      f"model_filter={MODEL_FILTER or '(both)'}")
print(f"[ncu] M_LIST={M_LIST}  mem_budget={MEM_BUDGET}GB")
print(f"[ncu] warmup_initial={INITIAL_WARMUP}  warmup_per_shape={WARMUP_PER_SHAPE}  "
      f"profile_launches={PROFILE_LAUNCHES}")

# ── Initial warmup (profiler OFF) ─────────────────────────────────────────
print(f"[ncu] initial warmup ({INITIAL_WARMUP} × {len(kernel_fns)} kernels on 1024x1024)")
A0 = torch.empty(1024, 1024, device=dev, dtype=torch.bfloat16); A0.normal_(0, 0.02)
B0 = torch.empty(1024, 1024, device=dev, dtype=torch.bfloat16); B0.normal_(0, 0.02)
for fn in kernel_fns.values():
    for _ in range(INITIAL_WARMUP):
        fn(A0, B0)
torch.cuda.synchronize()
del A0, B0
torch.cuda.empty_cache()

# ── Sweep ─────────────────────────────────────────────────────────────────
n_profiled = 0
n_skipped  = 0
for op_name, model, K, N in OPS:
    if OP_FILTER and op_name not in OP_FILTER:
        continue
    if MODEL_FILTER and model != MODEL_FILTER:
        continue
    op_s = OP_SHORT.get(op_name, op_name)
    md_s = MODEL_SHORT.get(model, model)

    for M in M_LIST:
        est_gb = (M * K + K * N + M * N) * 2 / 1024**3
        if est_gb > MEM_BUDGET:
            print(f"[skip mem ] {op_name:10s} {model:9s} M={M:>7}  est={est_gb:.2f}GB > {MEM_BUDGET}GB")
            n_skipped += 1
            continue

        try:
            A = torch.empty(M, K, device=dev, dtype=torch.bfloat16); A.normal_(0, 0.02)
            B = torch.empty(K, N, device=dev, dtype=torch.bfloat16); B.normal_(0, 0.02)
        except torch.cuda.OutOfMemoryError:
            print(f"[skip oom ] {op_name:10s} {model:9s} M={M:>7}  est={est_gb:.2f}GB")
            n_skipped += 1
            torch.cuda.empty_cache()
            continue

        # per-shape warmup — profiler still OFF (--profile-from-start no)
        try:
            for fn in kernel_fns.values():
                for _ in range(WARMUP_PER_SHAPE):
                    fn(A, B)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            print(f"[skip warm] {op_name:10s} {model:9s} M={M:>7}  est={est_gb:.2f}GB")
            n_skipped += 1
            del A, B
            torch.cuda.empty_cache()
            continue

        # PROFILE region — turn profiler ON only here.
        # NVTX range labels each kernel in the report.
        for k, fn in kernel_fns.items():
            tag = f"{k}__{op_s}{md_s}__M{M}"
            torch.cuda.nvtx.range_push(tag)
            cprof.start()
            for _ in range(PROFILE_LAUNCHES):
                fn(A, B)
            torch.cuda.synchronize()
            cprof.stop()
            torch.cuda.nvtx.range_pop()
            n_profiled += 1

        print(f"[ncu ok  ] {op_name:10s} {model:9s} M={M:>7}  K={K} N={N} est={est_gb:.2f}GB")
        del A, B
        torch.cuda.empty_cache()

print(f"\n[ncu] DONE  profiled_launches={n_profiled}  skipped={n_skipped}")
