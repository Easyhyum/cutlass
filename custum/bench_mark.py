#!/usr/bin/env python3
"""
Benchmark WMMA GEMM simulating ALL matmul operations in a single
Qwen3-8B forward pass, with varying __nanosleep durations.

Qwen3-8B architecture (from HuggingFace config):
    hidden_size        = 4096
    intermediate_size  = 12288
    num_attention_heads= 32
    num_key_value_heads= 8
    head_dim           = 128
    num_hidden_layers  = 36
    vocab_size         = 151936

Per-layer matmuls during PREFILL (batch B, sequence length S):
    Linear projections use M = B×S (all tokens flattened):
    1. Q   projection : (B*S, 4096) @ (4096, 4096)   → (B*S, 4096)
    2. K   projection : (B*S, 4096) @ (4096, 1024)   → (B*S, 1024)
    3. V   projection : (B*S, 4096) @ (4096, 1024)   → (B*S, 1024)
    4. Out projection : (B*S, 4096) @ (4096, 4096)   → (B*S, 4096)
    5. Gate projection: (B*S, 4096) @ (4096, 12288)  → (B*S, 12288)
    6. Up   projection: (B*S, 4096) @ (4096, 12288)  → (B*S, 12288)
    7. Down projection: (B*S, 12288)@ (12288, 4096)  → (B*S, 4096)

    Attention (BMM, per-sequence per-head):
    8. QK^T (per head): (S, 128)  @ (128, S)   → (S, S)    ×B×32 heads
    9. Attn·V (per hd): (S, S)    @ (S, 128)   → (S, 128)  ×B×32 heads

    Final (once):
   10. LM head        : (B*S, 4096) @ (4096, 151936) → (B*S, 151936)

Usage:
    cd /path/to/vllm-power-test/test
    python setup_wmma_sleep.py build_ext --inplace
    python benchmark_wmma_qwen3.py --device 3 --output-csv logs/qwen3_8b_forward.csv
"""

import argparse
import time
import os
import csv
import subprocess
import sys
from collections import defaultdict
import multiprocessing as mp
import torch
import pynvml
import ctypes

from gpu_profile import GPUMonitor
# ─────────────────── Qwen3-8B architecture constants ───────────────────
HIDDEN       = 4096
INTER        = 12288
NUM_HEADS    = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
NUM_LAYERS   = 36
ACTUAL_FORWARD_LAYERS = 32
VOCAB_SIZE   = 151936   # 151936 = 9496 × 16, WMMA-safe

device_id = 3
# monitor_graph = GPUMonitor(device_id=device_id)
def set_specific_clock(handle, TARGET_MEM_CLOCK = 9001, TARGET_SM_CLOCK = 1650, device_index=0):
    try:
        # device_name = pynvml.nvmlDeviceGetName(handle)
        
        # print(f"=== Configuring GPU: {device_name} ===")
        print(f"Target Memory Clock: {TARGET_MEM_CLOCK} MHz")
        print(f"Target SM Clock    : {TARGET_SM_CLOCK} MHz")
        try:
            applied_clocks = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)
            print(f"--> application Clock Setting: {applied_clocks} MHz")
        except Exception as e:
            print(f"--> application Clock Setting: Error - {e}")

        # 1. 클럭 설정 (Root 권한 필요)
        # pynvml.nvmlDeviceSetApplicationsClocks(handle, TARGET_MEM_CLOCK, TARGET_SM_CLOCK)
        
        # print("\n✅ Successfully set application clocks!")

        # 2. 적용 확인
        # 주의: 설정 직후에는 부하가 없으면 클럭이 낮게 보일 수 있습니다 (P-State 절전).
        # 확실한 확인을 위해 설정을 '조회'합니다.
        # applied_clocks = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)
        # print(f"--> After Application Clock Setting: {applied_clocks} MHz")

        # pynvml.nvmlDeviceSetMemoryLockedClocks(handle, TARGET_MEM_CLOCK, TARGET_MEM_CLOCK)
        # print("\n✅ Successfully set memory locked clocks!")

        pynvml.nvmlDeviceSetGpuLockedClocks(handle, TARGET_SM_CLOCK-2, TARGET_SM_CLOCK+10)
        print("\n✅ Successfully set sm locked clocks!")

        # min_clock, max_clock = pynvml.nvmlDeviceGetGpuLockedClocks(handle)
        # print(f"🔒 Locked Setting (Range): {min_clock} MHz ~ {max_clock} MHz")

        # Memory Clock Information
        try:
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            print(f"Memory Clock: {mem_clock} MHz")
        except Exception as e:
            print(f"Memory Clock: Error - {e}")

        # SM Clock Information
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            print(f"SM Clock: {sm_clock} MHz")
        except Exception as e:
            print(f"SM Clock: Error - {e}")
        

        
        clock_changed = True
    except pynvml.NVMLError as e:
        import traceback
        traceback.print_exc()
        raise SystemExit(f"Critical NVML Error: {e}")
def build_layer_matmuls(B, S):
    """Return list of (M, K, N, name, count_per_forward) for Qwen3-8B prefill.

    B: batch size
    S: sequence length (rounded up to multiple of 16)
    """
    q_dim  = NUM_HEADS    * HEAD_DIM   # 4096
    kv_dim = NUM_KV_HEADS * HEAD_DIM   # 1024

    # Round up to multiple of 16 for WMMA compatibility
    BS = ((B * S + 15) // 16) * 16     # flattened token count
    S_aligned = ((S + 15) // 16) * 16  # per-sequence length

    # Linear projections: M = B*S  (×NUM_LAYERS each)
    linear_per_layer = [
        # (BS, HIDDEN, q_dim,  "attn_q_proj"),
        # (BS, HIDDEN, kv_dim, "attn_k_proj"),
        # (BS, HIDDEN, kv_dim, "attn_v_proj"),
        # (BS, HIDDEN, q_dim,  "attn_o_proj"),
        # (BS, HIDDEN, q_dim + kv_dim * 2, "attn_qkv_proj"),
        # (BS, HIDDEN, INTER,  "mlp_gate_proj"),
        (BS, HIDDEN, INTER,  "mlp_up_proj"),
        # (BS, INTER,  HIDDEN, "mlp_down_proj"),
    ]
    matmuls = [(m, k, n, name, NUM_LAYERS) for m, k, n, name in linear_per_layer]

    # Attention BMM (per-sequence, per-head): count = B × NUM_HEADS × NUM_LAYERS
    # attn_count = B * NUM_HEADS * NUM_LAYERS
    # matmuls.append((S_aligned, HEAD_DIM, S_aligned, "attn_qk", attn_count))
    # matmuls.append((S_aligned, S_aligned, HEAD_DIM, "attn_av", attn_count))

    # LM head: once
    # matmuls.append((BS, HIDDEN, VOCAB_SIZE, "lm_head", 1))
    return matmuls


def build_qwen_forward_linear_matmuls(B, S, include_lm_head=True,
                                      num_layers=NUM_LAYERS):
    """Qwen3-8B forward에서 FlashAttention 자체를 제외한 linear matmul 목록.

    Attention Q/K/V/O projections and MLP projections are matrix multiplications;
    QK/AV attention BMM은 FlashAttention 측정값으로 따로 더한다.
    """
    q_dim = NUM_HEADS * HEAD_DIM
    kv_dim = NUM_KV_HEADS * HEAD_DIM
    BS = ((B * S + 15) // 16) * 16

    per_layer = [
        (BS, HIDDEN, q_dim, "attn_q_proj", num_layers),
        (BS, HIDDEN, kv_dim, "attn_k_proj", num_layers),
        (BS, HIDDEN, kv_dim, "attn_v_proj", num_layers),
        (BS, HIDDEN, q_dim, "attn_o_proj", num_layers),
        (BS, HIDDEN, INTER, "mlp_gate_proj", num_layers),
        (BS, HIDDEN, INTER, "mlp_up_proj", num_layers),
        (BS, INTER, HIDDEN, "mlp_down_proj", num_layers),
    ]
    if include_lm_head:
        per_layer.append((BS, HIDDEN, VOCAB_SIZE, "lm_head", 1))
    return per_layer


def matmul_flops(rows):
    return sum(2.0 * m * k * n * count for m, k, n, _name, count in rows)


def qwen_flash_attention_flops(B, S, num_layers=NUM_LAYERS):
    """Approximate QK^T + Attn*V FLOPs for all Qwen3-8B layers."""
    s_aligned = ((S + 15) // 16) * 16
    per_layer = 4.0 * B * NUM_HEADS * s_aligned * s_aligned * HEAD_DIM
    return per_layer * num_layers


def _sdpa_flash(q, k, v):
    """Run SDPA with FlashAttention backend only."""
    import torch.nn.functional as F

    with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False):
        try:
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True)
        except TypeError:
            if k.size(1) != q.size(1):
                repeat = q.size(1) // k.size(1)
                k = k.repeat_interleave(repeat, dim=1).contiguous()
                v = v.repeat_interleave(repeat, dim=1).contiguous()
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def make_qwen3_kernel_forward_fn(B, S, matmul_kernel, sleep_ns, sleep_freq,
                                 cutlass_sm80, persistent_sm80, args):
    """Actual CUDA forward kernel sequence for Qwen3-8B block stack.

    This uses synthetic weights/activations but executes the forward kernels in
    model order: Q/K/V projections, FlashAttention, O projection, MLP gate/up/down
    for ACTUAL_FORWARD_LAYERS. LM head is intentionally excluded because full prefill logits
    for B*S by VOCAB_SIZE are impractically large for the current benchmark shape.
    """
    q_dim = NUM_HEADS * HEAD_DIM
    kv_dim = NUM_KV_HEADS * HEAD_DIM
    BS = ((B * S + 15) // 16) * 16
    import torch.nn.functional as F

    hidden = torch.randn(BS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    weights = {
        "q": torch.randn(HIDDEN, q_dim, device="cuda", dtype=torch.bfloat16),
        "k": torch.randn(HIDDEN, kv_dim, device="cuda", dtype=torch.bfloat16),
        "v": torch.randn(HIDDEN, kv_dim, device="cuda", dtype=torch.bfloat16),
        "o": torch.randn(q_dim, HIDDEN, device="cuda", dtype=torch.bfloat16),
        "gate": torch.randn(HIDDEN, INTER, device="cuda", dtype=torch.bfloat16),
        "up": torch.randn(HIDDEN, INTER, device="cuda", dtype=torch.bfloat16),
        "down": torch.randn(INTER, HIDDEN, device="cuda", dtype=torch.bfloat16),
    }

    def _mm(a, b):
        if matmul_kernel == "cublas":
            return torch.mm(a, b)
        if matmul_kernel == "cutlass_sm80":
            if cutlass_sm80 is None:
                raise RuntimeError("cutlass_sm80 module is not available")
            return cutlass_sm80.gemm_sm80_v3(
                a.contiguous(), b.contiguous(),
                sleep_ns=sleep_ns, sleep_freq=sleep_freq)
        if matmul_kernel == "persistent_cta":
            if persistent_sm80 is None:
                raise RuntimeError("bf16_gemm_sm80_persistent module is not available")
            return persistent_sm80.gemm_sm80_persistent(
                a.contiguous(), b.contiguous(),
                sleep_ns, sleep_freq,
                args.persistent_throttle_mode,
                args.persistent_ctas_per_sm,
                args.persistent_chunk_tiles)
        raise ValueError(f"unknown matmul kernel: {matmul_kernel}")

    def _forward_once():
        x = hidden
        for _ in range(ACTUAL_FORWARD_LAYERS):
            q = _mm(x, weights["q"]).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = _mm(x, weights["k"]).view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = _mm(x, weights["v"]).view(B, S, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            attn = _sdpa_flash(q, k, v).transpose(1, 2).contiguous().view(BS, q_dim)
            _ = _mm(attn, weights["o"])

            gate = _mm(x, weights["gate"])
            up = _mm(x, weights["up"])
            mlp = F.silu(gate).mul_(up)
            _ = _mm(mlp, weights["down"])
        return x

    return _forward_once


def qwen3_forward_kernel_flops(B, S, include_lm_head=False):
    linear = matmul_flops(
        build_qwen_forward_linear_matmuls(
            B, S,
            include_lm_head=include_lm_head,
            num_layers=ACTUAL_FORWARD_LAYERS))
    return linear + qwen_flash_attention_flops(
        B, S, num_layers=ACTUAL_FORWARD_LAYERS)


def _fmt_ratio(v, base):
    return f"{(v / base):.2f}×" if base and base > 0 else "n/a"


def _fmt_sm_delta(v, base):
    if not base or base <= 0:
        return "n/a"
    return f"{(v / base):.2f}×, Δ{(v - base):+.0f}MHz"


def _cutlass_cmp_parts(gflops, sm_clock_mhz, cutlass_ref, label):
    if cutlass_ref is None:
        return []
    return [
        f"vs {label}: {_fmt_ratio(gflops, cutlass_ref['gflops'])}",
        f"SM/{label}: {_fmt_sm_delta(sm_clock_mhz, cutlass_ref['sm_clock_mhz'])}",
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  CUTLASS SM80 GEMM: M축 chunk + async multi-stream wrapper
# ─────────────────────────────────────────────────────────────────────────────
#  M (= batch_size × seq_len) 이 m_chunk(=2048) 보다 크면 cutlass 커널을
#  한 번에 띄우지 않고 M축을 m_chunk 단위로 잘라 각 chunk 를 별도 CUDA
#  stream 에서 비동기로 실행한 뒤 하나의 출력 버퍼로 합친다.
#  - 각 sub-stream 은 호출 직전 current stream 의 작업(A,B 준비)을 wait
#  - 모든 chunk 가 끝난 뒤 current stream 이 모든 sub-stream 을 wait
#  - 출력은 (M,N) 으로 한 번 사전 할당하고 각 stream 이 자기 슬라이스만
#    채워 넣음 → 별도의 cat copy 없이 view 슬라이스로 자연스럽게 결합
#  - stream 은 (device, n_chunks) 별로 persistent pool 에 캐싱
#  ※ cutlass_sm80.gemm_sm80_v3 는 set_sleep_params 가 __constant__ 메모리에
#    값을 쓰는 구조라 chunk 마다 다른 sleep 값을 섞으면 race 가능. 모든
#    chunk 가 같은 (sleep_ns, sleep_freq) 일 때만 사용한다 (이 함수도 그렇게
#    동작한다).
_CUTLASS_MULTISTREAM_POOL: dict = {}


def _get_cutlass_streams(device: torch.device, n: int):
    """device 별 persistent CUDA stream pool 에서 최소 n 개를 돌려준다."""
    key = (device.type, device.index if device.index is not None else -1)
    pool = _CUTLASS_MULTISTREAM_POOL.setdefault(key, [])
    while len(pool) < n:
        pool.append(torch.cuda.Stream(device=device))
    return pool[:n]


def cutlass_sm80_gemm_multistream(A: torch.Tensor,
                                  B: torch.Tensor,
                                  cutlass_sm80,
                                  sleep_ns: int = 0,
                                  sleep_freq: int = 0,
                                  m_chunk: int = 2048,
                                  out: "torch.Tensor | None" = None) -> torch.Tensor:
    """cutlass_sm80.gemm_sm80_v3 을 M-축 chunk + async multi-stream 으로 실행.

    A: (M,K) BF16, B: (K,N) BF16. M 이 m_chunk 이하이면 그냥 단일 호출로
    fast-path 처리하고, 그 외에는 ceil(M/m_chunk) 개의 chunk 로 나눠 각자
    별도 CUDA stream 에서 동시에 cutlass 커널을 띄운 뒤 (M,N) 출력 텐서에
    슬라이스 단위로 결과를 모은다. 반환값은 그 (M,N) 텐서 하나.
    """
    if cutlass_sm80 is None:
        raise RuntimeError("cutlass_sm80 module is not available")
    if A.dim() != 2 or B.dim() != 2:
        raise ValueError(
            f"expected 2D tensors, got A={tuple(A.shape)} B={tuple(B.shape)}")
    M, K = A.shape
    Kb, N = B.shape
    if K != Kb:
        raise ValueError(f"K mismatch: A.K={K} vs B.K={Kb}")
    if m_chunk <= 0:
        raise ValueError(f"m_chunk must be positive, got {m_chunk}")

    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

    if M <= m_chunk:
        return cutlass_sm80.gemm_sm80_v3(
            A, B, sleep_ns=sleep_ns, sleep_freq=sleep_freq)

    if out is None:
        out = torch.empty((M, N), dtype=A.dtype, device=A.device)
    else:
        if (tuple(out.shape) != (M, N)
                or out.dtype != A.dtype
                or out.device != A.device):
            raise ValueError(
                f"out mismatch: want ({M},{N}) {A.dtype} {A.device}, "
                f"got {tuple(out.shape)} {out.dtype} {out.device}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    num_chunks = (M + m_chunk - 1) // m_chunk
    streams = _get_cutlass_streams(A.device, num_chunks)
    current = torch.cuda.current_stream(A.device)

    for s in streams[:num_chunks]:
        s.wait_stream(current)

    for i in range(num_chunks):
        start = i * m_chunk
        end = min(start + m_chunk, M)
        s = streams[i]
        A_chunk = A[start:end]            # row-major contiguous view
        out_slice = out[start:end]        # (chunk, N) 출력 슬라이스
        with torch.cuda.stream(s):
            C_chunk = cutlass_sm80.gemm_sm80_v3(
                A_chunk, B,
                sleep_ns=sleep_ns, sleep_freq=sleep_freq)
            out_slice.copy_(C_chunk, non_blocking=True)

    for s in streams[:num_chunks]:
        current.wait_stream(s)

    return out


def measure_qwen3_actual_forward(B, S, matmul_kernel, sleep_ns, sleep_freq,
                                 cutlass_sm80, persistent_sm80, handle, args,
                                 cmp_parts=None):
    """Measure an actual synthetic Qwen3-8B forward kernel sequence."""
    try:
        cmp_parts = cmp_parts or []
        fn = make_qwen3_kernel_forward_fn(
            B, S, matmul_kernel, sleep_ns, sleep_freq,
            cutlass_sm80, persistent_sm80, args)
        # Warmup once to pay allocator/kernel selection cost outside measurement.
        fn()
        torch.cuda.synchronize()
        ni = 1
        el, en, pw, sm_clk, temp, pv_bef, ref_bef, pv_dur, ref_dur, pv_rat, _cap, _samples, gpu_ms = measure(
            fn, ni, handle, sample_interval=args.nvml_interval)
        flops = qwen3_forward_kernel_flops(B, S, include_lm_head=False)
        timing_ms = gpu_ms if gpu_ms > 0 else el
        gflops = flops * ni / (timing_ms / 1000.0) / 1e9
        nj = en * 1e6 / (flops * ni) if ni > 0 else 0.0
        print(
            f"  {'actual_forward':14s} | {matmul_kernel:12s} "
            f"sleep={sleep_ns:6d}ns freq={sleep_freq:4d} iters={ni:3d} | "
            f"wall={el:8.1f}ms gpu={gpu_ms:8.1f}ms "
            f"{gflops:8.1f}GF {pw:6.1f}W "
            f"{en:8.0f}mJ {nj:.3f}nJ/F  SM={sm_clk:.0f}MHz  T={temp:.0f}°C "
            f"PVrun={pv_dur}ns/{ref_dur}ns ({pv_rat * 100:.1f}% NVML ref)"
            f"{'  [' + ' | '.join(cmp_parts) + ']' if cmp_parts else ''}")
        return {
            "batch_size": B,
            "seq_len": S,
            "num_layers": ACTUAL_FORWARD_LAYERS,
            "matmul_kernel": matmul_kernel,
            "sleep_ns": sleep_ns,
            "sleep_freq": sleep_freq,
            "attention_kernel": "flashattention_sdpa",
            "num_iters": ni,
            "elapsed_ms": el,
            "gpu_elapsed_ms": gpu_ms,
            "energy_mj": en,
            "power_w": pw,
            "sm_clock_mhz": sm_clk,
            "temp_c": temp,
            "gflops": gflops,
            "nj_per_flop": nj,
            "total_flops": flops,
            "power_violation_before_ns": pv_bef,
            "reference_time_before_ns": ref_bef,
            "power_violation_during_ns": pv_dur,
            "reference_time_during_ns": ref_dur,
            "power_violation_ratio": pv_rat,
        }
    except Exception as e:
        print(
            f"  ⚠ actual forward skipped: kernel={matmul_kernel} "
            f"sleep={sleep_ns} freq={sleep_freq}: {e}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def qwen3_mlp_graph_flops(B, S, num_layers=ACTUAL_FORWARD_LAYERS):
    """FLOPs for (mlp_up -> mlp_gate -> mlp_down) repeated num_layers times."""
    bs = ((B * S + 15) // 16) * 16
    return 3.0 * 2.0 * bs * HIDDEN * INTER * num_layers


def make_qwen3_mlp_graph_replay_fn(B, S, matmul_kernel, sleep_ns, sleep_freq,
                                   cutlass_sm80, persistent_sm80, args):
    """Capture (mlp_up -> mlp_gate -> mlp_down) x ACTUAL_FORWARD_LAYERS as one CUDA Graph."""
    BS = ((B * S + 15) // 16) * 16
    hidden = torch.randn(BS, HIDDEN, device="cuda", dtype=torch.bfloat16)
    weights = {
        "up": torch.randn(HIDDEN, INTER, device="cuda", dtype=torch.bfloat16),
        "gate": torch.randn(HIDDEN, INTER, device="cuda", dtype=torch.bfloat16),
        "down": torch.randn(INTER, HIDDEN, device="cuda", dtype=torch.bfloat16),
    }

    def _mm(a, b):
        if matmul_kernel == "cublas":
            return torch.mm(a, b)
        if matmul_kernel == "cutlass_sm80":
            if cutlass_sm80 is None:
                raise RuntimeError("cutlass_sm80 module is not available")
            return cutlass_sm80.gemm_sm80_v3(
                a.contiguous(), b.contiguous(),
                sleep_ns=sleep_ns, sleep_freq=sleep_freq)
        if matmul_kernel == "persistent_cta":
            if persistent_sm80 is None:
                raise RuntimeError("bf16_gemm_sm80_persistent module is not available")
            return persistent_sm80.gemm_sm80_persistent(
                a.contiguous(), b.contiguous(),
                sleep_ns, sleep_freq,
                args.persistent_throttle_mode,
                args.persistent_ctas_per_sm,
                args.persistent_chunk_tiles)
        raise ValueError(f"unknown matmul kernel: {matmul_kernel}")

    def _body():
        x = hidden
        for _ in range(ACTUAL_FORWARD_LAYERS):
            up = _mm(x, weights["up"])
            _ = _mm(x, weights["gate"])
            x = _mm(up, weights["down"])
        return x

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(1):
            _body()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _body()
    torch.cuda.synchronize()

    static_refs = (hidden, weights)

    def _replay():
        _ = static_refs
        graph.replay()

    return _replay


def measure_qwen3_mlp_graph_replay(B, S, matmul_kernel, sleep_ns, sleep_freq,
                                   cutlass_sm80, persistent_sm80, handle, args,
                                   cmp_parts=None):
    """Measure replay of one captured MLP-only CUDA Graph workload."""
    try:
        cmp_parts = cmp_parts or []
        replay = make_qwen3_mlp_graph_replay_fn(
            B, S, matmul_kernel, sleep_ns, sleep_freq,
            cutlass_sm80, persistent_sm80, args)
        for _ in range(1):
            replay()
        torch.cuda.synchronize()
        time.sleep(1)
        ni = auto_iterations(replay, args.target_seconds, pilot_iters=3)
        ni = 1
        el, en, pw, sm_clk, temp, pv_bef, ref_bef, pv_dur, ref_dur, pv_rat, _cap, _samples, gpu_ms = measure(
            replay, ni, handle, sample_interval=args.nvml_interval)
        flops = qwen3_mlp_graph_flops(B, S)
        timing_ms = gpu_ms if gpu_ms > 0 else el
        gflops = flops * ni / (timing_ms / 1000.0) / 1e9
        nj = en * 1e6 / (flops * ni) if ni > 0 else 0.0
        print(
            f"  {'mlp_graph_replay':14s} | {matmul_kernel:12s} "
            f"sleep={sleep_ns:6d}ns freq={sleep_freq:4d} iters={ni:3d} | "
            f"wall={el:8.1f}ms gpu={gpu_ms:8.1f}ms "
            f"{gflops:8.1f}GF {pw:6.1f}W "
            f"{en:8.0f}mJ {nj:.3f}nJ/F  SM={sm_clk:.0f}MHz  T={temp:.0f}°C "
            f"PVrun={pv_dur}ns/{ref_dur}ns ({pv_rat * 100:.1f}% NVML ref)"
            f"{'  [' + ' | '.join(cmp_parts) + ']' if cmp_parts else ''}")
        return {
            "batch_size": B,
            "seq_len": S,
            "num_layers": ACTUAL_FORWARD_LAYERS,
            "matmul_kernel": matmul_kernel,
            "workload": "mlp_up_gate_down_x32_cuda_graph_replay",
            "sleep_ns": sleep_ns,
            "sleep_freq": sleep_freq,
            "num_iters": ni,
            "elapsed_ms": el,
            "gpu_elapsed_ms": gpu_ms,
            "energy_mj": en,
            "power_w": pw,
            "sm_clock_mhz": sm_clk,
            "temp_c": temp,
            "gflops": gflops,
            "nj_per_flop": nj,
            "total_flops": flops,
            "power_violation_before_ns": pv_bef,
            "reference_time_before_ns": ref_bef,
            "power_violation_during_ns": pv_dur,
            "reference_time_during_ns": ref_dur,
            "power_violation_ratio": pv_rat,
        }
    except Exception as e:
        print(
            f"  ⚠ MLP graph replay skipped: kernel={matmul_kernel} "
            f"sleep={sleep_ns} freq={sleep_freq}: {e}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


# ─────────────────────────── helpers ───────────────────────────

def auto_iterations(fn, target_seconds=2.0, pilot_iters=10):
    """Estimate iteration count to fill *target_seconds*."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(pilot_iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    per_iter = elapsed / pilot_iters
    if per_iter <= 0:
        return 1000
    return max(1, int(target_seconds / per_iter))


# ─────────────────────────────────────────────────────────────────────────────
#  Persistent NVML sampler process
#
#  프로세스는 프로그램 시작 시 딱 한 번 spawn 되고 종료 시까지 살아있는다.
#  measure() 는 cmd_q 에 명령을 보내고 result_q 에서 결과를 받기만 한다.
#
#  cmd_q (main → sampler):
#    ("start",)       : 샘플 수집 시작
#    ("stop",)        : 수집 중단, result_q 에 결과 push
#    ("exit",)        : 프로세스 종료
#
#  result_q (sampler → main):
#    (time_ms_list, power_mw_list, sm_list, temp_list) : stop 명령 처리 후 1회 put
#    power_mw_list[i] = mW; 기본 nvmlDeviceGetPowerUsage, 또는 --gpu-power-source
#      nvidia_smi_instant 일 때 nvidia-smi power.draw.instant 과 같은 계열.
#    time_ms_list[i] = 샘플 루프 시작(base) 기준 경과 시간(ms)
# ─────────────────────────────────────────────────────────────────────────────


def _instant_power_mw_nvidia_smi(device_idx: int) -> int:
    """nvidia-smi power.draw.instant (W) → 정수 mW. gpu_power_log CLI 와 정합."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_idx),
                "--query-gpu=power.draw.instant",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if proc.returncode != 0:
            return 0
        line = proc.stdout.strip().splitlines()[0].strip()
        return int(round(float(line) * 1000.0))
    except Exception:
        return 0

import ctypes

def get_fine_grained_power(device_handle=None):
    """
    pynvml 패키지를 완전히 우회하고 순수 ctypes로만 작동합니다.
    multiprocessing.Process 내부에서 독립적으로 실행되어도 에러가 나지 않습니다.
    """
    # 1. NVML C 라이브러리 직접 로드
    try:
        nvml_lib = ctypes.CDLL("libnvidia-ml.so")
    except OSError:
        nvml_lib = ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so")

    # 2. 하드웨어 샘플 구조체 정의
    class struct_c_nvmlSample_t(ctypes.Structure):
        _fields_ = [
            ('timestamp', ctypes.c_ulonglong),  # 마이크로초(us) 단위 내부 타임스탬프
            ('sampleValue', ctypes.c_double)    # 하드웨어 센서 즉시 전력 값 (Watts)
        ]

    # NVML 공식 상숫값 정의
    NVML_TOTAL_POWER_SAMPLES = 0 
    MAX_SAMPLES = 2000  # 한 번에 가져올 최대 링 버퍼 샘플 수
    
    sample_count = ctypes.c_uint(MAX_SAMPLES)
    samples_buffer = (struct_c_nvmlSample_t * MAX_SAMPLES)()

    # 3. NVML C API 명시적 바인딩 및 프로토타입 정의
    # nvmlDeviceGetSamples(nvmlDevice_t, nvmlSamplingType_t, unsigned int*, nvmlSample_t*)
    nvmlDeviceGetSamples = nvml_lib.nvmlDeviceGetSamples
    nvmlDeviceGetSamples.argtypes = [
        ctypes.c_void_p,                      # device handle
        ctypes.c_int,                         # sampling type
        ctypes.POINTER(ctypes.c_uint),        # sample count pointer
        ctypes.POINTER(struct_c_nvmlSample_t)  # samples array pointer
    ]
    nvmlDeviceGetSamples.restype = ctypes.c_int

    # 만약 기존 pynvml 장치 핸들이 넘어왔다면 내부 포인터 값만 정수형으로 추출
    if device_handle is not None:
        try:
            # pynvml 핸들 객체 내부에 숨겨진 실제 C 포인터 주소 추출 시도
            if hasattr(device_handle, 'value'):
                c_handle = ctypes.c_void_p(device_handle.value)
            else:
                c_handle = ctypes.c_void_p(int(device_handle))
        except Exception:
            # 추출 실패 시 0번 GPU 다이렉트 바인딩으로 안전장치 마련
            nvmlInit = nvml_lib.nvmlInit
            nvmlInit.restype = ctypes.c_int
            nvmlInit()
            
            nvmlDeviceGetHandleByIndex = nvml_lib.nvmlDeviceGetHandleByIndex
            nvmlDeviceGetHandleByIndex.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            nvmlDeviceGetHandleByIndex.restype = ctypes.c_int
            
            raw_handle = ctypes.c_void_p()
            nvmlDeviceGetHandleByIndex(3, ctypes.byref(raw_handle)) # 현재 타깃인 GPU 3번 적용
            c_handle = raw_handle
    else:
        # 핸들이 없으면 새로 초기화하여 3번 GPU 핸들을 직접 획득
        nvmlInit = nvml_lib.nvmlInit
        nvmlInit.restype = ctypes.c_int
        nvmlInit()
        
        nvmlDeviceGetHandleByIndex = nvml_lib.nvmlDeviceGetHandleByIndex
        nvmlDeviceGetHandleByIndex.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        nvmlDeviceGetHandleByIndex.restype = ctypes.c_int
        
        raw_handle = ctypes.c_void_p()
        nvmlDeviceGetHandleByIndex(3, ctypes.byref(raw_handle))
        c_handle = raw_handle

    # 4. 하드웨어 링 버퍼 덤프 실행
    result = nvmlDeviceGetSamples(c_handle, NVML_TOTAL_POWER_SAMPLES, ctypes.byref(sample_count), samples_buffer)
    
    if result == 0:
        print(f"\n[NVML 하드웨어 버퍼 덤프 성공] 총 {sample_count.value}개의 Blackwell 1ms 생데이터 확보.")
        
        # 상위 5개 샘플 테스트 출력
        for i in range(min(5, sample_count.value)):
            t_us = samples_buffer[i].timestamp
            p_w = samples_buffer[i].sampleValue
            print(f" -> Hardware Timestamp: {t_us} us | Raw Power: {p_w:.3f} W")
            
        # TODO: 스크립트의 기존 CSV 저장 컴포넌트가 있다면 samples_buffer 루프 데이터로 교체 연동
    else:
        print(f"[NVML 에러] 하드웨어 샘플 버퍼를 가져오지 못했습니다. 에러 코드: {result}")

def _nvml_sampler_proc(device_idx: int, interval: float,
                        cmd_q: mp.Queue, result_q: mp.Queue,
                        ready_ev: mp.Event, power_source: str = "nvml") -> None:

    """프로그램 시작 시 1회 spawn. cmd_q 명령으로 start/stop/exit 제어."""
    import queue as _queue
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    ready_ev.set()   # 초기화 완료 신호

    while True:
        cmd = cmd_q.get()          # "start" 대기 (blocking)
        if cmd == "exit":
            break

        # ── "start" 수신: 수집 루프 진입 ─────────────────────────────────
        # 한 루프마다 (시간, power_mW, SM MHz, °C) 동시에 1개씩만 append → 길이 항상 일치
        ts_samples: list = []
        power_mw_samples: list = []
        sm_samples: list = []
        temp_samples: list = []
        t_loop0 = time.perf_counter()
        while True:
            t_rel_ms = (time.perf_counter() - t_loop0) * 1000.0
            ts_samples.append(t_rel_ms)
            pw_mw = 0
            if power_source == "nvidia_smi_instant":
                pw_mw = _instant_power_mw_nvidia_smi(device_idx)
            else:
                try:
                    pw_mw = int(pynvml.nvmlDeviceGetPowerUsage(handle))
                    # get_fine_grained_power(handle)
                except pynvml.NVMLError:
                    pass
            power_mw_samples.append(pw_mw)
            sm = 0.0
            try:
                sm = float(pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_SM))
            except pynvml.NVMLError:
                pass
            sm_samples.append(sm)
            tc = 0.0
            try:
                tc = float(pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU))
            except pynvml.NVMLError:
                pass
            temp_samples.append(tc)

            # interval 동안 대기하되 "stop"/"exit" 가 오면 즉시 탈출
            try:
                next_cmd = cmd_q.get(timeout=interval)
                if next_cmd in ("stop", "exit"):
                    result_q.put(
                        (ts_samples, power_mw_samples, sm_samples, temp_samples))
                    if next_cmd == "exit":
                        pynvml.nvmlShutdown()
                        return
                    break          # "stop" → 외부 while 로 돌아가 다음 "start" 대기
            except _queue.Empty:
                pass               # timeout → 계속 수집

    pynvml.nvmlShutdown()


# ── 전역 sampler 핸들 (main() 에서 초기화) ──────────────────────────────────
_sampler_proc     = None  # mp.Process
_sampler_cmd_q    = None  # mp.Queue
_sampler_result_q = None  # mp.Queue


def start_sampler(device_idx: int, interval: float = 0.01,
                  power_source: str = "nvml") -> None:
    """프로그램 시작 시 1회 호출. sampler 프로세스를 spawn 하고 준비 대기."""
    global _sampler_proc, _sampler_cmd_q, _sampler_result_q
    ctx = mp.get_context("spawn")
    _sampler_cmd_q    = ctx.Queue()
    _sampler_result_q = ctx.Queue()
    ready_ev          = ctx.Event()
    _sampler_proc = ctx.Process(
        target=_nvml_sampler_proc,
        args=(
            device_idx, interval, _sampler_cmd_q, _sampler_result_q, ready_ev,
            power_source),
        daemon=True,
    )
    _sampler_proc.start()
    ready_ev.wait()   # 초기화 완료까지 대기


def stop_sampler() -> None:
    """프로그램 종료 시 1회 호출. sampler 프로세스를 정상 종료."""
    if _sampler_proc is not None:
        _sampler_cmd_q.put("exit")
        _sampler_proc.join(timeout=5.0)


def _nvml_power_violation_snapshot_ns(handle):
    """nvmlDeviceGetViolationStatus(NVML_PERF_POLICY_POWER).

    Returns:
        (violation_time_ns, reference_time_ns) 누적 카운터. 실패 시 (0, 0).

    See:
        https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
    """
    try:
        st = pynvml.nvmlDeviceGetViolationStatus(
            handle, pynvml.NVML_PERF_POLICY_POWER)
        # print(st)
        return int(st.violationTime), int(st.referenceTime)
    except Exception:
        return 0, 0


def _nvml_violation_delta_ns(v0: int, r0: int, v1: int, r1: int) -> tuple[int, int]:
    """누적 violation/reference 카운터 차이(ns). 음수는 0으로 클램프."""
    dv = max(0, v1 - v0)
    dr = max(0, r1 - r0)
    return dv, dr


def _nvml_power_management_limit_w(handle):
    """현재 GPU 전력 상한(NVML power management limit), 단위 W."""
    try:
        mw = int(pynvml.nvmlDeviceGetPowerManagementLimit(handle))
        return mw / 1000.0
    except Exception:
        return 0.0


# 커널 전·후 NVML 스트림에 idle→부스트 구간을 넣기 위한 여유 (detail 플롯용)
MEASURE_PRE_MARGIN_S = 0.5
MEASURE_POST_MARGIN_S = 0.5


def measure(
    fn,
    num_iters,
    handle,
    sample_interval=0.1,
    pre_measure_samples: int = 5,
):
    """Run *fn()* num_iters times.

    Returns:
        elapsed_ms, energy_mJ, avg_power_W, avg_sm_clock_mhz, avg_temp_c,
        power_violation_before_ns, reference_time_before_ns,
        power_violation_during_ns, reference_time_during_ns, power_violation_ratio,
        gpu_power_cap_w, nvml_samples, gpu_elapsed_ms.

        elapsed_ms     : wall-clock(ms). 전력/에너지·NVML 정합 기준.
        gpu_elapsed_ms : CUDA Event 기반 GPU 측 ni번 실행 시간(ms).
                         CPU launch overhead가 빠져 GFLOP/s 계산에 더 적합.

        gpu_power_cap_w: NVML nvmlDeviceGetPowerManagementLimit 기준 현재 설정 상한(W).

        nvml_samples: 해당 구간 NVML 주기 샘플 목록. 각 원소는
        sample_index, sample_time_ms(첫 샘플 대비 ms), sample_power_mw(NVML 정수 mW),
        sample_power_w(sample_power_mw/1000),
        sample_sm_clock_mhz, sample_temp_c.

    전력 위반·참조 시간은 ``nvmlDeviceGetViolationStatus(..., NVML_PERF_POLICY_POWER)``
    의 누적 카운터 차이(나노초 정수). ratio = violation_during_ns / reference_during_ns.

    NVML 시계열은 ``start`` 직후 약 MEASURE_PRE_MARGIN_S 초 idle·클럭 상승 구간,
    커널 실행, ``stop`` 직전 약 MEASURE_POST_MARGIN_S 초를 포함한다.
    요약 전력·SM·온도·에너지는 커널 실행 구간(시간 기준)에 해당하는 샘플만 평균한다.

    persistent sampler 프로세스에 start/stop 명령을 보내 GIL 간섭 없이 수집.
    """
    interval_s = float(sample_interval)

    torch.cuda.synchronize()
    gpu_power_cap_w = _nvml_power_management_limit_w(handle)

    # ── 측정 시작 전: idle 구간에서 violation/reference 누적 차이 (NVML)
    pv0_ns, pr0_ns = _nvml_power_violation_snapshot_ns(handle)
    for i in range(pre_measure_samples):
        if i < pre_measure_samples - 1:
            time.sleep(interval_s)
    pv1_ns, pr1_ns = _nvml_power_violation_snapshot_ns(handle)
    power_violation_before_ns, reference_time_before_ns = _nvml_violation_delta_ns(
        pv0_ns, pr0_ns, pv1_ns, pr1_ns)

    _sampler_cmd_q.put("start")

    # time.sleep(MEASURE_PRE_MARGIN_S)

    torch.cuda.synchronize()
    pv_run0_ns, pr_run0_ns = _nvml_power_violation_snapshot_ns(handle)

    # GPU-side timing via CUDA events — CPU launch overhead 제외 한 순수 GPU 실행 시간
    gpu_ev0 = torch.cuda.Event(enable_timing=True)
    gpu_ev1 = torch.cuda.Event(enable_timing=True)

    t0 = time.perf_counter()

    torch.cuda.cudart().cudaProfilerStart()
    gpu_ev0.record()
    for _ in range(num_iters):
        fn()
    gpu_ev1.record()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    t1 = time.perf_counter()
    gpu_elapsed_ms = gpu_ev0.elapsed_time(gpu_ev1)
    pv_run1_ns, pr_run1_ns = _nvml_power_violation_snapshot_ns(handle)

    # time.sleep(MEASURE_POST_MARGIN_S)

    _sampler_cmd_q.put("stop")

    elapsed_ms = (t1 - t0) * 1000.0

    power_violation_during_ns, reference_time_during_ns = _nvml_violation_delta_ns(
        pv_run0_ns, pr_run0_ns, pv_run1_ns, pr_run1_ns)
    power_violation_ratio = (
        (power_violation_during_ns / reference_time_during_ns)
        if reference_time_during_ns > 0 else 0.0)

    try:
        ts_samples, power_mw_samples, sm_samples, temp_samples = (
            _sampler_result_q.get(timeout=5.0))
    except Exception:
        ts_samples, power_mw_samples, sm_samples, temp_samples = [], [], [], []

    nvml_samples = []
    n = min(
        len(ts_samples), len(power_mw_samples), len(sm_samples), len(temp_samples))
    t_off0 = ts_samples[0] if n else 0.0
    for i in range(n):
        pmw = power_mw_samples[i]
        nvml_samples.append({
            "sample_index": i,
            "sample_time_ms": ts_samples[i] - t_off0,
            "sample_power_mw": pmw,
            "sample_power_w": pmw / 1000.0,
            "sample_sm_clock_mhz": sm_samples[i],
            "sample_temp_c": temp_samples[i],
        })

    # 기본: 전체 샘플 평균 (커널 구간 추출 실패 시)
    power_W = (
        (sum(power_mw_samples) / len(power_mw_samples) / 1000.0)
        if power_mw_samples else 0.0)
    avg_sm_mhz = (
        sum(sm_samples) / len(sm_samples) if sm_samples else 0.0)
    avg_temp_c = (
        sum(temp_samples) / len(temp_samples) if temp_samples else 0.0)

    # 커널 구간만 평균 (전후 margin 샘플 제외) — 요약 CSV·에너지와 커널 시간 정합
    pre_ms = MEASURE_PRE_MARGIN_S * 1000.0
    interval_ms = interval_s * 1000.0
    tol_lo = max(40.0, interval_ms * 1.5)
    tol_hi = max(100.0, interval_ms * 4.0)
    lo_t = pre_ms - tol_lo
    hi_t = pre_ms + elapsed_ms + tol_hi
    ks = [
        s for s in nvml_samples
        if lo_t <= s["sample_time_ms"] <= hi_t
    ]
    if ks:
        power_W = sum(s["sample_power_mw"] for s in ks) / len(ks) / 1000.0
        avg_sm_mhz = sum(s["sample_sm_clock_mhz"] for s in ks) / len(ks)
        avg_temp_c = sum(s["sample_temp_c"] for s in ks) / len(ks)

    energy_mJ = power_W * elapsed_ms

    return (
        elapsed_ms,
        energy_mJ,
        power_W,
        avg_sm_mhz,
        avg_temp_c,
        power_violation_before_ns,
        reference_time_before_ns,
        power_violation_during_ns,
        reference_time_during_ns,
        power_violation_ratio,
        gpu_power_cap_w,
        nvml_samples,
        gpu_elapsed_ms,
    )


# ── detail CSV (측정마다 즉시 기록; violation 컬럼 제외) ───────────────────────

DETAIL_VIOLATION_KEYS = frozenset({
    "power_violation_before_ns",
    "reference_time_before_ns",
    "power_violation_during_ns",
    "reference_time_during_ns",
    "power_violation_ratio",
})

DETAIL_CSV_COLUMNS = [
    "measure_id",
    "nvml_num_samples",
    "sample_index",
    "label",
    "pid",
    "test_case",
    "batch_size",
    "seq_len",
    "layer",
    "kernel",
    "M",
    "K",
    "N",
    "sleep_ns",
    "sleep_freq",
    "count",
    "num_iters",
    "elapsed_ms",
    "gpu_elapsed_ms",
    "gflops",
    "energy_mj",
    "nj_per_flop",
    "sm_clock_mhz",
    "temp_c",
    "gpu_power_cap_w",
    "sample_time_ms",
    "sample_power_mw",
    "sample_power_w",
    "sample_sm_clock_mhz",
    "sample_temp_c",
]

DETAIL_FLOAT_KEYS = frozenset({
    "elapsed_ms", "gpu_elapsed_ms", "gflops",
    "energy_mj", "nj_per_flop", "sm_clock_mhz", "temp_c", "gpu_power_cap_w",
    "sample_time_ms", "sample_power_w", "sample_sm_clock_mhz", "sample_temp_c",
})


class DetailCsvSink:
    """measure() 1회마다 NVML 시계열 행을 detail CSV에 바로 쓰고 flush."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(
            self._f, fieldnames=DETAIL_CSV_COLUMNS, extrasaction="ignore")
        self._w.writeheader()
        self._f.flush()
        self._measure_id = 0

    def append_measurement(
            self, rec: dict, run_label: str, test_case: str, pid: int) -> None:
        self._measure_id += 1
        mid = self._measure_id
        base = {
            k: v for k, v in rec.items()
            if k != "_nvml_samples" and k not in DETAIL_VIOLATION_KEYS}
        # detail에는 구간 평균 전력(power_w) 컬럼을 넣지 않음 — 샘플 열만 사용
        pw_run_avg = float(base.pop("power_w", 0.0))
        samples = rec.get("_nvml_samples") or []
        n = len(samples)

        def _round(row: dict) -> None:
            for k in DETAIL_FLOAT_KEYS:
                if k in row and isinstance(row[k], float):
                    row[k] = round(row[k], 4)

        if not samples:
            row = {
                **base,
                "measure_id": mid,
                "nvml_num_samples": 1,
                "sample_index": 0,
                "sample_time_ms": 0.0,
                "sample_power_w": pw_run_avg,
                "sample_power_mw": int(round(pw_run_avg * 1000)),
                "sample_sm_clock_mhz": base.get("sm_clock_mhz", 0.0),
                "sample_temp_c": base.get("temp_c", 0.0),
                "label": run_label,
                "pid": pid,
                "test_case": test_case,
            }
            _round(row)
            self._w.writerow(row)
        else:
            for s in samples:
                row = {
                    **base,
                    **s,
                    "measure_id": mid,
                    "nvml_num_samples": n,
                    "label": run_label,
                    "pid": pid,
                    "test_case": test_case,
                }
                _round(row)
                self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        if getattr(self, "_f", None) is not None:
            self._f.close()
            self._f = None


def commit_measure(
        results_list, rec, detail_sink, args, test_case, pid):
    """detail CSV 즉시 기록 후 메모리에서는 _nvml_samples 제거하여 results만 유지."""
    run_label = (args.run_label.strip() or test_case)
    if detail_sink is not None:
        detail_sink.append_measurement(rec, run_label, test_case, pid)
    results_list.append(
        {k: v for k, v in rec.items() if k != "_nvml_samples"})


def _sanitize_graph_filename(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))


def cuda_graph_capture_debug_dump(fn, dot_path: str) -> bool:
    return True
    """CUDA Graph 1회 캡처 후 debug_dump로 .dot 저장. 실패 시 False.

    PyTorch 기본 ``CUDAGraph(keep_graph=False)`` 는 capture 직후 내부
    ``cudaGraph_t`` 를 정리해 ``debug_dump`` 가 ``cudaGraphDebugDotPrint`` 를
    호출하지 못함 → ``keep_graph=True`` 로 캡처해야 .dot 이 실제로 생성됨.
    """
    torch.cuda.synchronize()
    dot_path = os.path.abspath(dot_path)
    g = None
    try:
        g = torch.cuda.CUDAGraph(keep_graph=True)
        g.enable_debug_mode()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        d = os.path.dirname(dot_path)
        if d:
            os.makedirs(d, exist_ok=True)
        g.debug_dump(dot_path)
        if os.path.isfile(dot_path) and os.path.getsize(dot_path) > 0:
            print(f"  CUDA Graph .dot 저장: {dot_path}")
            return True
        print(
            f"  ⚠ CUDA Graph debug_dump 후 파일 없음 또는 0바이트: {dot_path} "
            f"(드라이버/툴킷 버전에 따라 경로·권한을 확인)"
        )
        return False
    except Exception as e:
        print(f"  ⚠ CUDA Graph capture/debug_dump 실패 ({dot_path}): {e}")
        return False
    finally:
        if g is not None:
            try:
                g.reset()
            except Exception:
                pass


def cuda_graph_dot_path(
    dump_dir: str,
    batch_sz: int,
    seq_len: int,
    layer: str,
    kernel: str,
    sleep_ns=None,
    sleep_freq=None,
) -> str:
    sub = os.path.join(dump_dir, f"S{seq_len}_B{batch_sz}")
    base = f"{_sanitize_graph_filename(layer)}__{_sanitize_graph_filename(kernel)}"
    if sleep_ns is not None and sleep_freq is not None:
        base += f"__ns{sleep_ns}_freq{sleep_freq}"
    return os.path.join(sub, f"{base}.dot")


def print_gpu_info(handle):
    try:
        # Get GPU name
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        print(f"\n{'='*60}")
        print(f"GPU: {gpu_name}")
        print(f"{'='*60}")
        
        # SM Clock Information
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            print(f"SM Clock: {sm_clock} MHz")
        except Exception as e:
            print(f"SM Clock: Error - {e}")
        
        # Memory Clock Information
        try:
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            print(f"Memory Clock: {mem_clock} MHz")
        except Exception as e:
            print(f"Memory Clock: Error - {e}")
        
        # Graphics Clock Information
        try:
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            print(f"Graphics Clock: {graphics_clock} MHz")
        except Exception as e:
            print(f"Graphics Clock: Error - {e}")
        
        # Max Clock Information
        try:
            max_sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
            max_mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            print(f"Max SM Clock: {max_sm_clock} MHz")
            print(f"Max Memory Clock: {max_mem_clock} MHz")
        except Exception as e:
            print(f"Max Clocks: Error - {e}")
        
        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            print(f"Temperature: {temp}°C")
        except Exception as e:
            print(f"Temperature: Error - {e}")
        
        # Power Usage
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            print(f"Power Usage: {power:.2f} W / {power_limit:.2f} W")
        except Exception as e:
            print(f"Power: Error - {e}")
        
        # Utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU Utilization: {util.gpu}%")
            print(f"Memory Utilization: {util.memory}%")
        except Exception as e:
            print(f"Utilization: Error - {e}")
        
        # Memory Info
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = mem_info.used / (1024**3)
            total_gb = mem_info.total / (1024**3)
            print(f"Memory: {used_gb:.2f} GB / {total_gb:.2f} GB ({mem_info.used * 100 / mem_info.total:.1f}%)")
        except Exception as e:
            print(f"Memory Info: Error - {e}")
        
        # PCIe Throughput
        try:
            pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            print(f"PCIe TX: {pcie_tx} KB/s")
            print(f"PCIe RX: {pcie_rx} KB/s")
        except Exception as e:
            print(f"PCIe Throughput: Error - {e}")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error getting GPU info: {e}")

# ─────────────────────────── main ──────────────────────────────

def main():
    pid = os.getpid()
    print(f"pid: {pid}")
    parser = argparse.ArgumentParser(
        description="Qwen3-8B full-forward matmul benchmark with WMMA sleep")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-csv", default="logs/qwen3_8b_forward.csv")
    parser.add_argument(
        "--run-label",
        default="",
        help="실험 구분용 라벨; 비우면 test_case 문자열과 동일하게 기록됩니다.",
    )
    parser.add_argument("--target-seconds", type=float, default=3.0,
                        help="Target wall-time per measurement point")
    # parser.add_argument("--warmup-iters", type=int, default=50*50*50)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size B")
    parser.add_argument("--seq-len", type=int, default=4096,
                        help="Sequence length S (prefill context length)")
    parser.add_argument(
        "--cuda-graph-dump-dir",
        default="logs/cuda_graph_dump",
        help="CUDA Graph debug .dot 저장 디렉터리 (seq_len×batch_size별 하위 폴더)",
    )
    parser.add_argument("--pm", action="store_true", help="GPU Persistence Mode (keep GPU initialized after script ends, for testing)")
    parser.add_argument(
        "--gpu-power-source",
        choices=("nvml", "nvidia_smi_instant"),
        default="nvml",
        help=(
            "detail 의 sample_power_* 전력 소스. nvml 은 nvmlDeviceGetPowerUsage 로 "
            "드라이버가 짧게 평균·필터링한 값이라 power_limit 근처 스파이크가 "
            "nvidia-smi power.draw.instant 로 찍은 gpu_power_log 보다 덜 튀어 보일 수 있다. "
            "nvidia_smi_instant 은 동일 계열 값이지만 샘플마다 subprocess 비용이 있다."
        ),
    )
    parser.add_argument(
        "--nvml-interval",
        type=float,
        default=0.01,
        help=(
            "NVML 폴링 주기(초). 기본 0.1(100ms)은 드라이버 전력 갱신에 가깝고 "
            "gpu_power_log/nvidia-smi 로그와 비슷한 시간 간격이다. "
            "0.01처럼 매우 짧게 하면 같은 mW가 연속으로 반복되는 행이 많아질 수 있다 "
            "(NVML 순간값도 하드웨어 해상도가 mW 정수이며 갱신 주기가 있다)."
        ),
    )
    parser.add_argument(
        "--persistent-ctas-per-sm",
        type=int,
        default=2,
        help="persistent_cta 커널에서 launch할 SM당 CTA 수. 기본값은 2, 실험용으로 8까지 가능.",
    ),
    parser.add_argument(
        "--persistent-chunk-tiles",
        type=int,
        default=1,
        help="persistent_cta 커널에서 CTA가 atomicAdd 한 번으로 가져갈 tile 개수.",
    )
    parser.add_argument(
        "--persistent-throttle-mode",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="persistent_cta throttle 방식: 0=clock64 wait, 1=instruction loop, 2=__nanosleep.",
    )
    parser.add_argument(
        "-m",
        "--matmul-kernel-mask",
        "--kernel-mask",
        dest="matmul_kernel_mask",
        type=lambda x: int(x, 0),
        default=0x7,
        help=(
            "실행할 matmul 커널 bitmask. bit0=cuBLAS, bit1=cutlass_sm80, "
            "bit2=persistent_cta. 기본 0x7=전체, 0x3=cuBLAS+cutlass_sm80."
        ),
    )
    parser.add_argument(
        "-w",
        "--workload-mask",
        dest="workload_mask",
        type=lambda x: int(x, 0),
        default=0x7,
        help=(
            "실행할 workload bitmask. bit0=full_forward, bit1=matmul_only, "
            "bit2=mlp_graph_replay. 기본 0x7=전체."
        ),
    )
    args = parser.parse_args()
    test_case = "FULL"

    _raw_kernel_mask = int(args.matmul_kernel_mask)
    kernel_mask = _raw_kernel_mask
    if kernel_mask & ~0x7:
        raise SystemExit(
            f"Unsupported --matmul-kernel-mask={_raw_kernel_mask:#x}; "
            "valid bits are 0x1(cublas), 0x2(cutlass_sm80), 0x4(persistent_cta).")

    kernel_bits = {
        "cublas": 0x1,
        "cutlass_sm80": 0x2,
        "persistent_cta": 0x4,
    }

    def kernel_enabled(name: str) -> bool:
        return bool(kernel_mask & kernel_bits[name])

    _raw_workload_mask = int(args.workload_mask)
    workload_mask = _raw_workload_mask
    if workload_mask & ~0x7:
        raise SystemExit(
            f"Unsupported --workload-mask={_raw_workload_mask:#x}; "
            "valid bits are 0x1(full_forward), 0x2(matmul_only), "
            "0x4(mlp_graph_replay).")

    workload_bits = {
        "full_forward": 0x1,
        "matmul_only": 0x2,
        "mlp_graph_replay": 0x4,
    }

    def workload_enabled(name: str) -> bool:
        return bool(workload_mask & workload_bits[name])

    torch.cuda.set_device(0)

    try:
        import sleep_wmma as wmma_sleep_gemm
    except ImportError:
        wmma_sleep_gemm = None
        print("⚠ sleep_wmma not found. WMMA sleep kernels will be skipped.\n"
              "  Build: cd /workspace/custum && python setup_wmma_sleep.py build_ext --inplace")

    # BF16 WMMA 확장 모듈 (선택적 로드 – 없으면 해당 커널 건너뜀)
    try:
        import bf16_wmma_sleep as bf16_gemm
    except ImportError:
        bf16_gemm = None
        print("⚠ bf16_wmma_sleep not found.  BF16 WMMA kernel will be skipped.\n"
              "  Build: cd /workspace/custum && "
              "python setup_bf16_gemm.py build_ext --inplace")

    # CUTLASS SM80 BF16 GEMM (cuBLAS 동급 성능, BF16 정밀도 유지)
    cutlass_sm80 = None
    if kernel_enabled("cutlass_sm80"):
        try:
            import bf16_gemm_sm80 as cutlass_sm80
        except ImportError:
            cutlass_sm80 = None
            print("⚠ bf16_gemm_sm80 not found.  CUTLASS SM80 BF16 kernel will be skipped.\n"
                  "  Build: cd /workspace/custum && "
                  "python setup_bf16_sm80.py build_ext --inplace")
    if kernel_enabled("cutlass_sm80") and cutlass_sm80 is None:
        cutlass_sm80 = None
    elif not kernel_enabled("cutlass_sm80"):
        print("cutlass_sm80 disabled by --matmul-kernel-mask")

    # Custom BF16 GEMM with __nanosleep (cp.async + WMMA + nanosleep K-loop)
    # 이 커널만 sleep_freq > 0 으로 동작; 나머지 커널은 sleep_ns=0 고정.
    try:
        import bf16_gemm_custom as custom_bf16
    except ImportError:
        custom_bf16 = None
        print("⚠ bf16_gemm_custom not found. Custom BF16 nanosleep kernel will be skipped.\n"
              "  Build: cd /workspace/custum && "
              "python setup_bf16_custom.py build_ext --inplace")

    # PTX inline-asm BF16 GEMM (mma.m16n8k16 + ldmatrix + cp.async, 128×128×64)
    try:
        import bf16_gemm_sm80_kernel as ptx_sm80
    except ImportError:
        ptx_sm80 = None
        print("⚠ bf16_gemm_sm80_kernel not found.  PTX SM80 kernel will be skipped.\n"
              "  Build: cd /workspace/custum && "
              "python setup_bf16_sm80_kernel.py build_ext --inplace")

    # Persistent CTA BF16 GEMM: dense full tiles throttle, tail tiles unthrottled.
    persistent_sm80 = None
    if kernel_enabled("persistent_cta"):
        try:
            import bf16_gemm_sm80_persistent as persistent_sm80
        except ImportError:
            persistent_sm80 = None
            print("⚠ bf16_gemm_sm80_persistent not found. Persistent CTA kernel will be skipped.\n"
                  "  Build: cd /workspace/custum && "
                  "python setup_bf16_sm80_persistent.py build_ext --inplace")
    if kernel_enabled("persistent_cta") and persistent_sm80 is None:
        persistent_sm80 = None
    elif not kernel_enabled("persistent_cta"):
        print("persistent_cta disabled by --matmul-kernel-mask")

    # CUTLASS 3.x SM90 BF16 GEMM (wgmma + TMA, SM120 backward-compatible)
    try:
        import bf16_gemm_sm90 as cutlass_sm90
    except ImportError:
        cutlass_sm90 = None
        print("⚠ bf16_gemm_sm90 not found.  CUTLASS SM90 BF16 kernel will be skipped.\n"
              "  Build: cd /workspace/custum && "
              "python setup_bf16_sm90.py build_ext --inplace")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    device_name = pynvml.nvmlDeviceGetName(handle)

    # ── persistent sampler 프로세스 시작 (이후 measure() 가 재사용) ──────────
    try:
        dev_idx = pynvml.nvmlDeviceGetIndex(handle)
    except Exception:
        dev_idx = int(args.device)
    start_sampler(
        device_idx=dev_idx,
        interval=float(args.nvml_interval),
        power_source=args.gpu_power_source,
    )
    props = torch.cuda.get_device_properties(0)
    print(f"GPU        : {device_name}")
    print(f"SMs        : {props.multi_processor_count}")
    print(f"Model      : Qwen3-8B  ({NUM_LAYERS} layers)")
    print(f"Batch size : {args.batch_size}")
    print(f"Seq length : {args.seq_len}")
    print(f"Tokens (M) : {args.batch_size * args.seq_len}")
    print(f"Output     : {args.output_csv}")
    print(f"Power samp.: {args.gpu_power_source} (--gpu-power-source)")
    print(
        "Matmul mask: "
        f"raw={_raw_kernel_mask:#x} effective={kernel_mask:#x} "
        f"enabled={','.join(name for name in kernel_bits if kernel_enabled(name)) or 'none'}")
    print(
        "Workloads  : "
        f"raw={_raw_workload_mask:#x} effective={workload_mask:#x} "
        f"enabled={','.join(name for name in workload_bits if workload_enabled(name)) or 'none'}")

    # sleep_ns_list = [0] + [2 ** (i) for i in range(5, 19)]
    # sleep_ns_list = [2 * i for i in range(1, 200)]
    # sleep_ns_list = [0, 1200, 1400, 1600, 1800]
    # sleep_ns_list = sorted(sleep_ns_list, reverse=True)
    # sleep_ns_list = [0, 16, 32768, 131072]
    # sleep_ns_list   = [0, 500, 1000, 1500, 2000, 2500]
    # sleep_ns_list = [0, 100, 800, 1600, 2400, 3600, 7200, 16384]
    # sleep_ns_list = [0, 100, 200, 300]
    # sleep_ns_list = sorted(sleep_ns_list, reverse=True)
    # sleep_ns_list = [0, 100, 200, 500, 600, 700, 800]
    # sleep_ns_list = [0, 1021,1022,1023,1024] # warp 단위로 sleep 할 경우
    # sleep_ns_list = [0, 500, 1000, 1500, 2047, 2048] ## sm bubbling sleep
    # sleep_ns_list = [0,500, 1000, 1500, 2040, 2200, 2300, 2400] ## sm bubbling sleep
    # sleep_ns_list = [0,30000, 40000, 50000, 60000] ## sm 188
    # sleep_ns_list = [0, 2045, 2047, 2048] ## sm bubbling clock64
    # sleep_ns_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] ## sm step sleep
    sleep_ns_list = [0, 2045, 2047]
    sleep_ns_list = [0]
    # sleep_ns_list = [0, 1000, 2000, 3000, 4000, 5000]
    # sleep_ns_list = [0, 1000, 2000, 3000, 4000, 5000]
    # sleep_ns_list = [0, 9, 9, 9, 9]
    # sleep_ns_list = [0, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    # sleep_ns_list = [140,141,142,143,144,145,146,147,148,149,150]
    # sleep_freq_list = [0, 1, 4, 16, 64, 256, 1024]
    sleep_freq_list = [32768, 16384, 8192, 4096, 2048, 1024, 256, 64, 16, 4, 1, 0]
    sleep_freq_list = [1024, 256, 64, 16, 4, 1, 0]
    # sleep_freq_list = [1, 2, 4]
    sleep_freq_list = [2, 4, 6, 8, 10, 12]
    sleep_freq_list = [0]
    # clock_list = [2430, 2130, 1830, 1530, 1230, 930]
    clock_list = [2430]
    # clock_list = [2430]
    # freq=0 : non-custom 커널만 동작 (custom_bf16 제외)
    # freq>0 : custom_bf16 만 동작 (sleep_freq_list 중 >0 인 값 순회)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    results = []
    actual_forward_results = []
    mlp_graph_results = []
    detail_sink = None
    stem, _ext = os.path.splitext(args.output_csv)
    detail_csv_path = f"{stem}_{pid}_detail.csv"
    actual_forward_csv = f"{stem}_{pid}_actual_forward.csv"
    mlp_graph_csv = f"{stem}_{pid}_mlp_graph_replay.csv"
    detail_sink = DetailCsvSink(detail_csv_path)
    print(f"Detail CSV (streaming, NVML Δt={args.nvml_interval}s): {detail_csv_path}")

    # seq_len_list    = [1, 256, 512, 1024, 2048, 4096, 8192]
    # seq_len_list    = [256, 512, 1024, 2048, 4096, 8192]
    # seq_len_list    = [256, 8192]
    # seq_len_list = [256]
    # seq_len_list    = [8192, 1]
    seq_len_list    = [8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]
    # seq_len_list = [8192]
    # batch_size_list = [16, 32, 64, 128, 512, 1024]
    batch_size_list = [64]
    # batch_size_list = [16, 16, 16, 16, 16,16 ]
    # batch_size_list = [8, 16, 32 ,64]

    # for i in range(0, 16):
    #     # if i % 4 == 0:
    #     seq_len_list.append(8192)
        # else:
        #     seq_len_list.append(1)
        # batch_size_list.append(16)
    print(seq_len_list, batch_size_list)
    enable_persistence_mode = args.pm
    clock_changed = False
    persistence_mode = None
    try:
        if enable_persistence_mode:
            try:
                persistence_mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
                print(persistence_mode)
                print(f"Current Persistence Mode: {'Enabled' if persistence_mode == pynvml.NVML_FEATURE_ENABLED else 'Disabled'}")
            except Exception as e:
                print(f"Warning: Could not get persistence mode: {e}")
                
            if persistence_mode != pynvml.NVML_FEATURE_ENABLED:
                try:
                    pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
                    print("✅ Persistence Mode Enabled")
                    persistence_mode = pynvml.NVML_FEATURE_ENABLED
                except Exception as e:
                    print(f"Warning: Could not set persistence mode: {e}")

            #check boost mode (optional - not supported on datacenter GPUs)
            try:
                boost_mode = pynvml.nvmlDeviceGetAutoBoostedClocksEnabled(handle)
                print(f"Auto Boosted Clocks Enabled: {boost_mode}")
            except pynvml.NVMLError as e:
                # This is expected on datacenter GPUs like L40S, A100, H100, etc.
                if e.value == pynvml.NVML_ERROR_NOT_SUPPORTED:
                    print(f"ℹ️  Auto Boosted Clocks: Not supported on this GPU (expected for datacenter GPUs)")
                else:
                    print(f"Warning: Could not get auto boosted clocks status: {e}")


        # Print initial GPU info
        print("\nInitial GPU State:")
        print_gpu_info(handle)
        

        for S in seq_len_list:
            for batch_sz in batch_size_list:
                # if batch_sz * S >= 1024 * 512:
                #     print(f"  ⚠ Skipping B={batch_sz}, S={S} "
                #           f"(tokens={batch_sz*S} > {1024*1024})")
                #     continue
                # else:
                #     print(f"Operating B={batch_sz}, S={S} (tokens={batch_sz*S} <= {1024*1024})")
                matmuls = build_layer_matmuls(batch_sz, S)
                total_flops_fwd = sum(2.0 * m * k * n * cnt
                                      for m, k, n, _, cnt in matmuls)
                if workload_enabled("full_forward"):
                    print(f"\n{'='*72}")
                    print(
                        f"  Actual Qwen3-8B forward kernel sequence  "
                        f"B={batch_sz}  S={S}  layers={ACTUAL_FORWARD_LAYERS}  "
                        "(FlashAttention + real GEMM launches, no estimate)")
                    print(f"{'='*72}")
                    actual_cutlass_base = None
                    actual_cutlass_by_sleep = {}
                    for _kernel in ("cutlass_sm80", "cublas", "persistent_cta"):
                        if not kernel_enabled(_kernel):
                            continue
                        if _kernel == "cutlass_sm80" and cutlass_sm80 is None:
                            continue
                        if _kernel == "persistent_cta" and persistent_sm80 is None:
                            continue
                        _sleep_iter = (
                            sleep_ns_list
                            if _kernel in ("cutlass_sm80", "persistent_cta")
                            else [0])
                        _freq_iter = sleep_freq_list if _kernel != "cublas" else [0]
                        for _sns in _sleep_iter:
                            for _sfreq in _freq_iter:
                                row = measure_qwen3_actual_forward(
                                    batch_sz, S, _kernel, _sns, _sfreq,
                                    cutlass_sm80, persistent_sm80, handle, args)
                                if row is not None:
                                    if _kernel == "cutlass_sm80":
                                        ref = {
                                            "gflops": row["gflops"],
                                            "sm_clock_mhz": row["sm_clock_mhz"],
                                        }
                                        actual_cutlass_by_sleep[(_sns, _sfreq)] = ref
                                        if actual_cutlass_base is None and _sns == 0:
                                            actual_cutlass_base = ref
                                    cmp_parts = _cutlass_cmp_parts(
                                        row["gflops"], row["sm_clock_mhz"],
                                        actual_cutlass_base, "cutlass_sm80@0ns")
                                    same_sleep_ref = actual_cutlass_by_sleep.get((_sns, _sfreq))
                                    if same_sleep_ref is not None and _kernel != "cutlass_sm80":
                                        cmp_parts.extend(_cutlass_cmp_parts(
                                            row["gflops"], row["sm_clock_mhz"],
                                            same_sleep_ref,
                                            f"cutlass_sm80@sleep={_sns}ns"))
                                    if cmp_parts:
                                        print(f"    compare            | [{' | '.join(cmp_parts)}]")
                                    row["label"] = args.run_label.strip() or test_case
                                    row["pid"] = pid
                                    row["test_case"] = "QWEN3_8B_ACTUAL_FORWARD"
                                    actual_forward_results.append(row)
                                time.sleep(1)

                # 기준 토큰 수 (256 × 16 = 4096) 대비 비율로 warmup/ni 스케일
                if batch_sz * S >= 256 * 16:
                    _BASE_TOKENS   = 256 * 16
                    _tokens_scale  = _BASE_TOKENS / max(1, batch_sz * S)
                    _tokens_scale = 1
                    warmup_scaled  = max(1, int(args.warmup_iters * _tokens_scale))
                    warmup_scaled = 0
                    ni_base        = max(1, int(50 * 50 * _tokens_scale))
                    ni_base_kernel        = max(1, int(50 * 50 * _tokens_scale))
                    ni_base = 50
                    ni_base_kernel = 50
                    # ni_base_kernel = max(1, int(50 * 50 * 1.2 * 20 * _tokens_scale))
                    # ni_base = 1
                    # ni_base_kernel = 1
                else:
                    _BASE_TOKENS   = 8
                    _tokens_scale  = _BASE_TOKENS / max(1, batch_sz * S)
                    _tokens_scale = 1
                    warmup_scaled  = max(1, int(1280000 * _tokens_scale))
                    warmup_scaled = 0
                    ni_base        = max(1, int(128000 * _tokens_scale))
                    ni_base_kernel        = max(1, int(128000 * _tokens_scale))
                    ni_base = 50
                    ni_base_kernel = 50
                    # ni_base_kernel = max(1, int(64 * 1000 * 3 * _tokens_scale))
                    # ni_base = 1
                    # ni_base_kernel = 1
                warmup_scaled = 300
                ni_base = 300
                ni_base_kernel = 300
                print(f"\n{'#'*72}")
                print(f"  Qwen3-8B prefill  B={batch_sz}  S={S}  tokens={batch_sz*S}  "
                      f"total={total_flops_fwd/1e9:.1f} GFLOP")
                print(f"{'#'*72}")

                # ───────────────────────────────────────────
                #  Per-layer breakdown
                # ───────────────────────────────────────────
                for M, K, N, name, count in (
                    matmuls if workload_enabled("matmul_only") else []):
                        flops_per_gemm = 2.0 * M * K * N
                        tag = f"{name}(×{count})"

                        # Tile decomposition for wmma_gemm_sleep_kernel (16×16×16)
                        tilesM = M // 16
                        tilesN = N // 16
                        tilesK = K // 16
                        totalTiles = tilesM * tilesN
                        warpsPerBlk = 8 if totalTiles >= 1024 else 4
                        numBlocks = (totalTiles + warpsPerBlk - 1) // warpsPerBlk
                        totalWarps = numBlocks * warpsPerBlk

                        print(f"\n{'─'*72}")
                        print(f"  {tag}:  ({M}×{K}) @ ({K}×{N})  "
                            f"[{flops_per_gemm/1e9:.3f} GFLOP/op]")
                        print(f"  tiles: M={tilesM} N={tilesN} K={tilesK}  "
                            f"total_output_tiles={totalTiles}  "
                            f"warps/blk={warpsPerBlk}  blocks={numBlocks}  "
                            f"total_warps={totalWarps}")
                        print(f"{'─'*72}")

                        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
                        B = torch.randn(K, N, dtype=torch.float16, device="cuda")
                        # CUDA Graph 캡처용 고정 출력 버퍼 (매 호출 alloc 방지)
                        C_fp16 = torch.empty(M, N, dtype=torch.float16, device="cuda")
                        C_bf16 = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

                        # BF16 텐서 (cuBLAS BF16 / WMMA BF16 공용)
                        A_bf16 = A.bfloat16()
                        B_bf16 = B.bfloat16()

                        # FP8 tcgen05용 사전 변환 (weight B는 상수이므로 1회만 변환)
                        # _scaled_mm: A(M,K) row-major × B.T(K,N) col-major → (M,N)
                        # → B를 (N,K) row-major FP8로 저장
                        # _fp8_scale_one = torch.tensor(1.0, device="cuda", dtype=torch.float32)
                        # _B_fp8_NK = B_bf16.T.contiguous().to(torch.float8_e4m3fn)  # (N,K)

                        # def _fp8_gemm(A_b, B_fp8_NK, sa, sb):
                        #     Aq = A_b.to(torch.float8_e4m3fn)        # (M,K) BF16→FP8 (fast cast)
                        #     return torch._scaled_mm(Aq, B_fp8_NK.T,  # B.T: (K,N) col-major
                        #                             scale_a=sa, scale_b=sb,
                        #                             out_dtype=torch.bfloat16)

                        # ── GPU Boost Clock Burn-in ──────────────────────────────────────
                        # idle → boost clock 전환은 수백ms 걸릴 수 있어
                        # warmup_iters 만으로 부족할 때 이 루프가 보완한다.
                        # 최대 SM clock 의 90% 이상에 도달하거나 3초가 지나면 종료.
                        for clock in clock_list:
                            if enable_persistence_mode:
                                set_specific_clock(handle, 0, clock, 3)
                                clock_changed = True
                            _burn_target_pct = 0.90
                            _burn_timeout_s  = 3.0
                            _burn_start      = time.time()
                            try:
                                _max_sm_clk = pynvml.nvmlDeviceGetMaxClockInfo(
                                    handle, pynvml.NVML_CLOCK_SM)
                            except Exception:
                                _max_sm_clk = 2400
                            if kernel_enabled("cublas"):
                                while True:
                                    for _ in range(20):
                                        torch.mm(A_bf16, B_bf16, out=C_bf16)
                                    torch.cuda.synchronize()
                                    try:
                                        _cur_clk = pynvml.nvmlDeviceGetClockInfo(
                                            handle, pynvml.NVML_CLOCK_SM)
                                    except Exception:
                                        _cur_clk = _max_sm_clk
                                    if (_cur_clk >= _max_sm_clk * _burn_target_pct or
                                            time.time() - _burn_start >= _burn_timeout_s):
                                        break
                            # ─────────────────────────────────────────────────────────────────

                            gf = None
                            if kernel_enabled("cublas"):
                                # ── cuBLAS FP16 baseline ──
                                fn_cublas = lambda: torch.mm(A, B, out=C_fp16)
                                # ni = auto_iterations(fn_cublas, args.target_seconds)
                                # warmup_scaled = int(ni * 0.5)
                                warmup_scaled = 0
                                # for _ in range(warmup_scaled):
                                #     fn_cublas()
                                # ni = auto_iterations(fn_cublas, args.target_seconds)
                                # ni = ni_base
                                ni = 100
                                if M>= 524288:
                                    ni = 30
                                elif M>= 262144:
                                    ni = 60
                                elif M>= 131072:
                                    ni = 120
                                elif M >= 65536:
                                    ni = ni * 2
                                elif M>= 32768:
                                    ni = ni * 4
                                elif M>= 16384:
                                    ni = ni * 8
                                elif M>= 8192:
                                    ni = ni * 16
                                elif M>= 4096:
                                    ni = ni * 32
                                elif M>= 2048:
                                    ni = ni * 64
                                elif M>= 1024:
                                    ni = ni * 128
                                elif M>= 512:
                                    ni = ni * 256
                                elif M>= 256:
                                    ni = ni * 512
                                elif M>= 128:
                                    ni = ni * 1024
                                elif M>= 64:
                                    ni = ni * 2048
                                elif M>= 32:
                                    ni = ni * 4096
                                elif M>= 16:
                                    ni = ni * 8192
                                elif M>= 8:
                                    ni = ni * 16384
                                # _cg_fp16 = cuda_graph_dot_path(
                                #     args.cuda_graph_dump_dir, batch_sz, S, name, "cublas_fp16")
                                # cuda_graph_capture_debug_dump(fn_cublas, _cg_fp16)
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.nvtx.range_push(f"{tag} cuBLAS sleep=0ns")
                                el, en, pw, sm_clk, temp, pv_bef, ref_bef, pv_dur, ref_dur, pv_rat, gpu_power_cap_w, nvml_samples, gpu_ms = measure(
                                    fn_cublas, ni, handle, sample_interval=args.nvml_interval)
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.nvtx.range_pop()
                                timing_ms = gpu_ms if gpu_ms > 0 else el
                                gf = flops_per_gemm * ni / (timing_ms / 1000) / 1e9
                                nj = en * 1e6 / (flops_per_gemm * ni) if ni > 0 else 0
                                print(f"  {'cuBLAS':14s} | sleep=     0ns freq=   0  iters={ni:5d} | "
                                    f"wall={el:8.1f}ms gpu={gpu_ms:8.1f}ms "
                                    f"{gf:8.1f}GF {pw:6.1f}W "
                                    f"{en:8.0f}mJ {nj:.3f}nJ/F  SM={sm_clk:.0f}MHz  T={temp:.0f}°C"
                                    f"  PVpre={pv_bef}ns/{ref_bef}ns"
                                    f"  PVrun={pv_dur}ns/{ref_dur}ns ({pv_rat * 100:.1f}% NVML ref)")
                                rec = dict(
                                    batch_size=batch_sz, seq_len=S,
                                    layer=name, count=count,
                                    M=M, K=K, N=N, kernel="cublas",
                                    sleep_ns=0, sleep_freq=0, num_iters=ni,
                                    elapsed_ms=el, gpu_elapsed_ms=gpu_ms,
                                    gflops=gf, power_w=pw,
                                    energy_mj=en, nj_per_flop=nj, sm_clock_mhz=sm_clk,
                                    temp_c=temp,
                                    gpu_power_cap_w=gpu_power_cap_w,
                                    power_violation_before_ns=pv_bef,
                                    reference_time_before_ns=ref_bef,
                                    power_violation_during_ns=pv_dur,
                                    reference_time_during_ns=ref_dur,
                                    power_violation_ratio=pv_rat,
                                    _nvml_samples=nvml_samples)
                                commit_measure(results, rec, detail_sink, args, test_case, pid)
                                time.sleep(1)
                            else:
                                print("  cuBLAS skipped by --matmul-kernel-mask")
                            if kernel_enabled("cublas"):
                                # ── cuBLAS BF16 baseline ──
                                fn_cublas_bf16 = lambda: torch.mm(A_bf16, B_bf16, out=C_bf16)
                                # for _ in range(warmup_scaled):
                                #     fn_cublas_bf16()
                                ni_bf16 = auto_iterations(fn_cublas_bf16, args.target_seconds)
                                ni_bf16 = ni_base
                                _cg_bf16 = cuda_graph_dot_path(
                                    args.cuda_graph_dump_dir, batch_sz, S, name, "cublas_bf16")
                                cuda_graph_capture_debug_dump(fn_cublas_bf16, _cg_bf16)
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.nvtx.range_push(f"{tag} cuBLAS_BF16 sleep=0ns")
                            # el_b, en_b, pw_b, sm_clk_b, temp_b, pv_bef_b, ref_bef_b, pv_dur_b, ref_dur_b, pv_rat_b = measure(
                            #     fn_cublas_bf16, ni_bf16, handle)
                            # if torch.cuda.is_available():
                            #     torch.cuda.synchronize()
                            #     torch.cuda.nvtx.range_pop()
                            # gf_b = flops_per_gemm * ni_bf16 / (el_b / 1000) / 1e9
                            # nj_b = en_b * 1e6 / (flops_per_gemm * ni_bf16) if ni_bf16 > 0 else 0
                            # print(f"  {'cuBLAS_BF16':14s} | sleep=     0ns freq=   0  iters={ni_bf16:5d} | "
                            #     f"{el_b:8.1f}ms {gf_b:8.1f}GF {pw_b:6.1f}W "
                            #     f"{en_b:8.0f}mJ {nj_b:.3f}nJ/F  SM={sm_clk_b:.0f}MHz  T={temp_b:.0f}°C"
                            #     f"  PVpre={pv_bef_b}ns/{ref_bef_b}ns"
                            #     f"  PVrun={pv_dur_b}ns/{ref_dur_b}ns ({pv_rat_b * 100:.1f}% NVML ref)")
                            # results.append(dict(
                            #     batch_size=batch_sz, seq_len=S,
                            #     layer=name, count=count,
                            #     M=M, K=K, N=N, kernel="cublas_bf16",
                            #     sleep_ns=0, sleep_freq=0, num_iters=ni_bf16,
                            #     elapsed_ms=el_b, gflops=gf_b, power_w=pw_b,
                            #     energy_mj=en_b, nj_per_flop=nj_b, sm_clock_mhz=sm_clk_b,
                            #     temp_c=temp_b,
                            #     power_violation_before_ns=pv_bef_b,
                            #     reference_time_before_ns=ref_bef_b,
                            #     power_violation_during_ns=pv_dur_b,
                            #     reference_time_during_ns=ref_dur_b,
                            #     power_violation_ratio=pv_rat_b))

                            # ── wmma_gemm_sleep_kernel sweep ──
                            # WMMA kernel outputs FP32 → check if output fits in GPU memory
                            output_bytes = M * N * 4  # FP32
                            free_mem, total_mem = torch.cuda.mem_get_info()
                            wmma_skip = output_bytes > free_mem * 0.8  # leave 20% headroom
                            # if wmma_skip:
                            #     print(f"  ⚠ WMMA skipped: output {output_bytes/1e9:.1f}GB "
                            #         f"> avail {free_mem/1e9:.1f}GB (FP32 too large)")
                            #     continue

                            # ── 커널 sweep 테이블: (kernel_name, fn_factory, has_sleep) ──
                            # has_sleep=True  → sleep_ns_list 전체를 순회
                            # has_sleep=False → sleep_ns=0 한 번만 측정
                            kernel_sweep = []

                            if wmma_sleep_gemm is not None and hasattr(wmma_sleep_gemm, 'gemm_opt'):
                                kernel_sweep.append((
                                    "wmma_opt",
                                    lambda _s, _f: wmma_sleep_gemm.gemm_opt(A, B, _s, _f),
                                    True,   # sleep 지원: sleep_ns_list 전체를 순회
                                ))
                            # if hasattr(wmma_sleep_gemm, 'gemm_hiperf'):
                            #     kernel_sweep.append((
                            #         "wmma_hiperf",
                            #         lambda _s, _f: wmma_sleep_gemm.gemm_hiperf(A, B, _s, _f),
                            #         True,
                            #     ))
                            if wmma_sleep_gemm is not None:
                                kernel_sweep.append((
                                    "wmma_simple",
                                    lambda _s, _f: wmma_sleep_gemm.gemm(A, B, _s, _f, False),
                                    True,
                                ))

                            # ── BF16 WMMA kernels (nanosleep before store_matrix_sync) ────────
                            # if bf16_gemm is not None:
                            #     _ok16  = (M % 16  == 0 and N % 16  == 0 and K % 16  == 0)
                            #     _ok128 = (M % 128 == 0 and N % 128 == 0 and K % 64  == 0)

                            #     if _ok16:
                            #         kernel_sweep.append((
                            #             "wmma_bf16",       # naive: 참조/sleep 효과 측정용
                            #             lambda _s, _f, _A=A_bf16, _B=B_bf16:
                            #                 bf16_gemm.gemm(_A, _B, _s, _f),
                            #             True,
                            #         ))
                            #     else:
                            #         print(f"  ⚠ wmma_bf16 skipped: M/N/K must be multiples of 16")

                            #     if _ok128:
                            #         kernel_sweep.append((
                            #             "wmma_bf16_opt",   # high-perf: smem + float4 + double-buf
                            #             lambda _s, _f, _A=A_bf16, _B=B_bf16:
                            #                 bf16_gemm.gemm_opt(_A, _B, _s, _f),
                            #             True,
                            #         ))
                            #     else:
                            #         print(f"  ⚠ wmma_bf16_opt skipped: M must be ×128, N ×128, K ×64 "
                            #               f"(got {M},{N},{K})")

                            # ── CUTLASS SM80 BF16 GEMM (cuBLAS 동급, 순수 BF16) ─────────────────
                            # CUTLASS 2.x sm80 집합체: cp.async + mma.sync m16n8k16 + 3-stage pipeline
                            # has_sleep=False: CUTLASS 커널 내부에 sleep 삽입 불가
                            if kernel_enabled("cutlass_sm80") and cutlass_sm80 is not None:
                                _A_bf16_cont = A_bf16.contiguous()
                                _B_bf16_cont = B_bf16.contiguous()
                                # cutlass_sm80 기준값을 먼저 확보해 이후 커널 로그에서
                                # 같은 sleep 값 기준 성능/SM clock 비교가 가능하게 한다.
                                kernel_sweep.insert(0, (
                                    "cutlass_sm80",
                                    lambda _s, _f,
                                        _A=_A_bf16_cont, _B=_B_bf16_cont:
                                        cutlass_sm80.gemm_sm80_v3(_A, _B,
                                                                sleep_ns=_s,
                                                                sleep_freq=_f),
                                    True,   # sleep 지원: mma_multistage.h CUTLASS_SLEEP_ENABLED 블록
                                ))

                                # ── CUTLASS SM80 multi-stream variant ──────────────────────────
                                # M(=BS) > 2048 인 경우에만 의미 있는 변종.
                                # M축을 2048 단위 chunk 로 잘라 별도 CUDA stream 에서
                                # async 로 cutlass 커널을 동시에 띄우고, 사전 할당한
                                # (M,N) 출력 텐서의 슬라이스에 결과를 모은다.
                                if M >= 2048:
                                    kernel_sweep.insert(1, (
                                        "cutlass_sm80_multistream",
                                        lambda _s, _f,
                                            _A=_A_bf16_cont, _B=_B_bf16_cont:
                                            cutlass_sm80_gemm_multistream(
                                                _A, _B, cutlass_sm80,
                                                sleep_ns=_s, sleep_freq=_f,
                                                m_chunk=2048),
                                        True,
                                    ))

                            # ── CUTLASS 3.x SM90 BF16 GEMM (wgmma + TMA) ────────────────────────
                            # SM120에서 BF16은 SM90 wgmma 경로 사용 (tcgen05는 FP4/6/8 전용)
                            # cuBLAS가 절반 클럭에도 빠른 이유: wgmma IPC가 mma.sync의 ~2배
                            if cutlass_sm90 is not None:
                                _ok_sm90 = (M % 128 == 0 and N % 128 == 0 and K % 64 == 0)
                                if _ok_sm90:
                                    _A_sm90 = A_bf16.contiguous()
                                    _B_sm90 = B_bf16.contiguous()
                                    kernel_sweep.append((
                                        "cutlass_sm90",
                                        lambda _s, _f, _A=_A_sm90, _B=_B_sm90:
                                            cutlass_sm90.gemm_sm90_bf16(_A, _B),
                                        False,
                                    ))
                                else:
                                    print(f"  ⚠ cutlass_sm90 skipped: M,N must be ×128, K ×64 "
                                        f"(got {M},{N},{K})")

                            # ── PTX SM80 BF16 GEMM (mma.m16n8k16 + ldmatrix + cp.async) ──────────
                            # bf16_gemm_sm80_kernel.cu : 순수 PTX inline-asm 커널
                            # Block 128×128, BK=64, 4 warps, dynamic smem ~71 KB
                            # has_sleep=False: sleep 삽입 없음
                            if ptx_sm80 is not None:
                                # v2: BK=32 → K는 32 배수 필요 (v1: 64)
                                _ok_ptx = (M % 128 == 0 and N % 128 == 0 and K % 32 == 0)
                                if _ok_ptx:
                                    _A_ptx = A_bf16.contiguous()
                                    _B_ptx = B_bf16.contiguous()
                                    kernel_sweep.append((
                                        "ptx_sm80",
                                        lambda _s, _f, _A=_A_ptx, _B=_B_ptx:
                                            ptx_sm80.gemm_sm80_ptx(_A, _B),
                                        False,
                                    ))
                                else:
                                    print(f"  ⚠ ptx_sm80 skipped: M must be ×128, N ×128, K ×32 "
                                        f"(got {M},{N},{K})")

                            # ── Persistent CTA SM80 BF16 GEMM ─────────────────────────────────
                            # CTA가 atomic tile queue에서 chunk를 받아 연속 처리한다.
                            # full dense tile에서만 throttle을 적용하고 tail tile은 latency 보호를 위해 skip.
                            if kernel_enabled("persistent_cta") and persistent_sm80 is not None:
                                _ok_persistent = (N % 8 == 0 and K % 8 == 0)
                                if _ok_persistent:
                                    _A_persistent = A_bf16.contiguous()
                                    _B_persistent = B_bf16.contiguous()
                                    kernel_sweep.append((
                                        "persistent_cta",
                                        lambda _s, _f,
                                            _A=_A_persistent, _B=_B_persistent:
                                            persistent_sm80.gemm_sm80_persistent(
                                                _A,
                                                _B,
                                                _s,
                                                _f,
                                                args.persistent_throttle_mode,
                                                args.persistent_ctas_per_sm,
                                                args.persistent_chunk_tiles),
                                        True,
                                    ))
                                else:
                                    print(f"  ⚠ persistent_cta skipped: N and K must be multiples of 8 "
                                        f"(got {M},{N},{K})")

                            # ── Custom BF16 GEMM with __nanosleep ────────────────────────────────
                            # 이 커널만 sleep_freq > 0 으로 동작.
                            # K-루프 내부에 __nanosleep 삽입 → 파워 duty-cycle 직접 제어.
                            if custom_bf16 is not None:
                                _ok_custom = (M % 16 == 0 and N % 16 == 0 and K % 16 == 0)
                                if _ok_custom:
                                    kernel_sweep.append((
                                        "bf16_custom",   # nanosleep K-loop 커널 (has_sleep=True)
                                        lambda _s, _f, _A=A_bf16, _B=B_bf16:
                                            custom_bf16.gemm_custom(_A, _B,
                                                                    sleep_ns=_s,
                                                                    sleep_freq=_f),
                                        True,   # ← 유일하게 sleep_ns_list 전체를 순회
                                    ))
                                else:
                                    print(f"  ⚠ bf16_custom skipped: M/N/K must be multiples of 16 "
                                        f"(got {M},{N},{K})")

                            # ── FP8 tcgen05 (SM120 신형 텐서코어 경로) ─────────────────────────
                            # B는 이미 사전 변환(_B_fp8_NK), A만 런타임에 BF16→FP8 캐스팅
                            # has_sleep=False: cuBLAS 내부에서 sleep 삽입 불가 → sleep_ns=0 고정
                            # kernel_sweep.append((
                            #     "fp8_tcgen05",
                            #     lambda _s, _f,
                            #            _A=A_bf16, _Bq=_B_fp8_NK,
                            #            _sa=_fp8_scale_one, _sb=_fp8_scale_one:
                            #         _fp8_gemm(_A, _Bq, _sa, _sb),
                            #     False,   # sleep 미지원
                            # ))

                            # cuBLAS baseline: sleep 없음
                            gf_cublas = gf  # 앞서 측정한 cuBLAS GFLOP/s (비율 계산용)
                            cutlass_sm80_base = None
                            cutlass_sm80_by_sleep = {}

                            def _fmt_ratio(v, base):
                                return f"{(v / base):.2f}×" if base and base > 0 else "n/a"

                            def _fmt_sm_delta(v, base):
                                if not base or base <= 0:
                                    return "n/a"
                                return f"{(v / base):.2f}×, Δ{(v - base):+.0f}MHz"

                            # gf_cublas = 100
                            for kname, kfn, has_sleep in kernel_sweep:
                                sns_iter = sleep_ns_list
                                for sns in sns_iter:
                                    for sfreq in sleep_freq_list:
                                        # if kname == "ptx_sm80" or kname == "cutlass_sm80" or kname == "bf16_custom":
                                        # cutlass_sm80_multistream 은 cutlass_sm80 의 wrapper 이므로
                                        # cutlass_sm80 활성화 비트(0x2)에 연동되어 켜진다.
                                        _kparent = (
                                            "cutlass_sm80"
                                            if kname == "cutlass_sm80_multistream"
                                            else kname
                                        )
                                        if (
                                            kname in (
                                                "cutlass_sm80",
                                                "cutlass_sm80_multistream",
                                                "persistent_cta",
                                            )
                                            and kernel_enabled(_kparent)
                                        ):
                                            pass
                                        else:
                                            continue
                                        fn = lambda _s=sns, _f=sfreq: kfn(_s, _f)
                                        # ni = auto_iterations(fn, args.target_seconds)
                                        # warmup_scaled = int(ni * 0.5)
                                        warmup_scaled = 100
                                        ni = 100
                                        if M>= 524288:
                                            ni = 30
                                        elif M>= 262144:
                                            ni = 60
                                        elif M>= 131072:
                                            ni = 120
                                        elif M >= 65536:
                                            ni = ni * 2
                                        elif M>= 32768:
                                            ni = ni * 4
                                        elif M>= 16384:
                                            ni = ni * 8
                                        elif M>= 8192:
                                            ni = ni * 16
                                        elif M>= 4096:
                                            ni = ni * 32
                                        elif M>= 2048:
                                            ni = ni * 64
                                        elif M>= 1024:
                                            ni = ni * 128
                                        elif M>= 512:
                                            ni = ni * 256
                                        elif M>= 256:
                                            ni = ni * 512
                                        elif M>= 128:
                                            ni = ni * 1024
                                        elif M>= 64:
                                            ni = ni * 2048
                                        elif M>= 32:
                                            ni = ni * 4096
                                        elif M>= 16:
                                            ni = ni * 8192
                                        elif M>= 8:
                                            ni = ni * 16384
                                        for _ in range(warmup_scaled):
                                            fn()

                                        # ni = auto_iterations(fn, args.target_seconds)
                                        # ni = ni_base_kernel
                                        _cg_k = cuda_graph_dot_path(
                                            args.cuda_graph_dump_dir,
                                            batch_sz, S, name, kname,
                                            sleep_ns=sns, sleep_freq=sfreq)
                                        cuda_graph_capture_debug_dump(fn, _cg_k)
                                        if torch.cuda.is_available():
                                            torch.cuda.synchronize()
                                            torch.cuda.nvtx.range_push(
                                                f"{tag} {kname} sleep={sns}ns freq={sfreq}")
                                        el, en, pw, sm_clk, temp, pv_bef, ref_bef, pv_dur, ref_dur, pv_rat, gpu_power_cap_w, nvml_samples, gpu_ms = measure(
                                            fn, ni, handle, sample_interval=args.nvml_interval)
                                        if torch.cuda.is_available():
                                            torch.cuda.synchronize()
                                            torch.cuda.nvtx.range_pop()
                                        timing_ms = gpu_ms if gpu_ms > 0 else el
                                        gf_k = flops_per_gemm * ni / (timing_ms / 1000) / 1e9
                                        nj_k = en * 1e6 / (flops_per_gemm * ni) if ni > 0 else 0
                                        if kname == "cutlass_sm80":
                                            ref = {
                                                "gflops": gf_k,
                                                "sm_clock_mhz": sm_clk,
                                            }
                                            cutlass_sm80_by_sleep[(sns, sfreq)] = ref
                                            if cutlass_sm80_base is None and sns == 0:
                                                cutlass_sm80_base = ref

                                        cmp_parts = [
                                            f"vs cuBLAS: {_fmt_ratio(gf_k, gf_cublas)}",
                                        ]
                                        if cutlass_sm80_base is not None:
                                            cmp_parts.extend([
                                                "vs cutlass_sm80@0ns: "
                                                f"{_fmt_ratio(gf_k, cutlass_sm80_base['gflops'])}",
                                                "SM/cutlass_sm80@0ns: "
                                                f"{_fmt_sm_delta(sm_clk, cutlass_sm80_base['sm_clock_mhz'])}",
                                            ])

                                        same_sleep_ref = cutlass_sm80_by_sleep.get((sns, sfreq))
                                        if same_sleep_ref is not None and kname != "cutlass_sm80":
                                            cmp_parts.extend([
                                                f"vs cutlass_sm80@sleep={sns}ns: "
                                                f"{_fmt_ratio(gf_k, same_sleep_ref['gflops'])}",
                                                f"SM/cutlass_sm80@sleep={sns}ns: "
                                                f"{_fmt_sm_delta(sm_clk, same_sleep_ref['sm_clock_mhz'])}",
                                            ])

                                        print(f"  {kname:14s} | sleep={sns:6d}ns freq={sfreq:4d}  "
                                            f"iters={ni:5d} | "
                                            f"wall={el:8.1f}ms gpu={gpu_ms:8.1f}ms "
                                            f"{gf_k:8.1f}GF {pw:6.1f}W "
                                            f"{en:8.0f}mJ {nj_k:.3f}nJ/F  "
                                            f"SM={sm_clk:.0f}MHz  T={temp:.0f}°C  "
                                            f"PVpre={pv_bef}ns/{ref_bef}ns "
                                            f"PVrun={pv_dur}ns/{ref_dur}ns ({pv_rat * 100:.1f}% NVML ref)  "
                                            f"[{' | '.join(cmp_parts)}]")
                                        rec = dict(
                                            batch_size=batch_sz, seq_len=S,
                                            layer=name, count=count,
                                            M=M, K=K, N=N, kernel=kname,
                                            sleep_ns=sns, sleep_freq=sfreq, num_iters=ni,
                                            elapsed_ms=el, gpu_elapsed_ms=gpu_ms,
                                            gflops=gf_k, power_w=pw,
                                            energy_mj=en, nj_per_flop=nj_k,
                                            sm_clock_mhz=sm_clk, temp_c=temp,
                                            gpu_power_cap_w=gpu_power_cap_w,
                                            power_violation_before_ns=pv_bef,
                                            reference_time_before_ns=ref_bef,
                                            power_violation_during_ns=pv_dur,
                                            reference_time_during_ns=ref_dur,
                                            power_violation_ratio=pv_rat,
                                            _nvml_samples=nvml_samples)
                                        commit_measure(results, rec, detail_sink, args, test_case, pid)
                                    
                                        time.sleep(1)

                if workload_enabled("mlp_graph_replay"):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    time.sleep(1)
                    print(f"\n{'='*72}")
                    print(
                        "  MLP CUDA Graph replay workload  "
                        f"B={batch_sz}  S={S}  layers={ACTUAL_FORWARD_LAYERS}  "
                        "(mlp_up -> mlp_gate -> mlp_down) x 32")
                    print(f"{'='*72}")
                    graph_cutlass_base = None
                    graph_cutlass_by_sleep = {}
                    for _kernel in ("cutlass_sm80", "cublas", "persistent_cta"):
                        if not kernel_enabled(_kernel):
                            continue
                        if _kernel == "cutlass_sm80" and cutlass_sm80 is None:
                            continue
                        if _kernel == "persistent_cta" and persistent_sm80 is None:
                            continue
                        _sleep_iter = (
                            sleep_ns_list
                            if _kernel in ("cutlass_sm80", "persistent_cta")
                            else [0])
                        _freq_iter = sleep_freq_list if _kernel != "cublas" else [0]
                        for _sns in _sleep_iter:
                            for _sfreq in _freq_iter:
                                row = measure_qwen3_mlp_graph_replay(
                                    batch_sz, S, _kernel, _sns, _sfreq,
                                    cutlass_sm80, persistent_sm80, handle, args)
                                if row is not None:
                                    if _kernel == "cutlass_sm80":
                                        ref = {
                                            "gflops": row["gflops"],
                                            "sm_clock_mhz": row["sm_clock_mhz"],
                                        }
                                        graph_cutlass_by_sleep[(_sns, _sfreq)] = ref
                                        if graph_cutlass_base is None and _sns == 0:
                                            graph_cutlass_base = ref
                                    cmp_parts = _cutlass_cmp_parts(
                                        row["gflops"], row["sm_clock_mhz"],
                                        graph_cutlass_base, "cutlass_sm80@0ns")
                                    same_sleep_ref = graph_cutlass_by_sleep.get((_sns, _sfreq))
                                    if same_sleep_ref is not None and _kernel != "cutlass_sm80":
                                        cmp_parts.extend(_cutlass_cmp_parts(
                                            row["gflops"], row["sm_clock_mhz"],
                                            same_sleep_ref,
                                            f"cutlass_sm80@sleep={_sns}ns"))
                                    if cmp_parts:
                                        print(f"    compare            | [{' | '.join(cmp_parts)}]")
                                    row["label"] = args.run_label.strip() or test_case
                                    row["pid"] = pid
                                    row["test_case"] = "QWEN3_8B_MLP_GRAPH_REPLAY"
                                    mlp_graph_results.append(row)
                                time.sleep(1)
                    #  Full-forward estimate  (aggregate per sleep_ns)
                    # ───────────────────────────────────────────
                # print(f"\n{'='*72}")
                # print(f"  Full-forward estimate  (B={batch_sz}, S={S})")
                # print(f"{'='*72}")
                # print(f"  {'kernel':14s} | {'sleep_ns':>8s} {'freq':>5s} | "
                #     f"{'time_ms':>9s} {'power_W':>8s} {'energy_mJ':>10s} "
                #     f"{'GFLOPS':>8s} {'nJ/FLOP':>8s} {'SM_MHz':>8s} {'Temp_C':>7s}")

                # all_kernels = (
                #     [(0, 0, "cublas")]
                #     + [(0, 0, "cublas_bf16")]
                #     + [(0, 0, "cutlass_sm90")]  # CUTLASS 3.x wgmma+TMA (SM90/SM120)
                #     + [(0, 0, "ptx_sm80")]     # PTX inline-asm SM80 kernel
                #     + [(0, 0, "fp8_tcgen05")]
                #     + [(0, 0, "wmma_opt")]     # freq=0 고정 (non-custom)
                #     # cutlass_sm80: sleep_ns × (freq>0) 전체 조합 (CUTLASS_SLEEP_ENABLED)
                #     + [(s, f, "cutlass_sm80")
                #     for s in sleep_ns_list
                #     for f in sleep_freq_list if f > 0]
                #     # bf16_custom: sleep_ns × (freq>0) 전체 조합
                #     + [(s, f, "bf16_custom")
                #     for s in sleep_ns_list
                #     for f in sleep_freq_list if f > 0]
                # )
                # for sns, sfreq, kern in all_kernels:

                #     matched = [r for r in results
                #             if r["kernel"] == kern
                #             and r["sleep_ns"] == sns
                #             and r["sleep_freq"] == sfreq]
                #     if not matched:
                #         continue

                #     fwd_time_ms   = 0.0
                #     fwd_energy_mj = 0.0
                #     # SM clock & temperature: 측정 시간 가중 평균 (elapsed_ms 비례)
                #     sm_clk_sum    = 0.0
                #     temp_sum      = 0.0
                #     weight_sum    = 0.0
                #     for r in matched:
                #         per_op_ms = r["elapsed_ms"] / r["num_iters"]
                #         per_op_mj = r["energy_mj"] / r["num_iters"]
                #         w = per_op_ms * r["count"]
                #         fwd_time_ms   += w
                #         fwd_energy_mj += per_op_mj * r["count"]
                #         sm_clk_sum    += r.get("sm_clock_mhz", 0.0) * w
                #         temp_sum      += r.get("temp_c", 0.0) * w
                #         weight_sum    += w

                #     fwd_power  = fwd_energy_mj / fwd_time_ms if fwd_time_ms > 0 else 0
                #     fwd_gflops = total_flops_fwd / (fwd_time_ms / 1000) / 1e9 \
                #         if fwd_time_ms > 0 else 0
                #     fwd_nj     = fwd_energy_mj * 1e6 / total_flops_fwd \
                #         if total_flops_fwd > 0 else 0
                #     fwd_sm_mhz = sm_clk_sum / weight_sum if weight_sum > 0 else 0
                #     fwd_temp_c = temp_sum   / weight_sum if weight_sum > 0 else 0

                #     print(f"  {kern:14s} | {sns:8d} {sfreq:5d} | {fwd_time_ms:9.3f} "
                #         f"{fwd_power:8.1f} {fwd_energy_mj:10.1f} {fwd_gflops:8.1f} "
                #         f"{fwd_nj:8.4f} {fwd_sm_mhz:8.0f} {fwd_temp_c:7.1f}")

        # ═══════════════════════════════════════════════════════════
        #  Summary CSV (detail 은 측정마다 detail_csv_path 에 즉시 기록됨)
        # ═══════════════════════════════════════════════════════════
        if results:
            run_label = (args.run_label.strip() or test_case)

            summary_csv = f"{stem}_{pid}_summary.csv"

            # ── 요약: (batch, seq, kernel, sleep_ns, sleep_freq) 별 full-forward 스타일 집계 ──
            grp = defaultdict(list)
            for r in results:
                key = (
                    r["batch_size"],
                    r["seq_len"],
                    r["kernel"],
                    r["sleep_ns"],
                    r["sleep_freq"],
                )
                grp[key].append(r)

            summary_rows = []
            for key in sorted(grp.keys()):
                rows = grp[key]
                batch_size, seq_len, kernel, sleep_ns, sleep_freq = key
                fwd_time_ms = sum(
                    (row["elapsed_ms"] / max(1, row["num_iters"])) * row["count"]
                    for row in rows)
                fwd_energy_mj = sum(
                    (row["energy_mj"] / max(1, row["num_iters"])) * row["count"]
                    for row in rows)
                weight_sum = sum(
                    (row["elapsed_ms"] / max(1, row["num_iters"])) * row["count"]
                    for row in rows)
                sm_clk_sum = sum(
                    row.get("sm_clock_mhz", 0.0)
                    * (row["elapsed_ms"] / max(1, row["num_iters"])) * row["count"]
                    for row in rows)
                temp_sum = sum(
                    row.get("temp_c", 0.0)
                    * (row["elapsed_ms"] / max(1, row["num_iters"])) * row["count"]
                    for row in rows)
                total_flops = sum(
                    2.0 * row["M"] * row["K"] * row["N"] * row["count"]
                    for row in rows)
                fwd_power_w = (
                    fwd_energy_mj / fwd_time_ms if fwd_time_ms > 0 else 0.0)
                fwd_gflops = (
                    total_flops / (fwd_time_ms / 1000.0) / 1e9
                    if fwd_time_ms > 0 else 0.0)
                fwd_nj = (
                    fwd_energy_mj * 1e6 / total_flops if total_flops > 0 else 0.0)
                fwd_sm_mhz = sm_clk_sum / weight_sum if weight_sum > 0 else 0.0
                fwd_temp_c = temp_sum / weight_sum if weight_sum > 0 else 0.0

                summary_rows.append({
                    "label": run_label,
                    "pid": pid,
                    "test_case": test_case,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "kernel": kernel,
                    "sleep_ns": sleep_ns,
                    "sleep_freq": sleep_freq,
                    "layer_rows": len(rows),
                    "fwd_time_ms": round(fwd_time_ms, 4),
                    "fwd_energy_mj": round(fwd_energy_mj, 4),
                    "fwd_power_w": round(fwd_power_w, 4),
                    "fwd_gflops": round(fwd_gflops, 4),
                    "fwd_nj_per_flop": round(fwd_nj, 4),
                    "total_flops_fwd": round(total_flops, 2),
                    "fwd_sm_mhz": round(fwd_sm_mhz, 4),
                    "fwd_temp_c": round(fwd_temp_c, 4),
                })

            summary_fields = [
                "label", "pid", "test_case",
                "batch_size", "seq_len", "kernel", "sleep_ns", "sleep_freq",
                "layer_rows",
                "fwd_time_ms", "fwd_energy_mj", "fwd_power_w",
                "fwd_gflops", "fwd_nj_per_flop", "total_flops_fwd",
                "fwd_sm_mhz", "fwd_temp_c",
            ]
            with open(summary_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=summary_fields)
                w.writeheader()
                for sr in summary_rows:
                    w.writerow(sr)

            if actual_forward_results:
                actual_forward_fields = [
                    "label", "pid", "test_case",
                    "batch_size", "seq_len",
                    "num_layers",
                    "matmul_kernel", "sleep_ns", "sleep_freq",
                    "attention_kernel", "num_iters",
                    "elapsed_ms", "gpu_elapsed_ms", "energy_mj", "power_w",
                    "gflops", "nj_per_flop", "total_flops",
                    "sm_clock_mhz", "temp_c",
                    "power_violation_before_ns", "reference_time_before_ns",
                    "power_violation_during_ns", "reference_time_during_ns",
                    "power_violation_ratio",
                ]
                with open(actual_forward_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=actual_forward_fields)
                    w.writeheader()
                    for row in actual_forward_results:
                        out = dict(row)
                        for k in (
                            "elapsed_ms", "gpu_elapsed_ms", "energy_mj",
                            "power_w", "gflops",
                            "nj_per_flop", "total_flops", "sm_clock_mhz",
                            "temp_c", "power_violation_ratio"):
                            if k in out and isinstance(out[k], float):
                                out[k] = round(out[k], 4)
                        w.writerow(out)

            if mlp_graph_results:
                mlp_graph_fields = [
                    "label", "pid", "test_case",
                    "batch_size", "seq_len",
                    "num_layers", "workload",
                    "matmul_kernel", "sleep_ns", "sleep_freq",
                    "num_iters",
                    "elapsed_ms", "gpu_elapsed_ms", "energy_mj", "power_w",
                    "gflops", "nj_per_flop", "total_flops",
                    "sm_clock_mhz", "temp_c",
                    "power_violation_before_ns", "reference_time_before_ns",
                    "power_violation_during_ns", "reference_time_during_ns",
                    "power_violation_ratio",
                ]
                with open(mlp_graph_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=mlp_graph_fields)
                    w.writeheader()
                    for row in mlp_graph_results:
                        out = dict(row)
                        for k in (
                            "elapsed_ms", "gpu_elapsed_ms", "energy_mj",
                            "power_w", "gflops",
                            "nj_per_flop", "total_flops", "sm_clock_mhz",
                            "temp_c", "power_violation_ratio"):
                            if k in out and isinstance(out[k], float):
                                out[k] = round(out[k], 4)
                        w.writerow(out)

            print(f"\n✅ Detail (streaming NVML, label={run_label!r}) → {detail_csv_path}")
            print(f"✅ Summary (aggregated)                  → {summary_csv}")
            if actual_forward_results:
                print(f"✅ Actual forward (FlashAttention + GEMM) → {actual_forward_csv}")
            if mlp_graph_results:
                print(f"✅ MLP graph replay                      → {mlp_graph_csv}")
        else:
            run_label = (args.run_label.strip() or test_case)
            if actual_forward_results:
                actual_forward_fields = [
                    "label", "pid", "test_case",
                    "batch_size", "seq_len",
                    "num_layers",
                    "matmul_kernel", "sleep_ns", "sleep_freq",
                    "attention_kernel", "num_iters",
                    "elapsed_ms", "gpu_elapsed_ms", "energy_mj", "power_w",
                    "gflops", "nj_per_flop", "total_flops",
                    "sm_clock_mhz", "temp_c",
                    "power_violation_before_ns", "reference_time_before_ns",
                    "power_violation_during_ns", "reference_time_during_ns",
                    "power_violation_ratio",
                ]
                with open(actual_forward_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=actual_forward_fields)
                    w.writeheader()
                    for row in actual_forward_results:
                        out = dict(row)
                        for k in (
                            "elapsed_ms", "gpu_elapsed_ms", "energy_mj",
                            "power_w", "gflops",
                            "nj_per_flop", "total_flops", "sm_clock_mhz",
                            "temp_c", "power_violation_ratio"):
                            if k in out and isinstance(out[k], float):
                                out[k] = round(out[k], 4)
                        w.writerow(out)
                print(f"\n✅ Actual forward (FlashAttention + GEMM) → {actual_forward_csv}")

            if mlp_graph_results:
                mlp_graph_fields = [
                    "label", "pid", "test_case",
                    "batch_size", "seq_len",
                    "num_layers", "workload",
                    "matmul_kernel", "sleep_ns", "sleep_freq",
                    "num_iters",
                    "elapsed_ms", "gpu_elapsed_ms", "energy_mj", "power_w",
                    "gflops", "nj_per_flop", "total_flops",
                    "sm_clock_mhz", "temp_c",
                    "power_violation_before_ns", "reference_time_before_ns",
                    "power_violation_during_ns", "reference_time_during_ns",
                    "power_violation_ratio",
                ]
                with open(mlp_graph_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=mlp_graph_fields)
                    w.writeheader()
                    for row in mlp_graph_results:
                        out = dict(row)
                        for k in (
                            "elapsed_ms", "gpu_elapsed_ms", "energy_mj",
                            "power_w", "gflops",
                            "nj_per_flop", "total_flops", "sm_clock_mhz",
                            "temp_c", "power_violation_ratio"):
                            if k in out and isinstance(out[k], float):
                                out[k] = round(out[k], 4)
                        w.writerow(out)
                print(f"✅ MLP graph replay                      → {mlp_graph_csv}")

            if not actual_forward_results and not mlp_graph_results:
                print(f"\n⚠ No workload results were produced (label={run_label!r}).")
    finally:
        if detail_sink is not None:
            detail_sink.close()
        # Detail CSV 가 모두 디스크에 반영된 뒤 자동 플롯 (벤치와 동일한 python)
        _custum_dir = os.path.dirname(os.path.abspath(__file__))
        _plot_script = os.path.join(_custum_dir, "plot_detail_csv.py")
        if (
            results
            and detail_csv_path
            and os.path.isfile(detail_csv_path)
            and os.path.isfile(_plot_script)
        ):
            _png = os.path.splitext(detail_csv_path)[0] + "_plot.png"
            try:
                subprocess.run(
                    [sys.executable, _plot_script, detail_csv_path, "-o", _png],
                    check=False,
                    timeout=180,
                )
            except Exception as e:
                print(f"⚠ plot_detail_csv 자동 실행 실패: {e}")
        _actual_plot_script = os.path.join(_custum_dir, "plot_actual_forward_csv.py")
        if (
            actual_forward_csv
            and os.path.isfile(actual_forward_csv)
            and os.path.isfile(_actual_plot_script)
        ):
            _actual_png = os.path.splitext(actual_forward_csv)[0] + "_plot.png"
            try:
                subprocess.run(
                    [
                        sys.executable,
                        _actual_plot_script,
                        actual_forward_csv,
                        "-o",
                        _actual_png,
                    ],
                    check=False,
                    timeout=180,
                )
            except Exception as e:
                print(f"⚠ plot_actual_forward_csv 자동 실행 실패: {e}")
        stop_sampler()       # sampler 프로세스 정상 종료
        if clock_changed:
            print("\n1. Resetting GPU Locked Clocks...", end=" ")
            try:
                pynvml.nvmlDeviceResetGpuLockedClocks(handle)
                print("✅ Success")
            except pynvml.NVMLError as e:
                # 설정된 적이 없으면 에러가 날 수도 있음
                print(f"⚠️ Skipped ({e})")
            print("2. Resetting Memory Locked Clocks...", end=" ")
            try:
                # 메모리 고정 해제 API 호출
                pynvml.nvmlDeviceResetMemoryLockedClocks(handle)
                print("✅ Success")
            except pynvml.NVMLError as e:
                print(f"⚠️ Skipped ({e})")

        if persistence_mode == pynvml.NVML_FEATURE_ENABLED:
            try:
                pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_DISABLED)
                print("✅ Persistence Mode Disabled")
            except Exception as e:
                print(f"Warning: Could not set persistence mode: {e}")

        try:
            pynvml.nvmlShutdown()
            print("\npynvml shutdown complete")
        except Exception as e:
            print(f"Warning: pynvml shutdown error: {e}")
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    # spawn: CUDA context 를 자식 프로세스에 물려주지 않아 fork-safe
    mp.set_start_method("spawn", force=True)
    main()
