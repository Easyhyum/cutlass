#!/usr/bin/env python3
"""
Flash Attention Benchmark for Qwen3-8B attention layers on Blackwell SM120.

Qwen3-8B attention configuration:
    num_attention_heads = 32  (Q heads)
    num_key_value_heads = 8   (KV heads, GQA ratio = 4)
    head_dim             = 128
    num_hidden_layers    = 36

Attention FLOP counting (per layer, per call):
    Non-causal  : 4 × B × H_q × S_q × S_k × D
    Causal (≈½) : 2 × B × H_q × S²  × D        (S_q = S_k = S)

Kernels compared:
    sdpa_flash   – torch SDPA with Flash-Attention backend (cudnn / FA2)
    sdpa_math    – torch SDPA with math (reference, slow)
    sdpa_mem_eff – torch SDPA with memory-efficient backend
    flash_attn   – flash-attn package (if installed)
    xformers     – xformers.ops.memory_efficient_attention (if installed)

Usage:
    python bench_attn.py --device 3 --output-csv logs/attn_bench.csv
    python bench_attn.py --device 3 --batch-size 1 --seq-lens 512,1024,2048,4096 --causal
"""

import argparse
import time
import os
import csv
import multiprocessing as mp

import torch
import pynvml

# ─────────────────── Qwen3-8B attention constants ────────────────────
NUM_Q_HEADS  = 32
NUM_KV_HEADS = 8
HEAD_DIM     = 128
NUM_LAYERS   = 36
GQA_RATIO    = NUM_Q_HEADS // NUM_KV_HEADS  # 4


# ─────────────────────────── FLOP helpers ────────────────────────────

def attn_flops(B, H_q, S_q, S_k, D, causal):
    """
    근사 FLOP 수:
      QK^T  = 2 × B × H_q × S_q × S_k × D
      AV    = 2 × B × H_q × S_q × S_k × D
    Causal mask 적용 시 평균 절반의 원소만 계산 (S_q == S_k 가정).
    """
    total = 4.0 * B * H_q * S_q * S_k * D
    if causal and S_q == S_k:
        total *= 0.5
    return total


# ─────────────────────────────────────────────────────────────────────
#  auto_iterations
# ─────────────────────────────────────────────────────────────────────

def auto_iterations(fn, target_seconds=2.0, pilot_iters=10):
    """target_seconds 를 채우기 위한 반복 횟수 추정."""
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


# ─────────────────────────────────────────────────────────────────────
#  Persistent NVML sampler process  (bench_mark.py 와 동일한 구조)
# ─────────────────────────────────────────────────────────────────────

def _nvml_sampler_proc(device_idx, interval, cmd_q, result_q, ready_ev):
    import queue as _queue
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    ready_ev.set()

    while True:
        cmd = cmd_q.get()
        if cmd == "exit":
            break

        power_samples, sm_samples, temp_samples = [], [], []
        while True:
            try:
                power_samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            except pynvml.NVMLError:
                pass
            try:
                sm_samples.append(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
            except pynvml.NVMLError:
                pass
            try:
                temp_samples.append(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except pynvml.NVMLError:
                pass
            try:
                next_cmd = cmd_q.get(timeout=interval)
                if next_cmd in ("stop", "exit"):
                    result_q.put((power_samples, sm_samples, temp_samples))
                    if next_cmd == "exit":
                        pynvml.nvmlShutdown()
                        return
                    break
            except _queue.Empty:
                pass

    pynvml.nvmlShutdown()


_sampler_proc     = None
_sampler_cmd_q    = None
_sampler_result_q = None


def start_sampler(device_idx, interval=0.1):
    global _sampler_proc, _sampler_cmd_q, _sampler_result_q
    ctx = mp.get_context("spawn")
    _sampler_cmd_q    = ctx.Queue()
    _sampler_result_q = ctx.Queue()
    ready_ev          = ctx.Event()
    _sampler_proc = ctx.Process(
        target=_nvml_sampler_proc,
        args=(device_idx, interval, _sampler_cmd_q, _sampler_result_q, ready_ev),
        daemon=True,
    )
    _sampler_proc.start()
    ready_ev.wait()


def stop_sampler():
    if _sampler_proc is not None:
        _sampler_cmd_q.put("exit")
        _sampler_proc.join(timeout=5.0)


def measure(fn, num_iters, handle):
    """fn() 을 num_iters 회 실행하고 (elapsed_ms, energy_mJ, power_W, sm_mhz, temp_c) 반환."""
    _sampler_cmd_q.put("start")

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(num_iters):
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    t1 = time.perf_counter()
    _sampler_cmd_q.put("stop")

    elapsed_ms = (t1 - t0) * 1000.0
    try:
        power_samples, sm_samples, temp_samples = _sampler_result_q.get(timeout=5.0)
    except Exception:
        power_samples, sm_samples, temp_samples = [], [], []

    power_W    = sum(power_samples) / len(power_samples) if power_samples else 0.0
    energy_mJ  = power_W * elapsed_ms
    avg_sm_mhz = sum(sm_samples)    / len(sm_samples)    if sm_samples    else 0.0
    avg_temp_c = sum(temp_samples)  / len(temp_samples)  if temp_samples  else 0.0
    return elapsed_ms, energy_mJ, power_W, avg_sm_mhz, avg_temp_c


# ─────────────────────────────────────────────────────────────────────
#  GPU info printer
# ─────────────────────────────────────────────────────────────────────

def print_gpu_info(handle):
    try:
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        print(f"\n{'='*64}")
        print(f"GPU: {gpu_name}")
        print(f"{'='*64}")
        for label, clk_type in [("SM Clock", pynvml.NVML_CLOCK_SM),
                                 ("Memory Clock", pynvml.NVML_CLOCK_MEM)]:
            try:
                print(f"{label}: {pynvml.nvmlDeviceGetClockInfo(handle, clk_type)} MHz")
            except Exception as e:
                print(f"{label}: Error – {e}")
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            print(f"Temperature: {temp}°C")
        except Exception:
            pass
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            print(f"Power: {power:.1f} W / {limit:.1f} W")
        except Exception:
            pass
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"VRAM: {mem.used/2**30:.1f} / {mem.total/2**30:.1f} GB")
        except Exception:
            pass
        print(f"{'='*64}\n")
    except Exception as e:
        print(f"GPU info error: {e}")


# ─────────────────────────────────────────────────────────────────────
#  Attention config builder
# ─────────────────────────────────────────────────────────────────────

def build_attn_configs(B, seq_lens, causal):
    """
    Returns list of dicts, each describing one attention problem.
    두 가지 시나리오:
      prefill  : S_q == S_k == S  (전체 시퀀스 처리)
      decode   : S_q == 1,  S_k == S  (생성 단계)
    """
    configs = []
    for S in seq_lens:
        configs.append(dict(
            scenario="prefill",
            label=f"prefill_S{S}",
            B=B, H_q=NUM_Q_HEADS, H_kv=NUM_KV_HEADS, D=HEAD_DIM,
            S_q=S, S_k=S,
            causal=causal,
            count=NUM_LAYERS,
            flops=attn_flops(B, NUM_Q_HEADS, S, S, HEAD_DIM, causal),
        ))
    for S_k in seq_lens:
        configs.append(dict(
            scenario="decode",
            label=f"decode_Sk{S_k}",
            B=B, H_q=NUM_Q_HEADS, H_kv=NUM_KV_HEADS, D=HEAD_DIM,
            S_q=1, S_k=S_k,
            causal=False,          # decode: causal 불필요 (KV cache 사용)
            count=NUM_LAYERS,
            flops=attn_flops(B, NUM_Q_HEADS, 1, S_k, HEAD_DIM, False),
        ))
    return configs


# ─────────────────────────────────────────────────────────────────────
#  Tensor factories for each attention kernel
# ─────────────────────────────────────────────────────────────────────

def make_tensors(cfg, dtype, device):
    """
    Q  : (B, H_q,  S_q, D)
    K  : (B, H_kv, S_k, D)
    V  : (B, H_kv, S_k, D)
    """
    B, H_q, H_kv, D = cfg["B"], cfg["H_q"], cfg["H_kv"], cfg["D"]
    S_q, S_k = cfg["S_q"], cfg["S_k"]
    Q = torch.randn(B, H_q,  S_q, D, dtype=dtype, device=device)
    K = torch.randn(B, H_kv, S_k, D, dtype=dtype, device=device)
    V = torch.randn(B, H_kv, S_k, D, dtype=dtype, device=device)
    scale = D ** -0.5
    return Q, K, V, scale


def expand_kv_for_mha(K, V, H_q, H_kv):
    """GQA → MHA: K/V 를 H_q 사이즈로 repeat."""
    ratio = H_q // H_kv
    K_exp = K.repeat_interleave(ratio, dim=1)
    V_exp = V.repeat_interleave(ratio, dim=1)
    return K_exp, V_exp


# ─────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────

def main():
    pid = os.getpid()
    print(f"pid: {pid}")

    parser = argparse.ArgumentParser(
        description="Flash Attention benchmark for Qwen3-8B on Blackwell SM120")
    parser.add_argument("--device",        type=int,   default=3)
    parser.add_argument("--output-csv",    default="logs/attn_bench.csv")
    parser.add_argument("--target-seconds",type=float, default=2.0,
                        help="Target wall-time per measurement point")
    parser.add_argument("--warmup-iters",  type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=1)
    parser.add_argument("--seq-lens",      default="512,1024,2048,4096",
                        help="Comma-separated prefill sequence lengths")
    parser.add_argument("--causal",        action="store_true", default=True,
                        help="Use causal (autoregressive) mask")
    parser.add_argument("--no-causal",     dest="causal", action="store_false")
    parser.add_argument("--dtype",         default="bf16",
                        choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--fixed-iters",   type=int,   default=0,
                        help="0 = auto, >0 = fixed iteration count")
    args = parser.parse_args()

    torch_device = f"cuda:{args.device}"
    torch.cuda.set_device(args.device)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    # ── optional imports ──────────────────────────────────────────────
    try:
        import flash_attn
        from flash_attn import flash_attn_func
        HAS_FLASH_ATTN = True
        print(f"flash-attn {flash_attn.__version__} available")
    except ImportError:
        HAS_FLASH_ATTN = False
        print("⚠ flash-attn not installed  →  flash_attn kernel skipped")

    try:
        import xformers.ops as xops
        HAS_XFORMERS = True
        print("xformers available")
    except ImportError:
        HAS_XFORMERS = False
        print("⚠ xformers not installed   →  xformers kernel skipped")

    # ── SDPA 백엔드 지원 현황 출력 ────────────────────────────────────
    _probe_q = torch.randn(1, 1, 64, 128, dtype=torch.bfloat16,
                           device=torch_device)
    print("\n── SDPA backend availability ────────────────────────────────")
    for _name, _kwargs in [
        ("flash      ", dict(enable_flash=True,  enable_math=False,
                             enable_mem_efficient=False)),
        ("mem_eff    ", dict(enable_flash=False, enable_math=False,
                             enable_mem_efficient=True)),
        ("math       ", dict(enable_flash=False, enable_math=True,
                             enable_mem_efficient=False)),
        ("cudnn      ", dict(enable_flash=False, enable_math=False,
                             enable_mem_efficient=False, enable_cudnn=True)),
    ]:
        try:
            with torch.backends.cuda.sdp_kernel(**_kwargs):
                torch.nn.functional.scaled_dot_product_attention(
                    _probe_q, _probe_q, _probe_q, is_causal=True)
            _status = "✅ supported"
        except Exception as _e:
            _status = f"❌ {_e}"
        print(f"  {_name}: {_status}")

    # 인자 없이 호출했을 때 PyTorch가 선택하는 실제 백엔드 확인
    # torch.nn.attention.SDPBackend 로 실제 선택 추적
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        _chosen = None
        for _backend in [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                         SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]:
            try:
                with sdpa_kernel([_backend]):
                    torch.nn.functional.scaled_dot_product_attention(
                        _probe_q, _probe_q, _probe_q, is_causal=True)
                _chosen = _backend.name
                break
            except Exception:
                continue
        print(f"\n  → 기본 호출 시 선택되는 백엔드: {_chosen}")
    except ImportError:
        pass
    del _probe_q
    print("─────────────────────────────────────────────────────────────\n")

    # ── NVML init ────────────────────────────────────────────────────
    pynvml.nvmlInit()
    handle      = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    device_name = pynvml.nvmlDeviceGetName(handle)

    try:
        dev_idx = pynvml.nvmlDeviceGetIndex(handle)
    except Exception:
        dev_idx = args.device
    start_sampler(device_idx=dev_idx, interval=0.05)

    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU        : {device_name}")
    print(f"SMs        : {props.multi_processor_count}")
    print(f"Model      : Qwen3-8B attention layers  ({NUM_LAYERS} layers)")
    print(f"Q-heads    : {NUM_Q_HEADS}   KV-heads: {NUM_KV_HEADS}   "
          f"Head-dim: {HEAD_DIM}   GQA-ratio: {GQA_RATIO}")
    print(f"Batch size : {args.batch_size}")
    print(f"Seq lens   : {seq_lens}")
    print(f"Causal     : {args.causal}")
    print(f"Dtype      : {args.dtype}")
    print(f"Output     : {args.output_csv}")

    persistence_mode = None
    clock_changed    = False
    results          = []

    try:
        # ── persistence mode ─────────────────────────────────────────
        try:
            persistence_mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
            if persistence_mode != pynvml.NVML_FEATURE_ENABLED:
                pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_ENABLED)
                print("✅ Persistence Mode Enabled")
        except Exception as e:
            print(f"Warning: persistence mode: {e}")

        print("\nInitial GPU State:")
        print_gpu_info(handle)

        configs = build_attn_configs(args.batch_size, seq_lens, args.causal)

        for cfg in configs:
            B, H_q, H_kv, D = cfg["B"], cfg["H_q"], cfg["H_kv"], cfg["D"]
            S_q, S_k         = cfg["S_q"], cfg["S_k"]
            causal           = cfg["causal"]
            label            = cfg["label"]
            count            = cfg["count"]
            flops            = cfg["flops"]

            print(f"\n{'─'*72}")
            print(f"  {label}  "
                  f"B={B}  H_q={H_q}  H_kv={H_kv}  D={D}  "
                  f"S_q={S_q}  S_k={S_k}  causal={causal}")
            print(f"  FLOP/call = {flops/1e9:.3f} GFLOP   "
                  f"(×{count} layers/fwd)")
            print(f"{'─'*72}")

            # allocate once, reuse across kernels
            Q, K, V, scale = make_tensors(cfg, dtype, torch_device)
            # MHA 버전 (xformers / math backend 용)
            K_exp, V_exp = expand_kv_for_mha(K, V, H_q, H_kv)

            # ── causal mask (math backend 용) ─────────────────────
            causal_mask = None
            if causal and S_q > 1:
                causal_mask = torch.tril(
                    torch.ones(S_q, S_k, dtype=torch.bool, device=torch_device)
                ).unsqueeze(0).unsqueeze(0)   # (1,1,S_q,S_k)

            # ─────────────────────────────────────────────────────
            #  커널 스윕 정의
            #  (kernel_name, fn, description)
            # ─────────────────────────────────────────────────────
            kernel_sweep = []

            # 1. PyTorch SDPA – Flash Attention backend (SM80+에서 자동 선택)
            def _sdpa_flash():
                with torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=False,
                        enable_mem_efficient=False):
                    return torch.nn.functional.scaled_dot_product_attention(
                        Q, K_exp, V_exp,
                        attn_mask=None,
                        is_causal=causal,
                        scale=scale)
            kernel_sweep.append(("sdpa_flash", _sdpa_flash,
                                  "PyTorch SDPA (Flash backend)"))

            # 2. PyTorch SDPA – Memory-efficient backend
            def _sdpa_mem_eff():
                with torch.backends.cuda.sdp_kernel(
                        enable_flash=False,
                        enable_math=False,
                        enable_mem_efficient=True):
                    return torch.nn.functional.scaled_dot_product_attention(
                        Q, K_exp, V_exp,
                        attn_mask=None,
                        is_causal=causal,
                        scale=scale)
            kernel_sweep.append(("sdpa_mem_eff", _sdpa_mem_eff,
                                  "PyTorch SDPA (Memory-efficient backend)"))

            # 3. PyTorch SDPA – cuDNN backend (SM90+ / Blackwell에서 사용 가능)
            try:
                with torch.backends.cuda.sdp_kernel(
                        enable_flash=False,
                        enable_math=False,
                        enable_mem_efficient=False,
                        enable_cudnn=True):
                    _test = torch.nn.functional.scaled_dot_product_attention(
                        Q[:1], K_exp[:1], V_exp[:1],
                        is_causal=causal, scale=scale)
                def _sdpa_cudnn():
                    with torch.backends.cuda.sdp_kernel(
                            enable_flash=False,
                            enable_math=False,
                            enable_mem_efficient=False,
                            enable_cudnn=True):
                        return torch.nn.functional.scaled_dot_product_attention(
                            Q, K_exp, V_exp,
                            attn_mask=None,
                            is_causal=causal,
                            scale=scale)
                kernel_sweep.append(("sdpa_cudnn", _sdpa_cudnn,
                                      "PyTorch SDPA (cuDNN backend)"))
            except Exception as e:
                print(f"  ⚠ sdpa_cudnn: not supported – {e}")

            # 4. PyTorch SDPA – Math backend (참조용, 느림)
            #    S_q * S_k 이 너무 크면 메모리 부족 가능 → 512 이하만 실행
            if S_q <= 512 and S_k <= 512:
                def _sdpa_math():
                    with torch.backends.cuda.sdp_kernel(
                            enable_flash=False,
                            enable_math=True,
                            enable_mem_efficient=False):
                        return torch.nn.functional.scaled_dot_product_attention(
                            Q, K_exp, V_exp,
                            attn_mask=causal_mask if not causal else None,
                            is_causal=causal if causal_mask is None else False,
                            scale=scale)
                kernel_sweep.append(("sdpa_math", _sdpa_math,
                                      "PyTorch SDPA (Math baseline)"))

            # 5. flash-attn package
            #    API: flash_attn_func(q, k, v, dropout_p=0, softmax_scale=None, causal=False)
            #    q/k/v shape: (B, S, H, D)
            if HAS_FLASH_ATTN:
                # flash_attn uses GQA natively
                q_fa = Q.permute(0, 2, 1, 3).contiguous()    # (B, S_q, H_q,  D)
                k_fa = K.permute(0, 2, 1, 3).contiguous()    # (B, S_k, H_kv, D)
                v_fa = V.permute(0, 2, 1, 3).contiguous()
                def _flash_attn(_q=q_fa, _k=k_fa, _v=v_fa):
                    return flash_attn_func(_q, _k, _v,
                                           softmax_scale=scale,
                                           causal=causal)
                kernel_sweep.append(("flash_attn", _flash_attn,
                                      "flash-attn package (FA2/FA3)"))

            # 6. xformers memory_efficient_attention
            #    q/k/v shape: (B, S, H, D)
            if HAS_XFORMERS:
                q_xf = Q.permute(0, 2, 1, 3).contiguous()
                k_xf = K_exp.permute(0, 2, 1, 3).contiguous()
                v_xf = V_exp.permute(0, 2, 1, 3).contiguous()
                xf_attn_bias = (
                    xops.LowerTriangularMask() if causal else None
                )
                def _xformers(_q=q_xf, _k=k_xf, _v=v_xf, _bias=xf_attn_bias):
                    return xops.memory_efficient_attention(
                        _q, _k, _v,
                        attn_bias=_bias,
                        scale=scale)
                kernel_sweep.append(("xformers", _xformers,
                                      "xformers memory_efficient_attention"))

            # ─────────────────────────────────────────────────────
            #  warmup + benchmark each kernel
            # ─────────────────────────────────────────────────────
            baseline_tflops = None   # sdpa_flash 대비 비율 계산용

            for kname, kfn, kdesc in kernel_sweep:
                # ── sanity check (skip on error) ──────────────────
                try:
                    _ = kfn()
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"  ⚠ {kname:16s} SKIP: {e}")
                    continue

                # ── warmup ─────────────────────────────────────────
                for _ in range(args.warmup_iters):
                    kfn()
                torch.cuda.synchronize()

                # ── iteration count ───────────────────────────────
                if args.fixed_iters > 0:
                    ni = args.fixed_iters
                else:
                    ni = auto_iterations(kfn, args.target_seconds)

                # ── NVTX range + measure ──────────────────────────
                torch.cuda.nvtx.range_push(f"{label} {kname}")
                el, en, pw, sm_clk, temp = measure(kfn, ni, handle)
                torch.cuda.nvtx.range_pop()

                # ── metrics ──────────────────────────────────────
                per_iter_ms = el / ni
                tflops      = flops * ni / (el / 1000) / 1e12
                nj_per_flop = en * 1e6 / (flops * ni) if ni > 0 else 0.0

                if kname == "sdpa_flash":
                    baseline_tflops = tflops
                ratio = (tflops / baseline_tflops
                         if baseline_tflops and baseline_tflops > 0 else 0.0)

                print(f"  {kname:16s} | iters={ni:5d} | "
                      f"{per_iter_ms:7.3f}ms/it  {tflops:7.3f}TF/s  "
                      f"{pw:6.1f}W  {en:8.0f}mJ  {nj_per_flop:.3f}nJ/F  "
                      f"SM={sm_clk:.0f}MHz  T={temp:.0f}°C  "
                      f"[vs sdpa_flash: {ratio:.2f}×]")

                results.append(dict(
                    scenario     = cfg["scenario"],
                    label        = label,
                    kernel       = kname,
                    dtype        = args.dtype,
                    causal       = int(causal),
                    batch_size   = B,
                    H_q          = H_q,
                    H_kv         = H_kv,
                    head_dim     = D,
                    seq_q        = S_q,
                    seq_k        = S_k,
                    count        = count,
                    num_iters    = ni,
                    elapsed_ms   = round(el, 4),
                    per_iter_ms  = round(per_iter_ms, 4),
                    tflops       = round(tflops, 4),
                    power_w      = round(pw, 2),
                    energy_mj    = round(en, 2),
                    nj_per_flop  = round(nj_per_flop, 4),
                    sm_clock_mhz = round(sm_clk, 1),
                    temp_c       = round(temp, 1),
                    vs_baseline  = round(ratio, 4),
                ))

        # ─────────────────────────────────────────────────────────
        #  Full-forward time estimate  (Qwen3-8B, NUM_LAYERS 레이어)
        # ─────────────────────────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"  Full-forward attention estimate  "
              f"(B={args.batch_size}  {NUM_LAYERS} layers)")
        print(f"{'='*72}")
        header = (f"  {'kernel':16s} | {'scenario':18s} | "
                  f"{'time/lyr ms':>11s}  {'time_all ms':>11s}  "
                  f"{'TFLOPS':>7s}  {'power_W':>7s}  {'SM_MHz':>7s}")
        print(header)
        print("  " + "─" * (len(header) - 2))

        # 시나리오 × 커널 그룹
        all_kernels = sorted({r["kernel"] for r in results})
        all_scenarios = sorted({r["scenario"] for r in results},
                               key=lambda s: (0 if s == "prefill" else 1))

        for scenario in all_scenarios:
            for kname in all_kernels:
                matched = [r for r in results
                           if r["kernel"] == kname
                           and r["scenario"] == scenario]
                if not matched:
                    continue

                total_ms    = 0.0
                total_flops = 0.0
                sm_sum      = 0.0
                pw_sum      = 0.0
                w_sum       = 0.0

                for r in matched:
                    w          = r["per_iter_ms"] * r["count"]
                    total_ms  += w
                    total_flops += r["tflops"] * (r["elapsed_ms"] / 1000)
                    sm_sum    += r["sm_clock_mhz"] * w
                    pw_sum    += r["power_w"] * w
                    w_sum     += w

                avg_sm = sm_sum / w_sum if w_sum else 0
                avg_pw = pw_sum / w_sum if w_sum else 0
                total_tflops = total_flops / (total_ms / 1000) if total_ms else 0

                # representative label (e.g. prefill_S2048)
                rep_label = matched[0]["label"]
                print(f"  {kname:16s} | {rep_label:18s} | "
                      f"{total_ms/NUM_LAYERS:11.3f}  {total_ms:11.3f}  "
                      f"{total_tflops:7.3f}  {avg_pw:7.1f}  {avg_sm:7.0f}")

        # ─────────────────────────────────────────────────────────
        #  CSV export
        # ─────────────────────────────────────────────────────────
        if results:
            os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
            out_path = args.output_csv.replace(".csv", f"_{pid}.csv")
            fieldnames = list(results[0].keys())
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(results)
            print(f"\n✅ Results saved to {out_path}")

    finally:
        stop_sampler()

        if clock_changed:
            print("\nResetting GPU Locked Clocks...", end=" ")
            try:
                pynvml.nvmlDeviceResetGpuLockedClocks(handle)
                print("✅")
            except pynvml.NVMLError as e:
                print(f"⚠ ({e})")

        if persistence_mode == pynvml.NVML_FEATURE_ENABLED:
            try:
                pynvml.nvmlDeviceSetPersistenceMode(handle, pynvml.NVML_FEATURE_DISABLED)
                print("✅ Persistence Mode Disabled")
            except Exception as e:
                print(f"Warning: {e}")

        try:
            pynvml.nvmlShutdown()
            print("pynvml shutdown complete")
        except Exception:
            pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
