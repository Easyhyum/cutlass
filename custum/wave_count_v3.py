#!/usr/bin/env python3
"""
Estimate CTA waves for custum/bf16_gemm_sm80.cu::gemm_sm80_v3.

The v3 kernel uses:
  ThreadblockShape = 128 x 128 x 64
  WarpShape        = 64  x 64  x 32
  InstructionShape = 16  x 8   x 16
  Stages           = 3

Examples:
  python custum/wave_count_v3.py --a 4096 8192 --b 8192 512
  python custum/wave_count_v3.py --m 4096 --n 512 --k 8192 --regs-per-thread 128
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# cudaDeviceAttr enum values used as fallbacks when torch does not expose them.
# Values are stable in the CUDA runtime API for the attributes below.
CUDA_ATTRS = {
    "multi_processor_count": 16,
    "max_threads_per_multiprocessor": 39,
    "max_shared_memory_per_block_optin": 97,
    "max_shared_memory_per_multiprocessor": 81,
    "max_registers_per_multiprocessor": 82,
    "max_blocks_per_multiprocessor": 106,
}


@dataclass(frozen=True)
class V3Shape:
    tb_m: int = 128
    tb_n: int = 128
    tb_k: int = 64
    warp_m: int = 64
    warp_n: int = 64
    warp_k: int = 32
    instr_m: int = 16
    instr_n: int = 8
    instr_k: int = 16
    stages: int = 3
    element_bytes: int = 2  # BF16
    threads_per_warp: int = 32

    @property
    def warp_count_m(self) -> int:
        return self.tb_m // self.warp_m

    @property
    def warp_count_n(self) -> int:
        return self.tb_n // self.warp_n

    @property
    def warp_count_k(self) -> int:
        return self.tb_k // self.warp_k

    @property
    def warps_per_cta(self) -> int:
        return self.warp_count_m * self.warp_count_n * self.warp_count_k

    @property
    def threads_per_cta(self) -> int:
        return self.warps_per_cta * self.threads_per_warp

    @property
    def k_partitions(self) -> int:
        return self.warp_count_k

    @property
    def mma_per_warp_tile(self) -> int:
        return (
            (self.warp_m // self.instr_m)
            * (self.warp_n // self.instr_n)
            * (self.warp_k // self.instr_k)
        )

    @property
    def mainloop_smem_bytes(self) -> int:
        a_bytes = self.tb_m * self.tb_k * self.stages * self.element_bytes
        b_bytes = self.tb_k * self.tb_n * self.stages * self.element_bytes
        return a_bytes + b_bytes


@dataclass
class DeviceInfo:
    index: int
    name: str
    sm_count: int
    regs_per_sm: Optional[int]
    smem_per_sm: Optional[int]
    smem_per_block_optin: Optional[int]
    max_threads_per_sm: Optional[int]
    max_blocks_per_sm: Optional[int]


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def load_cudart() -> Optional[ctypes.CDLL]:
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def cuda_device_get_attribute(device: int, attr: int) -> Optional[int]:
    cudart = load_cudart()
    if cudart is None:
        return None

    value = ctypes.c_int()
    fn = cudart.cudaDeviceGetAttribute
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    err = fn(ctypes.byref(value), attr, device)
    return value.value if err == 0 else None


def get_device_info(device: int) -> DeviceInfo:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not available from PyTorch.")

    props = torch.cuda.get_device_properties(device)

    def prop_or_attr(prop_name: str, attr_name: str) -> Optional[int]:
        value = getattr(props, prop_name, None)
        if value is not None:
            return int(value)
        attr = CUDA_ATTRS.get(attr_name)
        return cuda_device_get_attribute(device, attr) if attr is not None else None

    return DeviceInfo(
        index=device,
        name=props.name,
        sm_count=int(props.multi_processor_count),
        regs_per_sm=prop_or_attr("regs_per_multiprocessor", "max_registers_per_multiprocessor"),
        smem_per_sm=prop_or_attr("shared_memory_per_multiprocessor", "max_shared_memory_per_multiprocessor"),
        smem_per_block_optin=prop_or_attr(
            "shared_memory_per_block_optin", "max_shared_memory_per_block_optin"
        ),
        max_threads_per_sm=prop_or_attr("max_threads_per_multi_processor", "max_threads_per_multiprocessor"),
        max_blocks_per_sm=prop_or_attr("max_blocks_per_multi_processor", "max_blocks_per_multiprocessor"),
    )


def find_module_so(module_name: str) -> Optional[Path]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None

    path = getattr(module, "__file__", None)
    if not path:
        return None

    so_path = Path(path)
    return so_path if so_path.exists() else None


def parse_cuobjdump_regs(so_path: Path, symbol_hint: str = "128, 128, 64") -> Optional[int]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return None

    commands = [
        [cuobjdump, "--dump-resource-usage", "--demangle", str(so_path)],
        [cuobjdump, "--dump-resource-usage", str(so_path)],
    ]

    output = ""
    for cmd in commands:
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            break
        except Exception:
            continue

    if not output:
        return None

    # Split into function-like chunks. cuobjdump formatting varies by CUDA version.
    chunks = re.split(r"(?=\n\s*(?:Function|Name|symbol)\s*[:=])", output, flags=re.IGNORECASE)

    candidates = []
    for chunk in chunks:
        if symbol_hint in chunk or "128x128" in chunk or "gemm_sm80_v3" in chunk:
            candidates.append(chunk)
    if not candidates:
        candidates = chunks

    reg_patterns = [
        r"\bREG(?:ISTERS)?\s*[:=]\s*(\d+)",
        r"\breg(?:ister)?s?\s*[:=]\s*(\d+)",
        r"\b(\d+)\s+registers?\b",
    ]

    for chunk in candidates:
        for pattern in reg_patterns:
            match = re.search(pattern, chunk, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))

    return None


def occupancy_limits(
    shape: V3Shape,
    device: DeviceInfo,
    smem_per_cta: int,
    regs_per_thread: Optional[int],
) -> tuple[int, dict[str, Optional[int]]]:
    limits: dict[str, Optional[int]] = {}

    if device.smem_per_sm:
        limits["smem"] = device.smem_per_sm // smem_per_cta
    else:
        limits["smem"] = None

    if device.regs_per_sm and regs_per_thread:
        regs_per_cta = regs_per_thread * shape.threads_per_cta
        limits["regs"] = device.regs_per_sm // regs_per_cta
    else:
        limits["regs"] = None

    if device.max_threads_per_sm:
        limits["threads"] = device.max_threads_per_sm // shape.threads_per_cta
    else:
        limits["threads"] = None

    limits["blocks"] = device.max_blocks_per_sm

    finite_limits = [value for value in limits.values() if value is not None]
    if not finite_limits:
        raise RuntimeError("Could not determine any occupancy limit from the device.")

    resident = max(0, min(finite_limits))
    return resident, limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate CTA waves for CUTLASS BF16 gemm_sm80_v3."
    )
    parser.add_argument("--a", nargs=2, type=int, metavar=("M", "K"), help="A matrix shape [M K]")
    parser.add_argument("--b", nargs=2, type=int, metavar=("K", "N"), help="B matrix shape [K N]")
    parser.add_argument("--m", type=int, help="GEMM M")
    parser.add_argument("--n", type=int, help="GEMM N")
    parser.add_argument("--k", type=int, help="GEMM K")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--regs-per-thread",
        type=int,
        default=None,
        help="Override compiled kernel register count per thread.",
    )
    parser.add_argument(
        "--module",
        default="bf16_gemm_sm80",
        help="Python extension module name to locate the built .so for cuobjdump parsing.",
    )
    parser.add_argument(
        "--so",
        type=Path,
        default=None,
        help="Path to built extension .so. Used to parse registers with cuobjdump.",
    )
    parser.add_argument(
        "--smem-per-cta",
        type=int,
        default=None,
        help="Override shared memory bytes per CTA.",
    )
    parser.add_argument("--stages", type=int, default=3, help="Pipeline stages for v3")
    return parser.parse_args()


def resolve_mnk(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.a or args.b:
        if not (args.a and args.b):
            raise ValueError("--a and --b must be provided together.")
        m, k_a = args.a
        k_b, n = args.b
        if k_a != k_b:
            raise ValueError(f"K mismatch: A is [M,{k_a}], B is [{k_b},N].")
        return m, n, k_a

    if args.m is None or args.n is None or args.k is None:
        raise ValueError("Provide either --a M K --b K N or --m M --n N --k K.")
    return args.m, args.n, args.k


def main() -> None:
    args = parse_args()
    m, n, k = resolve_mnk(args)

    shape = V3Shape(stages=args.stages)
    device = get_device_info(args.device)

    so_path = args.so or find_module_so(args.module)
    parsed_regs = parse_cuobjdump_regs(so_path) if so_path else None
    regs_per_thread = args.regs_per_thread or parsed_regs

    cta_m = ceil_div(m, shape.tb_m)
    cta_n = ceil_div(n, shape.tb_n)
    cta_k_iters = ceil_div(k, shape.tb_k)
    total_ctas = cta_m * cta_n

    smem_per_cta = args.smem_per_cta or shape.mainloop_smem_bytes
    resident_cta_per_sm, limits = occupancy_limits(shape, device, smem_per_cta, regs_per_thread)
    concurrent_ctas = resident_cta_per_sm * device.sm_count
    waves = math.ceil(total_ctas / concurrent_ctas) if concurrent_ctas else math.inf
    active_ctas_first_wave = min(total_ctas, concurrent_ctas)
    active_sms_first_wave = ceil_div(active_ctas_first_wave, resident_cta_per_sm) if resident_cta_per_sm else 0
    idle_sms_first_wave = max(0, device.sm_count - active_sms_first_wave)

    total_warps = total_ctas * shape.warps_per_cta
    total_mma = total_ctas * cta_k_iters * shape.warps_per_cta * shape.mma_per_warp_tile

    print("=== gemm_sm80_v3 wave estimate ===")
    print(f"Device                 : cuda:{device.index} {device.name}")
    print(f"SM count               : {device.sm_count}")
    print(f"Registers / SM         : {device.regs_per_sm if device.regs_per_sm is not None else 'unknown'}")
    print(f"Shared memory / SM     : {device.smem_per_sm if device.smem_per_sm is not None else 'unknown'} B")
    print(f"Max threads / SM       : {device.max_threads_per_sm if device.max_threads_per_sm is not None else 'unknown'}")
    print(f"Max blocks / SM        : {device.max_blocks_per_sm if device.max_blocks_per_sm is not None else 'unknown'}")
    print()
    print(f"A shape                : [{m}, {k}]")
    print(f"B shape                : [{k}, {n}]")
    print(f"C shape                : [{m}, {n}]")
    print(f"ThreadblockShape       : {shape.tb_m} x {shape.tb_n} x {shape.tb_k}")
    print(f"WarpShape              : {shape.warp_m} x {shape.warp_n} x {shape.warp_k}")
    print(f"InstructionShape       : {shape.instr_m} x {shape.instr_n} x {shape.instr_k}")
    print(f"Stages                 : {shape.stages}")
    print()
    print(f"CTA tiles M x N        : {cta_m} x {cta_n}")
    print(f"Total CTAs             : {total_ctas}")
    print(f"CTA K iterations       : {cta_k_iters}")
    print(f"Warps / CTA            : {shape.warps_per_cta} ({shape.warp_count_m}x{shape.warp_count_n}x{shape.warp_count_k})")
    print(f"Threads / CTA          : {shape.threads_per_cta}")
    print(f"MMA instructions / warp tile : {shape.mma_per_warp_tile}")
    print(f"Total warp instances   : {total_warps}")
    print(f"Total mma.sync estimate: {total_mma}")
    print()
    print(f"Shared memory / CTA    : {smem_per_cta} B ({smem_per_cta / 1024:.1f} KiB)")
    if regs_per_thread is None:
        print("Registers / thread     : unknown (pass --regs-per-thread or provide --so for cuobjdump)")
    else:
        source = "override" if args.regs_per_thread else f"cuobjdump: {so_path}"
        print(f"Registers / thread     : {regs_per_thread} ({source})")
        print(f"Registers / CTA        : {regs_per_thread * shape.threads_per_cta}")
    print()
    print("Resident CTA / SM limits:")
    for key in ("smem", "regs", "threads", "blocks"):
        print(f"  {key:8s}: {limits[key] if limits[key] is not None else 'unknown'}")
    print(f"Resident CTAs / SM     : {resident_cta_per_sm}")
    print(f"Concurrent CTAs        : {concurrent_ctas}")
    print(f"Estimated waves        : {waves}")
    print()
    print("CTX summary (CTX == CTA/threadblock):")
    print(f"  CTX per SM concurrently : {resident_cta_per_sm}")
    print(f"  Total grid CTX          : {total_ctas}")
    print(f"  Total concurrent CTX    : {concurrent_ctas}")
    print(f"  Active CTX in 1st wave  : {active_ctas_first_wave}")
    print(f"  Active SMs in 1st wave  : {active_sms_first_wave} / {device.sm_count}")
    print(f"  Idle SMs in 1st wave    : {idle_sms_first_wave}")

    if regs_per_thread is None:
        print()
        print("NOTE: register pressure was not included. Use a ptxas/cuobjdump register count for exact occupancy.")
    if device.smem_per_block_optin and smem_per_cta > device.smem_per_block_optin:
        print()
        print(
            "WARNING: estimated smem/CTA exceeds max opt-in smem/block. "
            "The kernel may not launch with this shape on this device."
        )


if __name__ == "__main__":
    main()
