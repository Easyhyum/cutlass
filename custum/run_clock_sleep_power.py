#!/usr/bin/env python3
"""Run clock64 busy-wait and __nanosleep kernels for GPU power profiling.

Build first:
    python setup_clock_sleep_power.py build_ext --inplace

Examples:
    python run_clock_sleep_power.py --device 0 --num-blocks 256 --threads-per-block 256
    python run_clock_sleep_power.py --mode clock64 --cycles 1000000 --repeats 100
    python run_clock_sleep_power.py --mode nanosleep --sleep-ns 1000 --repeats 1000
"""

from __future__ import annotations

import argparse
import csv
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import torch

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    pynvml = None
    NVML_AVAILABLE = False


class PowerMonitor:
    def __init__(self, device_id: int, interval_s: float) -> None:
        self.device_id = device_id
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: Any | None = None
        self.energy_start_mj: int | None = None
        self.energy_end_mj: int | None = None

    def __enter__(self) -> "PowerMonitor":
        if not NVML_AVAILABLE:
            return self

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(3)
        self.energy_start_mj = self._read_energy_mj()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self._handle is not None:
            self.energy_end_mj = self._read_energy_mj()
            pynvml.nvmlShutdown()

    def _read_energy_mj(self) -> int | None:
        try:
            return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception:
            return None

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.samples.append(
                    {
                        "timestamp": time.time(),
                        "power_w": pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0,
                        "sm_clock_mhz": float(
                            pynvml.nvmlDeviceGetClockInfo(
                                self._handle, pynvml.NVML_CLOCK_SM
                            )
                        ),
                        "mem_clock_mhz": float(
                            pynvml.nvmlDeviceGetClockInfo(
                                self._handle, pynvml.NVML_CLOCK_MEM
                            )
                        ),
                        "gpu_util_pct": float(util.gpu),
                        "mem_util_pct": float(util.memory),
                    }
                )
            except Exception as err:
                self.samples.append({"timestamp": time.time(), "error": str(err)})
            time.sleep(self.interval_s)

    def summary(self) -> dict[str, float | int | None]:
        valid = [s for s in self.samples if "power_w" in s]
        if not valid:
            return {
                "power_samples": 0,
                "power_w_mean": None,
                "power_w_min": None,
                "power_w_max": None,
                "sm_clock_mhz_mean": None,
                "gpu_util_pct_mean": None,
                "energy_j": None,
            }

        energy_j = None
        if self.energy_start_mj is not None and self.energy_end_mj is not None:
            energy_j = (self.energy_end_mj - self.energy_start_mj) / 1000.0

        return {
            "power_samples": len(valid),
            "power_w_mean": statistics.fmean(s["power_w"] for s in valid),
            "power_w_min": min(s["power_w"] for s in valid),
            "power_w_max": max(s["power_w"] for s in valid),
            "sm_clock_mhz_mean": statistics.fmean(s["sm_clock_mhz"] for s in valid),
            "gpu_util_pct_mean": statistics.fmean(s["gpu_util_pct"] for s in valid),
            "energy_j": energy_j,
        }


def format_optional(value: float | int | None, suffix: str = "", digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def run_mode(args: argparse.Namespace, mode: str, module: Any) -> dict[str, Any]:
    torch.cuda.synchronize(args.device)

    # for _ in range(1):
    #     launch_kernel(args, mode, module)
    # torch.cuda.synchronize(args.device)

    times_ms: list[float] = []
    last_cycles = None
    started = time.perf_counter()

    with PowerMonitor(args.device, args.sample_interval) as monitor:
        # while time.perf_counter() - started < args.duration:
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        last_cycles = launch_kernel(args, mode, module)
        end.record()
        torch.cuda.synchronize(args.device)
        times_ms.append(begin.elapsed_time(end))
            

    elapsed_s = time.perf_counter() - started
    cycle_values = last_cycles.detach().cpu().double() if last_cycles is not None else None
    power = monitor.summary()
    kernel_ms_mean = statistics.fmean(times_ms) if times_ms else 0.0

    result = {
        "mode": mode,
        "num_blocks": args.num_blocks,
        "threads_per_block": args.threads_per_block,
        "repeats": args.repeats,
        "launches": len(times_ms),
        "elapsed_s": elapsed_s,
        "kernel_ms_mean": kernel_ms_mean,
        "kernel_ms_min": min(times_ms) if times_ms else 0.0,
        "kernel_ms_max": max(times_ms) if times_ms else 0.0,
        "block_cycles_mean": float(cycle_values.mean()) if cycle_values is not None else 0.0,
        "block_cycles_min": float(cycle_values.min()) if cycle_values is not None else 0.0,
        "block_cycles_max": float(cycle_values.max()) if cycle_values is not None else 0.0,
        **power,
    }
    if mode == "clock64":
        result["cycles"] = args.cycles
        result["sleep_ns"] = None
    else:
        result["cycles"] = None
        result["sleep_ns"] = args.sleep_ns
    time.sleep(3.0)
    return result


def launch_kernel(args: argparse.Namespace, mode: str, module: Any) -> torch.Tensor:
    if mode == "clock64":
        return module.clock64_busy_wait(
            args.num_blocks,
            args.threads_per_block,
            args.cycles,
            args.repeats,
        )
    if mode == "nanosleep":
        return module.nanosleep(
            args.num_blocks,
            args.threads_per_block,
            args.sleep_ns,
            args.repeats,
        )
    raise ValueError(f"unknown mode: {mode}")


def print_result(result: dict[str, Any]) -> None:
    print("=" * 80)
    print(f"mode                 : {result['mode']}")
    print(f"launch config        : blocks={result['num_blocks']} threads/block={result['threads_per_block']}")
    print(f"delay config         : cycles={result['cycles']} sleep_ns={result['sleep_ns']} repeats={result['repeats']}")
    print(f"launches / elapsed   : {result['launches']} / {result['elapsed_s']:.3f} s")
    print(
        "kernel wall time     : "
        f"mean={result['kernel_ms_mean']:.4f} ms "
        f"min={result['kernel_ms_min']:.4f} ms "
        f"max={result['kernel_ms_max']:.4f} ms"
    )
    print(
        "per-block cycles     : "
        f"mean={result['block_cycles_mean']:.1f} "
        f"min={result['block_cycles_min']:.1f} "
        f"max={result['block_cycles_max']:.1f}"
    )
    print(
        "power                : "
        f"mean={format_optional(result['power_w_mean'], ' W')} "
        f"min={format_optional(result['power_w_min'], ' W')} "
        f"max={format_optional(result['power_w_max'], ' W')} "
        f"samples={result['power_samples']}"
    )
    print(
        "gpu status           : "
        f"sm_clock_mean={format_optional(result['sm_clock_mhz_mean'], ' MHz', 1)} "
        f"gpu_util_mean={format_optional(result['gpu_util_pct_mean'], ' %', 1)} "
        f"energy={format_optional(result['energy_j'], ' J')}"
    )


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in results for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["clock64", "nanosleep", "both"], default="both")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=3000)
    parser.add_argument("--threads-per-block", type=int, default=8 * 32)
    parser.add_argument("--cycles", type=int, default=3000000000)
    parser.add_argument("--sleep-ns", type=int, default=3000000000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.device)

    import clock_sleep_power

    modes = ["clock64", "nanosleep"] if args.mode == "both" else [args.mode]
    results = [run_mode(args, mode, clock_sleep_power) for mode in modes]

    for result in results:
        print_result(result)

    if args.csv is not None:
        write_csv(args.csv, results)
        print(f"wrote csv: {args.csv}")


if __name__ == "__main__":
    main()
