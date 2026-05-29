#!/usr/bin/env python3
"""Plot *_actual_forward.csv written by bench_mark.py.

The CSV has one row per actual synthetic Qwen forward run, so this plot uses
sleep_ns as the x axis instead of a sample timeline.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict


def _fget(row: dict, key: str, default: float = float("nan")) -> float:
    v = row.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _label(kernel: str, sleep_freq: str) -> str:
    if kernel == "cublas":
        return "cublas"
    return f"{kernel} freq={sleep_freq}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot actual forward CSV to PNG")
    ap.add_argument("actual_forward_csv", help="Path to *_actual_forward.csv")
    ap.add_argument(
        "-o", "--output",
        help="Output PNG (default: <actual_forward_stem>_plot.png)",
    )
    args = ap.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("plot_actual_forward_csv: need matplotlib:", e, file=sys.stderr)
        return 1

    path = args.actual_forward_csv
    if not os.path.isfile(path):
        print("plot_actual_forward_csv: file not found:", path, file=sys.stderr)
        return 1

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("plot_actual_forward_csv: no data rows in", path, file=sys.stderr)
        return 1

    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("matmul_kernel", "?"), row.get("sleep_freq", ""))].append(row)

    out_path = args.output
    if not out_path:
        base, _ = os.path.splitext(path)
        out_path = base + "_plot.png"

    colors = {
        "cublas": "#2563eb",
        "cutlass_sm80": "#dc2626",
    }
    fig, (ax_perf, ax_power) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7.0),
        dpi=120,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
        facecolor="white",
    )
    ax_gf = ax_perf.twinx()
    ax_sm = ax_power.twinx()

    handles = []
    labels = []
    for (kernel, sleep_freq), group_rows in sorted(groups.items()):
        group_rows = sorted(group_rows, key=lambda r: _fget(r, "sleep_ns", 0.0))
        x = [_fget(r, "sleep_ns") for r in group_rows]
        elapsed = [_fget(r, "elapsed_ms") for r in group_rows]
        gflops = [_fget(r, "gflops") for r in group_rows]
        power = [_fget(r, "power_w") for r in group_rows]
        sm = [_fget(r, "sm_clock_mhz") for r in group_rows]
        color = colors.get(kernel, "#64748b")
        marker = "o" if kernel == "cublas" else "s"
        name = _label(kernel, sleep_freq)

        h, = ax_perf.plot(
            x, elapsed,
            color=color,
            marker=marker,
            lw=2.0,
            label=f"{name} latency",
            zorder=4,
        )
        ax_gf.plot(
            x, gflops,
            color=color,
            marker=marker,
            lw=1.5,
            ls="--",
            alpha=0.85,
            label=f"{name} GFLOPS",
            zorder=3,
        )
        ax_power.plot(
            x, power,
            color=color,
            marker=marker,
            lw=2.0,
            label=f"{name} power",
            zorder=4,
        )
        ax_sm.plot(
            x, sm,
            color=color,
            marker=marker,
            lw=1.5,
            ls="--",
            alpha=0.85,
            label=f"{name} SM clock",
            zorder=3,
        )
        handles.append(h)
        labels.append(name)

    for ax in (ax_perf, ax_power):
        ax.set_facecolor("white")
        ax.grid(True, axis="y", alpha=0.35, color="#cbd5e1")
        ax.grid(True, axis="x", alpha=0.18, color="#cbd5e1")
        ax.margins(x=0.04)

    ax_perf.set_ylabel("elapsed_ms (lower is better)", color="#0f172a")
    ax_gf.set_ylabel("GFLOPS", color="#475569")
    ax_power.set_ylabel("power_w", color="#0f172a")
    ax_sm.set_ylabel("sm_clock_mhz", color="#475569")
    ax_power.set_xlabel("sleep_ns")

    r0 = rows[0]
    title = (
        f"{os.path.basename(path)} | "
        f"B={r0.get('batch_size', '')}, S={r0.get('seq_len', '')}, "
        f"layers={r0.get('num_layers', '')}, attention={r0.get('attention_kernel', '')}"
    )
    fig.suptitle(title, fontsize=11, fontweight="semibold", y=0.995)

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=min(4, len(handles)),
            framealpha=0.92,
            fontsize=8.5,
        )

    # Line style note: solid for latency/power, dashed for GFLOPS/SM clock.
    fig.text(
        0.5,
        0.02,
        "Solid: elapsed_ms / power_w   Dashed: GFLOPS / SM clock",
        ha="center",
        fontsize=8,
        color="#475569",
    )

    d = os.path.dirname(os.path.abspath(out_path)) or "."
    if d:
        os.makedirs(d, exist_ok=True)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.075, right=0.925)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    print(f"plot_actual_forward_csv: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
