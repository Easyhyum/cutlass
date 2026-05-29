#!/usr/bin/env python3
"""
지정된 디렉토리(또는 단일 파일)의 power CSV들을 읽어 시계열 PNG로 저장한다.

CSV 형식 예시(헤더에 단위 포함):
    timestamp, clocks.current.sm [MHz], power.draw.instant [W],
    power.draw.average [W], power.limit [W],
    clocks_event_reasons.sw_power_cap, clocks_event_reasons_counters.sw_power_cap [us]

플롯 구성 (CSV 한 개당 PNG 한 장):
    - X 축: row index (1 .. N)
    - 왼쪽 Y 축 (0 ~ 1000):
        * Power = power.draw.instant [W]
        * Cap   = power.limit [W]
    - 오른쪽 Y 축 (0 ~ 3000):
        * Freq     = clocks.current.sm [MHz]
        * Max freq = --max-freq (default 2430 MHz) 고정 수평선

사용 예:
    python plot_power_csv.py custum/profile/run_20260519_072901_cublas
    python plot_power_csv.py custum/profile/run_20260519_072901_cublas \
        --max-freq 2430 --outdir custum/profile/run_20260519_072901_cublas/plots
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import List, Optional, Tuple


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _to_float(value: Optional[str]) -> float:
    """문자열에서 첫 숫자를 추출해 float로 변환. 실패 시 NaN."""
    if value is None:
        return float("nan")
    s = str(value).strip()
    if not s:
        return float("nan")
    m = _NUM_RE.search(s)
    if not m:
        return float("nan")
    try:
        return float(m.group(0))
    except ValueError:
        return float("nan")


def _normalize_key(key: str) -> str:
    """헤더 키를 단위/공백 제거하여 표준화."""
    k = key.strip()
    k = re.sub(r"\s*\[[^\]]*\]\s*", "", k)
    return k.lower()


def _find_column(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    """후보 이름들 중 fieldnames에 존재하는 것을 반환(단위 무시, 대소문자 무시)."""
    norm_to_orig = {_normalize_key(fn): fn for fn in fieldnames}
    for c in candidates:
        nc = _normalize_key(c)
        if nc in norm_to_orig:
            return norm_to_orig[nc]
    return None


def _read_series(
    csv_path: str,
) -> Tuple[List[float], List[float], List[float]]:
    """CSV 한 개를 읽어 (power_draw, power_limit, sm_clock) 리스트 반환."""
    powers: List[float] = []
    caps: List[float] = []
    freqs: List[float] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        fields = reader.fieldnames or []
        if not fields:
            return powers, caps, freqs

        col_power = _find_column(
            fields,
            ["power.draw.instant", "power.draw", "power.draw.average"],
        )
        col_cap = _find_column(fields, ["power.limit"])
        col_freq = _find_column(fields, ["clocks.current.sm", "clocks.sm"])

        if col_power is None or col_cap is None or col_freq is None:
            print(
                f"[warn] {csv_path}: required columns not found "
                f"(power={col_power}, cap={col_cap}, freq={col_freq})",
                file=sys.stderr,
            )
            return powers, caps, freqs

        for row in reader:
            powers.append(_to_float(row.get(col_power)))
            caps.append(_to_float(row.get(col_cap)))
            freqs.append(_to_float(row.get(col_freq)))

    return powers, caps, freqs


def _plot_one(
    csv_path: str,
    out_png: str,
    max_freq: float,
    power_ylim: Tuple[float, float],
    freq_ylim: Tuple[float, float],
) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    powers, caps, freqs = _read_series(csv_path)
    n = len(powers)
    if n == 0:
        print(f"[skip] {csv_path}: no data rows", file=sys.stderr)
        return False

    x = list(range(1, n + 1))

    fig, ax_left = plt.subplots(figsize=(12, 5))
    ax_right = ax_left.twinx()

    line_power, = ax_left.plot(
        x, powers, color="tab:red", linewidth=1.4, label="Power (power.draw.instant) [W]"
    )
    line_cap, = ax_left.plot(
        x,
        caps,
        color="tab:orange",
        linewidth=1.2,
        linestyle="--",
        label="Cap (power.limit) [W]",
    )

    line_freq, = ax_right.plot(
        x,
        freqs,
        color="tab:blue",
        linewidth=1.2,
        label="Freq (clocks.current.sm) [MHz]",
    )
    line_max = ax_right.axhline(
        y=max_freq,
        color="tab:green",
        linewidth=1.0,
        linestyle=":",
        label=f"Max freq = {max_freq:g} MHz",
    )

    ax_left.set_xlabel("row index")
    ax_left.set_ylabel("Power [W]")
    ax_right.set_ylabel("Frequency [MHz]")

    ax_left.set_ylim(power_ylim)
    ax_right.set_ylim(freq_ylim)
    ax_left.set_xlim(1, max(n, 2))

    ax_left.grid(True, which="both", linestyle=":", alpha=0.4)

    title = os.path.basename(csv_path)
    ax_left.set_title(title)

    handles = [line_power, line_cap, line_freq, line_max]
    labels = [h.get_label() for h in handles]
    ax_left.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[ok] {csv_path} -> {out_png}")
    return True


def _collect_csvs(path: str, exclude: List[str]) -> List[str]:
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    exclude_norm = {e.lower() for e in exclude}
    out: List[str] = []
    for name in sorted(os.listdir(path)):
        if not name.lower().endswith(".csv"):
            continue
        if name.lower() in exclude_norm:
            continue
        out.append(os.path.join(path, name))
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot per-CSV power/frequency time series from nvidia-smi style logs.",
    )
    p.add_argument(
        "input",
        help="CSV file 또는 CSV들이 들어있는 디렉토리",
    )
    p.add_argument(
        "--max-freq",
        type=float,
        default=2430.0,
        help="오른쪽 Y축에 표시할 Max freq 고정 값(MHz). default=2430",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="PNG 출력 디렉토리. default=입력 디렉토리(또는 CSV가 있는 디렉토리)",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=["phases.csv"],
        help="제외할 파일명(디렉토리 입력일 때만 적용). default=['phases.csv']",
    )
    p.add_argument(
        "--power-ymin", type=float, default=0.0, help="왼쪽 Y축 최소값. default=0",
    )
    p.add_argument(
        "--power-ymax", type=float, default=1000.0, help="왼쪽 Y축 최대값. default=1000",
    )
    p.add_argument(
        "--freq-ymin", type=float, default=0.0, help="오른쪽 Y축 최소값. default=0",
    )
    p.add_argument(
        "--freq-ymax", type=float, default=3000.0, help="오른쪽 Y축 최대값. default=3000",
    )
    args = p.parse_args()

    try:
        csv_paths = _collect_csvs(args.input, args.exclude)
    except FileNotFoundError:
        print(f"[err] input not found: {args.input}", file=sys.stderr)
        return 2

    if not csv_paths:
        print(f"[err] no csv files found under: {args.input}", file=sys.stderr)
        return 2

    if args.outdir:
        outdir = args.outdir
    elif os.path.isdir(args.input):
        outdir = args.input
    else:
        outdir = os.path.dirname(os.path.abspath(args.input))

    ok = 0
    for cp in csv_paths:
        stem = os.path.splitext(os.path.basename(cp))[0]
        out_png = os.path.join(outdir, f"{stem}_plot.png")
        try:
            if _plot_one(
                cp,
                out_png,
                max_freq=args.max_freq,
                power_ylim=(args.power_ymin, args.power_ymax),
                freq_ylim=(args.freq_ymin, args.freq_ymax),
            ):
                ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[err] {cp}: {e}", file=sys.stderr)

    print(f"[done] plotted {ok}/{len(csv_paths)} files -> {outdir}")
    return 0 if ok == len(csv_paths) else 1


if __name__ == "__main__":
    sys.exit(main())
