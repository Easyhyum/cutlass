#!/usr/bin/env python3
"""
nvidia-smi 가 만든 CSV (--format=csv -lms …) 에 전력 피크 추적용 열을 붙인다.

왜 필요한가
  ``nvidia-smi --query-gpu=…`` 에는 ``nvidia-smi -q -d POWER`` 안의
  "Power Samples" 블록(구간 Max/Min/Avg) 과 동일한 필드가 **없다**.
  드라이버가 더 촘촘히 모은 통계와 완전히 같게 만들 수는 없다.

이 스크립트가 하는 일
  로그에 이미 찍힌 ``power.draw.instant`` 값만 사용해 다음 열을 추가한다.

  - ``power.draw.instant_cummax [W]``
      로그 첫 데이터 행부터 **현재 행까지** instant 의 최댓값(누적 최대).
      파일 전체의 피크는 마지막 행의 이 열에서 보면 된다.

  선택 ``--rolling N``
  - ``power.draw.instant_rollmax [W]``
      **가장 최근 N개 샘플** 구간에서의 instant 최댓값
      (-lms 100 이면 대략 N×0.1초 구간; 예: N=1180 ≈ 118초).

샘플 간격 사이의 아주 짧은 스파이크는 CSV 샘플에 안 나올 수 있다.
더 촘촘히 보려면 ``-lms`` 를 줄이거나 NVML/pynvml 을 쓴다.

사용 예::

  nvidia-smi -i 3 --query-gpu=timestamp,clocks.current.sm,power.draw.instant,... \\
    --format=csv -lms 100 \\
    | python3 gpu_power_log_augment.py > gpu_power_log_with_max.csv

  python3 gpu_power_log_augment.py gpu_power_log_mma_9.csv -o gpu_power_log_mma_9_max.csv
  python3 gpu_power_log_augment.py log.csv --rolling 1180 -o log_roll.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import deque


def _find_instant_column(header: list[str]) -> int | None:
    for i, h in enumerate(header):
        if "power.draw.instant" in h.strip().lower():
            return i
    return None


def _parse_watts(cell: str) -> float:
    s = (cell or "").strip().replace(",", "")
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return float("nan")
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "nvidia-smi CSV 에 power.draw.instant 기준 누적 최대(및 선택적 롤링 최대) 열 추가"
        )
    )
    ap.add_argument(
        "input",
        nargs="?",
        default="-",
        help="입력 CSV 경로 (기본: stdin)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="-",
        help="출력 경로 (기본: stdout)",
    )
    ap.add_argument(
        "--rolling",
        type=int,
        metavar="N",
        default=0,
        help="최근 N개 샘플 구간 instant 최대 열 추가 (0이면 생략)",
    )
    args = ap.parse_args()

    fin = sys.stdin if args.input == "-" else open(args.input, newline="", encoding="utf-8")
    fout = sys.stdout if args.output == "-" else open(args.output, "w", newline="", encoding="utf-8")

    try:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        rows = iter(reader)
        try:
            header = next(rows)
        except StopIteration:
            return 0

        idx = _find_instant_column(header)
        if idx is None:
            print(
                "gpu_power_log_augment: 헤더에 power.draw.instant 열이 없습니다.",
                file=sys.stderr,
            )
            return 1

        extra_cummax = "power.draw.instant_cummax [W]"
        extras = [extra_cummax]
        if args.rolling > 0:
            extras.append("power.draw.instant_rollmax [W]")

        writer.writerow(header + extras)

        cummax = float("-inf")
        buf: deque[float] | None = deque(maxlen=args.rolling) if args.rolling > 0 else None

        for row in rows:
            if not row:
                continue
            while len(row) <= idx:
                row.append("")
            inst = _parse_watts(row[idx])
            extra: list[str] = []
            if inst == inst:  # finite (not NaN)
                cummax = max(cummax, inst)
                extra.append(f"{cummax:.2f} W")
                if buf is not None:
                    buf.append(inst)
                    extra.append(f"{max(buf):.2f} W")
            else:
                extra.append("")
                if buf is not None:
                    extra.append("")
            writer.writerow(row + extra)
    finally:
        if fin is not sys.stdin:
            fin.close()
        if fout is not sys.stdout:
            fout.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
