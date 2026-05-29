#!/usr/bin/env python3
"""
bench_mark.py 가 쓰는 detail CSV 를 읽어 NVML 샘플 시계열을 PNG 로 저장한다.

matplotlib 가 필요하다. bench_mark.py 는 종료 시 벤치와 동일한 인터프리터
(``sys.executable``)로 이 스크립트를 자동 호출한다.

레이아웃 (한 장의 PNG):
  - 위 패널: sample_sm_clock_mhz (주파수) + 2430 MHz 참조 점선
  - 아래 패널: sample_power_w + gpu_power_cap_w 수평선(있을 때)
  - y 축: Frequency 1800–2800 MHz, Power 250–800 W (고정)
  - x 축: 샘플 행 인덱스 (1 … N, 연속 NVML 샘플)
  - measure_id 구간마다 배경색으로 kernel 구분; 위·아래 패널 구간 상단에 sleep_ns 표기
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys


def _fget(row: dict, key: str, default: float = float("nan")) -> float:
    v = row.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main() -> int:
    p = argparse.ArgumentParser(description="Plot bench_mark detail CSV to PNG")
    p.add_argument("detail_csv", help="Path to *_detail.csv")
    p.add_argument(
        "-o", "--output",
        help="Output PNG (default: <detail_stem>_plot.png next to CSV)",
    )
    args = p.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator
        from matplotlib.transforms import blended_transform_factory
    except ImportError as e:
        print("plot_detail_csv: need matplotlib:", e, file=sys.stderr)
        return 1

    path = args.detail_csv
    if not os.path.isfile(path):
        print("plot_detail_csv: file not found:", path, file=sys.stderr)
        return 1

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("plot_detail_csv: no data rows in", path, file=sys.stderr)
        return 1

    n = len(rows)
    # 참고 플롯처럼 샘플 인덱스는 1 … N
    xs = list(range(1, n + 1))
    power = [_fget(r, "sample_power_w") for r in rows]
    sm_mhz = [_fget(r, "sample_sm_clock_mhz") for r in rows]

    has_cap = "gpu_power_cap_w" in rows[0]
    cap_val = _fget(rows[0], "gpu_power_cap_w") if has_cap else float("nan")

    # 참고 이미지 스타일 (진한 파워 / 옅은 주파수 / 캡 초록)
    C_POWER = "#1e3a8a"
    C_FREQ = "#38bdf8"
    C_CAP = "#166534"
    SM_REF_CLOCK_MHZ = 2430  # Frequency 패널 참조 수평선 (MHz)
    BG_KERNEL = {
        "cublas": "#dbeafe",
        "cutlass_sm80": "#fee2e2",
    }
    BG_DEFAULT = "#f1f5f9"

    out_path = args.output
    if not out_path:
        base, _ = os.path.splitext(path)
        out_path = base + "_plot.png"

    fig, (ax_freq, ax_pwr) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(14, 7.5),
        dpi=120,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.06},
        facecolor="white",
    )

    # 세그먼트: measure_id 연속 구간
    i0 = 0
    seg_labels: list[tuple[str, str, str]] = []  # (measure_id, kernel, sleep_ns)
    segment_annots: list[tuple[float, str]] = []  # (x_center, sleep_ns 문자열)
    while i0 < n:
        i1 = i0 + 1
        mid0 = rows[i0].get("measure_id", "")
        while i1 < n and rows[i1].get("measure_id", "") == mid0:
            i1 += 1

        kern = rows[i0].get("kernel", "?")
        sns = rows[i0].get("sleep_ns", "")
        seg_labels.append((str(mid0), kern, str(sns)))

        x_lo = xs[i0] - 0.5
        x_hi = xs[i1 - 1] + 0.5
        bg = BG_KERNEL.get(kern, BG_DEFAULT)
        for ax in (ax_freq, ax_pwr):
            ax.axvspan(x_lo, x_hi, facecolor=bg, alpha=0.55, zorder=0, lw=0)

        x_center = 0.5 * (x_lo + x_hi)
        segment_annots.append((x_center, str(sns)))

        seg_x = xs[i0:i1]
        seg_p = power[i0:i1]
        seg_s = sm_mhz[i0:i1]

        ax_freq.plot(
            seg_x,
            seg_s,
            color=C_FREQ,
            lw=1.35,
            solid_capstyle="round",
            zorder=3,
        )
        ax_pwr.plot(
            seg_x,
            seg_p,
            color=C_POWER,
            lw=1.35,
            solid_capstyle="round",
            zorder=3,
        )

        # 구간 경계 세로선 (옅게)
        if i1 < n:
            boundary = xs[i1 - 1] + 0.5
            for ax in (ax_freq, ax_pwr):
                ax.axvline(
                    boundary,
                    color="#94a3b8",
                    lw=0.85,
                    ls=":",
                    alpha=0.65,
                    zorder=2,
                )

        i0 = i1

    if has_cap and math.isfinite(cap_val):
        ax_pwr.axhline(
            cap_val,
            color=C_CAP,
            lw=1.45,
            ls="--",
            alpha=0.92,
            zorder=4,
        )

    ax_freq.axhline(
        SM_REF_CLOCK_MHZ,
        color="#a855f7",
        lw=1.45,
        ls="--",
        alpha=0.92,
        zorder=4,
    )

    # 축·그리드 (참고: 흰 배경, 수평 그리드)
    for ax in (ax_freq, ax_pwr):
        ax.set_facecolor("white")
        ax.grid(True, axis="y", alpha=0.35, color="#cbd5e1", zorder=1)
        ax.grid(False, axis="x")

    ax_freq.set_ylabel("Frequency (MHz)", fontsize=10, color="#0c4a6e")
    ax_freq.tick_params(axis="y", labelcolor="#0c4a6e")
    ax_pwr.set_ylabel("Power (W)", fontsize=10, color=C_POWER)
    ax_pwr.tick_params(axis="y", labelcolor=C_POWER)
    ax_pwr.set_xlabel("Sample index (detail CSV row)", fontsize=10)
    ax_pwr.xaxis.set_major_locator(MaxNLocator(nbins=14, prune=None))

    # y 범위: SM clock / 전력은 고정 스케일 (요청 값)
    ax_freq.set_ylim(1800, 2800)
    ax_pwr.set_ylim(30, 800)

    # 기본 x margins(~5%) 때문에 좌우가 과하게 비어 보임 → 샘플 열(1…N)에 맞춤
    for ax in (ax_freq, ax_pwr):
        ax.margins(x=0)
        ax.set_xlim(0.5, n + 0.5)

    # 각 세그먼트(배경 구간) 상단에 sleep_ns — 위·아래 패널 모두
    trans_top_f = blended_transform_factory(ax_freq.transData, ax_freq.transAxes)
    trans_top_p = blended_transform_factory(ax_pwr.transData, ax_pwr.transAxes)
    for xc, sns in segment_annots:
        label = f"sleep_ns={sns}"
        for ax, trans in ((ax_freq, trans_top_f), (ax_pwr, trans_top_p)):
            ax.text(
                xc,
                0.97,
                label,
                transform=trans,
                ha="center",
                va="top",
                fontsize=7.5,
                color="#0f172a",
                fontweight="semibold",
                zorder=6,
                clip_on=False,
            )

    r0 = rows[0]
    seq = r0.get("seq_len", "")
    bat = r0.get("batch_size", "")
    mx_f = max((v for v in sm_mhz if math.isfinite(v)), default=0.0)
    cap_part = f", Cap {cap_val:.0f}" if has_cap and math.isfinite(cap_val) else ""
    title = (
        f"Seq {seq}  Batch {bat}{cap_part},  "
        f"Fre {mx_f:.0f}  |  {os.path.basename(path)}"
    )
    fig.suptitle(title, fontsize=11, fontweight="semibold", y=0.995)

    # 범례: 참고 그림처럼 상단 중앙 — 라인 전용 handle (구간 색은 배경으로 설명)
    leg_handles = [
        Line2D([0], [0], color=C_POWER, lw=2.0, label="W"),
        Line2D([0], [0], color=C_FREQ, lw=2.0, label="Frequency"),
        Line2D(
            [0],
            [0],
            color="#a855f7",
            lw=1.5,
            ls="--",
            label=f"{SM_REF_CLOCK_MHZ} MHz",
        ),
    ]
    if has_cap and math.isfinite(cap_val):
        leg_handles.insert(
            1,
            Line2D([0], [0], color=C_CAP, lw=1.5, ls="--", label="Cap"),
        )

    fig.legend(
        handles=leg_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(leg_handles),
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        fontsize=9,
    )

    # 구간 설명 (kernel / sleep_ns) — 그림 아래 작은 글씨
    _show = min(16, len(seg_labels))
    seg_txt = "  |  ".join(
        f"M{s[0]} {s[1]} sleep_ns={s[2]}" for s in seg_labels[:_show]
    )
    if len(seg_labels) > _show:
        seg_txt += f"  … (+{len(seg_labels) - _show} segments)"
    fig.text(0.5, 0.02, seg_txt, ha="center", fontsize=6.5, color="#64748b")

    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.055, right=0.985)
    d = os.path.dirname(os.path.abspath(out_path)) or "."
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(
        out_path,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05,
    )
    plt.close(fig)
    print(f"plot_detail_csv: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
