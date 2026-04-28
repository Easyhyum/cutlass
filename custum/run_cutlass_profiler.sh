#!/usr/bin/env bash
# =====================================================================
#  CUTLASS Profiler vs cuBLAS – Qwen3-8B 주요 GEMM shape 비교
#
#  사전 조건:
#    /workspace/build/tools/profiler/cutlass_profiler 가 빌드돼 있어야 함
#    (이전 build 로그 기준 SM80 커널 포함)
#
#  사용법:
#    bash run_cutlass_profiler.sh [출력디렉터리]
# =====================================================================
set -euo pipefail

PROFILER="/workspace/build/tools/profiler/cutlass_profiler"
OUTDIR="${1:-/workspace/custum/logs/cutlass}"
mkdir -p "$OUTDIR"

if [[ ! -x "$PROFILER" ]]; then
  echo "ERROR: cutlass_profiler not found at $PROFILER"
  echo "Build it first:"
  echo "  cd /workspace/build && make cutlass_profiler -j\$(nproc)"
  exit 1
fi

echo "============================================================"
echo " cutlass_profiler: $PROFILER"
echo " Output dir     : $OUTDIR"
echo "============================================================"

# ── 공통 옵션 ──────────────────────────────────────────────────────
#  --verification-enabled=false : 검증 생략 → 순수 성능만 측정
#  --warmup-iterations=20
#  --profiling-iterations=100
#  --providers=cutlass,cublas : CUTLASS 와 cuBLAS 모두 출력
COMMON_OPTS=(
  --warmup-iterations=20
  --profiling-iterations=100
  --verification-enabled=false
  --providers=cutlass,cublas
  --library-algo-mode=best
)

# ── Qwen3-8B MLP 주요 shape (FP16, NN layout) ─────────────────────
# batch_size=1, seq_len=4096 → M=4096
# mlp_up_proj  : (4096, 4096) @ (4096, 12288)  → M=4096, K=4096, N=12288
# mlp_down_proj: (4096,12288) @ (12288, 4096)  → M=4096, K=12288, N=4096

declare -A SHAPES=(
  ["mlp_up_proj"]="--m=4096 --n=12288 --k=4096"
  ["mlp_down_proj"]="--m=4096 --n=4096 --k=12288"
  ["attn_qkv_4096x4096"]="--m=4096 --n=4096 --k=4096"
  ["attn_o_proj"]="--m=4096 --n=4096 --k=4096"
)

# ── FP16 TensorOp 커널 패턴 (SM80 s16816gemm 계열이 가장 강력) ───
#  너무 많은 커널을 빌드하면 시간이 오래 걸리므로 대표 패턴만 사용
KERNEL_FILTER="cutlass_tensorop_s*gemm_f16_*_nt_align8"

for name in "${!SHAPES[@]}"; do
  shape_args="${SHAPES[$name]}"
  outcsv="$OUTDIR/${name}.csv"

  echo ""
  echo "────────────────────────────────────────"
  echo " Shape: $name  ($shape_args)"
  echo " Output: $outcsv"
  echo "────────────────────────────────────────"

  # shellcheck disable=SC2086
  "$PROFILER" \
    --operation=gemm \
    --kernels="$KERNEL_FILTER" \
    $shape_args \
    --A=f16:column \
    --B=f16:row \
    --C=f32:column \
    --alpha=1 --beta=0 \
    "${COMMON_OPTS[@]}" \
    --output="$outcsv" \
    || true   # 일부 커널 미지원 시에도 계속

  echo "  → 저장 완료: $outcsv"
done

# ── 결과 요약 ────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " 요약: 각 shape 의 Top-5 커널 (GFLOP/s 기준)"
echo "============================================================"
for name in "${!SHAPES[@]}"; do
  outcsv="$OUTDIR/${name}.csv"
  if [[ -f "$outcsv" ]]; then
    echo ""
    echo "── $name ──"
    # CSV에서 Runtime_ms, GFLOPs 열을 파싱해 상위 5개 출력
    # 헤더 출력 후 GFLOPs 역순 정렬
    head -1 "$outcsv"
    tail -n +2 "$outcsv" \
      | sort -t',' -k$(head -1 "$outcsv" | tr ',' '\n' | grep -n "GFLOPs\|Math" | head -1 | cut -d: -f1) -rn \
      | head -5 \
      || true
  fi
done

echo ""
echo "✅ 완료. CSV 파일들: $OUTDIR/"
