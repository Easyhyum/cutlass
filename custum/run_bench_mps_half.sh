#!/usr/bin/env bash
# =====================================================================
#  NVIDIA MPS (Multi-Process Service) 로 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
#  를 50→60→70→80→90→100 으로 sweep 하면서 custum/bench_mark.py 를
#  반복 실행하고, 각 구간을 별도 power_profile CSV 로 분리 기록한다.
#
#  핵심:
#    - CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=N  → 클라이언트 컨텍스트가 쓸
#      수 있는 SM warp slot 을 GPU 전체의 N% 로 제한 (Volta+).
#    - MPS 제어 daemon 은 한 번만 띄우고 모든 phase 에서 재사용. 매
#      iteration 직전에 환경변수만 새 값으로 export → 다음에 만들어지는
#      CUDA context 에 새 비율이 적용됨.
#    - phase 마다 nvidia-smi 를 새 파일로 띄워 CSV 가 분리되며, 추가로
#      phases.csv 에 (pct, start_iso, end_iso, exit_code) 를 남겨 후처리
#      때 시간 윈도우로 잘라 합치기 쉽도록 한다.
#
#  사용법:
#    bash custum/run_bench_mps_half.sh                       # GPU 0, 50..100
#    TARGET_GPU=3 bash custum/run_bench_mps_half.sh          # 다른 GPU
#    THREAD_PCT_LIST="50 75 100" bash custum/run_bench_mps_half.sh
#    bash custum/run_bench_mps_half.sh -m 0x2 -w 0x2         # 추가 인자 패스스루
#
#  결과물 (예):
#    custum/profile/run_<timestamp>/
#      ├ phases.csv                # 구간별 시작/종료 timestamp + exit code
#      ├ power_pct050.csv          # nvidia-smi -lms 100 sample (50%)
#      ├ power_pct060.csv
#      ├ ...
#      └ power_pct100.csv
# =====================================================================
set -euo pipefail

# ── 사용자 설정 (환경변수로 override 가능) ─────────────────────────
TARGET_GPU="${TARGET_GPU:-0}"          # 물리 GPU 인덱스
# 공백 구분 정수 리스트. 기본은 50,60,70,80,90,100.
THREAD_PCT_LIST="${THREAD_PCT_LIST:-50 60 70 80 90 100}"
# 각 phase 사이 cool-down (초). nvidia-smi 종료 + 다음 시작 사이 휴지.
PHASE_COOLDOWN="${PHASE_COOLDOWN:-2}"
MPS_BASE="${MPS_BASE:-/tmp/nvidia-mps-gpu${TARGET_GPU}-$$}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SCRIPT="${SCRIPT_DIR}/bench_mark.py"

NVSMI_PID=""   # nvidia-smi 백그라운드 PID (아직 시작 전 → 빈 값)

if [[ ! -f "${BENCH_SCRIPT}" ]]; then
  echo "[mps] ERROR: ${BENCH_SCRIPT} 를 찾을 수 없습니다." >&2
  exit 1
fi
if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "[mps] ERROR: nvidia-cuda-mps-control 바이너리가 PATH 에 없습니다." >&2
  echo "         CUDA 드라이버/툴킷 설치 및 PATH 를 확인하세요." >&2
  exit 1
fi
# THREAD_PCT_LIST 파싱/검증
read -r -a THREAD_PCT_ARR <<< "${THREAD_PCT_LIST}"
if (( ${#THREAD_PCT_ARR[@]} == 0 )); then
  echo "[mps] ERROR: THREAD_PCT_LIST 가 비어 있습니다." >&2
  exit 1
fi
for _pct in "${THREAD_PCT_ARR[@]}"; do
  if ! [[ "${_pct}" =~ ^[0-9]+$ ]] || (( _pct < 1 || _pct > 100 )); then
    echo "[mps] ERROR: THREAD_PCT_LIST 항목은 1~100 사이의 정수여야 합니다 (입력값: ${_pct})." >&2
    exit 1
  fi
done

# ── MPS 전용 파이프/로그 경로 (사용자/PID 별 격리) ─────────────────
export CUDA_MPS_PIPE_DIRECTORY="${MPS_BASE}/pipe"
export CUDA_MPS_LOG_DIRECTORY="${MPS_BASE}/log"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

# ── 정리 루틴 ─────────────────────────────────────────────────────
COMPUTE_MODE_CHANGED=0
MPS_DAEMON_STARTED=0

restore_compute_mode() {
  local SET_DEFAULT=(nvidia-smi -i "${TARGET_GPU}" -c DEFAULT)
  if [[ ${EUID} -eq 0 ]]; then
    "${SET_DEFAULT[@]}" >/dev/null 2>&1 || true
  elif sudo -n true 2>/dev/null; then
    sudo "${SET_DEFAULT[@]}" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  set +e
  echo "[mps] cleanup: MPS daemon 종료 및 환경 복원..."
  if [[ ${MPS_DAEMON_STARTED} -eq 1 ]]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    for _ in $(seq 1 25); do
      pgrep -f "nvidia-cuda-mps-(control|server)" >/dev/null 2>&1 || break
      sleep 0.2
    done
  fi
  if [[ ${COMPUTE_MODE_CHANGED} -eq 1 ]]; then
    restore_compute_mode
  fi
  # nvidia-smi 모니터링 정리 (시작 전이면 NVSMI_PID 가 빈 문자열)
  if [[ -n "${NVSMI_PID:-}" ]]; then
    kill "${NVSMI_PID}" 2>/dev/null || true
    wait "${NVSMI_PID}" 2>/dev/null || true
    echo "■ nvidia-smi stopped (PID=${NVSMI_PID})"
  fi
  # 격리된 임시 디렉터리는 남겨두면 다음 디버깅에 도움이 되므로 그대로 둔다.
  echo "[mps] cleanup done. (pipe=${CUDA_MPS_PIPE_DIRECTORY}, log=${CUDA_MPS_LOG_DIRECTORY})"
}
trap cleanup EXIT INT TERM

# ── 1) GPU compute mode = EXCLUSIVE_PROCESS (권장) ────────────────
echo "[mps] target GPU=${TARGET_GPU}, SM percentage sweep=[${THREAD_PCT_LIST}]"
SET_EXCL=(nvidia-smi -i "${TARGET_GPU}" -c EXCLUSIVE_PROCESS)
if [[ ${EUID} -eq 0 ]]; then
  if "${SET_EXCL[@]}" >/dev/null 2>&1; then
    COMPUTE_MODE_CHANGED=1
    echo "[mps] compute mode -> EXCLUSIVE_PROCESS (root)"
  else
    echo "[mps] WARN: EXCLUSIVE_PROCESS 설정 실패. 계속 진행."
  fi
elif sudo -n true 2>/dev/null; then
  if sudo "${SET_EXCL[@]}" >/dev/null 2>&1; then
    COMPUTE_MODE_CHANGED=1
    echo "[mps] compute mode -> EXCLUSIVE_PROCESS (sudo)"
  else
    echo "[mps] WARN: sudo 로도 compute mode 변경 실패. 계속 진행."
  fi
else
  echo "[mps] WARN: root/sudo 권한 없음 -> compute mode 변경 생략."
  echo "         (컨테이너 안이라면 호스트에서 미리"
  echo "          'nvidia-smi -i ${TARGET_GPU} -c EXCLUSIVE_PROCESS' 설정 권장)"
fi

# ── 2) 혹시 떠 있을지 모르는 이전 MPS daemon 정리 ─────────────────
echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
sleep 0.3

# ── 3) MPS control daemon 기동 (해당 GPU 만 관리) ─────────────────
echo "[mps] starting nvidia-cuda-mps-control daemon"
CUDA_VISIBLE_DEVICES="${TARGET_GPU}" nvidia-cuda-mps-control -d
# nvidia-cuda-mps-control -d
MPS_DAEMON_STARTED=1

# daemon 이 파이프를 만들 때까지 잠시 대기
for _ in $(seq 1 25); do
  if [[ -S "${CUDA_MPS_PIPE_DIRECTORY}/control" ]]; then
    break
  fi
  sleep 0.2
done
if [[ ! -S "${CUDA_MPS_PIPE_DIRECTORY}/control" ]]; then
  echo "[mps] ERROR: MPS control pipe 가 생성되지 않았습니다."
  echo "         로그: ${CUDA_MPS_LOG_DIRECTORY}"
  exit 2
fi

# daemon 상태 확인 (best-effort)
echo "[mps] daemon status:"
echo get_server_list | nvidia-cuda-mps-control 2>/dev/null || true

# ── 4) 클라이언트 공통 환경 (CUDA_VISIBLE_DEVICES 등) ──────────────
export CUDA_VISIBLE_DEVICES="${TARGET_GPU}"

echo "[mps] common env:"
echo "       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "       CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY}"
echo "       CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY}"
echo "       THREAD_PCT_LIST=${THREAD_PCT_LIST}"

# ── 5) PCT sweep 실행 + 구간별 power profile ──────────────────────
RUN_TS="$(date +%Y%m%d_%H%M%S)"
PROFILE_DIR="${SCRIPT_DIR}/profile/run_${RUN_TS}"
mkdir -p "${PROFILE_DIR}"
PHASES_CSV="${PROFILE_DIR}/phases.csv"
echo "phase_pct,gpu_log,start_iso,end_iso,duration_sec,exit_code" > "${PHASES_CSV}"

echo "[mps] sweep output dir: ${PROFILE_DIR}"

LAST_RC=0
for PCT in "${THREAD_PCT_ARR[@]}"; do
  PCT_TAG="$(printf 'pct%03d' "${PCT}")"
  GPU_LOG="${PROFILE_DIR}/power_${PCT_TAG}.csv"

  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="${PCT}"

  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo " phase: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=${PCT}%"
  echo " gpu log : ${GPU_LOG}"
  echo "════════════════════════════════════════════════════════════"

  # phase 전용 nvidia-smi 모니터 시작
  nvidia-smi -i "${TARGET_GPU}" \
      --query-gpu=timestamp,clocks.current.sm,power.draw.instant,power.draw.average,power.limit,clocks_event_reasons.sw_power_cap,clocks_event_reasons_counters.sw_power_cap \
      --format=csv \
      -lms 100 > "${GPU_LOG}" &
  NVSMI_PID=$!
  echo "▶ nvidia-smi started (PID=${NVSMI_PID})"
  sleep 1   # nvidia-smi 가 첫 sample 찍을 시간 확보

  start_iso="$(date -Iseconds)"
  start_epoch="$(date +%s)"

  set +e
  python "${BENCH_SCRIPT}" "$@"
  RC=$?
  set -e

  end_iso="$(date -Iseconds)"
  end_epoch="$(date +%s)"
  dur=$(( end_epoch - start_epoch ))

  # phase 종료 후 잔향을 살짝 더 기록한 뒤 nvidia-smi 종료
  sleep 1
  kill "${NVSMI_PID}" 2>/dev/null || true
  wait "${NVSMI_PID}" 2>/dev/null || true
  echo "■ nvidia-smi stopped (PID=${NVSMI_PID})"
  NVSMI_PID=""

  echo "${PCT},$(basename "${GPU_LOG}"),${start_iso},${end_iso},${dur},${RC}" >> "${PHASES_CSV}"
  echo "[mps] phase ${PCT}% done — exit=${RC}, dur=${dur}s"

  LAST_RC="${RC}"
  if (( RC != 0 )); then
    echo "[mps] WARN: phase ${PCT}% 실패 (exit=${RC}). 다음 phase 계속 진행."
  fi

  # 다음 phase 사이 cool-down
  if (( PHASE_COOLDOWN > 0 )); then
    sleep "${PHASE_COOLDOWN}"
  fi
done

echo ""
echo "[mps] sweep complete."
echo "       phases summary : ${PHASES_CSV}"
echo "       per-phase logs : ${PROFILE_DIR}/power_pct*.csv"
echo "[mps] last bench_mark.py exit code: ${LAST_RC}"

# ── 6) sweep 결과 자동 plot ─────────────────────────────────────────
#   PLOT_SCRIPT     : custum/plot_power_csv.py (없으면 skip)
#   PLOT_MAX_FREQ   : 오른쪽 Y축 Max freq 고정값 (default 2430)
#   PLOT_EXTRA_ARGS : 추가 인자 패스스루 (예: "--power-ymax 800")
PLOT_SCRIPT="${SCRIPT_DIR}/plot_power_csv.py"
PLOT_MAX_FREQ="${PLOT_MAX_FREQ:-2430}"
PLOT_EXTRA_ARGS="${PLOT_EXTRA_ARGS:-}"
if [[ -f "${PLOT_SCRIPT}" ]]; then
  echo ""
  echo "[plot] running ${PLOT_SCRIPT} on ${PROFILE_DIR} (max_freq=${PLOT_MAX_FREQ})"
  # shellcheck disable=SC2086
  python "${PLOT_SCRIPT}" "${PROFILE_DIR}" --max-freq "${PLOT_MAX_FREQ}" ${PLOT_EXTRA_ARGS} \
    || echo "[plot] WARN: plot step failed (matplotlib 미설치 또는 데이터 문제일 수 있음)."
else
  echo "[plot] SKIP: ${PLOT_SCRIPT} 가 없습니다."
fi

exit "${LAST_RC}"
