#!/usr/bin/env bash
set -euo pipefail

IMAGE="cutlass-ljh:latest"
WORKDIR="/workspace"
USE_IPC_HOST=1           # 1 = use --ipc=host
USE_ULIMIT_MEMLOCK=1     # 1 = use --ulimit memlock=-1:-1

## NAME을 현재 user 이름으로 설정
########################################
# 3) SET NAME
########################################
NAME=cutlass-ljh-$(whoami)
TARGET_GPUS=3
GPU_OPTS=("-e" "CUDA_VISIBLE_DEVICES=${TARGET_GPUS}")
########################################
# 4) Execution flags
########################################
IPC_FLAG=""; [[ "${USE_IPC_HOST}" -eq 1 ]] && IPC_FLAG="--ipc=host"
ULIMIT_FLAG=""; [[ "${USE_ULIMIT_MEMLOCK}" -eq 1 ]] && ULIMIT_FLAG="--ulimit memlock=-1:-1"

########################################
# 5) docker run
########################################
CMD=(docker run -d -it
  --name "${NAME}"
  --gpus "device=${TARGET_GPUS}"
  --privileged
  "${GPU_OPTS[@]}"
  ${IPC_FLAG}
  ${ULIMIT_FLAG}
  -w "${WORKDIR}"
  -v "${PWD}":/workspace
  -v /data:/data
  -v /sys/kernel/debug:/sys/kernel/debug:ro
  "${IMAGE}"
  bash
)

echo "== Container Name : ${NAME}"
echo "== GPU     : ${TARGET_GPUS}"
echo "== Command to run =="
printf '%q ' "${CMD[@]}"; echo
exec "${CMD[@]}"