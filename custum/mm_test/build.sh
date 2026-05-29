#!/usr/bin/env bash
# One-line nvcc build for mm_test (BF16 cuBLAS GEMM)
# Target: RTX PRO 6000 Blackwell = sm_120
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${ARCH:-sm_120}"
NVCC="${NVCC:-nvcc}"

set -x
"$NVCC" -O3 -std=c++17 -arch="$ARCH" \
        -Xcompiler -Wall \
        mm_test.cu -o mm_test \
        -lcublas
