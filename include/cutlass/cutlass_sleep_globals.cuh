#pragma once
// ── CUTLASS sleep params ──────────────────────────────────────────────────────
// bf16_gemm_sm80.cu 가 커널 런치 직전에 cudaMemcpyToSymbol 로 값을 업로드.
// mma_multistage.h 와 bf16_gemm_sm80.cu 양쪽에서 이 헤더를 include 하지만,
// #pragma once 덕분에 한 translation unit 내에서 딱 한 번만 정의됨.
//
// 활성화: 빌드 플래그 -DCUTLASS_SLEEP_ENABLED
#ifdef CUTLASS_SLEEP_ENABLED
__constant__ unsigned int kCutlassSleepNs   = 0u;  // sleep 지속 시간(ns), 0=disable
__constant__ unsigned int kCutlassSleepFreq = 1u;  // 매 N번째 warp_mma_k 마다 sleep
#endif
