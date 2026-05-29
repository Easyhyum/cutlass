#pragma once
// ── CUTLASS sleep params (Method A: SM-staggered nanosleep) ──────────────────
// bf16_gemm_sm80.cu 가 커널 런치 직전에 cudaMemcpyToSymbol 로 값을 업로드.
// mma_multistage.h 와 bf16_gemm_sm80.cu 양쪽에서 이 헤더를 include 하지만,
// #pragma once 덕분에 한 translation unit 내에서 딱 한 번만 정의됨.
//
// 활성화: 빌드 플래그 -DCUTLASS_SLEEP_ENABLED
//
// 동작 (Method A):
//   sleep_ns(sm) = kCutlassSleepNs                                   (베이스, 모든 SM 공통)
//               + (sm_id % kCutlassSleepStaggerMod) * kCutlassSleepStaggerNs   (SM-staggered)
//
//   - 프롤로그 (gemm_iters 진입 직후 1회): one-shot offset
//       SM이 mainloop에 들어가는 시점을 SM ID에 따라 다르게 함
//       → device-level 동시 HMMA 발사 감소 → power spike 분산
//   - 페리오드릭 (매 kCutlassSleepFreq 외부 iter 마다 1회): 지속적 phase 유지
//
//   모든 파라미터가 0 이면 baseline 동작 (sleep 없음, 종전과 동일).
#ifdef CUTLASS_SLEEP_ENABLED
__constant__ unsigned int kCutlassSleepNs        = 0u;  // base sleep (ns), 0=disable
__constant__ unsigned int kCutlassSleepFreq      = 1u;  // 매 N번째 outer iter 마다 periodic sleep
__constant__ unsigned int kCutlassSleepStaggerNs = 0u;  // per-SM-phase additional ns
__constant__ unsigned int kCutlassSleepStaggerMod= 1u;  // SM phase 개수 (예: 8 = 0~7 phase)
#endif
