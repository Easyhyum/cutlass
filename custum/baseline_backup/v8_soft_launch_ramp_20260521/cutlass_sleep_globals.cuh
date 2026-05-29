#pragma once
// ── CUTLASS sleep params — v4 (pipeline-aware spin only) ─────────────────────
// 이전 sleep_ns / sleep_freq 코드는 모두 제거되었고, 유일한 hook은
// mma_multistage.h의 gmem_wait() 직후 SM-staggered clock64 busy-spin.
//
// 이전 버전 (v3 with sleep residue) 백업:
//   /workspace/custum/baseline_backup/v3_pipeline_aware_with_sleep_residue_20260521/
//
// 동작:
//   spin_cycles(this CTA) = (smid % StaggerMod) * StaggerNs (cycles 단위)
//   StaggerNs=0 또는 StaggerMod<=1 이면 baseline 동작 (spin 없음).
//
// 활성화: 빌드 플래그 -DCUTLASS_SLEEP_ENABLED
#ifdef CUTLASS_SLEEP_ENABLED
__constant__ unsigned int kCutlassSleepStaggerNs  = 0u;   // v7 ring rotation: ns per warp sleep
__constant__ unsigned int kCutlassSleepStaggerMod = 1u;   // legacy / enable flag
// v8 soft-launch ramp params:
__constant__ unsigned int kCutlassRampPeakNs      = 0u;   // initial sleep at outer_iter=0
__constant__ unsigned int kCutlassRampIters       = 1u;   // ramp duration in outer iters
#endif
