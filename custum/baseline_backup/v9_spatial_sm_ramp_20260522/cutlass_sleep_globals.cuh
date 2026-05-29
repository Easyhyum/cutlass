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

// v8 (A) one-shot ramp params (linear activity model):
//   activity[k] = min(100, start_pct + k * step_pct)
//   sleep_ns[k] = iter_time_ns * (100 - activity[k]) / activity[k]   if activity<100
//                 0                                                   otherwise
//   start_pct  = activity at outer_iter k=0 (e.g. 70 = 70% of full HMMA rate)
//   step_pct   = per-iter activity increase (e.g. 5 → 70, 75, 80, 85, 90, 95, 100)
//   iter_time_ns = nominal mainloop outer iter time (host estimate)
//   start_pct = 100  →  ramp disabled (no sleep, no extra work)
__constant__ unsigned int kCutlassRampStartPct    = 100u;   // 0..100; 100 = ramp OFF
__constant__ unsigned int kCutlassRampStepPct     = 100u;   // activity %/iter increment
__constant__ unsigned int kCutlassRampIterTimeNs  = 0u;     // nominal outer iter time (ns)
#endif
