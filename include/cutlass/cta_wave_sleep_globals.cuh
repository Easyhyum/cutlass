#pragma once
// ── Wave-aware per-CTA sleep globals ──────────────────────────────────────
//
// Activated by build flag  -DCUTLASS_WAVE_SLEEP_ENABLED.
//
// Idea: when the launched grid spans ≥ 3 hardware waves, we can treat the
// three groups very differently:
//
//   * Wave 0 (first wave):  graduated staircase by SM id — the first N_thr
//                           SMs start immediately, the remaining SMs each
//                           sleep (smid - N_thr) * step_ns at operator()
//                           entry. Used to RAMP UP device-level activity.
//
//   * Wave k  (1 ≤ k ≤ N-2): a hash-selected fraction P% of the CTAs sleep
//                            ConstNs nanoseconds at entry. Used to keep the
//                            SM-level utilization noisy / spread over time
//                            during the "steady state".
//
//   * Wave N-1 (last wave):  no sleep at all — the kernel should drain as
//                            fast as possible.
//
// Wave index is computed device-side as
//      wave_idx = (bz*gridDim.y + by) * gridDim.x + bx
//                  ────────── linear blockIdx ──────────
//                                  / kWaveSleepNSm
// which is correct as long as the swizzle dispatches CTAs in row-major
// linear order — which we verified is the case for both
// ThreadblockSwizzleStreamK and GemmIdentityThreadblockSwizzle on RTX PRO
// 6000 Blackwell (probe data, monotone_frac ≈ 1 for M ≤ 131072).
//
// Host setter: probe_streamk.cu :: configure_wave_sleep(...).
//
#ifdef CUTLASS_WAVE_SLEEP_ENABLED
#include <cstdint>

__constant__ int          kWaveSleepNumWaves         = 0;   // total #waves the grid spans; 0 → feature disabled
__constant__ int          kWaveSleepNSm              = 0;   // CTAs per wave (= multi_processor_count)
__constant__ int          kWaveSleepFirstWaveSmidThr = 0;   // smid < thr → no delay in wave 0
__constant__ unsigned int kWaveSleepFirstWaveStepNs  = 0;   // ns per smid step beyond threshold in wave 0
__constant__ unsigned int kWaveSleepMidWavePct       = 0;   // 0..100 : fraction of mid-wave CTAs that sleep
__constant__ unsigned int kWaveSleepMidWaveNs        = 0;   // ns sleep when selected
__constant__ unsigned int kWaveSleepHashSeed         = 0xC0FFEE11u;

// Mode:
//   0 = wave-0-only staircase (post-prologue), optional mid bubble in mac_loop_iter
//   1 = ALL-wave staircase   (applied at every mac_loop_iter start, no mid bubble)
//   4 = UNIFORM per-iter      (every CTA × every mac_loop_iter sleeps mid_ns;
//                              no staircase, no hash gate — purest duty-cycle dial)
//   5 = EVERY-N-TH iter       (every CTA, but sleep only when
//                              (gemm_k_iterations % N == 0), N = 100/mid_pct.
//                              mid_pct=50→N=2, 25→4, 10→10, etc. Bypasses
//                              the nanosleep quantum by reducing frequency.)
//   6 = mode 5 WITHOUT __syncthreads — only thread-0's warp stalls; the
//                              other warps in the CTA keep executing. SM-level
//                              power dip is ~1/4 of mode 5 (one of four warps).
//   7 = SM GATING (MPS-style) — CTAs landing on smid >= first_smid_thr return
//                              immediately. first_smid_thr is the active cutoff:
//                                 thr = n_sm * active_pct / 100
//                              e.g. active_pct=70, n_sm=188 → thr=131 →
//                              SMs 0..130 run mainloop, SMs 131..187 skip.
//                              Spatial 70% active replicates the MPS-70% power
//                              point (~600W observed) without nanosleep.
//   8 = ROTATIONAL stagger    — at each mac_loop_iter, 1/N of the SMs sleep
//                              briefly while the rest issue MMA. N=mid_pct
//                              (e.g. mid_pct=4 → 25% SMs sleep, 75% MMA at any
//                              moment). Each SM idles every N-th iter, so
//                              100% SM utilisation is preserved but the
//                              instantaneous MMA-power peak is cut by 1/N.
//                              Sleep duration = mid_ns (use small, e.g. 200ns).
//   9 = MEMORY-PRESSURE        — at each mac_loop_iter, every CTA issues
//                              `mid_pct` extra global memory loads from a
//                              host-supplied dummy buffer at randomized
//                              offsets — injects TLB/L2 pressure so part of
//                              the SM activity shifts from MMA to memory
//                              (~380W class) rather than full MMA (~700W).
//                              first_smid_thr serves as the cycle-stride.
//  10 = UNIFIED 3-PHASE wave-sleep with explicit last-wave skip:
//                                  wave 0      : smid-keyed staircase delay
//                                                at operator() entry (post-
//                                                prologue), ONCE per CTA —
//                                                first-wave burst prevention.
//                                  wave 1..N-2 : per-iter mid bubble inside
//                                                mac_loop_iter — hash-gated
//                                                mid_pct% of CTAs sleep mid_ns
//                                                ns at the START of every
//                                                outer K iter.  Hash mixes
//                                                gemm_k_iterations so the set
//                                                of CTAs selected ROTATES
//                                                across iters → no single SM
//                                                is consistently the one
//                                                sleeping (tail-latency
//                                                balanced across SMs).
//                                  wave N-1    : no sleep (drain fast).
//                              Differs from mode 0 only in: (a) last-wave skip
//                              is explicit in mode 10 (mode 0 also skips via
//                              the same `_wsm_wave_idx < N-1` guard), (b)
//                              meant to be paired with the sticky one-shot
//                              semantic in the host setter so the sleep
//                              persists across all kernel launches in a burst
//                              (not just the first).
__constant__ int          kWaveSleepMode             = 0;

// Pointer to a host-allocated dummy buffer used by mode 9 (memory pressure).
// Buffer size in 4-byte units is kMemPressureWords (set via host setter).
__device__ unsigned int*  g_mem_pressure_buf         = nullptr;
__constant__ int          kMemPressureWords          = 0;

// Staircase shape — applied to wave-0 (mode 0) and to every wave (mode 1):
//   0 = linear:    delay = slot * step_ns
//   1 = quartile:  4 step levels — SMs binned into 4 groups
//   2 = octile:    8 step levels (smoother than quartile, more gradual ramp)
//   3 = 2-step:    bottom half delay=0, top half delay=(n_ramp/2)*step_ns (sharpest)
__constant__ int          kWaveSleepShape            = 0;

// Helper: compute the entry-time delay for this CTA given (smid, thr, step_ns,
// n_sm, shape). Called in mma_multistage.h.
__device__ __forceinline__ unsigned int cta_wave_sleep_delay_ns(
    int smid_sm, int thr, unsigned int step_ns, int n_sm, int shape) {
    if (smid_sm < thr) return 0u;
    unsigned int const slot   = static_cast<unsigned int>(smid_sm - thr);
    unsigned int const n_ramp = static_cast<unsigned int>(n_sm - thr);
    if (n_ramp == 0u) return 0u;
    switch (shape) {
        case 1: {                                   // quartile (4 bins)
            unsigned int const q = (slot * 4u) / n_ramp;
            return q * (n_ramp / 4u) * step_ns;
        }
        case 2: {                                   // octile (8 bins, smoother)
            unsigned int const q = (slot * 8u) / n_ramp;
            return q * (n_ramp / 8u) * step_ns;
        }
        case 3: {                                   // 2-step (sharp)
            if (slot < (n_ramp / 2u)) return 0u;
            return (n_ramp / 2u) * step_ns;
        }
        default:                                    // 0 = linear
            return slot * step_ns;
    }
}
#endif
