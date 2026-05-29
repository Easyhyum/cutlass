#pragma once
// ── CTA dispatch probe globals ─────────────────────────────────────────────
//
// Activated by build flag  -DCUTLASS_CTA_PROBE_ENABLED.
//
// When enabled, MmaMultistage::operator() (mma_multistage.h) records the
// physical SM id, the globaltimer at entry, and blockIdx.{x,y,z} for each
// CTA, then RETURNS immediately — the actual GEMM mainloop is skipped.
//
// The grid shape is preserved (CUTLASS launches the same number of CTAs
// it would for a real GEMM), so the hardware dispatcher's CTA→SM mapping
// can be observed without burning the real mainloop work.
//
// Host side (probe_streamk.cu) sets the four output pointers + the bound:
//   cudaMemcpyToSymbol(kCtaProbeOutSmid,  &dev_ptr, sizeof(void*));
//   ...
//   cudaMemcpyToSymbol(kCtaProbeMaxCtas, &n_ctas, sizeof(int));
//
#ifdef CUTLASS_CTA_PROBE_ENABLED
#include <cstdint>

__device__ int*                g_cta_probe_smid_out  = nullptr;
__device__ unsigned long long* g_cta_probe_start_out = nullptr;
__device__ unsigned long long* g_cta_probe_end_out   = nullptr;
__device__ int*                g_cta_probe_bx_out    = nullptr;
__device__ int*                g_cta_probe_by_out    = nullptr;
__device__ int*                g_cta_probe_bz_out    = nullptr;
__constant__ int               kCtaProbeMaxCtas      = 0;
#endif
