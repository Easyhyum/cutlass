/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

// ── CUTLASS_SLEEP_ENABLED : runtime symbol 정의용 ──────────────────────────
// ── CUTLASS_METHOD_A_BM_ENABLED : v7 ring-rotating warp mis-aligned bubble ──
//
//   v7 디자인:
//     hook: mac_loop_iter()의 warp_mma_k 루프 내부, HMMA 직후.
//     Gate: warp_mma_k == ((warp_in_cta + time_phase) & (kWGI-1))
//     time_phase = gemm_k_iterations & (kWGI-1)  → 매 outer iter 회전
//     → 같은 warp의 sleep 위치가 outer iter 마다 회전 (동일 지점 반복 없음)
//     → 같은 시점에 warp 절반만 sleep → 다른 절반은 HMMA 계속 → mis-align 유지
//     → mac_loop_iter 내부에 __syncthreads 없음 (gmem_wait 만 sync) → drift 보존
//     → 모든 warp의 총 sleep 시간 동일 → 동시 완료
//
//   왜 별도 build flag?
//     v6 시도: CUTLASS_SLEEP_ENABLED 단일 flag로 인라인. baseline -15% regression
//     원인 — __constant__ memory gated 조건문이 inner unrolled loop의 컴파일러
//     최적화 (register alloc, unroll) 방해.
//     해결: CUTLASS_METHOD_A_BM_ENABLED 가 없으면 inner-loop 코드가 아예 존재
//     하지 않음 → baseline 100% 보존.
//     setup_bf16_sm80*.py 에서 두 binary 빌드 권장: 하나는 baseline (이 flag
//     없음), 하나는 Method A (flag 있음).
//
//   파라미터:
//     kCutlassSleepStaggerNs = sleep duration (ns); 0 = no sleep (Method A off)
//     kCutlassSleepStaggerMod = legacy (현재 v7에서는 미사용)
//
//   이전 버전 백업:
//     v3_pipeline_aware_with_sleep_residue_20260521/  (sleep residue)
//     v4_pipeline_aware_only_20260521/                (clock64 spin)
//     v5_warp_nanosleep_20260521/                     (post-gmem_wait, all-warps)
//     v6_warp_mis_aligned_20260521/                   (no temporal rotation)
#ifdef CUTLASS_SLEEP_ENABLED
#include "cutlass/cutlass_sleep_globals.cuh"
#endif

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/threadblock/mma_base.h"

#if defined(CUTLASS_MMA_MAC_LOOP_TIMING)
#include <stdio.h>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Used for partial specialization
    typename Enable = bool>
class MmaMultistage : 
  public MmaBase<Shape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = MmaBase<Shape_, Policy_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    // Optional staged-accumulation (e.g., tf32x3 kernels) for improved numerical
    // accuracy, where each mainloop iteration first accumulates into a temporary
    // set of freshly-cleared accumulators, which are subsequently added to the
    // final accumulator set.
    static bool const kStagedAccumulation = arch::detail::UseStagedAccumulation<Operator>::value;
  };

 private:


  // Structure encapsulating pipeline state live from one iteration to the next
  struct PipeState {

    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

    /// Temporary accumulator to facilitate staged-accumulation
    FragmentC tmp_accum_;

    /// Pair of A fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentA warp_loaded_frag_A_[2];
    WarpTransformedFragmentA warp_transformed_frag_A_[2];

    /// Pair of B fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentB warp_loaded_frag_B_[2];
    WarpTransformedFragmentB warp_transformed_frag_B_[2];
  };


 private:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma_;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  /// Shared memory write stage index
  int smem_write_stage_idx_;

  /// Shared memory read stage index
  int smem_read_stage_idx_;


public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
      smem_write_stage_idx_(0),
      smem_read_stage_idx_(0)
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  /// Advance shared memory read-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_read_stage()
  {
    ++smem_read_stage_idx_;

    if (smem_read_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      this->warp_tile_iterator_A_.add_tile_offset({0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset({-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      smem_read_stage_idx_ = 0;
    }
  }

  /// Advance global memory read-iterators and shared memory write-iterators to the stage
  CUTLASS_DEVICE
  void advance_smem_write_stage(
    IteratorA &iterator_A,
    IteratorB &iterator_B)
  {
    // Advance global iterators
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});

    // Advance shared iterators
    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_.add_tile_offset({1, 0});

    // Increment shared memory write stage index
    ++smem_write_stage_idx_;

    if (smem_write_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
      smem_write_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          }

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          }

          ++iterator_B;
        }
        ++this->smem_iterator_B_;
      }
    }
  }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA &iterator_A,      ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,      ///< [in|out] iterator over B operand in global memory
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {

      // Disable global fetching if done with global fetch iterations
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorA::Element>::value *
              IteratorA::ThreadMap::kElementsPerAccess /
              IteratorA::kAccessesPerVector / 8;

          int src_bytes = (iterator_A.valid() ? kSrcBytes : 0);

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v, iterator_A.get(), iterator_A.valid());

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }

      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB::Element>::value *
              IteratorB::ThreadMap::kElementsPerAccess /
              IteratorB::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B.get(), iterator_B.valid());

          ++iterator_B;
        }

        ++this->smem_iterator_B_;
      }

      // Move to the next write stage
      advance_smem_write_stage(iterator_A, iterator_B);

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // Optionally clear the remaining stages of SMEM. This is a functional requirement for
    // some kernels so that all accumulator elements outside the GEMM footprint are zero.
    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);
      typename IteratorA::AccessType zero_A;

      zero_A.clear();
      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }
  }


  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait()
  {
    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();
  }


  /// Perform a threadblock mainloop iteration of matrix multiply-accumulate
  CUTLASS_DEVICE
  void mac_loop_iter(
    PipeState &pipe_state,          ///< [in|out] loop-carried pipeline state
    FragmentC &accum,               ///< [in|out] destination accumulator tile
    IteratorA &iterator_A,          ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B,          ///< [in|out] iterator over B operand in global memory
    int &gemm_k_iterations)
  {
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

      // Load the next warp-tile's A fragment from shared memory
      this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      // Load the next warp-tile's B fragment from shared memory
      this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_B_;

      // Except for the first warp-tile, all warp-tiles convert their incoming shared memory fragments as necessary
      if (warp_mma_k > 0) {
        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
      }

      // Execute the current warp-tile of MMA operations
      if (Detail::kStagedAccumulation) {
        warp_mma_(
          pipe_state.tmp_accum_,
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.tmp_accum_
        );

        if (warp_mma_k == 0) {
          plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(
          accum,
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          accum
        );
      }

#if defined(CUTLASS_METHOD_A_BM_ENABLED)
      // ── Method A v7 : ring-rotating warp mis-aligned bubble sync ───────────
      //  변경 사유 (v6 → v7):
      //    1) v6의 inner-loop conditional 블록 자체가 baseline -15% regression.
      //       해결: 별도 build flag (CUTLASS_METHOD_A_BM_ENABLED) 로 가드 →
       //      flag 없으면 코드 완전 부재, baseline 그대로.
      //    2) 사용자 지적: "하나의 warp가 매번 동일지점에서 sleep 되면 안 됨".
      //       해결: time_phase (gemm_k_iterations 기반)로 매 outer iter 마다
      //       각 warp의 sleep 위치가 회전. 결과적으로 모든 warp가 같은 총
      //       시간 sleep 하므로 동시 완료 보장.
      //
      //  Gate:  warp_mma_k == ((warp_in_cta + time_phase) & (kWGI - 1))
      //  Effect (kWGI=2, 4 warps):
      //    outer iter t  (time_phase=0): warps 0,2 sleep at slot 0; warps 1,3 sleep at slot 1
      //    outer iter t+1(time_phase=1): warps 0,2 sleep at slot 1; warps 1,3 sleep at slot 0
      //    → warp 0 의 sleep 위치가 회전 (slot 0 → 1 → 0 → 1 ...)
      //    → 매 시점 HMMA 발사 warp 수 = N/2 (peak 절반)
      //    → 매 warp 총 sleep 시간 동일 → 동시 완료
      if (kCutlassSleepStaggerNs > 0u) {
        unsigned int const tid_v7 =
            threadIdx.z * blockDim.y * blockDim.x +
            threadIdx.y * blockDim.x +
            threadIdx.x;
        unsigned int const warp_in_cta_v7 = tid_v7 >> 5u;
        unsigned int const kWGImask = Base::kWarpGemmIterations - 1u;
        unsigned int const time_phase =
            static_cast<unsigned int>(gemm_k_iterations) & kWGImask;
        unsigned int const my_slot =
            (warp_in_cta_v7 + time_phase) & kWGImask;
        if (static_cast<unsigned int>(warp_mma_k) == my_slot) {
          __nanosleep(kCutlassSleepStaggerNs);
        }
      }
#endif

      // Except for the last warp-tile, all warp-tiles issue their share of
      // global->shared fragment copies
      if (warp_mma_k < Base::kWarpGemmIterations - 1) {

        int group_start_iteration_A, group_start_iteration_B;
        group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(
            iterator_A,
            iterator_B,
            group_start_iteration_A,
            group_start_iteration_B);
      }

      // The second-to-last warp-tile also:
      //   - performs the last warp-tile's share of global->shared fragment copies
      //   - moves to the next global fetch stage
      if (warp_mma_k + 2 == Base::kWarpGemmIterations) {

        // Performs the last warp-tile's share of global->shared fragment copies
        int group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
        int group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(
          iterator_A,
          iterator_B,
          group_start_iteration_A,
          group_start_iteration_B);

        // Inserts a memory fence between stages of cp.async instructions.
        cutlass::arch::cp_async_fence();

        // Wait until we have at least one completed global fetch stage
        gmem_wait();

        // (v6: post-gmem_wait sleep removed. Sleep moved INTO warp_mma_k loop
        //  with warp_in_cta-gated position for true warp mis-alignment.)

        // Move to the next global fetch stage
        advance_smem_write_stage(iterator_A, iterator_B);
        advance_smem_read_stage();

        // Disable global fetching when done with global fetch iterations
        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B.clear_mask(gemm_k_iterations == 0);
      }
      // The last warp-tile also converts the shared memory fragments used by
      // the first warp-tile of the next iteration, if necessary (so we can
      // immediately start issuing MMA instructions at the top of the loop )
      if (warp_mma_k + 1 == Base::kWarpGemmIterations) {

        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      }

    }
  }


  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
      int gemm_k_iterations,        ///< number of threadblock mainloop iterations
      FragmentC &accum,             ///< [in|out] accumulator tile
      IteratorA &iterator_A,        ///< [in|out] iterator over A operand in global memory
      IteratorB &iterator_B)        ///< [in|out] iterator over B operand in global memory
  {
    PipeState pipe_state;
    // (v4: tid/warp_in_cta/smid extraction moved into mac_loop_iter where the
    //  pipeline-aware spin needs smid. Removing dead variable here so that any
    //  register pressure or compiler scheduling artifact from holding smid in
    //  a register across the whole mainloop is eliminated.)
    // Disable global fetching if done with global fetch iterations
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    // Load first warp-tile's A fragment from shared memory
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;

    // Load first warp-tile's B fragment from shared memory
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++this->warp_tile_iterator_B_;

    // Transform, if necessary, the first warp-tile's shared memory fragments
    warp_mma_.transform(
      pipe_state.warp_transformed_frag_A_[0],
      pipe_state.warp_transformed_frag_B_[0],
      pipe_state.warp_loaded_frag_A_[0],
      pipe_state.warp_loaded_frag_B_[0]);

    if (Detail::kStagedAccumulation) {
      pipe_state.tmp_accum_.clear();
    }

    // Mainloop
#if defined(CUTLASS_METHOD_A_RAMP_ENABLED)
    // v8 soft-launch ramp: kernel 진입 직후 첫 kCutlassRampIters outer iter
    // 동안 __nanosleep 시간을 큰 값에서 0 으로 선형 감소.
    // 효과: kernel 시작 시 SMs가 풀 HMMA 발사를 즉시 안 함 → 평균 power
    // 가 step 이 아닌 ramp 로 상승 → DVFS 가 gradient 따라 clock 을 천천히
    // 낮춤 → 급격한 throttle drop 회피, 안정 clock 정착.
    // outer-loop scope 이므로 warp_mma_k inner-unrolled loop 영향 없음 →
    // baseline regression 없음.
    unsigned int outer_iter_v8 = 0u;
#endif
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
#if defined(CUTLASS_METHOD_A_RAMP_ENABLED)
      if (kCutlassRampPeakNs > 0u && outer_iter_v8 < kCutlassRampIters) {
        // Linear ramp: iter 0 → peak,  iter ramp_iters-1 → small.
        unsigned int const remaining = kCutlassRampIters - outer_iter_v8;
        unsigned int const ramp_ns =
            (kCutlassRampPeakNs * remaining) / kCutlassRampIters;
        if (ramp_ns > 0u) {
          __nanosleep(ramp_ns);
        }
      }
      ++outer_iter_v8;
#endif
      mac_loop_iter(
        pipe_state,
        accum,
        iterator_A,
        iterator_B,
        gemm_k_iterations);
    }

    if (Detail::kStagedAccumulation) {
      plus<FragmentC> plus_accum;
      accum = plus_accum(accum, pipe_state.tmp_accum_);
    }

    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

  }


  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    // Catch-up the smem-read iterator to the smem-write iterator (so this class can be reused for another tile's prologue)

    // First, increment remaining warp tiles to get to the next full stage.  (Ideally we would
    // just decrement one tile, but not all iterators implement --() decrement.)
    #pragma unroll
    for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
    {
      this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k);

      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;
    }
    smem_read_stage_idx_++;

    // Then wrap back two full stages (one for the tile advancing we just did, and one to catch the write iterators)
    static const int kStageIters = Policy::kPartitionsK * Base::kWarpGemmIterations;
    if (smem_read_stage_idx_ > 1)
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, (-2 * kStageIters)});
      this->warp_tile_iterator_B_.add_tile_offset({(-2 * kStageIters), 0});
    }
    else
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, ((Base::kStages - 2) * kStageIters)});
      this->warp_tile_iterator_B_.add_tile_offset({((Base::kStages - 2) * kStageIters), 0});
    }
    smem_read_stage_idx_ = smem_write_stage_idx_;
  }


  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      ///< initial value of accumulator
      FragmentC const &src_accum) {

    // (v4: prologue stagger removed — wash out in mainloop made it ineffective.
    //  pipeline-aware spin in mac_loop_iter handles persistent SM stagger.)

    // Prologue (start fetching iterations of global fragments into shared memory)
    prologue(iterator_A, iterator_B, gemm_k_iterations);

    // Wait until we have at least one completed global fetch stage
    gmem_wait();

    // Initialize destination accumulators with source accumulators
    accum = src_accum;

    // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
  }

  // Expose pipeline state via alias without changing its original access level
  using PublicPipeState = PipeState;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

