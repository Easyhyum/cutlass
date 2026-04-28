/*
 * BF16 WMMA GEMM with configurable __nanosleep  (PyTorch Extension)
 *
 * SM120 (Blackwell GeForce / RTX PRO 6000) BF16 matmul 전용.
 *
 * ── 성능 격차 원인 및 해소 ───────────────────────────────────────────────────
 *  [naive] global mem 직접 접근: 한 번 쓴 A[row][k] 를 다른 warp가 또 로드
 *          → 메모리 대역폭 포화, arithmetic intensity 극히 낮음
 *  [opt]  shared mem 128×128 블로킹 + float4(128-bit) 로드 + double buffering
 *          → L1 smem 재사용, 동기화 오버헤드 절반, 메모리 효율 4×
 *
 * Exposed Python functions:
 *   gemm(A, B, sleep_ns=0, sleep_freq=1)
 *       BF16 × BF16 → FP32  naive (1-warp-per-16×16, 교육/참조용)
 *   gemm_opt(A, B, sleep_ns=0, sleep_freq=1)
 *       BF16 × BF16 → FP32  high-perf (128×128 smem, float4, double-buf)
 *   gemm_bf16(A, B, sleep_ns=0, sleep_freq=1)
 *       BF16 × BF16 → BF16  (gemm_opt 결과를 BF16으로 다운캐스트)
 *
 * Build:
 *   python setup_bf16_gemm.py build_ext --inplace
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// ======================== 공통 상수 ========================
static constexpr int WMMA_M    = 16;
static constexpr int WMMA_N    = 16;
static constexpr int WMMA_K    = 16;
static constexpr int WARP_SIZE = 32;

// ======================== Kernel 1: Naive BF16 WMMA ========================
// 특징: 1 warp = 1 (16×16) 출력 타일, global mem 직접 접근
// 용도: 참조/교육용, nanosleep 삽입 효과 확인
// 성능: cuBLAS 대비 ~1/7 (메모리 바운드)
__global__ void wmma_bf16_naive_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    int M, int N, int K,
    unsigned int sleep_ns,
    unsigned int sleep_freq)
{
  const int warp_x   = threadIdx.x / WARP_SIZE;
  const int warp_y   = threadIdx.y;
  const int row_tile = (blockIdx.y * blockDim.y + warp_y) * WMMA_M;
  const int col_tile = (blockIdx.x * blockDim.x / WARP_SIZE + warp_x) * WMMA_N;

  if (row_tile >= M || col_tile >= N) return;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  for (int k = 0; k < K; k += WMMA_K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::load_matrix_sync(a_frag, A + row_tile * K + k, K);
    wmma::load_matrix_sync(b_frag, B + k * N + col_tile, N);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
  }

  // ── nanosleep before store ───────────────────────────────────────────────
  if (sleep_ns > 0u) {
    const unsigned int warp_id =
        (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x / WARP_SIZE * blockDim.y)
        + (unsigned int)(warp_y * (blockDim.x / WARP_SIZE) + warp_x);
    if (warp_id % sleep_freq == 0u)
      __nanosleep(sleep_ns);
  }
  wmma::store_matrix_sync(C + row_tile * N + col_tile, acc, N, wmma::mem_row_major);
}

// ======================== Kernel 2: High-Perf BF16 WMMA ========================
//
// sleep_wmma.cu 의 wmma_gemm_hiperf_kernel 을 BF16 로 포팅.
//
// 최적화 기법:
//   1. 128×128 출력 블록: 각 블록이 대형 타일 담당 → launch overhead 감소
//   2. Shared mem double buffering: 현재 K 타일 MMA 와 다음 K 타일 로드 중첩
//   3. float4 (128-bit) 벡터 로드: 한 명령으로 8 BF16 원소 → 4× 로드 효율
//   4. Warp당 2×4 WMMA 타일 (32×64 출력): 레지스터 재사용 극대화
//   5. 동적 shared mem (cudaFuncAttributeMaxDynamicSharedMemorySize) 로 76KB 확보
//
// 레이아웃:
//   Block: (256 threads = 8 warps), 블록당 128×128 출력
//   Warp:  warpM(0..3) × warpN(0..1) → 32×64 출력 (2×4 WMMA 타일)
//   Smem:  A[2][128][80], B[2][64][144]  (half2→bfloat16 동일 2B)
//
static constexpr int HP_BLOCK_M  = 128;
static constexpr int HP_BLOCK_N  = 128;
static constexpr int HP_BLOCK_K  = 64;
static constexpr int HP_WARPS_M  = 4;
static constexpr int HP_WARPS_N  = 2;
static constexpr int HP_WARPS    = HP_WARPS_M * HP_WARPS_N;   // 8 warps
static constexpr int HP_WARP_M   = HP_BLOCK_M / HP_WARPS_M;   // 32 → 2 WMMA tiles
static constexpr int HP_WARP_N   = HP_BLOCK_N / HP_WARPS_N;   // 64 → 4 WMMA tiles
static constexpr int HP_MMA_M    = HP_WARP_M / WMMA_M;        // 2
static constexpr int HP_MMA_N    = HP_WARP_N / WMMA_N;        // 4
static constexpr int HP_PAD      = 16;  // 16B 정렬을 위한 패딩

static constexpr int HP_SMEM_A_STRIDE   = HP_BLOCK_K + HP_PAD;           // 80
static constexpr int HP_SMEM_B_STRIDE   = HP_BLOCK_N + HP_PAD;           // 144
static constexpr int HP_SMEM_A_BUF_BF16 = HP_BLOCK_M * HP_SMEM_A_STRIDE; // 10240 BF16
static constexpr int HP_SMEM_B_BUF_BF16 = HP_BLOCK_K * HP_SMEM_B_STRIDE; // 9216  BF16
static constexpr size_t HP_SMEM_BYTES   =
    2 * (HP_SMEM_A_BUF_BF16 + HP_SMEM_B_BUF_BF16) * sizeof(__nv_bfloat16); // ~76 KB

__global__ void __launch_bounds__(HP_WARPS * WARP_SIZE)
wmma_bf16_hiperf_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    int M, int N, int K,
    unsigned int sleep_ns,
    unsigned int sleep_freq)
{
  // ── 동적 shared memory ─────────────────────────────────────────────────
  extern __shared__ char _smem_raw[];
  __nv_bfloat16* smemA = reinterpret_cast<__nv_bfloat16*>(_smem_raw);
  __nv_bfloat16* smemB = smemA + 2 * HP_SMEM_A_BUF_BF16;

  const int tid      = threadIdx.x;
  const int warpId   = tid / WARP_SIZE;
  const int warpM    = warpId / HP_WARPS_N;  // 0..3
  const int warpN    = warpId % HP_WARPS_N;  // 0..1

  const int blockRow = blockIdx.y * HP_BLOCK_M;
  const int blockCol = blockIdx.x * HP_BLOCK_N;
  const int numThr   = HP_WARPS * WARP_SIZE;  // 256

  // ── accumulator 초기화 ──────────────────────────────────────────────────
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
      acc[HP_MMA_M][HP_MMA_N];
  #pragma unroll
  for (int mi = 0; mi < HP_MMA_M; mi++)
    #pragma unroll
    for (int ni = 0; ni < HP_MMA_N; ni++)
      wmma::fill_fragment(acc[mi][ni], 0.0f);

  // ── float4 로드 헬퍼 ────────────────────────────────────────────────────
  // float4 = 16 bytes = 8 × __nv_bfloat16 (2B each)
  auto load_A = [&](int kStart, int buf) __attribute__((always_inline)) {
    constexpr int TOTAL_F4 = HP_BLOCK_M * HP_BLOCK_K / 8;  // 1024
    __nv_bfloat16* dst = smemA + buf * HP_SMEM_A_BUF_BF16;
    #pragma unroll 4
    for (int i = tid; i < TOTAL_F4; i += numThr) {
      const int row = i / (HP_BLOCK_K / 8);
      const int col = (i % (HP_BLOCK_K / 8)) * 8;
      const int gr  = blockRow + row;
      const int gc  = kStart   + col;
      float4 val = {};
      if (gr < M && gc + 7 < K) {
        val = *reinterpret_cast<const float4*>(A + gr * K + gc);
      } else if (gr < M) {
        const __nv_bfloat16* src = A + gr * K + gc;
        __nv_bfloat16* tmp = reinterpret_cast<__nv_bfloat16*>(&val);
        #pragma unroll
        for (int j = 0; j < 8 && gc + j < K; j++) tmp[j] = src[j];
      }
      *reinterpret_cast<float4*>(dst + row * HP_SMEM_A_STRIDE + col) = val;
    }
  };

  auto load_B = [&](int kStart, int buf) __attribute__((always_inline)) {
    constexpr int TOTAL_F4 = HP_BLOCK_K * HP_BLOCK_N / 8;  // 1024
    __nv_bfloat16* dst = smemB + buf * HP_SMEM_B_BUF_BF16;
    #pragma unroll 4
    for (int i = tid; i < TOTAL_F4; i += numThr) {
      const int row = i / (HP_BLOCK_N / 8);
      const int col = (i % (HP_BLOCK_N / 8)) * 8;
      const int gr  = kStart   + row;
      const int gc  = blockCol + col;
      float4 val = {};
      if (gr < K && gc + 7 < N) {
        val = *reinterpret_cast<const float4*>(B + gr * N + gc);
      } else if (gr < K) {
        const __nv_bfloat16* src = B + gr * N + gc;
        __nv_bfloat16* tmp = reinterpret_cast<__nv_bfloat16*>(&val);
        #pragma unroll
        for (int j = 0; j < 8 && gc + j < N; j++) tmp[j] = src[j];
      }
      *reinterpret_cast<float4*>(dst + row * HP_SMEM_B_STRIDE + col) = val;
    }
  };

  // ── 첫 번째 K 타일 프리페치 ────────────────────────────────────────────
  load_A(0, 0);
  load_B(0, 0);
  __syncthreads();

  // ── 메인 K 루프: double buffering ─────────────────────────────────────
  for (int kStart = 0; kStart < K; kStart += HP_BLOCK_K) {
    const int curBuf = (kStart / HP_BLOCK_K) & 1;
    const int nxtBuf = curBuf ^ 1;

    // 다음 K 타일 소프트웨어 프리페치 (현재 타일 MMA 와 중첩)
    const int nextK = kStart + HP_BLOCK_K;
    if (nextK < K) {
      load_A(nextK, nxtBuf);
      load_B(nextK, nxtBuf);
    }

    // ── 현재 버퍼로 WMMA 연산 ─────────────────────────────────────────
    const __nv_bfloat16* curA = smemA + curBuf * HP_SMEM_A_BUF_BF16;
    const __nv_bfloat16* curB = smemB + curBuf * HP_SMEM_B_BUF_BF16;

    #pragma unroll
    for (int kk = 0; kk < HP_BLOCK_K; kk += WMMA_K) {
      #pragma unroll
      for (int mi = 0; mi < HP_MMA_M; mi++) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> a_frag;
        wmma::load_matrix_sync(
            a_frag,
            curA + (warpM * HP_WARP_M + mi * WMMA_M) * HP_SMEM_A_STRIDE + kk,
            HP_SMEM_A_STRIDE);

        #pragma unroll
        for (int ni = 0; ni < HP_MMA_N; ni++) {
          wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                         __nv_bfloat16, wmma::row_major> b_frag;
          wmma::load_matrix_sync(
              b_frag,
              curB + kk * HP_SMEM_B_STRIDE + warpN * HP_WARP_N + ni * WMMA_N,
              HP_SMEM_B_STRIDE);

          wmma::mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
        }
      }
    }
    __syncthreads();
  }

  // ── nanosleep: wmma::store_matrix_sync 직전 삽입 ────────────────────────
  if (sleep_ns > 0u) {
    const unsigned int warp_id =
        (blockIdx.y * gridDim.x + blockIdx.x) * HP_WARPS
        + (unsigned int)warpId;
    if (warp_id % sleep_freq == 0u)
      __nanosleep(sleep_ns);
  }

  // ── 결과 저장 ──────────────────────────────────────────────────────────
  #pragma unroll
  for (int mi = 0; mi < HP_MMA_M; mi++) {
    #pragma unroll
    for (int ni = 0; ni < HP_MMA_N; ni++) {
      const int outRow = blockRow + warpM * HP_WARP_M + mi * WMMA_M;
      const int outCol = blockCol + warpN * HP_WARP_N + ni * WMMA_N;
      if (outRow < M && outCol < N)
        wmma::store_matrix_sync(C + outRow * N + outCol,
                                acc[mi][ni], N, wmma::mem_row_major);
    }
  }
}

// ======================== 유틸리티 ========================

static void check_bf16_2d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(),                   name, " must be CUDA tensor");
  TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be BF16");
  TORCH_CHECK(t.dim() == 2,                  name, " must be 2-D");
  TORCH_CHECK(t.is_contiguous(),             name, " must be contiguous");
}

__global__ void fp32_to_bf16_kernel(const float* __restrict__ src,
                                    __nv_bfloat16* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// ======================== C++ 래퍼 ========================

// ── gemm (naive): BF16 × BF16 → FP32 ─────────────────────────────────────
torch::Tensor wmma_gemm_bf16_naive(
    torch::Tensor A, torch::Tensor B,
    int sleep_ns = 0, int sleep_freq = 1)
{
  check_bf16_2d(A, "A"); check_bf16_2d(B, "B");
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  TORCH_CHECK(B.size(0) == K, "K mismatch");
  TORCH_CHECK(M%16==0 && N%16==0 && K%16==0, "M/N/K must be multiples of 16");
  TORCH_CHECK(sleep_freq >= 1, "sleep_freq >= 1");

  auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

  constexpr int WX = 4, WY = 4;
  dim3 block(WX * WARP_SIZE, WY);
  dim3 grid((N / WMMA_N + WX - 1) / WX, (M / WMMA_M + WY - 1) / WY);

  wmma_bf16_naive_kernel<<<grid, block>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
      C.data_ptr<float>(),
      M, N, K, (unsigned)sleep_ns, (unsigned)sleep_freq);
  return C;
}

// ── gemm_opt (high-perf): BF16 × BF16 → FP32 ─────────────────────────────
torch::Tensor wmma_gemm_bf16_opt(
    torch::Tensor A, torch::Tensor B,
    int sleep_ns = 0, int sleep_freq = 1)
{
  check_bf16_2d(A, "A"); check_bf16_2d(B, "B");
  const int M = A.size(0), K = A.size(1), N = B.size(1);
  TORCH_CHECK(B.size(0) == K, "K mismatch");
  TORCH_CHECK(M % HP_BLOCK_M == 0, "M must be multiple of ", HP_BLOCK_M);
  TORCH_CHECK(N % HP_BLOCK_N == 0, "N must be multiple of ", HP_BLOCK_N);
  TORCH_CHECK(K % HP_BLOCK_K == 0, "K must be multiple of ", HP_BLOCK_K);
  TORCH_CHECK(sleep_freq >= 1, "sleep_freq >= 1");

  auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(A.device()));

  dim3 block(HP_WARPS * WARP_SIZE);  // 256 threads, 1-D
  dim3 grid(N / HP_BLOCK_N, M / HP_BLOCK_M);

  // 동적 smem 크기 확장 (96 KB)
  auto fn = wmma_bf16_hiperf_kernel;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)HP_SMEM_BYTES);

  fn<<<grid, block, HP_SMEM_BYTES>>>(
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()),
      C.data_ptr<float>(),
      M, N, K, (unsigned)sleep_ns, (unsigned)sleep_freq);
  return C;
}

// ── gemm_bf16: BF16 × BF16 → BF16 (opt 사용, FP32 acc 후 다운캐스트) ──────
torch::Tensor wmma_gemm_bf16_out(
    torch::Tensor A, torch::Tensor B,
    int sleep_ns = 0, int sleep_freq = 1)
{
  auto C_fp32 = wmma_gemm_bf16_opt(A, B, sleep_ns, sleep_freq);
  const int n = C_fp32.numel();
  auto C_bf16 = torch::empty_like(C_fp32, torch::dtype(torch::kBFloat16));
  fp32_to_bf16_kernel<<<(n+255)/256, 256>>>(
      C_fp32.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(C_bf16.data_ptr<at::BFloat16>()), n);
  return C_bf16;
}

// ======================== PyBind11 ========================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "BF16 WMMA GEMM kernels with nanosleep for Blackwell SM120\n\n"
            "  gemm     : naive (global mem, 교육용)\n"
            "  gemm_opt : high-perf (128×128 smem, float4, double-buf)\n"
            "  gemm_bf16: gemm_opt + FP32→BF16 다운캐스트 출력";

  m.def("gemm",
        &wmma_gemm_bf16_naive,
        "Naive BF16×BF16→FP32 WMMA (nanosleep before store)",
        py::arg("A"), py::arg("B"),
        py::arg("sleep_ns") = 0, py::arg("sleep_freq") = 1);

  m.def("gemm_opt",
        &wmma_gemm_bf16_opt,
        "High-perf BF16×BF16→FP32 WMMA (128×128 smem, float4, double-buf, nanosleep before store)",
        py::arg("A"), py::arg("B"),
        py::arg("sleep_ns") = 0, py::arg("sleep_freq") = 1);

  m.def("gemm_bf16",
        &wmma_gemm_bf16_out,
        "High-perf BF16×BF16→BF16 WMMA (FP32 acc, nanosleep before store)",
        py::arg("A"), py::arg("B"),
        py::arg("sleep_ns") = 0, py::arg("sleep_freq") = 1);
}
