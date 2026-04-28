/***************************************************************************************************
 * BF16 GEMM – NVIDIA Blackwell SM120 (RTX PRO 6000 / RTX 50 GeForce)
 *
 *   D = alpha * A * B + beta * C
 *
 * ── SM120 BF16 주의사항 ─────────────────────────────────────────────────────
 *  SM120의 새로운 tcgen05 Tensor Core는 FP4/FP6/FP8(narrow precision)만 지원.
 *  BF16은 cuBLAS를 통해 최적의 레거시 HMMA 경로가 자동 선택된다.
 *
 * ── 구현 방식 ────────────────────────────────────────────────────────────────
 *  1. cuBLAS BF16 GEMM    : cublasGemmEx  (CUBLAS_COMPUTE_32F_FAST_16BF)
 *  2. CUDA WMMA BF16 GEMM : wmma API 직접 사용 (교육용 / 커스텀 커널 기반)
 *
 * ── 빌드 ─────────────────────────────────────────────────────────────────────
 *  nvcc -O3 -arch=sm_120 \
 *       -I/workspace/include -I/workspace/tools/util/include \
 *       blackwell_bf16_gemm_sm120.cu -lcublas -o blackwell_bf16_gemm_sm120
 *
 * ── 실행 ─────────────────────────────────────────────────────────────────────
 *  ./blackwell_bf16_gemm_sm120 --m=8192 --n=8192 --k=8192 --iterations=50
 *
 **************************************************************************************************/

#include <iostream>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cublas_v2.h>

#include "cutlass/util/command_line.h"

using namespace nvcuda;

/////////////////////////////////////////////////////////////////////////////////////////////////
// 매크로 헬퍼
/////////////////////////////////////////////////////////////////////////////////////////////////

#define CUDA_CHECK(x)                                                         \
  do {                                                                        \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,             \
              cudaGetErrorString(_e)); std::exit(1); }                        \
  } while(0)

#define CUBLAS_CHECK(x)                                                       \
  do {                                                                        \
    cublasStatus_t _s = (x);                                                  \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                        \
      fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,(int)_s);  \
      std::exit(1); }                                                         \
  } while(0)

/////////////////////////////////////////////////////////////////////////////////////////////////
// 커맨드라인 옵션
/////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool  help       = false;
  int   m          = 8192;
  int   n          = 8192;
  int   k          = 8192;
  float alpha      = 1.0f;
  float beta       = 0.0f;
  int   iterations = 50;

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    cmd.get_cmd_line_argument("m",          m,          8192);
    cmd.get_cmd_line_argument("n",          n,          8192);
    cmd.get_cmd_line_argument("k",          k,          8192);
    cmd.get_cmd_line_argument("alpha",      alpha,      1.0f);
    cmd.get_cmd_line_argument("beta",       beta,       0.0f);
    cmd.get_cmd_line_argument("iterations", iterations, 50);
  }

  void print_usage() const {
    std::cout
      << "\nBF16 GEMM for Blackwell SM120\n\n"
      << "  --m=<int>          M 차원         (기본: 8192)\n"
      << "  --n=<int>          N 차원         (기본: 8192)\n"
      << "  --k=<int>          K 차원         (기본: 8192)\n"
      << "  --alpha=<float>    알파 스케일    (기본: 1.0)\n"
      << "  --beta=<float>     베타 스케일    (기본: 0.0)\n"
      << "  --iterations=<int> 반복 횟수      (기본: 50)\n\n";
  }

  double tflops(double ms) const {
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// 방법 1 : cuBLAS BF16 GEMM
//   - SM120에서 cuBLAS가 최적 커널 자동 선택
//   - CUBLAS_COMPUTE_32F_FAST_16BF : FP32 누산 + BF16 tensor core 가속
/////////////////////////////////////////////////////////////////////////////////////////////////

void run_cublas_bf16_gemm(const Options& opt)
{
  const int M = opt.m, N = opt.n, K = opt.k;
  const size_t bytes_A = (size_t)M * K * sizeof(__nv_bfloat16);
  const size_t bytes_B = (size_t)K * N * sizeof(__nv_bfloat16);
  const size_t bytes_C = (size_t)M * N * sizeof(__nv_bfloat16);

  // ── 호스트 메모리 초기화 ───────────────────────────────────────────────────
  __nv_bfloat16 *h_A = new __nv_bfloat16[M * K];
  __nv_bfloat16 *h_B = new __nv_bfloat16[K * N];
  __nv_bfloat16 *h_C = new __nv_bfloat16[M * N];
  for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16((float)(rand() % 10 - 5) * 0.1f);
  for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16((float)(rand() % 10 - 5) * 0.1f);
  for (int i = 0; i < M * N; ++i) h_C[i] = __float2bfloat16(0.0f);

  // ── 디바이스 메모리 ──────────────────────────────────────────────────────────
  __nv_bfloat16 *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
  CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

  // ── cuBLAS 핸들 ──────────────────────────────────────────────────────────────
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // ── 스케일 (float) ────────────────────────────────────────────────────────────
  const float alpha_f = opt.alpha, beta_f = opt.beta;

  // ── Warmup ───────────────────────────────────────────────────────────────────
  CUBLAS_CHECK(cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,  // A[MxK] * B[KxN] (column-major notation: No-trans)
    N, M, K,                   // cuBLAS는 column-major: (N, M, K)
    &alpha_f,
    d_B, CUDA_R_16BF, N,       // B 가 첫 번째 인자 (column-major 규칙)
    d_A, CUDA_R_16BF, K,
    &beta_f,
    d_C, CUDA_R_16BF, N,
    CUBLAS_COMPUTE_32F_FAST_16BF,       // FP32 누산 + BF16 tensor core 가속
    CUBLAS_GEMM_DEFAULT_TENSOR_OP       // 자동 최적 알고리즘
  ));
  CUDA_CHECK(cudaDeviceSynchronize());

  // ── 성능 측정 ────────────────────────────────────────────────────────────────
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  for (int i = 0; i < opt.iterations; ++i) {
    CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K,
      &alpha_f,
      d_B, CUDA_R_16BF, N,
      d_A, CUDA_R_16BF, K,
      &beta_f,
      d_C, CUDA_R_16BF, N,
      CUBLAS_COMPUTE_32F_FAST_16BF,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
  }
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float elapsed_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
  double avg_ms = elapsed_ms / opt.iterations;

  std::cout << "\n[cuBLAS BF16 GEMM]\n";
  std::cout << "  문제 크기   : " << M << " x " << N << " x " << K << "\n";
  std::cout << "  평균 시간   : " << avg_ms        << " ms\n";
  std::cout << "  성능        : " << opt.tflops(avg_ms) << " TFLOPS\n";

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  delete[] h_A; delete[] h_B; delete[] h_C;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// 방법 2 : CUDA WMMA BF16 GEMM 커스텀 커널
//   - nvcuda::wmma API를 직접 사용 (16x16x16 타일)
//   - 교육용 / 커스텀 커널 작성 참고용
//   - cuBLAS 대비 낮은 성능이나 완전한 제어 가능
/////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// ── 커널: 각 warp가 WMMA_M×WMMA_N 출력 타일 담당 ──────────────────────────
__global__ void wmma_bf16_gemm_kernel(
    const __nv_bfloat16* __restrict__ A,   // [M x K], RowMajor
    const __nv_bfloat16* __restrict__ B,   // [K x N], RowMajor
    float*               __restrict__ C,   // [M x N], RowMajor (FP32 출력)
    int M, int N, int K,
    float alpha, float beta,
    unsigned int sleep_ns   = 0,           // store 직전 nanosleep 시간 (ns)
    unsigned int sleep_freq = 1)           // sleep_freq 타일마다 1회 sleep
{
  // warp 2D 위치
  const int warp_row = (blockIdx.y * blockDim.y + threadIdx.y);
  const int warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

  const int row_tile = warp_row * WMMA_M;
  const int col_tile = warp_col * WMMA_N;

  if (row_tile >= M || col_tile >= N) return;

  // accumulator fragment (FP32)
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  // K 방향 루프
  for (int k = 0; k < K; k += WMMA_K) {
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   __nv_bfloat16, wmma::row_major> b_frag;

    // A 타일 로드: A[row_tile : row_tile+16][k : k+16]
    wmma::load_matrix_sync(a_frag, A + row_tile * K + k, K);
    // B 타일 로드: B[k : k+16][col_tile : col_tile+16]
    wmma::load_matrix_sync(b_frag, B + k * N + col_tile, N);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // epilogue: D = alpha * acc + beta * C (FP32 출력)
  if (beta != 0.0f) {
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::load_matrix_sync(c_frag, C + row_tile * N + col_tile, N, wmma::mem_row_major);
    for (int i = 0; i < acc_frag.num_elements; ++i)
      acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
  } else {
    for (int i = 0; i < acc_frag.num_elements; ++i)
      acc_frag.x[i] *= alpha;
  }

  // ── nanosleep: sleep_freq 타일마다 1회 sleep (MMA 부하 사이클 제어) ──────
  // warp 전역 ID를 기준으로 sleep_freq 간격마다 __nanosleep 삽입
  if (sleep_ns > 0) {
    // const int warp_id = blockIdx.x * (blockDim.x / 32) * blockDim.y
    //                   + blockIdx.y * blockDim.y
    //                   + threadIdx.y
    //                   + threadIdx.x / 32;
    // if ((unsigned int)warp_id % sleep_freq == 0) {
      __nanosleep(sleep_ns);
    // }
  }

  wmma::store_matrix_sync(C + row_tile * N + col_tile, acc_frag, N, wmma::mem_row_major);
}

void run_wmma_bf16_gemm(const Options& opt)
{
  const int M = opt.m, N = opt.n, K = opt.k;

  // ── 호스트 초기화 ─────────────────────────────────────────────────────────
  __nv_bfloat16 *h_A = new __nv_bfloat16[M * K];
  __nv_bfloat16 *h_B = new __nv_bfloat16[K * N];
  float         *h_C = new float[M * N]();
  for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16((float)(rand()%10-5)*0.1f);
  for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16((float)(rand()%10-5)*0.1f);

  // ── 디바이스 메모리 ───────────────────────────────────────────────────────
  __nv_bfloat16 *d_A, *d_B;
  float         *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, (size_t)M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, (size_t)K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)));

  // ── 그리드 / 블록 설정 ────────────────────────────────────────────────────
  // 블록 = (128, 4) threads → 16 warps/block → 16 개의 WMMA_N 타일
  // Y=4 warps, X=4 warps (각 warp는 32 threads)
  const int WARPS_X = 4, WARPS_Y = 4;
  dim3 block(WARPS_X * 32, WARPS_Y);
  dim3 grid(
    (N + WMMA_N * WARPS_X - 1) / (WMMA_N * WARPS_X),
    (M + WMMA_M * WARPS_Y - 1) / (WMMA_M * WARPS_Y)
  );

  // ── Warmup ───────────────────────────────────────────────────────────────
  wmma_bf16_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K,
                                         opt.alpha, opt.beta, 0u, 1u);
  CUDA_CHECK(cudaDeviceSynchronize());

  // ── 성능 측정 ────────────────────────────────────────────────────────────
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  for (int i = 0; i < opt.iterations; ++i) {
    wmma_bf16_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K,
                                           opt.alpha, opt.beta, 0u, 1u);
  }
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float elapsed_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
  double avg_ms = elapsed_ms / opt.iterations;

  std::cout << "\n[WMMA BF16 GEMM (커스텀 커널)]\n";
  std::cout << "  문제 크기   : " << M << " x " << N << " x " << K << "\n";
  std::cout << "  평균 시간   : " << avg_ms            << " ms\n";
  std::cout << "  성능        : " << opt.tflops(avg_ms) << " TFLOPS\n";

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  delete[] h_A; delete[] h_B; delete[] h_C;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// main
/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args)
{
  Options opt;
  opt.parse(argc, args);
  if (opt.help) { opt.print_usage(); return 0; }

  // GPU 정보 출력
  cudaDeviceProp props;
  int dev;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&props, dev));
  std::cout << "GPU  : " << props.name
            << "  (SM " << props.major << props.minor << ")\n";
  std::cout << "VRAM : " << props.totalGlobalMem / (1024*1024*1024) << " GB\n";

  std::cout << "\n── SM120 BF16 GEMM 비교 ──────────────────────────────────────────\n";
  std::cout << "  SM120 new tcgen05 Tensor Core = FP4/FP6/FP8 전용\n";
  std::cout << "  BF16 = HMMA (레거시 Tensor Core) 경로 사용\n";

  // ── 1. cuBLAS BF16 (권장) ─────────────────────────────────────────────────
  run_cublas_bf16_gemm(opt);

  // ── 2. WMMA 커스텀 커널 BF16 (교육/커스텀용) ──────────────────────────────
  // 큰 행렬은 시간이 오래 걸리므로 작은 크기로 실행
  Options wmma_opt = opt;
  wmma_opt.m = std::min(opt.m, 4096);
  wmma_opt.n = std::min(opt.n, 4096);
  wmma_opt.k = std::min(opt.k, 4096);
  wmma_opt.iterations = std::min(opt.iterations, 20);
  run_wmma_bf16_gemm(wmma_opt);

  return 0;
}
