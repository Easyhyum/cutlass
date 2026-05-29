/**
 * cta_probe.cu
 *
 * Lightweight probe kernel: each CTA records (smid, globaltimer-start,
 * globaltimer-end, blockIdx.{x,y,z}). No GEMM math — we just want to see
 * how the hardware dispatcher places CTAs across SMs/waves for a given grid.
 *
 * The grid shape is supplied by the host so the caller can reproduce the
 * grid that the CUTLASS 128x128 GEMM kernel would use, without actually
 * running the GEMM.
 *
 * Build:
 *   python setup_cta_probe.py build_ext --inplace
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ __forceinline__ uint64_t get_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void cta_probe_kernel(
    int*                smid_out,
    unsigned long long* start_out,
    unsigned long long* end_out,
    int*                bx_out,
    int*                by_out,
    int*                bz_out,
    int                 grid_x,
    int                 grid_y,
    int                 busy_clocks)
{
    // single thread per CTA records — no atomics, no shared memory needed.
    if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) return;

    const uint64_t t0 = get_globaltimer();
    const int linear =
        (blockIdx.z * grid_y + blockIdx.y) * grid_x + blockIdx.x;

    smid_out[linear]  = static_cast<int>(get_smid());
    start_out[linear] = t0;
    bx_out[linear]    = static_cast<int>(blockIdx.x);
    by_out[linear]    = static_cast<int>(blockIdx.y);
    bz_out[linear]    = static_cast<int>(blockIdx.z);

    // tiny busy-wait so wave boundaries are visible in start-time ordering
    if (busy_clocks > 0) {
        const long long c0 = clock64();
        while ((clock64() - c0) < (long long)busy_clocks) { /* spin */ }
    }

    end_out[linear] = get_globaltimer();
}

void cta_probe(int grid_x, int grid_y, int grid_z,
               int threads_per_cta,
               int busy_clocks,
               torch::Tensor smid,
               torch::Tensor start_t,
               torch::Tensor end_t,
               torch::Tensor bx,
               torch::Tensor by,
               torch::Tensor bz)
{
    TORCH_CHECK(smid.is_cuda() && smid.dtype() == torch::kInt32, "smid must be cuda int32");
    TORCH_CHECK(start_t.dtype() == torch::kInt64, "start_t must be int64");
    TORCH_CHECK(end_t.dtype() == torch::kInt64, "end_t must be int64");

    dim3 grid(static_cast<unsigned>(grid_x),
              static_cast<unsigned>(grid_y),
              static_cast<unsigned>(grid_z));
    dim3 block(static_cast<unsigned>(threads_per_cta), 1u, 1u);

    auto stream = at::cuda::getCurrentCUDAStream();
    cta_probe_kernel<<<grid, block, 0, stream>>>(
        smid.data_ptr<int>(),
        reinterpret_cast<unsigned long long*>(start_t.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long*>(end_t.data_ptr<int64_t>()),
        bx.data_ptr<int>(), by.data_ptr<int>(), bz.data_ptr<int>(),
        grid_x, grid_y, busy_clocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cta_probe", &cta_probe,
          "Probe CTA-to-SM dispatch (records smid + globaltimer per CTA)",
          py::arg("grid_x"), py::arg("grid_y"), py::arg("grid_z"),
          py::arg("threads_per_cta"),
          py::arg("busy_clocks"),
          py::arg("smid"), py::arg("start_t"), py::arg("end_t"),
          py::arg("bx"), py::arg("by"), py::arg("bz"));
}
