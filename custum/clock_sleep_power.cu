#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ void consume_clock_result(
    unsigned long long* out,
    int block_id,
    unsigned long long value) {
    if (threadIdx.x == 0) {
        out[block_id] = value;
    }
}

__global__ void clock64_busy_wait_kernel(
    unsigned long long* __restrict__ elapsed_cycles,
    unsigned long long cycles,
    int repeats) {

    for (int i = 0; i < cycles; ++i) {
        asm volatile("nop;");
        // asm volatile("": : : "memory");
    }
}

__global__ void nanosleep_kernel(
    unsigned long long* __restrict__ elapsed_cycles,
    unsigned int sleep_ns,
    int repeats) {

    for (int i = 0; i < repeats; ++i) {
        __nanosleep(sleep_ns);
    }
}

void check_launch_config(int num_blocks, int threads_per_block, int repeats) {
    TORCH_CHECK(num_blocks > 0, "num_blocks must be > 0");
    TORCH_CHECK(threads_per_block > 0, "threads_per_block must be > 0");
    TORCH_CHECK(threads_per_block <= 1024, "threads_per_block must be <= 1024");
    TORCH_CHECK(repeats > 0, "repeats must be > 0");
}

torch::Tensor make_output_tensor(int num_blocks) {
    auto options = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(torch::kCUDA);
    return torch::empty({num_blocks}, options);
}

}  // namespace

torch::Tensor run_clock64_busy_wait(
    int num_blocks,
    int threads_per_block,
    unsigned long long cycles,
    int repeats) {
    check_launch_config(num_blocks, threads_per_block, repeats);
    TORCH_CHECK(cycles > 0, "cycles must be > 0");

    auto out = make_output_tensor(num_blocks);
    auto stream = at::cuda::getCurrentCUDAStream();

    clock64_busy_wait_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()),
        cycles,
        repeats);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor run_nanosleep(
    int num_blocks,
    int threads_per_block,
    unsigned int sleep_ns,
    int repeats) {
    check_launch_config(num_blocks, threads_per_block, repeats);
    TORCH_CHECK(sleep_ns > 0, "sleep_ns must be > 0");

    auto out = make_output_tensor(num_blocks);
    auto stream = at::cuda::getCurrentCUDAStream();

    nanosleep_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()),
        sleep_ns,
        repeats);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "clock64_busy_wait",
        &run_clock64_busy_wait,
        "Launch a clock64-based busy-wait kernel and return per-block elapsed cycles",
        py::arg("num_blocks"),
        py::arg("threads_per_block"),
        py::arg("cycles"),
        py::arg("repeats") = 1);

    m.def(
        "nanosleep",
        &run_nanosleep,
        "Launch a __nanosleep kernel and return per-block elapsed cycles",
        py::arg("num_blocks"),
        py::arg("threads_per_block"),
        py::arg("sleep_ns"),
        py::arg("repeats") = 1);
}
