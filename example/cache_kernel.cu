#include "torch/extension.h"
#include <cassert>
#include <cstdint>
#include <map>

void swap_block(
    torch::Tensor& src,
    torch::Tensor& dst,
    const std::map<int, int>& block_mapping
) {
    torch::Device src_divice = src.device();
    torch::Device dst_divice = dst.device();
    cudaMemcpyKind memcpy_kind;
    if (src_device.is_cuda() && dst_device.is_cuda()) { 
        memcpy_kind = cudaMemcpyDeviceToDevice;
    } else if (src_device.is_cuda() && dst_device.is_cpu()) {
        memcpy_kind = cudaMemcpyDeviceToHost;
    } else if (src_device.is_cpu() && dst_device.is_cuda()) {
        memcpy_kind = cudaMemcpyHostToDevice;
    } else {
        assert(false && "Unsupported device type");
    }

    void* src_ptr = src.data_ptr();
    void* dst_ptr = dst.data_ptr();

    const int64_t block_bytes = src.numel() * src.element_size();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (const auto& pair : block_mapping) {
        int64_t src_block = pair.first;
        int64_t dst_block = pair.second;
        int64_t src_offset = src_block * block_bytes;
        int64_t dst_offset = dst_block * block_bytes;

        cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset, block_bytes, memcpy_kind, stream);
    }
}


