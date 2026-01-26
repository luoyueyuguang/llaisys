#include "op.hpp"
#include "./cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // TO_BE_IMPLEMENTED();
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val -> dtype(), vals -> dtype());
    switch (vals -> deviceType()) {
        case LLAISYS_DEVICE_CPU:
            llaisys::ops::cpu::argmax(
                vals -> data(),
                max_val -> data(),
                reinterpret_cast<int64_t *>(max_idx -> data()),
                vals -> dtype(),
                vals -> numel()
            );
            break;
        #if ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_GPU:
            TO_BE_IMPLEMENTED();
        #endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
