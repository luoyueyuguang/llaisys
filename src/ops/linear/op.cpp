#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: all tensors must be contiguous.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    size_t m = in->shape()[0];
    size_t k = in->shape()[1];
    size_t n = weight->shape()[0];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), m, n, k);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(), m, n, k);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
