#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t size) {
    // #pragma unroll
    for (size_t i = 0; i < size; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            float gate_val = llaisys::utils::cast<float>(gate[i]);
            float up_val = llaisys::utils::cast<float>(up[i]);
            float sigmoid_val = gate_val / (1.0f + std::exp(-gate_val));
            out[i] = llaisys::utils::cast<T>(sigmoid_val * up_val);
        } else {
            float gate_val = static_cast<float>(gate[i]);
            float up_val = static_cast<float>(up[i]);
            float sigmoid_val = gate_val / (1.0f + std::exp(-gate_val));
            out[i] = static_cast<T>(sigmoid_val * up_val);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate), reinterpret_cast<const float *>(up), size);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(gate), reinterpret_cast<const llaisys::bf16_t *>(up), size);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(gate), reinterpret_cast<const llaisys::fp16_t *>(up), size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu