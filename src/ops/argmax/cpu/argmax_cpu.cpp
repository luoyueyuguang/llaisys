#include "./argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void argmax_(T *max_val, int64_t *max_idx, const T *vals, size_t size) {
    max_val[0] = vals[0];
    max_idx[0] = 0;
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) { 
        for (size_t i = 1; i < size; i++) {
            if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(max_val[0])) {
                max_val[0] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(vals[i]));
                max_idx[0] = i;
            }
        }
    } else {
        for (size_t i = 1; i < size; i++) {
            if (vals[i] > max_val[0]) {
                max_val[0] = vals[i];
                max_idx[0] = i;
            }
        }
    }
}

namespace llaisys::ops::cpu {
    void argmax(std::byte *vals, std::byte *max_val, int64_t *max_idx, llaisysDataType_t type, size_t size) {
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return argmax_(reinterpret_cast<float *>(max_val), max_idx, reinterpret_cast<const float *>(vals), size);
        case LLAISYS_DTYPE_BF16:
            return argmax_(reinterpret_cast<llaisys::bf16_t *>(max_val), max_idx, reinterpret_cast<const llaisys::bf16_t *>(vals), size);
        case LLAISYS_DTYPE_F16:
            return argmax_(reinterpret_cast<llaisys::fp16_t *>(max_val), max_idx, reinterpret_cast<const llaisys::fp16_t *>(vals), size);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
}