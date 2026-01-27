#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t num_heads, size_t head_dim, float theta) {
    const size_t half_head_dim = head_dim / 2;
    for(size_t i = 0; i < seq_len; i++) {
        float posi = llaisys::utils::cast<float>(pos_ids[i]);
        for(size_t h = 0; h < num_heads; h++) {
            for(size_t d = 0; d < half_head_dim; d++) {
                float phi = posi  / std::pow(theta, 2.0f * static_cast<float>(d) / static_cast<float>(head_dim));
                float cos_val = std::cos(phi);
                float sin_val = std::sin(phi);
                size_t idx = i * num_heads * head_dim + h * head_dim + d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    float a = llaisys::utils::cast<float>(in[idx]);
                    float b = llaisys::utils::cast<float>(in[idx + half_head_dim]);
                    out[idx] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                    out[idx + half_head_dim] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
                } else if constexpr (std::is_same_v<T, float>) {
                    out[idx] = in[idx] * cos_val - in[idx + half_head_dim] * sin_val;
                    out[idx + half_head_dim] = in[idx + half_head_dim] * cos_val + in[idx] * sin_val;
                }
            }
        }
    }
    // TO_BE_IMPLEMENTED();
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_len, size_t num_heads, size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), seq_len, num_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu