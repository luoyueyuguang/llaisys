#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t index_size, size_t embed_dim) {
    for (size_t i = 0; i < index_size; i++) {
        int64_t idx = index[i];
        std::copy(weight + idx * embed_dim, weight + (idx + 1) * embed_dim, out + i * embed_dim);
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t index_size, size_t embed_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight), index_size, embed_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::bf16_t *>(weight), index_size, embed_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::fp16_t *>(weight), index_size, embed_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu