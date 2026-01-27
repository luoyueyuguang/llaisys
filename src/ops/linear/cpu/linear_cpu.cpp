#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_nobias_(T *out, const T *in, const T *weight, size_t m, size_t n, size_t k) {
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            float val = 0.0f;
            for(size_t l = 0; l < k; l++) {
                val += llaisys::utils::cast<float>(in[i * k + l]) * llaisys::utils::cast<float>(weight[j * k + l]);
            }
            out[i * n + j] = llaisys::utils::cast<T>(val);
        }
    }
}

void linear_nobias_(float *out, const float *in, const float *weight, size_t m, size_t n, size_t k) {
    const size_t block_size = 32;
    for(size_t i = 0; i < m; i += block_size) {
        for(size_t j = 0; j < n; j += block_size) {
            for(size_t l = 0; l < k; l += block_size) {
                size_t iend = std::min(i + block_size, m);
                size_t jend = std::min(j + block_size, n);
                size_t kend = std::min(k, l + block_size);
                for(size_t ii = i; ii < iend; ii++) {
                    for(size_t jj = j; jj < jend; jj++) {
                        float val = 0.0f;
                        for(size_t kk = l; kk < kend; kk++) {
                            val += in[ii * k + kk] * weight[jj * k + kk];
                        }
                        out[ii * n + jj] += val;
                    }
                }
            }
        }
    }
}

template <typename T>
void linear_bias_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    //给out初始化0
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            float val = llaisys::utils::cast<float>(bias[j]);
            for(size_t l = 0; l < k; l++) {
                val += llaisys::utils::cast<float>(in[i * k + l]) * llaisys::utils::cast<float>(weight[j * k + l]);
            }
            out[i * n + j] = llaisys::utils::cast<T>(val);
        }
    }
}

void linear_bias_(float *out, const float *in, const float *weight, const float *bias, size_t m, size_t n, size_t k) {
    // 计算out = in @ weight^T + bias，weight未转置
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            out[i * n + j] = bias[j];
        }
    }
    const size_t block_size = 32;
    for(size_t i = 0; i < m; i += block_size) {
        for(size_t j = 0; j < n; j += block_size) {
            for(size_t l = 0; l < k; l += block_size) {
                size_t iend = std::min(i + block_size, m);
                size_t jend = std::min(j + block_size, n);
                size_t kend = std::min(k, l + block_size);
                for(size_t ii = i; ii < iend; ii++) {
                    for(size_t jj = j; jj < jend; jj++) {
                        float val = 0.0f;
                        for(size_t kk = l; kk < kend; kk++) {
                            val += in[ii * k + kk] * weight[jj * k + kk];
                        }
                        out[ii * n + jj] += val;
                    }
                }
            }
        }
    }
}

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t n, size_t k) {
    // 计算out = in @ weight^T + bias，
    if(bias == nullptr) {
        linear_nobias_(out, in, weight, m, n, k);
    } else {
        linear_bias_(out, in, weight, bias, m, n, k);
    }
        
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t m, size_t n, size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu