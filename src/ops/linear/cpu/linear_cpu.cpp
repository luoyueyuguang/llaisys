#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// Check for AVX2 and FMA support
#if defined(__AVX2__) && defined(__FMA__)
#define LLAISYS_USE_AVX2 1
#include <immintrin.h>
#else
#define LLAISYS_USE_AVX2 0
#endif

// OpenMP support
#if defined(_OPENMP) && _OPENMP >= 201307
#define LLAISYS_USE_OPENMP 1
#include <omp.h>
#else
#define LLAISYS_USE_OPENMP 0
#endif

// AVX2 optimized inner kernel for float matrix multiplication
// Uses FMA (fused multiply-add) for better performance
#if LLAISYS_USE_AVX2
inline void linear_avx2_kernel(
    const float* __restrict__ A,  // Input: [m, k]
    const float* __restrict__ B,  // Weight: [n, k] (untransposed)
    float* __restrict__ C,        // Output: [m, n]
    const float* __restrict__ bias,
    size_t m, size_t n, size_t k,
    size_t row_start, size_t row_end,
    size_t col_start, size_t col_end
) {
    // Block size for cache efficiency
    constexpr size_t BLOCK_M = 64;
    constexpr size_t BLOCK_N = 256;
    constexpr size_t BLOCK_K = 64;
    
    for (size_t ii = row_start; ii < row_end; ii += BLOCK_M) {
        size_t iend = std::min(ii + BLOCK_M, row_end);
        
        for (size_t jj = col_start; jj < col_end; jj += BLOCK_N) {
            size_t jend = std::min(jj + BLOCK_N, col_end);
            
            // Initialize output block with bias (or zero when bias is null)
            for (size_t i = ii; i < iend; i++) {
                for (size_t j = jj; j < jend; j++) {
                    C[i * n + j] = bias ? bias[j] : 0.0f;
                }
            }
            
            // Blocked matrix multiplication
            for (size_t kk = 0; kk < k; kk += BLOCK_K) {
                size_t kend = std::min(kk + BLOCK_K, k);
                
                for (size_t i = ii; i < iend; i++) {
                    for (size_t kk_inner = kk; kk_inner < kend; kk_inner++) {
                        // Broadcast A[i, kk_inner] - load once, use many times
                        __m256 a_vec = _mm256_set1_ps(A[i * k + kk_inner]);
                        
                        // Process 8 columns at a time with AVX2
                        size_t j = jj;
                        alignas(32) float b_block[8];
                        for (; j + 8 <= jend; j += 8) {
                            for (int lane = 0; lane < 8; ++lane) {
                                b_block[lane] = B[(j + static_cast<size_t>(lane)) * k + kk_inner];
                            }
                            __m256 b_vec = _mm256_loadu_ps(b_block);
                            __m256 c_vec = _mm256_loadu_ps(&C[i * n + j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&C[i * n + j], c_vec);
                        }
                        
                        // Handle remaining columns (< 8)
                        for (; j < jend; j++) {
                            C[i * n + j] += A[i * k + kk_inner] * B[j * k + kk_inner];
                        }
                    }
                }
            }
        }
    }
}
#endif // LLAISYS_USE_AVX2

// OpenMP parallelized version with optional AVX2
inline void linear_optimized_(
    float *out, 
    const float *in, 
    const float *weight, 
    const float *bias, 
    size_t m, size_t n, size_t k
) {
    // If bias provided, initialize output with bias
    if (bias != nullptr) {
        #if LLAISYS_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                out[i * n + j] = bias[j];
            }
        }
    } else {
        #if LLAISYS_USE_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                out[i * n + j] = 0.0f;
            }
        }
    }
    
#if LLAISYS_USE_AVX2
    // Use AVX2 optimized path
    #if LLAISYS_USE_OPENMP
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < m; i += 64) {
            size_t iend = std::min(i + 64, m);
            for (size_t j = 0; j < n; j += 256) {
                size_t jend = std::min(j + 256, n);
                
                for (size_t kk = 0; kk < k; kk += 64) {
                    size_t kend = std::min(kk + 64, k);
                    
                    for (size_t ii = i; ii < iend; ii++) {
                        for (size_t kk_inner = kk; kk_inner < kend; kk_inner++) {
                            __m256 a_vec = _mm256_set1_ps(in[ii * k + kk_inner]);
                            
                            size_t jj = j;
                            alignas(32) float b_block[8];
                            for (; jj + 8 <= jend; jj += 8) {
                                for (int lane = 0; lane < 8; ++lane) {
                                    b_block[lane] = weight[(jj + static_cast<size_t>(lane)) * k + kk_inner];
                                }
                                __m256 b_vec = _mm256_loadu_ps(b_block);
                                __m256 c_vec = _mm256_loadu_ps(&out[ii * n + jj]);
                                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                                _mm256_storeu_ps(&out[ii * n + jj], c_vec);
                            }
                            
                            for (; jj < jend; jj++) {
                                out[ii * n + jj] += in[ii * k + kk_inner] * weight[jj * k + kk_inner];
                            }
                        }
                    }
                }
            }
        }
    }
    #else
    // Single-threaded AVX2
    linear_avx2_kernel(in, weight, out, bias, m, n, k, 0, m, 0, n);
    #endif
#else
    // Fallback: OpenMP parallelized without AVX2
    #if LLAISYS_USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float val = 0.0f;
            #pragma omp simd reduction(+:val)
            for (size_t l = 0; l < k; l++) {
                val += in[i * k + l] * weight[j * k + l];
            }
            out[i * n + j] += val;
        }
    }
#endif
}

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
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            float val = 0.0f;
            for(size_t l = 0; l < k; l++) {
                val += in[i * k + l] * weight[j * k + l];
            }
            out[i * n + j] = val;
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
    if constexpr (std::is_same_v<T, float>) {
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < n; j++) {
                float val = bias == nullptr ? 0.0f : bias[j];
                for(size_t l = 0; l < k; l++) {
                    val += in[i * k + l] * weight[j * k + l];
                }
                out[i * n + j] = val;
            }
        }
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for(size_t i = 0; i < m; i++) {
            for(size_t j = 0; j < n; j++) {
                float val = bias == nullptr ? 0.0f : llaisys::utils::cast<float>(bias[j]);
                for(size_t l = 0; l < k; l++) {
                    val += llaisys::utils::cast<float>(in[i * k + l]) * llaisys::utils::cast<float>(weight[j * k + l]);
                }
                out[i * n + j] = llaisys::utils::cast<T>(val);
            }
        }
    }
        
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t m, size_t n, size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_optimized_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
