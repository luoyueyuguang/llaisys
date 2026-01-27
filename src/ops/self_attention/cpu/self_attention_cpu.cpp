#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, size_t seq_len, size_t num_heads, size_t d, size_t total_len, size_t nkvhead, size_t dv, float scale) {
    /*
    q：查询张量。形状应该是 [seqlen, nhead, d]。你暂时可以假设张量是连续的。
    k：键张量。形状应该是 [total_len, nkvhead, d]。你暂时可以假设张量是连续的。
    v：值张量。形状应该是 [total_len, nkvhead, dv]。你暂时可以假设张量是连续的。
    scale：缩放因子。在大多数情况下取值为 1/d。
    O = QK^T∗scale
    A = causalsoftmax(O).V
    */

    std::vector<float> scores(total_len, 0.0f);
    int64_t past_len = static_cast<int64_t>(total_len) - static_cast<int64_t>(seq_len);
    past_len = std::max<int64_t>(0, past_len);

    const size_t heads_per_kv = num_heads / nkvhead;
    for(size_t seq = 0; seq < seq_len; seq++){
        for(size_t head = 0; head < num_heads; head++){
            size_t kvhead = head / heads_per_kv;
            size_t tend = std::min(static_cast<size_t>(past_len) + seq + 1, total_len);
            for(size_t total_j = 0; total_j < tend; total_j++){
                float score = 0.0f;
                for(size_t dim = 0; dim < d; dim++){
                    size_t q_idx = seq * num_heads * d + head * d + dim;
                    size_t k_idx = total_j * nkvhead * d + kvhead * d + dim;
                    if constexpr(std::is_same_v<T, float>) {
                        score += q[q_idx] * k[k_idx];
                    } else if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        score += llaisys::utils::cast<float>(q[q_idx]) * llaisys::utils::cast<float>(k[k_idx]);
                    }
                }
                scores[total_j] = score * scale;
            }
             // Softmax over [0..max_t]
            float max_score = -INFINITY;
            for (size_t t = 0; t < tend; ++t) {
               max_score = std::max(max_score, scores[t]);
            }
            float sum_exp = 0.0f;
            for (size_t t = 0; t < tend; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }
            float inv_sum = (sum_exp == 0.0f) ? 0.0f : (1.0f / sum_exp);
            size_t attn_idx = seq * num_heads * dv + head * dv;
            for (size_t dvi = 0; dvi < dv; ++dvi) {
                float attn_val_ij = 0.0f;
                for (size_t t = 0; t < tend; ++t) {
                    size_t v_idx = t * nkvhead * dv + kvhead * dv + dvi;
                    if constexpr(std::is_same_v<T, float>) {
                        attn_val_ij += scores[t] * v[v_idx];
                    } else if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        attn_val_ij += scores[t] * llaisys::utils::cast<float>(v[v_idx]);
                    }
                }
                if constexpr(std::is_same_v<T, float>) {
                    attn_val[attn_idx + dvi] = attn_val_ij * inv_sum;
                } else if constexpr(std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[attn_idx + dvi] = llaisys::utils::cast<T>(attn_val_ij * inv_sum);
                }
            }
        }
    }
   


}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, size_t seq_len, size_t num_heads, size_t d, size_t total_len, size_t nkvhead, size_t dv, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), seq_len, num_heads, d, total_len, nkvhead, dv, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), seq_len, num_heads, d, total_len, nkvhead, dv, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), seq_len, num_heads, d, total_len, nkvhead, dv, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu