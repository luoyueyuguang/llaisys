#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "Self Attention: all tensors must be contiguous.");
    /*
    attn_val：结果注意力值张量。形状应该是[seqlen, nhead, dv]。你暂时可以假设张量是连续的。
    q：查询张量。形状应该是 [seqlen, nhead, d]。你暂时可以假设张量是连续的。
    k：键张量。形状应该是 [total_len, nkvhead, d]。你暂时可以假设张量是连续的。
    v：值张量。形状应该是 [total_len, nkvhead, dv]。你暂时可以假设张量是连续的。
    scale：缩放因子。在大多数情况下取值为 1/d。
    */

    size_t seq_len = q->shape()[0];
    size_t num_heads = q->shape()[1];
    size_t d = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    size_t dv = v->shape()[2];

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), seq_len, num_heads, d, total_len, nkvhead, dv, scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), seq_len, num_heads, d, total_len, nkvhead, dv, scale);
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
