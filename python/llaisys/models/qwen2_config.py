from ctypes import POINTER, Structure, c_int, c_size_t, c_float, c_void_p, c_int64, byref, cast

from llaisys.libllaisys.llaisys_types import (
    llaisysDeviceType_t,
    llaisysDataType_t
)

#endif // LLAISYS_MODELS_QWEN2_H

from llaisys.libllaisys.tensor import llaisysTensor_t


class LlaisysQWen2Meta(Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]

class LlaisysQWen2Weights(Structure):
    _fields_ = [
        ("in_embed", llaisysTensor_t),
        ("out_embed", llaisysTensor_t),
        ("out_norm_w", llaisysTensor_t),
        ("attn_norm_w", POINTER(llaisysTensor_t)),
        ("attn_q_w", POINTER(llaisysTensor_t)),
        ("attn_q_b", POINTER(llaisysTensor_t)),
        ("attn_k_w", POINTER(llaisysTensor_t)),
        ("attn_k_b", POINTER(llaisysTensor_t)),
        ("attn_v_w", POINTER(llaisysTensor_t)),
        ("attn_v_b", POINTER(llaisysTensor_t)),
        ("attn_o_w", POINTER(llaisysTensor_t)),
        ("mlp_norm_w", POINTER(llaisysTensor_t)),
        ("mlp_gate_w", POINTER(llaisysTensor_t)),
        ("mlp_up_w", POINTER(llaisysTensor_t)),
        ("mlp_down_w", POINTER(llaisysTensor_t)),
    ]

class LlaisysQwen2Model(Structure):
    _fields_ = [
        ("meta", LlaisysQWen2Meta),
        ("weights", POINTER(LlaisysQWen2Weights)),

        ("device", llaisysDeviceType_t),
        ("device_ids", POINTER(c_int)),
        ("ndevice", c_int),

        #定义kv cache相关
        ("k_cache", POINTER(llaisysTensor_t)),
        ("v_cache", POINTER(llaisysTensor_t)),

        # size_t cached = 0;
    ]

def load_qwen2(lib):
    lib.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQWen2Meta),
        c_size_t,
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]

    lib.llaisysQwen2ModelCreate.restype = POINTER(LlaisysQwen2Model)

    lib.llaisysQwen2ModelDestroy.argtypes = [
        POINTER(LlaisysQwen2Model),
    ]
    lib.llaisysQwen2ModelDestroy.restype = None

    lib.llaisysQwen2ModelWeights.argtypes = [
        POINTER(LlaisysQwen2Model),
    ]
    lib.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQWen2Weights)

    lib.llaisysQwen2ModelInfer.argtypes = [
        POINTER(LlaisysQwen2Model),
        POINTER(c_int64),
        c_size_t,
        c_size_t,
    ]
    lib.llaisysQwen2ModelInfer.restype = c_int64