from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType, llaisysTensor_t
from .qwen2_config import *
from .utils import *

from pathlib import Path
import safetensors
import json
import logging
import sys

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # 明确指定使用重定向后的 stdout
        logging.FileHandler("qwen2.log", encoding="utf-8"),  # 文件也指定编码
    ],
)

logger = logging.getLogger(__name__)

load_qwen2(LIB_LLAISYS)


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor
        assert device == DeviceType.CPU, "Only CPU is supported for now"

        self.model_path = Path(model_path)
        self.device = device
        # self.model:LlaisysQwen2Model = {}

    def create_model(self, kv_cache_size: int = 0):
        # Load model meta
        with open(self.model_path / "config.json", "r") as f:
            self.config = json.load(f)
        # 将信息打印为绿色字体，并在两行横线之间打印

        self.meta = LlaisysQWen2Meta(
            dtype=torch_dtype_to_llaisys_dtype(self.config),
            nlayer=self.config["num_hidden_layers"],
            hs=self.config["hidden_size"],
            nh=self.config["num_attention_heads"],
            nkvh=self.config["num_key_value_heads"],
            dh=self.config["hidden_size"] // self.config["num_attention_heads"],
            di=self.config["intermediate_size"],
            maxseq=self.config["max_position_embeddings"],
            voc=self.config["vocab_size"],
            episilon=self.config["rms_norm_eps"],
            theta=self.config["rope_theta"],
            end_token=self.config["eos_token_id"],
        )

        # Load model weights
        state_dict = {}
        self.weights: LlaisysQWen2Weights = {}
        for file in sorted(self.model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="torch", device="cpu")
            for name_ in data_.keys():
                state_dict[name_] = data_.get_tensor(name_)
                # TODO: load the weights to self.weights
        # 将信息打印为绿色字体，并在两行横线之间打印

        # c_int数组
        device_ids = (c_int * 1)(0)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self.meta),
            c_size_t(kv_cache_size),
            self.device,
            device_ids,
            len(device_ids),
        )

        self.weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)

        logger.info(
            f"Loading in_embed with shape: {state_dict['model.embed_tokens.weight'].shape}"
        )
        logger.info(f"in_embed dtype: {state_dict['model.embed_tokens.weight'].dtype}")
        LIB_LLAISYS.tensorLoad(
            getattr(self.weights.contents, "in_embed"),
            state_dict["model.embed_tokens.weight"].data_ptr(),
        )

        logger.info(
            f"Loading out_embed with shape: {state_dict['lm_head.weight'].shape}"
        )
        logger.info(f"out_embed dtype: {state_dict['lm_head.weight'].dtype}")
        LIB_LLAISYS.tensorLoad(
            getattr(self.weights.contents, "out_embed"),
            state_dict["lm_head.weight"].data_ptr(),
        )

        logger.info(
            f"Loading out_norm_w with shape: {state_dict['model.norm.weight'].shape}"
        )
        logger.info(f"out_norm_w dtype: {state_dict['model.norm.weight'].dtype}")
        LIB_LLAISYS.tensorLoad(
            getattr(self.weights.contents, "out_norm_w"),
            state_dict["model.norm.weight"].data_ptr(),
        )

        # refernce:https://github.com/liulog/llaisys/blob/main/python/llaisys/models/qwen2.py
        def load_layer_array(field_name, base_name):
            arr_ptr = getattr(self.weights.contents, field_name)
            arr_type = llaisysTensor_t * self.meta.nlayer
            arr = cast(arr_ptr, POINTER(arr_type)).contents

            for i in range(self.meta.nlayer):
                tensor_name = f"model.layers.{i}.{base_name}"
                LIB_LLAISYS.tensorLoad(arr[i], state_dict[tensor_name].data_ptr())

        load_layer_array("attn_norm_w", "input_layernorm.weight")
        load_layer_array("attn_q_w", "self_attn.q_proj.weight")
        load_layer_array("attn_q_b", "self_attn.q_proj.bias")
        load_layer_array("attn_k_w", "self_attn.k_proj.weight")
        load_layer_array("attn_k_b", "self_attn.k_proj.bias")
        load_layer_array("attn_v_w", "self_attn.v_proj.weight")
        load_layer_array("attn_v_b", "self_attn.v_proj.bias")
        load_layer_array("attn_o_w", "self_attn.o_proj.weight")

        load_layer_array("mlp_norm_w", "post_attention_layernorm.weight")
        load_layer_array("mlp_gate_w", "mlp.gate_proj.weight")
        load_layer_array("mlp_up_w", "mlp.up_proj.weight")
        load_layer_array("mlp_down_w", "mlp.down_proj.weight")

        # weights_to_print = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        # logger.info(weights_to_print)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # TODO: Implement generate function
        logger.info(f"max_new_tokens: {max_new_tokens}")
        tokens = list(inputs)
        kv_cache_size = len(tokens) + max_new_tokens
        self.create_model(kv_cache_size)

        # prefill
        logger.info("prefill start")
        ntoken = len(tokens)
        token_ids = (c_int64 * ntoken)(*tokens)
        past_len = c_size_t(0)
        try:
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model, token_ids, ntoken, past_len
            )
            logger.info(f"prefill completed, next_token: {next_token}")
        except Exception as e:
            logger.error(f"Error during prefill: {e}")
            raise
        tokens.append(next_token)
        logger.info("prefill end")
        logger.info("decode start")
        # decode
        for _ in range(max_new_tokens - 1):
            logger.info(f"next_token: {next_token}")
            if next_token == self.meta.end_token:
                break
            logger.info(f"tokens: {tokens}")
            ntoken = 1
            token_ids = (c_int64 * 1)(next_token)
            past_len = c_size_t(len(tokens) - 1)
            logger.info(f"past_len: {past_len}")
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model, token_ids, ntoken, past_len
            )
            tokens.append(next_token)

        LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)

        return tokens
