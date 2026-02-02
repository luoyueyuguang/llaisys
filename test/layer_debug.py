#!/usr/bin/env python3

import sys

sys.path.append("../../python")
sys.path.append(".")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import llaisys
import numpy as np
from test_utils import *


def load_hf_model_and_inputs():
    """加载HuggingFace模型和准备输入"""
    model_path = "DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    # 准备输入
    prompt = "Hello"
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt")

    return tokenizer, model, inputs[0]  # 单个序列


def load_our_model(model_path):
    """加载我们的模型"""
    our_model = llaisys.models.Qwen2(model_path, llaisys.DeviceType.CPU)
    return our_model


def compare_embeddings():
    """对比embedding层输出"""
    tokenizer, hf_model, input_ids = load_hf_model_and_inputs()
    our_model = load_our_model("DeepSeek-R1-Distill-Qwen-1.5B")

    print("=== Comparing Embedding Layer ===")

    # HuggingFace embedding
    with torch.no_grad():
        hf_embeddings = hf_model.model.embed_tokens(input_ids)

    print(f"HF embeddings shape: {hf_embeddings.shape}")
    print(f"HF embeddings dtype: {hf_embeddings.dtype}")
    print(f"HF embeddings [0,0:5]: {hf_embeddings[0, 0:5]}")

    # 获取embedding权重
    our_model.create_model(len(input_ids) + 1)

    # 由于我们的模型没有直接获取中间输出的API，我需要修改C++代码来添加调试
    print("Need to add intermediate output debugging to C++ code")


def test_individual_ops():
    """测试单个操作的正确性"""
    print("=== Testing Individual Ops ===")

    # 测试Linear操作
    device = llaisys_device("cpu")

    # 创建测试数据
    input_shape = [1, 1536]  # qwen2的hidden_size
    weight_shape = [1536, 1536]  # 方形权重用于测试

    x_torch, x = random_tensor(input_shape, "f32", "cpu")
    weight_torch, weight = random_tensor(weight_shape, "f32", "cpu")

    # 我们的linear - 需要输出tensor
    output_shape = [1, 1536]
    out_torch, out_tensor = random_tensor(output_shape, "f32", "cpu")

    # 调用linear操作
    llaisys.Ops.linear(out_tensor, x, weight, bias=None)

    # PyTorch的linear
    torch_out = torch.nn.functional.linear(x_torch, weight_torch)

    # 对比 - 使用check_equal函数
    if check_equal(out_tensor, torch_out, atol=1e-5, rtol=1e-5):
        print("✅ Linear op test PASSED")
    else:
        print("❌ Linear op test FAILED")


if __name__ == "__main__":
    test_individual_ops()
    compare_embeddings()
