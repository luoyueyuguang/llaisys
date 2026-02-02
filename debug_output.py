#!/usr/bin/env python3

import gc
from test_utils import *

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def load_llaisys_model(model_path, device_name):
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def compare_outputs(prompt="Hello", model_path=None, device_name="cpu"):
    print(f"Comparing outputs for prompt: '{prompt}'")

    # 加载tokenizer
    tokenizer, hf_model, model_path = load_hf_model(model_path, device_name)

    # 准备输入
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(hf_model.device)

    print(f"Input tokens: {inputs[0].tolist()}")
    print(f"Input text: {repr(input_content)}")

    # HuggingFace前向传播
    print("\n=== HuggingFace Forward Pass ===")
    with torch.no_grad():
        hf_outputs = hf_model(inputs)
        hf_logits = hf_outputs.logits[0, -1, :]  # 获取最后一个位置的logits

        # 获取top-10 tokens
        hf_top_values, hf_top_indices = torch.topk(hf_logits, 10)
        print("Top 10 HuggingFace logits:")
        for i, (idx, val) in enumerate(zip(hf_top_indices, hf_top_values)):
            token_str = tokenizer.decode([idx.item()])
            print(f"  {i + 1}. Token {idx.item()} ('{token_str}'): {val.item():.4f}")

        # Argmax结果
        hf_argmax = torch.argmax(hf_logits).item()
        print(
            f"HuggingFace argmax token: {hf_argmax} ('{tokenizer.decode([hf_argmax])}')"
        )

    del hf_model
    gc.collect()

    # 自定义实现
    print("\n=== Custom Implementation ===")
    custom_model = load_llaisys_model(model_path, device_name)

    # 只运行一次推理看结果
    tokens = tokenizer.encode(input_content)
    print(f"Running inference with tokens: {tokens}")

    # 这里我们需要修改一下来获取输出，而不是调用generate
    # 但由于没有现成的接口，我们先比较generate的第一个token

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default="./DeepSeek-R1-Distill-Qwen-1.5B", type=str)
    parser.add_argument("--prompt", default="Hello", type=str)

    args = parser.parse_args()

    compare_outputs(args.prompt, args.model, args.device)
