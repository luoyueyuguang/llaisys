from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile llaisys linear op")
    parser.add_argument("--m", type=int, default=128, help="batch dimension")
    parser.add_argument("--k", type=int, default=4096, help="input features")
    parser.add_argument("--n", type=int, default=4096, help="output features")
    parser.add_argument(
        "--dtype",
        type=str,
        default="f32",
        choices=["f32", "f16", "bf16"],
        help="data type used for the tensors",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="number of warmup iterations before timing",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="number of timed repetitions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, os.path.join(repo_root, "test"))

    import torch
    import llaisys
    from test_utils import benchmark, random_tensor

    device = "cpu"
    out_shape = (args.m, args.n)
    x_shape = (args.m, args.k)
    w_shape = (args.n, args.k)

    print(
        f"Profiling linear: out={out_shape}, x={x_shape}, w={w_shape}, dtype={args.dtype}, repeats={args.repeat}"
    )

    x, x_ = random_tensor(x_shape, args.dtype, device, scale=0.1)
    w, w_ = random_tensor(w_shape, args.dtype, device, scale=0.01)
    bias, bias_ = random_tensor((args.n,), args.dtype, device)
    out, out_ = random_tensor(out_shape, args.dtype, device)

    def torch_linear():
        torch.nn.functional.linear(x, w, bias, out=out)

    def llaisys_linear():
        llaisys.Ops.linear(out_, x_, w_, bias_)

    benchmark(torch_linear, llaisys_linear, device, warmup=args.warmup, repeat=args.repeat)


if __name__ == "__main__":
    main()
