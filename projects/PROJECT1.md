# Project #1 – CPU Optimization Report

## Overview

- Added an opt-in OpenMP build flag (`--openmp`, enabled by default) so the CPU
  kernels can run with multi-threaded loops when the toolchain supports it.
- Fixed the `linear` operator to keep the bias values when the AVX path is
  enabled and made the vectorized kernel respect the row-major layout used by
  PyTorch tensors.
- Introduced `scripts/profile_linear.py` to run repeatable micro-benchmarks and
  compare LLAISYS against PyTorch without modifying the official tests.

## Build instructions

```bash
# Configure without OpenMP (baseline)
source ~/.xmake/profile && xmake f --openmp=n -c
xmake && xmake install

# Configure with OpenMP (optimized)
source ~/.xmake/profile && xmake f --openmp=y -c
xmake && xmake install
```

The flag only affects the CPU static libraries and the shared library that is
copied into `python/llaisys/libllaisys/` during `xmake install`, so the Python
tests will automatically pick up the chosen configuration.

## Benchmark

Command used for both runs:

```bash
PYTHONPATH=python python scripts/profile_linear.py \
  --m 128 --k 4096 --n 4096 --warmup 1 --repeat 5
```

| Build            | Torch (ms) | LLAISYS (ms) |
|------------------|-----------:|-------------:|
| `--openmp=n`     |     26.44  |      1923.68 |
| `--openmp=y`     |     36.90  |       977.21 |

The multi-threaded build halves the execution time of the linear operator on
the same workload, while still matching PyTorch numerically (validated with
`python test/ops/linear.py`).
