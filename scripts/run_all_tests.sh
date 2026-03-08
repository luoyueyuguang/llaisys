#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

function log() {
    printf '\n[run_all_tests] %s\n' "$*"
}

if ! command -v xmake >/dev/null 2>&1; then
    if [ -f "$HOME/.xmake/profile" ]; then
        set +u
        # shellcheck source=/dev/null
        source "$HOME/.xmake/profile"
        set -u
    fi
fi

if ! command -v xmake >/dev/null 2>&1; then
    echo "xmake is required but not found in PATH" >&2
    exit 1
fi

OPENMP_FLAG=${OPENMP_FLAG:-y}

log "Configuring xmake (openmp=${OPENMP_FLAG})"
xmake f --openmp="${OPENMP_FLAG}" -c

log "Building native library"
xmake

log "Installing shared library to python package"
xmake install

export PYTHONPATH="$ROOT_DIR/python:${PYTHONPATH:-}"

function run() {
    log "Running: $*"
    "$@"
}

run python test/test_tensor.py
run python test/ops/linear.py
run python test/test_runtime.py --device cpu

log "All tests finished successfully"
