#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Resolve repo root (parent of riscv/).
cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

docker run --rm -v "$REPO_ROOT:$REPO_ROOT" -w "$REPO_ROOT" kleidicv-riscv-dev:latest \
  bash -c '
    set -euo pipefail
    rm -rf riscv/smoke/build
    cmake -S riscv/smoke -B riscv/smoke/build -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE="$(pwd)/riscv/cmake/toolchain-rv64gcv.cmake" \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build riscv/smoke/build
    file riscv/smoke/build/hello_rvv
  '
