#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
# SPDX-License-Identifier: Apache-2.0
#
# Configure + build the RISC-V parallel build root and run ctest under qemu.
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

VLEN="${VLEN:-256}"

docker run --rm -v "$REPO_ROOT:$REPO_ROOT" -w "$REPO_ROOT" \
  -e VLEN="$VLEN" \
  kleidicv-riscv-dev:latest bash -c '
    set -euo pipefail
    BUILD_DIR=riscv/library/build
    rm -rf "$BUILD_DIR"
    cmake -S riscv/library -B "$BUILD_DIR" -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE="$(pwd)/riscv/cmake/toolchain-rv64gcv.cmake" \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR"
    file "$BUILD_DIR"/libkleidicv.a "$BUILD_DIR"/test/test_absdiff
    echo "--- ctest (qemu vlen='"$VLEN"') ---"
    cd "$BUILD_DIR"
    ctest --output-on-failure
  '
