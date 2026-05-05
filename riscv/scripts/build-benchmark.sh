#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build the upstream kleidicv-benchmark binary against our RISC-V backend.
#
# Pulls Google Benchmark v1.9.2 from the network (configure-time FetchContent),
# so the dev container needs working internet on first invocation. Subsequent
# rebuilds reuse the cached download.
#
# Output: riscv/library/build-bench/upstream_benchmark/kleidicv-benchmark
#
# Run on real hardware as:
#   ./kleidicv-benchmark                                  # default 1280x720
#   ./kleidicv-benchmark --image_width=512 --image_height=512
#   ./kleidicv-benchmark --benchmark_filter='saturating_add_u8.*'
# Force scalar baseline:
#   KLEIDICV_FORCE_SCALAR=1 ./kleidicv-benchmark ...

set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

VLEN="${VLEN:-256}"

docker run --rm -v "$REPO_ROOT:$REPO_ROOT" -w "$REPO_ROOT" \
  -e VLEN="$VLEN" \
  kleidicv-riscv-dev:latest bash -c '
    set -euo pipefail
    BUILD_DIR=riscv/library/build-bench
    cmake -S riscv/library -B "$BUILD_DIR" -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE="$(pwd)/riscv/cmake/toolchain-rv64gcv.cmake" \
      -DCMAKE_BUILD_TYPE=Release \
      -DKLEIDICV_UPSTREAM_BENCHMARK=ON \
      -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON \
      -DBENCHMARK_ENABLE_TESTING=OFF \
      -DBENCHMARK_ENABLE_GTEST_TESTS=OFF
    cmake --build "$BUILD_DIR" --target kleidicv-benchmark
    echo
    echo "kleidicv-benchmark built:"
    file "$BUILD_DIR"/upstream_benchmark/kleidicv-benchmark
  '
