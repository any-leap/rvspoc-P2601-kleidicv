#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

# VLEN can be 128/256/512 (etc). Default 256 for fast iteration.
VLEN="${VLEN:-256}"

docker run --rm -v "$REPO_ROOT:$REPO_ROOT" -w "$REPO_ROOT" kleidicv-riscv-dev:latest \
  qemu-riscv64 -L /usr/riscv64-linux-gnu \
    -cpu "rv64,v=true,vlen=${VLEN},zba=true,zbb=true,zbs=true" \
    riscv/smoke/build/hello_rvv
