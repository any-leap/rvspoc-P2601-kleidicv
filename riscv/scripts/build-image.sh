#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

cd "$(dirname "$0")/.."
docker build --progress=plain -t kleidicv-riscv-dev:latest -f docker/Dockerfile docker/
echo "Built image: kleidicv-riscv-dev:latest"
