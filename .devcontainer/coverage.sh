#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -eu

BUILD_ID="kleidicv-coverage"

COVERAGE="ON" \
CMAKE_CXX_FLAGS="--target=aarch64-linux-gnu -fprofile-instr-generate -fcoverage-mapping" \
CMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -static -fuse-ld=lld -fprofile-instr-generate" \
EXTRA_CMAKE_ARGS="-DKLEIDICV_ENABLE_SME2=ON -DKLEIDICV_LIMIT_SME2_TO_SELECTED_ALGORITHMS=OFF -DKLEIDICV_ENABLE_SME=ON -DKLEIDICV_LIMIT_SVE2_TO_SELECTED_ALGORITHMS=OFF" \
./scripts/build.sh kleidicv-test

# Clean any coverage results from previous runs
LLVM_PROFILE_DIR="build/${BUILD_ID}/profiles"
rm -rf "${LLVM_PROFILE_DIR}"
mkdir -p "${LLVM_PROFILE_DIR}"

LONG_VECTOR_TESTS="GRAY2.*:RGB*:Yuv*:Rgb*:Resize*"
SME_API_TESTS="SmeApi*"
TEST_DIR="build/${BUILD_ID}/test"

LLVM_PROFILE_FILE="${LLVM_PROFILE_DIR}/framework-%p.profraw" qemu-aarch64 ${TEST_DIR}/framework/kleidicv-framework-test
LLVM_PROFILE_FILE="${LLVM_PROFILE_DIR}/api-neon-%p.profraw" qemu-aarch64 -cpu cortex-a35 ${TEST_DIR}/api/kleidicv-api-test
LLVM_PROFILE_FILE="${LLVM_PROFILE_DIR}/unit-neon-%p.profraw" qemu-aarch64 -cpu cortex-a35 ${TEST_DIR}/unit_neon/kleidicv-neon-unit-test
LLVM_PROFILE_FILE="${LLVM_PROFILE_DIR}/api-sve128-%p.profraw" qemu-aarch64 -cpu max,sve128=on,sme=off ${TEST_DIR}/api/kleidicv-api-test --vector-length=16
LLVM_PROFILE_FILE="${LLVM_PROFILE_DIR}/api-sve2048-%p.profraw" qemu-aarch64 -cpu max,sve2048=on,sve-default-vector-length=256,sme=off \
  ${TEST_DIR}/api/kleidicv-api-test --gtest_filter="${LONG_VECTOR_TESTS}" --vector-length=256
LLVM_PROFILE_FILE="${LLVM_PROFILE_DIR}/api-sme-%p.profraw" KLEIDICV_PREFER_SME_BACKEND=ON qemu-aarch64 -cpu max,sve128=on,sme512=on ${TEST_DIR}/api/kleidicv-api-test --vector-length=64
LLVM_PROFILE_FILE="${LLVM_PROFILE_DIR}/api-sme-api-%p.profraw" KLEIDICV_PREFER_SME_BACKEND=OFF qemu-aarch64 -cpu max,sve128=on,sme512=on ${TEST_DIR}/api/kleidicv-api-test --gtest_filter="${SME_API_TESTS}" --vector-length=64
LLVM_PROFILE_FILE="${LLVM_PROFILE_DIR}/api-sme2-%p.profraw" KLEIDICV_PREFER_SME_BACKEND=ON armie -mvl=16 -msvl=64 -mfeatures=scripts/armie_features.txt \
  ${TEST_DIR}/api/kleidicv-api-test --vector-length=64

# Generate test coverage report
LLVM_COV=llvm-cov scripts/generate_coverage_report.py "build/${BUILD_ID}"
