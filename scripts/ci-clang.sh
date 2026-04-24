#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -exu

# Ensure we're at the root of the repo.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Ensure we're doing a clean build.
rm -rf build/ci
mkdir -p build/ci

if ! command -v qemu-aarch64; then
  apt-get update
  apt-get -y --no-install-recommends install qemu-user
fi

# Force ccache for all CMake builds.
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Build with Clang.
cmake -S . -B build/ci/clang -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
  -DCMAKE_CXX_CLANG_TIDY="clang-tidy-${CLANG_TIDY_VERSION}" \
  -DCMAKE_CXX_FLAGS="--target=aarch64-linux-gnu -fcoverage-mapping -fprofile-instr-generate" \
  -DCMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -static -fuse-ld=lld -fprofile-instr-generate" \
  -DKLEIDICV_ENABLE_SME2=ON \
  -DKLEIDICV_LIMIT_SME2_TO_SELECTED_ALGORITHMS=OFF \
  -DKLEIDICV_ENABLE_SME=ON \
  -DKLEIDICV_LIMIT_SVE2_TO_SELECTED_ALGORITHMS=OFF \
  -DKLEIDICV_CHECK_BANNED_FUNCTIONS=ON

# Workaround to avoid applying clang-tidy to files in external projects.
echo '{"Checks": "-*,cppcoreguidelines-avoid-goto"}'>build/ci/clang/_deps/.clang-tidy

ninja -C build/ci/clang/

# Run tests on Clang build.
LONG_VECTOR_TESTS="GRAY2.*:RGB*:Yuv*:Rgb*:Resize*"
SME_API_TESTS="SmeApi*"
PROFILE_DIR=build/ci/clang/profiles
TEST_DIR=build/ci/clang/test
mkdir -p "${PROFILE_DIR}"
TESTRESULT=0
LLVM_PROFILE_FILE="${PROFILE_DIR}/framework-%p.profraw" qemu-aarch64 ${TEST_DIR}/framework/kleidicv-framework-test --gtest_output=xml:build/ci/test-results/clang-framework/ || TESTRESULT=1
LLVM_PROFILE_FILE="${PROFILE_DIR}/unit-neon-%p.profraw" qemu-aarch64 -cpu cortex-a35 ${TEST_DIR}/unit_neon/kleidicv-neon-unit-test --gtest_output=xml:build/ci/test-results/clang-unit-neon/ || TESTRESULT=1
LLVM_PROFILE_FILE="${PROFILE_DIR}/api-neon-%p.profraw" qemu-aarch64 -cpu cortex-a35 ${TEST_DIR}/api/kleidicv-api-test --gtest_output=xml:build/ci/test-results/clang-neon/ || TESTRESULT=1
# To test whether the right backend is chosen KLEIDICV_PREFER_SME_BACKEND is set while there is no SME backend.
LLVM_PROFILE_FILE="${PROFILE_DIR}/api-sve128-%p.profraw" KLEIDICV_PREFER_SME_BACKEND=ON qemu-aarch64 -cpu max,sve128=on,sme=off \
  ${TEST_DIR}/api/kleidicv-api-test --gtest_output=xml:build/ci/test-results/clang-sve128/ --vector-length=16 || TESTRESULT=1
LLVM_PROFILE_FILE="${PROFILE_DIR}/api-sve2048-%p.profraw" qemu-aarch64 -cpu max,sve2048=on,sve-default-vector-length=256,sme=off \
  ${TEST_DIR}/api/kleidicv-api-test --gtest_filter="${LONG_VECTOR_TESTS}" --gtest_output=xml:build/ci/test-results/clang-sve2048/ --vector-length=256 || TESTRESULT=1
LLVM_PROFILE_FILE="${PROFILE_DIR}/api-sme-%p.profraw" KLEIDICV_PREFER_SME_BACKEND=ON qemu-aarch64 -cpu max,sve128=on,sme512=on \
  ${TEST_DIR}/api/kleidicv-api-test --gtest_output=xml:build/ci/test-results/clang-sme/ --vector-length=64 || TESTRESULT=1
LLVM_PROFILE_FILE="${PROFILE_DIR}/api-sme-api-%p.profraw" KLEIDICV_PREFER_SME_BACKEND=OFF qemu-aarch64 -cpu max,sve128=on,sme512=on \
  ${TEST_DIR}/api/kleidicv-api-test --gtest_filter="${SME_API_TESTS}" --gtest_output=xml:build/ci/test-results/clang-sme-api/ --vector-length=64 || TESTRESULT=1
LLVM_PROFILE_FILE="${PROFILE_DIR}/api-sme2-%p.profraw" KLEIDICV_PREFER_SME_BACKEND=ON armie -mvl=16 -msvl=64 -mfeatures=scripts/armie_features.txt \
  ${TEST_DIR}/api/kleidicv-api-test --gtest_output=xml:build/ci/test-results/clang-sme2/ --vector-length=64 || TESTRESULT=1

scripts/prefix_testsuite_names.py build/ci/test-results/clang-unit-neon/kleidicv-neon-unit-test.xml "clang-unit-neon."
scripts/prefix_testsuite_names.py build/ci/test-results/clang-neon/kleidicv-api-test.xml "clang-neon."
scripts/prefix_testsuite_names.py build/ci/test-results/clang-sve128/kleidicv-api-test.xml "clang-sve128."
scripts/prefix_testsuite_names.py build/ci/test-results/clang-sve2048/kleidicv-api-test.xml "clang-sve2048."
scripts/prefix_testsuite_names.py build/ci/test-results/clang-sme/kleidicv-api-test.xml "clang-sme."
scripts/prefix_testsuite_names.py build/ci/test-results/clang-sme-api/kleidicv-api-test.xml "clang-sme-api."
scripts/prefix_testsuite_names.py build/ci/test-results/clang-sme2/kleidicv-api-test.xml "clang-sme2."

# Generate test coverage report.
LLVM_COV=llvm-cov scripts/generate_coverage_report.py build/ci

# Sanitizers don't work when run through qemu so must be run natively.
if [[ $(dpkg --print-architecture) = arm64 ]]; then
  # Clang address & undefined behaviour sanitizers.
  cmake -S . -B build/ci/sanitize -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
    -DKLEIDICV_ENABLE_SME=OFF \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all -Wno-pass-failed"
  ninja -C build/ci/sanitize kleidicv-api-test
  build/ci/sanitize/test/api/kleidicv-api-test
fi

# Build benchmarks and without continuous load/store code path, just to prevent bitrot.
cmake -S . -B build/ci/build-benchmark -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
  -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-aarch64 \
  -DCMAKE_CXX_COMPILER_TARGET=aarch64-linux-gnu \
  -DCMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -static -fuse-ld=lld" \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DKLEIDICV_BENCHMARK=ON \
  -DKLEIDICV_ENABLE_SME=ON \
  -DKLEIDICV_ENABLE_SME2=ON \
  -DKLEIDICV_LIMIT_SME2_TO_SELECTED_ALGORITHMS=OFF \
  -DKLEIDICV_LIMIT_SVE2_TO_SELECTED_ALGORITHMS=OFF \
  -DKLEIDICV_NEON_USE_CONTINUOUS_MULTIVEC_LS=OFF
ninja -C build/ci/build-benchmark kleidicv-benchmark

# Build examples to prevent bitrot.
cmake -S ./examples/extract_one_operation -B build/ci/extract_example -G Ninja \
  -DCMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -fuse-ld=lld"
ninja -C build/ci/extract_example

# Generate documentation.
doxygen

exit $TESTRESULT
