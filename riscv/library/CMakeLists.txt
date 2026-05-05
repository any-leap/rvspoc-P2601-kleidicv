# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Parallel CMake root for the RISC-V backend of KleidiCV.
#
# Why a parallel root?
# The upstream top-level CMakeLists.txt unconditionally builds AArch64 OBJECT
# libraries (kleidicv_neon, kleidicv_sve2, ...) and links them into the public
# kleidicv static library. Until enough operators are ported, that tree cannot
# even configure under a riscv64 toolchain. We avoid touching upstream by
# building the riscv64 deliverable from this directory; eventually this root
# folds back into upstream once feature parity is reached.
#
# Layout assumption: this file lives at <repo>/riscv/library/, and upstream
# headers are at <repo>/kleidicv/include/.

cmake_minimum_required(VERSION 3.16)

project("KleidiCV-RISCV" CXX)

if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
  message(FATAL_ERROR
    "This build root only supports riscv64. Pass -DCMAKE_TOOLCHAIN_FILE="
    "<repo>/riscv/cmake/toolchain-rv64gcv.cmake.")
endif()

# The library uses RVV intrinsics throughout the *_rvv.cpp paths, so it must
# be compiled with `-march=rv64gcv*` (the toolchain file sets this). The
# public header (kleidicv.h) does NOT enforce this — consumers can include
# it from plain RV64GC code; only the build of this library requires it.
include(CheckCXXSourceCompiles)
set(_old_flags "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS_INIT}")
check_cxx_source_compiles("
#if !defined(__riscv_v) || (__riscv_v < 1000000)
#error need RVV 1.0 (-march=rv64gcv...)
#endif
int main() { return 0; }
" KLEIDICV_RISCV_TOOLCHAIN_HAS_RVV)
set(CMAKE_REQUIRED_FLAGS "${_old_flags}")
if(NOT KLEIDICV_RISCV_TOOLCHAIN_HAS_RVV)
  message(FATAL_ERROR
    "Toolchain doesn't enable RVV 1.0 — pass -march=rv64gcv... via the "
    "toolchain file (see riscv/cmake/toolchain-rv64gcv.cmake).")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
set(KLEIDICV_UPSTREAM_INCLUDE "${REPO_ROOT}/kleidicv/include")

# Generate kleidicv/config.h with every AArch64-specific feature OFF.
# Upstream's config.h.in is reusable as-is — every #cmakedefine01 left unset
# expands to 0, which is exactly what we want on RISC-V.
configure_file(
  "${KLEIDICV_UPSTREAM_INCLUDE}/kleidicv/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/kleidicv/config.h"
)

set(KLEIDICV_RISCV_CXX_FLAGS
  "-fno-stack-protector"
  "-fno-exceptions"
  "-fno-rtti"
  "-Wall"
  "-Wextra"
  "-Wold-style-cast"
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND KLEIDICV_RISCV_CXX_FLAGS "-O0" "-g")
else()
  list(APPEND KLEIDICV_RISCV_CXX_FLAGS "-O2" "-g0")
endif()

add_library(kleidicv STATIC
  src/dispatch.cpp
  src/absdiff_scalar.cpp
  src/absdiff_rvv.cpp
  src/absdiff_api.cpp
  src/gray_to_rgb_scalar.cpp
  src/gray_to_rgb_rvv.cpp
  src/gray_to_rgb_api.cpp
  src/sum_scalar.cpp
  src/sum_rvv.cpp
  src/sum_api.cpp
  src/add_scalar.cpp
  src/add_rvv.cpp
  src/add_api.cpp
  src/sub_scalar.cpp
  src/sub_rvv.cpp
  src/sub_api.cpp
  src/bitwise_and_scalar.cpp
  src/bitwise_and_rvv.cpp
  src/bitwise_and_api.cpp
  src/threshold_binary_scalar.cpp
  src/threshold_binary_rvv.cpp
  src/threshold_binary_api.cpp
  src/compare_scalar.cpp
  src/compare_rvv.cpp
  src/compare_api.cpp
  src/in_range_scalar.cpp
  src/in_range_rvv.cpp
  src/in_range_api.cpp
  src/add_abs_with_threshold_scalar.cpp
  src/add_abs_with_threshold_rvv.cpp
  src/add_abs_with_threshold_api.cpp
  src/multiply_scalar.cpp
  src/multiply_rvv.cpp
  src/multiply_api.cpp
  src/scale_scalar.cpp
  src/scale_rvv.cpp
  src/scale_api.cpp
  src/exp_scalar.cpp
  src/exp_rvv.cpp
  src/exp_api.cpp
  src/gray_to_rgba_scalar.cpp
  src/gray_to_rgba_rvv.cpp
  src/gray_to_rgba_api.cpp
  src/split_merge_scalar.cpp
  src/split_merge_rvv.cpp
  src/split_merge_api.cpp
  src/rgb_to_rgb_scalar.cpp
  src/rgb_to_rgb_rvv.cpp
  src/rgb_to_rgb_api.cpp
  src/float_conv_scalar.cpp
  src/float_conv_rvv.cpp
  src/float_conv_api.cpp
  src/rgb_to_yuv_scalar.cpp
  src/rgb_to_yuv_rvv.cpp
  src/rgb_to_yuv_api.cpp
  src/yuv_to_rgb_scalar.cpp
  src/yuv_to_rgb_rvv.cpp
  src/yuv_to_rgb_api.cpp
  src/min_max_scalar.cpp
  src/min_max_rvv.cpp
  src/min_max_api.cpp
  src/sobel_scalar.cpp
  src/sobel_rvv.cpp
  src/sobel_api.cpp
  src/scharr_scalar.cpp
  src/scharr_rvv.cpp
  src/scharr_api.cpp
  src/blur_and_downsample_scalar.cpp
  src/blur_and_downsample_rvv.cpp
  src/blur_and_downsample_api.cpp
  src/separable_filter_2d_scalar.cpp
  src/separable_filter_2d_rvv.cpp
  src/separable_filter_2d_api.cpp
  src/gaussian_blur_scalar.cpp
  src/gaussian_blur_rvv.cpp
  src/median_blur_scalar.cpp
  src/median_blur_api.cpp
  src/morphology_scalar.cpp
  src/morphology_rvv.cpp
  src/resize_linear_scalar.cpp
  src/resize_linear_rvv.cpp
  src/remap_s16.cpp
  src/transpose_rotate.cpp
  src/warp_perspective.cpp
  src/heavy_stubs.cpp
  src/standalone_lucas_kanade_alg_scalar.cpp
  src/standalone_lucas_kanade_alg_rvv.cpp
  src/standalone_lucas_kanade_alg_api.cpp
  src/optical_flow_pyramid.cpp
)
target_include_directories(kleidicv PUBLIC
  "${KLEIDICV_UPSTREAM_INCLUDE}"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
)
target_compile_options(kleidicv PRIVATE ${KLEIDICV_RISCV_CXX_FLAGS})

# --- tests ---
enable_testing()
add_subdirectory(test)

# --- upstream kleidicv-benchmark (Google Benchmark) ---
#
# Off by default because configure-time fetches Google Benchmark v1.9.2 from
# the network. Turn on with -DKLEIDICV_UPSTREAM_BENCHMARK=ON to produce the
# `kleidicv-benchmark` binary that exercises every public C API across ranges
# of image sizes, matching the upstream Arm reference benchmark suite.
#
# The benchmark binary intentionally calls APIs that may return
# KLEIDICV_ERROR_NOT_IMPLEMENTED for multi-channel filter inputs; those
# benchmarks skip / fail at runtime rather than at link time. Use Google
# Benchmark's `--benchmark_filter=` to limit runs to the supported subset.
option(KLEIDICV_UPSTREAM_BENCHMARK
  "Build the upstream kleidicv-benchmark Google-Benchmark suite for riscv64"
  OFF)
if(KLEIDICV_UPSTREAM_BENCHMARK)
  set(KLEIDICV_BENCHMARK ON CACHE BOOL "" FORCE)
  add_subdirectory("${REPO_ROOT}/benchmark"
    "${CMAKE_BINARY_DIR}/upstream_benchmark")
endif()
