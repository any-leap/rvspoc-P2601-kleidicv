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
)
target_include_directories(kleidicv PUBLIC
  "${KLEIDICV_UPSTREAM_INCLUDE}"
  "${CMAKE_CURRENT_BINARY_DIR}/include"
)
target_compile_options(kleidicv PRIVATE ${KLEIDICV_RISCV_CXX_FLAGS})

# --- tests ---
enable_testing()
add_subdirectory(test)
