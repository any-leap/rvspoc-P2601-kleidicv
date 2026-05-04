# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)
project(hello_rvv C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(hello_rvv hello_rvv.c)
target_compile_options(hello_rvv PRIVATE -O2 -Wall -Wextra)

enable_testing()
add_test(NAME hello_rvv_runs COMMAND hello_rvv)
set_tests_properties(hello_rvv_runs PROPERTIES
    PASS_REGULAR_EXPRESSION "match: yes")
