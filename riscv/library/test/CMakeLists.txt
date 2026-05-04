# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
# SPDX-License-Identifier: Apache-2.0

add_executable(test_absdiff test_absdiff.cpp)
target_link_libraries(test_absdiff PRIVATE kleidicv)
add_test(NAME absdiff_basic COMMAND test_absdiff)
