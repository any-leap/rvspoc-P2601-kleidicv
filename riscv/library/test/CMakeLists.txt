# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
# SPDX-License-Identifier: Apache-2.0

add_executable(test_absdiff test_absdiff.cpp)
target_link_libraries(test_absdiff PRIVATE kleidicv)

add_test(NAME absdiff_rvv COMMAND test_absdiff)
add_test(NAME absdiff_scalar_forced COMMAND test_absdiff)
set_tests_properties(absdiff_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")
