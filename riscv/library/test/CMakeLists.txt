# SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
# SPDX-License-Identifier: Apache-2.0

add_executable(test_absdiff test_absdiff.cpp)
target_link_libraries(test_absdiff PRIVATE kleidicv)

add_test(NAME absdiff_rvv COMMAND test_absdiff)
add_test(NAME absdiff_scalar_forced COMMAND test_absdiff)
set_tests_properties(absdiff_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_gray_to_rgb test_gray_to_rgb.cpp)
target_link_libraries(test_gray_to_rgb PRIVATE kleidicv)

add_test(NAME gray_to_rgb_rvv COMMAND test_gray_to_rgb)
add_test(NAME gray_to_rgb_scalar_forced COMMAND test_gray_to_rgb)
set_tests_properties(gray_to_rgb_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_sum test_sum.cpp)
target_link_libraries(test_sum PRIVATE kleidicv)

add_test(NAME sum_rvv COMMAND test_sum)
add_test(NAME sum_scalar_forced COMMAND test_sum)
set_tests_properties(sum_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")
