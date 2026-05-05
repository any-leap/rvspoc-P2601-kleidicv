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

add_executable(test_add test_add.cpp)
target_link_libraries(test_add PRIVATE kleidicv)
add_test(NAME add_rvv COMMAND test_add)
add_test(NAME add_scalar_forced COMMAND test_add)
set_tests_properties(add_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_sub test_sub.cpp)
target_link_libraries(test_sub PRIVATE kleidicv)
add_test(NAME sub_rvv COMMAND test_sub)
add_test(NAME sub_scalar_forced COMMAND test_sub)
set_tests_properties(sub_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_bitwise_and test_bitwise_and.cpp)
target_link_libraries(test_bitwise_and PRIVATE kleidicv)
add_test(NAME bitwise_and_rvv COMMAND test_bitwise_and)
add_test(NAME bitwise_and_scalar_forced COMMAND test_bitwise_and)
set_tests_properties(bitwise_and_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_threshold_binary test_threshold_binary.cpp)
target_link_libraries(test_threshold_binary PRIVATE kleidicv)
add_test(NAME threshold_binary_rvv COMMAND test_threshold_binary)
add_test(NAME threshold_binary_scalar_forced COMMAND test_threshold_binary)
set_tests_properties(threshold_binary_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_compare test_compare.cpp)
target_link_libraries(test_compare PRIVATE kleidicv)
add_test(NAME compare_rvv COMMAND test_compare)
add_test(NAME compare_scalar_forced COMMAND test_compare)
set_tests_properties(compare_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_in_range test_in_range.cpp)
target_link_libraries(test_in_range PRIVATE kleidicv)
add_test(NAME in_range_rvv COMMAND test_in_range)
add_test(NAME in_range_scalar_forced COMMAND test_in_range)
set_tests_properties(in_range_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_add_abs_with_threshold test_add_abs_with_threshold.cpp)
target_link_libraries(test_add_abs_with_threshold PRIVATE kleidicv)
add_test(NAME add_abs_with_threshold_rvv COMMAND test_add_abs_with_threshold)
add_test(NAME add_abs_with_threshold_scalar_forced COMMAND test_add_abs_with_threshold)
set_tests_properties(add_abs_with_threshold_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_multiply test_multiply.cpp)
target_link_libraries(test_multiply PRIVATE kleidicv)
add_test(NAME multiply_rvv COMMAND test_multiply)
add_test(NAME multiply_scalar_forced COMMAND test_multiply)
set_tests_properties(multiply_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_scale test_scale.cpp)
target_link_libraries(test_scale PRIVATE kleidicv)
add_test(NAME scale_rvv COMMAND test_scale)
add_test(NAME scale_scalar_forced COMMAND test_scale)
set_tests_properties(scale_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

add_executable(test_exp test_exp.cpp)
target_link_libraries(test_exp PRIVATE kleidicv)
add_test(NAME exp_rvv COMMAND test_exp)
add_test(NAME exp_scalar_forced COMMAND test_exp)
set_tests_properties(exp_scalar_forced PROPERTIES
  ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")

foreach(t gray_to_rgba split_merge rgb_to_rgb float_conv yuv min_max filters transform separable optical_flow)
  add_executable(test_${t} test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE kleidicv)
  add_test(NAME ${t}_rvv COMMAND test_${t})
  add_test(NAME ${t}_scalar_forced COMMAND test_${t})
  set_tests_properties(${t}_scalar_forced PROPERTIES
    ENVIRONMENT "KLEIDICV_FORCE_SCALAR=1")
endforeach()

# Microbenchmark — built but not added to ctest. Run manually:
#   ./riscv/library/build/test/bench_kleidicv 20      # rvv path
#   KLEIDICV_FORCE_SCALAR=1 ./.../bench_kleidicv 20   # scalar baseline
add_executable(bench_kleidicv bench_kleidicv.cpp)
target_link_libraries(bench_kleidicv PRIVATE kleidicv)
