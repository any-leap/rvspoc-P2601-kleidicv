// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_OPENCV_CONFORMITY_TESTS_H_
#define KLEIDICV_OPENCV_CONFORMITY_TESTS_H_

#include <vector>

#include "utils.h"

std::vector<test>& binary_op_tests_get();
std::vector<test>& yuv42x_to_rgb_tests_get();
std::vector<test>& morphology_tests_get();
std::vector<test>& separable_filter_2d_tests_get();
std::vector<test>& gaussian_blur_tests_get();
std::vector<test>& rgb2yuv_tests_get();
std::vector<test>& yuv2rgb_tests_get();
std::vector<test>& sobel_tests_get();
std::vector<test>& sum_tests_get();
std::vector<test>& exp_tests_get();
std::vector<test>& float_conversion_tests_get();
std::vector<test>& resize_tests_get();
std::vector<test>& scale_tests_get();
std::vector<test>& min_max_tests_get();
std::vector<test>& in_range_tests_get();
std::vector<test>& remap_tests_get();
std::vector<test>& warp_perspective_tests_get();
std::vector<test>& blur_and_downsample_tests_get();
std::vector<test>& scharr_interleaved_tests_get();
std::vector<test>& median_blur_tests_get();
std::vector<test>& rgb2yuv420_tests_get();
std::vector<test>& rgb_to_yuv_422_tests_get();
std::vector<test>& standalone_lucas_kanade_alg_tests_get();
std::vector<test>& rotate_tests_get();
std::vector<test>& transpose_tests_get();
std::vector<test>& add_padding_by_copy_tests_get();

#if MANAGER
std::vector<manager_only_test>& add_padding_by_copy_manager_only_tests_get();
std::vector<manager_only_test>& build_optical_flow_pyr_lk_pyramid_tests_get();
std::vector<manager_only_test>& calc_optical_flow_pyr_lk_tests_get();
#endif

#endif  // KLEIDICV_OPENCV_CONFORMITY_TESTS_H_
