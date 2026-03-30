// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <type_traits>
#include <vector>

#include "opencv2/core/hal/interface.h"
#include "opencv2/imgproc/hal/interface.h"
#include "utils.h"

// Factor is interpreted as 1/1000, i.e. 500 for 0.5
template <int Factor, int Type>
cv::Mat exec_resize(cv::Mat& input_mat) {
  cv::Mat result(input_mat.size().height * Factor / 1000,
                 input_mat.size().width * Factor / 1000, input_mat.type());
  resize(input_mat, result, result.size(), 0, 0, Type);
  return result;
}

template <int Type>
cv::Mat exec_resize_random_scale(cv::Mat& input_mat) {
  double scale_x =
      *reinterpret_cast<double*>(&input_mat.at<uint8_t>(input_mat.rows - 2, 0));
  double scale_y =
      *reinterpret_cast<double*>(&input_mat.at<uint8_t>(input_mat.rows - 1, 0));
  cv::Mat input_mat_real = input_mat.rowRange(0, input_mat.rows - 2);
  cv::Mat result;
  resize(input_mat_real, result,
         cv::Size(input_mat.cols * scale_x, input_mat.rows * scale_y), 0, 0,
         Type);
  return result;
}

template <int Type>
cv::Mat exec_resize_to_third(cv::Mat& input_mat) {
  cv::Mat result;
  resize(input_mat, result, cv::Size(input_mat.cols / 3, input_mat.rows / 3), 0,
         0, Type);
  return result;
}

#if MANAGER
template <typename T>
static T get_threshold(int, size_t);

template <>
float get_threshold(int, size_t) {
  return 0.01F;
}

template <>
uint8_t get_threshold(int Factor, size_t channels) {
  if (Factor == 500 ||
      ((Factor == 2000 || Factor == 4000) && (channels == 1))) {
    return 1;
  }

  // OpenCV uses 7 bit fractionl part for interpolation, that results in a
  // bigger error
  return 4;
}

template <int Factor, int Type, int MinSize, int MaxSize, size_t Format>
bool test_resize(int index, RecreatedMessageQueue& request_queue,
                 RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t h = MinSize; h <= MaxSize; h += 2) {
    for (size_t w = MinSize; w <= MaxSize; w += 3) {
      cv::Mat input_mat;
      if constexpr (Factor < 1000) {
        input_mat.create(h * 1000 / Factor, w * 1000 / Factor, Format);
      } else {
        input_mat.create(h, w, Format);
      }
      if constexpr (CV_MAT_DEPTH(Format) == CV_32F) {
        rng.fill(input_mat, cv::RNG::UNIFORM, 1, 1000000);
      } else if constexpr (CV_MAT_DEPTH(Format) == CV_8U) {
        rng.fill(input_mat, cv::RNG::UNIFORM, 0, 255);
      } else {
        return true;
      }
      cv::Mat actual_mat = exec_resize<Factor, Type>(input_mat);
      cv::Mat expected_mat = get_expected_from_subordinate(
          index, request_queue, reply_queue, input_mat);

      bool success = (CV_MAT_DEPTH(Format) == CV_32F &&
                      !are_float_matrices_different<float>(
                          get_threshold<float>(Factor, CV_MAT_CN(Format)),
                          actual_mat, expected_mat)) ||
                     (CV_MAT_DEPTH(Format) == CV_8U &&
                      !are_matrices_different<uint8_t>(
                          get_threshold<uint8_t>(Factor, CV_MAT_CN(Format)),
                          actual_mat, expected_mat));
      if (!success) {
        fail_print_matrices(h, w, input_mat, actual_mat, expected_mat);
        return true;
      }
    }
  }
  return false;
}

template <int Type, size_t Format>
bool test_resize_random_scale(int index, RecreatedMessageQueue& request_queue,
                              RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t i = 0; i < 1; ++i) {
    size_t src_width = 1000;
    size_t src_height = 10;
    // One extra line is allocated to be sure factor can be placed next to the
    // real input
    cv::Mat input_mat(src_height + 2, src_width, Format);
    rng.fill(input_mat, cv::RNG::UNIFORM, 0, 255);

    double scale_x = rng.uniform(0.3, 1.0);  // [0.3, 1.0)
    double scale_y = rng.uniform(0.2, 1.0);  // [0.2, 1.0)

    // scale_x and scale_y is embedded into the input matrix
    *reinterpret_cast<double*>(&input_mat.at<uint8_t>(input_mat.rows - 2, 0)) =
        scale_x;
    *reinterpret_cast<double*>(&input_mat.at<uint8_t>(input_mat.rows - 1, 0)) =
        scale_y;

    cv::Mat actual_mat = exec_resize_random_scale<Type>(input_mat);
    cv::Mat expected_mat = get_expected_from_subordinate(
        index, request_queue, reply_queue, input_mat);

    if (are_matrices_different<uint8_t>(10, actual_mat, expected_mat)) {
      fail_print_matrices(actual_mat.rows, actual_mat.cols, input_mat,
                          actual_mat, expected_mat);
      return true;
    }
  }
  return false;
}

template <int Type, size_t Format>
bool test_resize_to_third(int index, RecreatedMessageQueue& request_queue,
                          RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (size_t i = 0; i <= 10; ++i) {
    size_t dst_width = rng.uniform(100, 200);  // [100, 200)
    size_t dst_height = 10;
    cv::Mat input_mat(dst_height * 3, dst_width * 3, Format);
    rng.fill(input_mat, cv::RNG::UNIFORM, 0, 255);

    cv::Mat actual_mat = exec_resize_to_third<Type>(input_mat);
    cv::Mat expected_mat = get_expected_from_subordinate(
        index, request_queue, reply_queue, input_mat);

    if (are_matrices_different<uint8_t>(5, actual_mat, expected_mat)) {
      fail_print_matrices(dst_height, dst_width, input_mat, actual_mat,
                          expected_mat);
      return true;
    }
  }
  return false;
}
#endif

std::vector<test>& resize_tests_get() {
  // clang-format off
  static std::vector<test> tests = {
    TEST("Resize2x2 float32, INTER_LINEAR", (test_resize<2000, CV_HAL_INTER_LINEAR, 5, 16, CV_32FC1>), (exec_resize<2000, CV_HAL_INTER_LINEAR>)),
    TEST("Resize4x4 float32, INTER_LINEAR", (test_resize<4000, CV_HAL_INTER_LINEAR, 5, 16, CV_32FC1>), (exec_resize<4000, CV_HAL_INTER_LINEAR>)),
    TEST("Resize8x8 float32, INTER_LINEAR", (test_resize<8000, CV_HAL_INTER_LINEAR, 5, 16, CV_32FC1>), (exec_resize<8000, CV_HAL_INTER_LINEAR>)),

    TEST("Resize2x2 uint8, INTER_LINEAR", (test_resize<2000, CV_HAL_INTER_LINEAR, 5, 16, CV_8UC1>), (exec_resize<2000, CV_HAL_INTER_LINEAR>)),
    TEST("Resize4x4 uint8, INTER_LINEAR", (test_resize<4000, CV_HAL_INTER_LINEAR, 5, 16, CV_8UC1>), (exec_resize<4000, CV_HAL_INTER_LINEAR>)),

    TEST("Resize0.5x0.5 uint8, INTER_AREA, 1 channel", (test_resize<500, CV_HAL_INTER_AREA, 5, 32, CV_8UC1>), (exec_resize<500, CV_HAL_INTER_AREA>)),
    TEST("Resize0.5x0.5 uint8, INTER_LINEAR, 1 channel", (test_resize<500, CV_HAL_INTER_LINEAR, 5, 32, CV_8UC1>), (exec_resize<500, CV_HAL_INTER_LINEAR>)),
    TEST("Resize0.777x0.777 uint8, INTER_LINEAR, 1 channel", (test_resize<777, CV_HAL_INTER_LINEAR, 21, 64, CV_8UC1>), (exec_resize<777, CV_HAL_INTER_LINEAR>)),
    TEST("Resize0.444x0.444 uint8, INTER_LINEAR, 1 channel", (test_resize<444, CV_HAL_INTER_LINEAR, 21, 64, CV_8UC1>), (exec_resize<444, CV_HAL_INTER_LINEAR>)),
    TEST("Resize0.3x0.3 uint8, INTER_LINEAR, 1 channel", (test_resize_to_third<CV_HAL_INTER_LINEAR, CV_8UC1>), (exec_resize_to_third<CV_HAL_INTER_LINEAR>)),
    TEST("Resize random downscale uint8, INTER_LINEAR, 1 channel", (test_resize_random_scale<CV_HAL_INTER_LINEAR, CV_8UC1>), (exec_resize_random_scale<CV_HAL_INTER_LINEAR>)),

    TEST("Resize0.5x0.5 uint8, INTER_AREA, 2 channels", (test_resize<500, CV_HAL_INTER_AREA, 5, 32, CV_8UC2>), (exec_resize<500, CV_HAL_INTER_AREA>)),
    TEST("Resize0.5x0.5 uint8, INTER_LINEAR, 2 channels", (test_resize<500, CV_HAL_INTER_LINEAR, 5, 32, CV_8UC2>), (exec_resize<500, CV_HAL_INTER_LINEAR>)),
    TEST("Resize0.777x0.777 uint8, INTER_LINEAR, 2 channels", (test_resize<777, CV_HAL_INTER_LINEAR, 21, 64, CV_8UC2>), (exec_resize<777, CV_HAL_INTER_LINEAR>)),
    TEST("Resize0.444x0.444 uint8, INTER_LINEAR, 2 channels", (test_resize<444, CV_HAL_INTER_LINEAR, 21, 64, CV_8UC2>), (exec_resize<444, CV_HAL_INTER_LINEAR>)),
    TEST("Resize0.3x0.3 uint8, INTER_LINEAR, 2 channels", (test_resize_to_third<CV_HAL_INTER_LINEAR, CV_8UC2>), (exec_resize_to_third<CV_HAL_INTER_LINEAR>)),
    TEST("Resize random downscale uint8, INTER_LINEAR, 2 channels", (test_resize_random_scale<CV_HAL_INTER_LINEAR, CV_8UC2>), (exec_resize_random_scale<CV_HAL_INTER_LINEAR>)),

    TEST("Resize0.5x0.5 uint8, INTER_AREA, 3 channels", (test_resize<500, CV_HAL_INTER_AREA, 5, 32, CV_8UC3>), (exec_resize<500, CV_HAL_INTER_AREA>)),
    TEST("Resize0.5x0.5 uint8, INTER_LINEAR, 3 channels", (test_resize<500, CV_HAL_INTER_LINEAR, 5, 32, CV_8UC3>), (exec_resize<500, CV_HAL_INTER_LINEAR>)),
    TEST("Resize0.777x0.777 uint8, INTER_LINEAR, 3 channels", (test_resize<777, CV_HAL_INTER_LINEAR, 21, 64, CV_8UC3>), (exec_resize<777, CV_HAL_INTER_LINEAR>)),
    TEST("Resize0.444x0.444 uint8, INTER_LINEAR, 3 channels", (test_resize<444, CV_HAL_INTER_LINEAR, 21, 64, CV_8UC3>), (exec_resize<444, CV_HAL_INTER_LINEAR>)),
    TEST("Resize0.333x0.333 uint8, INTER_LINEAR, 3 channels", (test_resize_to_third<CV_HAL_INTER_LINEAR, CV_8UC3>), (exec_resize_to_third<CV_HAL_INTER_LINEAR>)),
    TEST("Resize random downscale uint8, INTER_LINEAR, 3 channels", (test_resize_random_scale<CV_HAL_INTER_LINEAR, CV_8UC3>), (exec_resize_random_scale<CV_HAL_INTER_LINEAR>)),

    TEST("Resize0.5x0.5 uint8, INTER_AREA, 4 channels", (test_resize<500, CV_HAL_INTER_AREA, 5, 32, CV_8UC4>), (exec_resize<500, CV_HAL_INTER_AREA>)),
    TEST("Resize0.5x0.5 uint8, INTER_LINEAR, 4 channels", (test_resize<500, CV_HAL_INTER_LINEAR, 5, 32, CV_8UC4>), (exec_resize<500, CV_HAL_INTER_LINEAR>)),
};
  // clang-format on
  return tests;
}
