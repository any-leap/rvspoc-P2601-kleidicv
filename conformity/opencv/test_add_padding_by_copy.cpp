// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/hal/interface.h"
#include "tests.h"

#if MANAGER
namespace kleidicv::hal {
int add_padding_by_copy(const uchar* src_data, size_t src_step, int src_type,
                        int src_width, int src_height, int src_full_width,
                        int src_full_height, int src_roi_x, int src_roi_y,
                        uchar* dst_data, size_t dst_step, int top, int bottom,
                        int left, int right, int border_type,
                        const double border_value[4]);
}
#endif

namespace {

enum class BorderValueMode : uint8_t {
  kRandom,
  kUniform,
  kNonUniform,
};

struct AddPaddingByCopyMetadata {
  int roi_x;
  int roi_y;
  int roi_width;
  int roi_height;
  int top;
  int bottom;
  int left;
  int right;
  double border_value[4];
};

struct AddPaddingByCopyScenario {
  int whole_width;
  int whole_height;
  int roi_x;
  int roi_y;
  int roi_width;
  int roi_height;
  int top;
  int bottom;
  int left;
  int right;
};

struct PaddingCase {
  int top;
  int bottom;
  int left;
  int right;
};

#if MANAGER
constexpr std::array<int, 4> kGeneratedWidths = {1, 2, 5, 16};
constexpr std::array<int, 4> kGeneratedHeights = {1, 2, 5, 16};
constexpr std::array<PaddingCase, 6> kGeneratedPaddingCases = {{
    {0, 0, 0, 0},
    {1, 0, 0, 0},
    {0, 1, 3, 0},
    {2, 0, 0, 4},
    {2, 3, 4, 1},
    {5, 4, 3, 6},
}};
#endif

static int metadata_rows_for(size_t row_bytes) {
  return static_cast<int>((sizeof(AddPaddingByCopyMetadata) + row_bytes - 1) /
                          row_bytes);
}

static AddPaddingByCopyMetadata read_metadata(const cv::Mat& input) {
  AddPaddingByCopyMetadata metadata{};
  const int metadata_rows = metadata_rows_for(input.step);
  std::memcpy(&metadata, input.ptr(input.rows - metadata_rows),
              sizeof(metadata));
  return metadata;
}

#if MANAGER
static cv::Mat pack_input_with_metadata(
    const cv::Mat& whole, const AddPaddingByCopyMetadata& metadata) {
  const int metadata_rows = metadata_rows_for(whole.step);
  cv::Mat input(whole.rows + metadata_rows, whole.cols, whole.type());
  whole.copyTo(input.rowRange(0, whole.rows));
  std::memset(input.ptr(whole.rows), 0, metadata_rows * input.step);
  std::memcpy(input.ptr(whole.rows), &metadata, sizeof(metadata));
  return input;
}
#endif

template <typename T>
static void fill_random(cv::RNG& rng, cv::Mat& mat) {
  if constexpr (std::is_same_v<T, float>) {
    rng.fill(mat, cv::RNG::UNIFORM, -100.0f, 100.0f);
  } else {
    rng.fill(mat, cv::RNG::UNIFORM, std::numeric_limits<T>::min(),
             std::numeric_limits<T>::max());
  }
}

template <typename T>
static double sample_border_component(cv::RNG& rng) {
  if constexpr (std::is_same_v<T, float>) {
    return static_cast<double>(rng.uniform(-20.0f, 20.0f));
  } else {
    const int low = std::is_signed_v<T> ? -20 : 0;
    const int high = 21;
    return static_cast<double>(rng.uniform(low, high));
  }
}

template <typename T, size_t Channels>
static cv::Scalar make_border_value(cv::RNG& rng, BorderValueMode mode) {
  cv::Scalar value{};

  switch (mode) {
    case BorderValueMode::kUniform: {
      const double uniform_value = sample_border_component<T>(rng);
      value = cv::Scalar(uniform_value, uniform_value, uniform_value,
                         uniform_value);
      break;
    }
    case BorderValueMode::kNonUniform:
      if constexpr (std::is_same_v<T, float>) {
        value = cv::Scalar(-17.5, -3.25, 9.75, 18.5);
      } else if constexpr (std::is_signed_v<T>) {
        value = cv::Scalar(-17, -3, 5, 19);
      } else {
        value = cv::Scalar(1, 7, 13, 19);
      }
      break;
    case BorderValueMode::kRandom:
      for (int i = 0; i < 4; ++i) {
        value[i] = sample_border_component<T>(rng);
      }
      break;
  }

  if constexpr (Channels > 4) {
    if (mode != BorderValueMode::kNonUniform) {
      value[1] = value[0];
      value[2] = value[0];
      value[3] = value[0];
    }
  }

  return value;
}

#if MANAGER
static AddPaddingByCopyScenario make_generated_scenario(const int width,
                                                        const int height,
                                                        const PaddingCase& pad,
                                                        const bool submatrix) {
  if (submatrix) {
    return {width + 8, height + 7, 2,          2,        width,
            height,    pad.top,    pad.bottom, pad.left, pad.right};
  }

  return {width,  height,  0,          0,        width,
          height, pad.top, pad.bottom, pad.left, pad.right};
}
#endif

template <int BorderType, bool Isolated, bool UseSubmatrix>
static cv::Mat exec_add_padding_by_copy(cv::Mat& input);

#if MANAGER
template <typename TypeParam>
static bool matrices_match(cv::Mat& actual, cv::Mat& expected) {
  return !are_matrices_different<TypeParam>(0, actual, expected);
}

template <int BorderType, typename TypeParam, size_t Channels, bool Isolated,
          bool UseSubmatrix>
static bool run_add_padding_by_copy_case(
    int index, RecreatedMessageQueue& request_queue,
    RecreatedMessageQueue& reply_queue,
    const AddPaddingByCopyScenario& scenario, cv::RNG& rng,
    const BorderValueMode border_value_mode) {
  const int type = get_opencv_matrix_type<TypeParam, Channels>();
  cv::Mat whole(scenario.whole_height, scenario.whole_width, type);
  fill_random<TypeParam>(rng, whole);

  AddPaddingByCopyMetadata metadata{};
  metadata.roi_x = scenario.roi_x;
  metadata.roi_y = scenario.roi_y;
  metadata.roi_width = scenario.roi_width;
  metadata.roi_height = scenario.roi_height;
  metadata.top = scenario.top;
  metadata.bottom = scenario.bottom;
  metadata.left = scenario.left;
  metadata.right = scenario.right;

  const cv::Scalar border_value =
      make_border_value<TypeParam, Channels>(rng, border_value_mode);
  metadata.border_value[0] = border_value[0];
  metadata.border_value[1] = border_value[1];
  metadata.border_value[2] = border_value[2];
  metadata.border_value[3] = border_value[3];

  cv::Mat input = pack_input_with_metadata(whole, metadata);
  cv::Mat actual =
      exec_add_padding_by_copy<BorderType, Isolated, UseSubmatrix>(input);
  cv::Mat expected =
      get_expected_from_subordinate(index, request_queue, reply_queue, input);

  if (!matrices_match<TypeParam>(actual, expected)) {
    cv::Mat debug_input =
        whole(cv::Rect(metadata.roi_x, metadata.roi_y, metadata.roi_width,
                       metadata.roi_height));
    fail_print_matrices(debug_input.rows, debug_input.cols, debug_input, actual,
                        expected);
    return true;
  }

  return false;
}
#endif

template <int BorderType, bool Isolated, bool UseSubmatrix>
static cv::Mat exec_add_padding_by_copy(cv::Mat& input) {
  AddPaddingByCopyMetadata metadata = read_metadata(input);
  const int metadata_rows = metadata_rows_for(input.step);

  // Clone the packed payload so ROI parentage reflects only the synthetic
  // source image and not the extra metadata rows appended for transport.
  cv::Mat whole = input.rowRange(0, input.rows - metadata_rows).clone();
  cv::Mat roi = UseSubmatrix
                    ? whole(cv::Rect(metadata.roi_x, metadata.roi_y,
                                     metadata.roi_width, metadata.roi_height))
                    : whole;

  cv::Mat result;
  cv::copyMakeBorder(
      roi, result, metadata.top, metadata.bottom, metadata.left, metadata.right,
      BorderType | (Isolated ? cv::BORDER_ISOLATED : 0),
      cv::Scalar(metadata.border_value[0], metadata.border_value[1],
                 metadata.border_value[2], metadata.border_value[3]));
  return result;
}

}  // namespace

#if MANAGER
template <int BorderType, typename TypeParam, size_t Channels, bool Isolated,
          bool UseSubmatrix, BorderValueMode ValueMode>
bool test_add_padding_by_copy(int index, RecreatedMessageQueue& request_queue,
                              RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);

  for (int height : kGeneratedHeights) {
    for (int width : kGeneratedWidths) {
      for (const auto& pad : kGeneratedPaddingCases) {
        const AddPaddingByCopyScenario scenario =
            make_generated_scenario(width, height, pad, UseSubmatrix);

        if (run_add_padding_by_copy_case<BorderType, TypeParam, Channels,
                                         Isolated, UseSubmatrix>(
                index, request_queue, reply_queue, scenario, rng, ValueMode)) {
          return true;
        }
      }
    }
  }

  return false;
}

template <int BorderType, typename TypeParam, size_t Channels, bool Isolated,
          bool UseSubmatrix, BorderValueMode ValueMode, int WholeWidth,
          int WholeHeight, int RoiX, int RoiY, int RoiWidth, int RoiHeight,
          int Top, int Bottom, int Left, int Right>
bool test_add_padding_by_copy_exact_case(int index,
                                         RecreatedMessageQueue& request_queue,
                                         RecreatedMessageQueue& reply_queue) {
  cv::RNG rng(0);
  const AddPaddingByCopyScenario scenario{WholeWidth, WholeHeight, RoiX, RoiY,
                                          RoiWidth,   RoiHeight,   Top,  Bottom,
                                          Left,       Right};
  return run_add_padding_by_copy_case<BorderType, TypeParam, Channels, Isolated,
                                      UseSubmatrix>(
      index, request_queue, reply_queue, scenario, rng, ValueMode);
}

template <size_t Channels>
bool test_add_padding_by_copy_wrap_partial_borrow(
    int index, RecreatedMessageQueue& request_queue,
    RecreatedMessageQueue& reply_queue) {
  constexpr int kWholeWidth = 10;
  constexpr int kWholeHeight = 7;
  constexpr int kType = get_opencv_matrix_type<uint8_t, Channels>();

  cv::Mat whole(kWholeHeight, kWholeWidth, kType);
  for (int y = 0; y < whole.rows; ++y) {
    auto* row = whole.ptr<uint8_t>(y);
    for (int x = 0; x < whole.cols; ++x) {
      for (size_t c = 0; c < Channels; ++c) {
        row[x * Channels + c] = static_cast<uint8_t>(
            (y * whole.cols * static_cast<int>(Channels) +
             x * static_cast<int>(Channels) + static_cast<int>(c) * 17 + 13) %
            251);
      }
    }
  }

  AddPaddingByCopyMetadata metadata{};
  metadata.roi_x = 1;
  metadata.roi_y = 1;
  metadata.roi_width = 6;
  metadata.roi_height = 5;
  metadata.top = 2;
  metadata.bottom = 2;
  metadata.left = 3;
  metadata.right = 4;

  cv::Mat input = pack_input_with_metadata(whole, metadata);
  cv::Mat actual =
      exec_add_padding_by_copy<cv::BORDER_WRAP, false, true>(input);
  cv::Mat expected =
      get_expected_from_subordinate(index, request_queue, reply_queue, input);

  if (!matrices_match<uint8_t>(actual, expected)) {
    cv::Mat debug_input =
        whole(cv::Rect(metadata.roi_x, metadata.roi_y, metadata.roi_width,
                       metadata.roi_height));
    fail_print_matrices(debug_input.rows, debug_input.cols, debug_input, actual,
                        expected);
    return true;
  }

  return false;
}

template <typename TypeParam, size_t Channels>
bool test_add_padding_by_copy_hal_rejects_transparent_border() {
  cv::Mat input(9, 12, get_opencv_matrix_type<TypeParam, Channels>(),
                cv::Scalar::all(5));
  cv::Mat output(13, 17, input.type(), cv::Scalar::all(0));
  const double border_value[4] = {0.0, 0.0, 0.0, 0.0};

  return kleidicv::hal::add_padding_by_copy(
             input.ptr(), input.step, input.type(), input.cols, input.rows,
             input.cols, input.rows, 0, 0, output.ptr(), output.step, 1, 3, 2,
             3, CV_HAL_BORDER_TRANSPARENT,
             border_value) != CV_HAL_ERROR_NOT_IMPLEMENTED;
}
#endif

#define COPY_BORDER_TEST(border, type, channels, isolated, submatrix, label) \
  TEST("copyMakeBorder " #border ", " #channels " channel (" label           \
       "), isolated=" #isolated ", submatrix=" #submatrix,                   \
       (test_add_padding_by_copy<cv::border, type, channels, isolated,       \
                                 submatrix, BorderValueMode::kRandom>),      \
       (exec_add_padding_by_copy<cv::border, isolated, submatrix>))

#define COPY_BORDER_UNIFORM_TEST(border, type, channels, isolated, submatrix, \
                                 label)                                       \
  TEST("copyMakeBorder " #border ", " #channels " channel (" label            \
       "), isolated=" #isolated ", submatrix=" #submatrix,                    \
       (test_add_padding_by_copy<cv::border, type, channels, isolated,        \
                                 submatrix, BorderValueMode::kUniform>),      \
       (exec_add_padding_by_copy<cv::border, isolated, submatrix>))

#define COPY_BORDER_EXACT_TEST(name, border, type, channels, isolated,        \
                               submatrix, whole_width, whole_height, roi_x,   \
                               roi_y, roi_width, roi_height, top, bottom,     \
                               left, right)                                   \
  TEST(name,                                                                  \
       (test_add_padding_by_copy_exact_case<                                  \
           cv::border, type, channels, isolated, submatrix,                   \
           BorderValueMode::kRandom, whole_width, whole_height, roi_x, roi_y, \
           roi_width, roi_height, top, bottom, left, right>),                 \
       (exec_add_padding_by_copy<cv::border, isolated, submatrix>))

#define COPY_BORDER_UNIFORM_EXACT_TEST(name, border, type, channels, isolated, \
                                       submatrix, whole_width, whole_height,   \
                                       roi_x, roi_y, roi_width, roi_height,    \
                                       top, bottom, left, right)               \
  TEST(name,                                                                   \
       (test_add_padding_by_copy_exact_case<                                   \
           cv::border, type, channels, isolated, submatrix,                    \
           BorderValueMode::kUniform, whole_width, whole_height, roi_x, roi_y, \
           roi_width, roi_height, top, bottom, left, right>),                  \
       (exec_add_padding_by_copy<cv::border, isolated, submatrix>))

#define COPY_BORDER_NONUNIFORM_EXACT_TEST(                                 \
    name, border, type, channels, isolated, submatrix, whole_width,        \
    whole_height, roi_x, roi_y, roi_width, roi_height, top, bottom, left,  \
    right)                                                                 \
  TEST(name,                                                               \
       (test_add_padding_by_copy_exact_case<                               \
           cv::border, type, channels, isolated, submatrix,                \
           BorderValueMode::kNonUniform, whole_width, whole_height, roi_x, \
           roi_y, roi_width, roi_height, top, bottom, left, right>),       \
       (exec_add_padding_by_copy<cv::border, isolated, submatrix>))

std::vector<test>& add_padding_by_copy_tests_get() {
  static std::vector<test> tests = {
      COPY_BORDER_TEST(BORDER_CONSTANT, uint8_t, 1, false, false, "U8"),
      COPY_BORDER_TEST(BORDER_CONSTANT, uint8_t, 3, false, false, "U8"),
      COPY_BORDER_TEST(BORDER_CONSTANT, uint16_t, 2, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_CONSTANT, int16_t, 4, false, false, "S16"),
      COPY_BORDER_TEST(BORDER_CONSTANT, float, 1, false, false, "F32"),
      COPY_BORDER_UNIFORM_TEST(BORDER_CONSTANT, uint8_t, 5, false, false, "U8"),
      COPY_BORDER_UNIFORM_TEST(BORDER_CONSTANT, float, 6, false, false, "F32"),

      COPY_BORDER_TEST(BORDER_REPLICATE, uint8_t, 1, false, false, "U8"),
      COPY_BORDER_TEST(BORDER_REPLICATE, uint8_t, 3, false, false, "U8"),
      COPY_BORDER_TEST(BORDER_REFLECT, uint8_t, 1, false, false, "U8"),
      COPY_BORDER_TEST(BORDER_REFLECT, uint8_t, 3, false, false, "U8"),
      COPY_BORDER_TEST(BORDER_REFLECT_101, uint8_t, 1, false, false, "U8"),
      COPY_BORDER_TEST(BORDER_REFLECT_101, uint8_t, 3, false, false, "U8"),
      COPY_BORDER_TEST(BORDER_WRAP, uint8_t, 1, false, false, "U8"),
      COPY_BORDER_TEST(BORDER_WRAP, uint8_t, 3, false, false, "U8"),

      COPY_BORDER_TEST(BORDER_REPLICATE, uint16_t, 1, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REPLICATE, uint16_t, 2, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REPLICATE, uint16_t, 3, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REPLICATE, uint16_t, 4, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REFLECT, uint16_t, 1, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REFLECT, uint16_t, 2, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REFLECT, uint16_t, 3, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REFLECT, uint16_t, 4, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REFLECT_101, uint16_t, 1, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REFLECT_101, uint16_t, 2, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REFLECT_101, uint16_t, 3, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_REFLECT_101, uint16_t, 4, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_WRAP, uint16_t, 1, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_WRAP, uint16_t, 2, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_WRAP, uint16_t, 3, false, false, "U16"),
      COPY_BORDER_TEST(BORDER_WRAP, uint16_t, 4, false, false, "U16"),

      COPY_BORDER_TEST(BORDER_CONSTANT, uint8_t, 1, false, true, "U8"),
      COPY_BORDER_TEST(BORDER_CONSTANT, uint8_t, 1, true, true, "U8"),
      COPY_BORDER_TEST(BORDER_REPLICATE, uint8_t, 1, false, true, "U8"),
      COPY_BORDER_TEST(BORDER_REPLICATE, uint8_t, 1, true, true, "U8"),
      COPY_BORDER_TEST(BORDER_REFLECT, uint8_t, 1, false, true, "U8"),
      COPY_BORDER_TEST(BORDER_REFLECT, uint8_t, 1, true, true, "U8"),
      COPY_BORDER_TEST(BORDER_REFLECT_101, uint8_t, 1, false, true, "U8"),
      COPY_BORDER_TEST(BORDER_REFLECT_101, uint8_t, 1, true, true, "U8"),
      COPY_BORDER_TEST(BORDER_WRAP, uint8_t, 1, false, true, "U8"),
      COPY_BORDER_TEST(BORDER_WRAP, uint8_t, 3, false, true, "U8"),
      COPY_BORDER_TEST(BORDER_WRAP, uint8_t, 1, true, true, "U8"),
      COPY_BORDER_UNIFORM_TEST(BORDER_CONSTANT, uint8_t, 5, false, true, "U8"),

      TEST("copyMakeBorder BORDER_WRAP partial borrow, 1 channel (U8)",
           (test_add_padding_by_copy_wrap_partial_borrow<1>),
           (exec_add_padding_by_copy<cv::BORDER_WRAP, false, true>)),
      TEST("copyMakeBorder BORDER_WRAP partial borrow, 3 channel (U8)",
           (test_add_padding_by_copy_wrap_partial_borrow<3>),
           (exec_add_padding_by_copy<cv::BORDER_WRAP, false, true>)),

      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 1 channel (U16), "
          "vector-sized constant fill",
          BORDER_CONSTANT, uint16_t, 1, false, false, 65, 37, 0, 0, 65, 37, 7,
          5, 33, 31),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 3 channel (U16), "
          "vector-sized constant fill",
          BORDER_CONSTANT, uint16_t, 3, false, false, 41, 29, 0, 0, 41, 29, 9,
          11, 17, 19),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 3 channel (F32), "
          "vector-sized constant fill",
          BORDER_CONSTANT, float, 3, false, false, 63, 47, 0, 0, 63, 47, 7, 9,
          21, 23),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 4 channel (F32), "
          "large heap-backed constant row",
          BORDER_CONSTANT, float, 4, false, false, 129, 64, 0, 0, 129, 64, 17,
          19, 32, 33),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 1 channel (U8), "
          "single-channel repeat vector minus one",
          BORDER_CONSTANT, uint8_t, 1, false, false, 64, 11, 0, 0, 64, 11, 1, 2,
          15, 15),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 1 channel (U8), "
          "single-channel repeat exact vector",
          BORDER_CONSTANT, uint8_t, 1, false, false, 64, 11, 0, 0, 64, 11, 1, 2,
          16, 16),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 1 channel (U8), "
          "single-channel repeat vector plus one",
          BORDER_CONSTANT, uint8_t, 1, false, false, 64, 11, 0, 0, 64, 11, 1, 2,
          17, 17),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 3 channel (U8), "
          "interleaved repeat vector minus one",
          BORDER_CONSTANT, uint8_t, 3, false, false, 40, 13, 0, 0, 40, 13, 1, 2,
          15, 15),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 3 channel (U8), "
          "interleaved repeat exact vector",
          BORDER_CONSTANT, uint8_t, 3, false, false, 40, 13, 0, 0, 40, 13, 1, 2,
          16, 16),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 3 channel (U8), "
          "interleaved repeat vector plus one",
          BORDER_CONSTANT, uint8_t, 3, false, false, 40, 13, 0, 0, 40, 13, 1, 2,
          17, 17),

      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_REFLECT targeted, 1 channel (U8), "
          "large fast-path reflection",
          BORDER_REFLECT, uint8_t, 1, false, false, 255, 255, 0, 0, 255, 255,
          23, 21, 255, 255),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_REFLECT targeted, 1 channel (U8), "
          "single-pixel fallback",
          BORDER_REFLECT, uint8_t, 1, false, false, 1, 1, 0, 0, 1, 1, 9, 10, 7,
          8),

      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_REFLECT_101 targeted, 4 channel (U16), "
          "large fast-path reverse",
          BORDER_REFLECT_101, uint16_t, 4, false, false, 129, 64, 0, 0, 129, 64,
          29, 31, 32, 31),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_REFLECT_101 targeted, 4 channel (F32), "
          "equal-width fallback",
          BORDER_REFLECT_101, float, 4, false, false, 255, 1, 0, 0, 255, 1, 7,
          9, 255, 255),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_REFLECT_101 targeted, 1 channel (U8), "
          "single-channel reverse vector minus one",
          BORDER_REFLECT_101, uint8_t, 1, false, false, 64, 11, 0, 0, 64, 11, 1,
          2, 15, 15),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_REFLECT_101 targeted, 1 channel (U8), "
          "single-channel reverse exact vector",
          BORDER_REFLECT_101, uint8_t, 1, false, false, 64, 11, 0, 0, 64, 11, 1,
          2, 16, 16),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_REFLECT_101 targeted, 1 channel (U8), "
          "single-channel reverse vector plus one",
          BORDER_REFLECT_101, uint8_t, 1, false, false, 64, 11, 0, 0, 64, 11, 1,
          2, 17, 17),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_REFLECT_101 targeted, 3 channel (U8), "
          "interleaved reverse vector minus one",
          BORDER_REFLECT_101, uint8_t, 3, false, false, 40, 13, 0, 0, 40, 13, 1,
          2, 15, 15),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_REFLECT_101 targeted, 3 channel (U8), "
          "interleaved reverse exact vector",
          BORDER_REFLECT_101, uint8_t, 3, false, false, 40, 13, 0, 0, 40, 13, 1,
          2, 16, 16),
      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_REFLECT_101 targeted, 3 channel (U8), "
          "interleaved reverse vector plus one",
          BORDER_REFLECT_101, uint8_t, 3, false, false, 40, 13, 0, 0, 40, 13, 1,
          2, 17, 17),

      COPY_BORDER_EXACT_TEST(
          "copyMakeBorder BORDER_WRAP targeted, 4 channel (U16), "
          "oversized fallback",
          BORDER_WRAP, uint16_t, 4, false, false, 255, 63, 0, 0, 255, 63, 13,
          17, 256, 255),
      COPY_BORDER_UNIFORM_EXACT_TEST(
          "copyMakeBorder BORDER_CONSTANT targeted, 7 channel (U8), "
          "uniform scalar supported",
          BORDER_CONSTANT, uint8_t, 7, false, false, 17, 12, 0, 0, 17, 12, 4, 3,
          5, 2),
  };
  return tests;
}

#if MANAGER
std::vector<manager_only_test>& add_padding_by_copy_manager_only_tests_get() {
  static std::vector<manager_only_test> tests = {
      MANAGER_ONLY_TEST(
          "HAL copyMakeBorder rejects BORDER_TRANSPARENT (U8)",
          (test_add_padding_by_copy_hal_rejects_transparent_border<uint8_t,
                                                                   1>)),
      MANAGER_ONLY_TEST(
          "HAL copyMakeBorder rejects BORDER_TRANSPARENT (F32)",
          (test_add_padding_by_copy_hal_rejects_transparent_border<float, 3>)),
  };
  return tests;
}
#endif
