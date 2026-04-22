// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "../../kleidicv/src/transform/add_padding_by_copy_internal.h"
#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/transform/add_padding_by_copy.h"
#include "test_config.h"

namespace {

struct AddPaddingByCopyParams {
  size_t width;
  size_t height;
  size_t top;
  size_t bottom;
  size_t left;
  size_t right;
  size_t pixel_size;
  size_t src_padding;
  size_t dst_padding;
  kleidicv_border_type_t border_type;
};

constexpr std::array<kleidicv_border_type_t, 5> kBorderTypes = {
    KLEIDICV_BORDER_TYPE_CONSTANT, KLEIDICV_BORDER_TYPE_REPLICATE,
    KLEIDICV_BORDER_TYPE_REFLECT,  KLEIDICV_BORDER_TYPE_WRAP,
    KLEIDICV_BORDER_TYPE_REVERSE,
};

constexpr std::array<size_t, 5> kPixelSizes = {1, 2, 3, 4, 8};

int border_interpolate(int p, int len, kleidicv_border_type_t border_type) {
  if (0 <= p && p < len) {
    return p;
  }

  if (border_type == KLEIDICV_BORDER_TYPE_REPLICATE) {
    return p < 0 ? 0 : len - 1;
  }

  if (border_type == KLEIDICV_BORDER_TYPE_REFLECT ||
      border_type == KLEIDICV_BORDER_TYPE_REVERSE) {
    const int delta = border_type == KLEIDICV_BORDER_TYPE_REVERSE ? 1 : 0;
    if (len == 1) {
      return 0;
    }
    do {
      if (p < 0) {
        p = -p - 1 + delta;
      } else {
        p = len - 1 - (p - len) - delta;
      }
    } while (p < 0 || p >= len);
    return p;
  }

  if (border_type == KLEIDICV_BORDER_TYPE_WRAP) {
    if (p < 0) {
      p -= ((p - len + 1) / len) * len;
    }
    if (p >= len) {
      p %= len;
    }
    return p;
  }

  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT) {
    return -1;
  }

  return -2;
}

class AddPaddingByCopyTest : public testing::Test {
 protected:
  static std::array<uint8_t, 16> border_value() {
    return {11,  29,  47, 83, 101, 149, 173, 211,
            227, 241, 13, 31, 59,  71,  97,  131};
  }

  static void fill_source(test::Array2D<uint8_t>& src, size_t pixel_size) {
    src.fill([pixel_size](size_t row, size_t column) -> std::optional<uint8_t> {
      const size_t pixel = column / pixel_size;
      const size_t channel = column % pixel_size;
      return static_cast<uint8_t>((row * 37 + pixel * 19 + channel * 53 + 17) &
                                  0xff);
    });
  }

  static void calculate_expected(const test::Array2D<uint8_t>& src,
                                 test::Array2D<uint8_t>& expected,
                                 const AddPaddingByCopyParams& params,
                                 const uint8_t* border_value) {
    const size_t dst_width = params.width + params.left + params.right;
    const size_t dst_height = params.height + params.top + params.bottom;

    for (size_t dst_y = 0; dst_y < dst_height; ++dst_y) {
      const int src_y = border_interpolate(
          static_cast<int>(dst_y) - static_cast<int>(params.top),
          static_cast<int>(params.height), params.border_type);
      for (size_t dst_x = 0; dst_x < dst_width; ++dst_x) {
        const int src_x = border_interpolate(
            static_cast<int>(dst_x) - static_cast<int>(params.left),
            static_cast<int>(params.width), params.border_type);
        for (size_t byte = 0; byte < params.pixel_size; ++byte) {
          uint8_t value = border_value[byte];
          if (src_x >= 0 && src_y >= 0) {
            value =
                *src.at(static_cast<size_t>(src_y),
                        static_cast<size_t>(src_x) * params.pixel_size + byte);
          }
          *expected.at(dst_y, dst_x * params.pixel_size + byte) = value;
        }
      }
    }
  }

  static void run_reference_test(const AddPaddingByCopyParams& params) {
    SCOPED_TRACE(::testing::Message()
                 << "w=" << params.width << ", h=" << params.height
                 << ", top=" << params.top << ", bottom=" << params.bottom
                 << ", left=" << params.left << ", right=" << params.right
                 << ", pixel_size=" << params.pixel_size << ", src_padding="
                 << params.src_padding << ", dst_padding=" << params.dst_padding
                 << ", border_type=" << static_cast<int>(params.border_type));
    const size_t src_row_width = params.width * params.pixel_size;
    const size_t dst_width = params.width + params.left + params.right;
    const size_t dst_height = params.height + params.top + params.bottom;
    const size_t dst_row_width = dst_width * params.pixel_size;

    test::Array2D<uint8_t> src{src_row_width, params.height, params.src_padding,
                               params.pixel_size};
    test::Array2D<uint8_t> actual{dst_row_width, dst_height, params.dst_padding,
                                  params.pixel_size};
    test::Array2D<uint8_t> expected{dst_row_width, dst_height,
                                    params.dst_padding, params.pixel_size};

    const auto border = border_value();
    fill_source(src, params.pixel_size);
    actual.fill(0xa5);
    calculate_expected(src, expected, params, border.data());

    ASSERT_EQ(
        KLEIDICV_OK,
        kleidicv_add_padding_by_copy(
            src.data(), src.stride(), actual.data(), actual.stride(),
            params.width, params.height, params.top, params.bottom, params.left,
            params.right, params.pixel_size, params.border_type,
            params.border_type == KLEIDICV_BORDER_TYPE_CONSTANT ? border.data()
                                                                : nullptr));

    EXPECT_EQ_ARRAY2D(expected, actual);
  }

  struct RawBuffer {
    std::vector<uint8_t> storage;
    uint8_t* data;
  };

  static RawBuffer make_raw_buffer(size_t size, size_t alignment, bool aligned,
                                   uint8_t fill_value) {
    RawBuffer buffer{std::vector<uint8_t>(size + alignment, fill_value),
                     nullptr};
    const uintptr_t base = reinterpret_cast<uintptr_t>(buffer.storage.data());
    const uintptr_t aligned_base =
        (base + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
    size_t offset = static_cast<size_t>(aligned_base - base);
    if (!aligned) {
      offset += 1;
    }
    buffer.data = buffer.storage.data() + offset;
    return buffer;
  }

  static bool is_aligned(const uint8_t* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
  }

  static void copy_active_rows_to_raw(const test::Array2D<uint8_t>& src,
                                      uint8_t* dst, size_t dst_stride) {
    for (size_t row = 0; row < src.height(); ++row) {
      for (size_t column = 0; column < src.width(); ++column) {
        dst[row * dst_stride + column] = *src.at(row, column);
      }
    }
  }

  static void copy_active_rows_from_raw(const uint8_t* src, size_t src_stride,
                                        test::Array2D<uint8_t>& dst) {
    for (size_t row = 0; row < dst.height(); ++row) {
      for (size_t column = 0; column < dst.width(); ++column) {
        *dst.at(row, column) = src[row * src_stride + column];
      }
    }
  }

  static void expect_padding_untouched(const uint8_t* data, size_t stride,
                                       size_t active_width, size_t height,
                                       uint8_t padding_value) {
    for (size_t row = 0; row < height; ++row) {
      for (size_t offset = active_width; offset < stride; ++offset) {
        EXPECT_EQ(padding_value, data[row * stride + offset]);
      }
    }
  }

  static void run_indexed_alignment_test(size_t pixel_size, bool src_aligned,
                                         bool dst_aligned) {
    constexpr uint8_t kRawPaddingValue = 0xa5;
    const AddPaddingByCopyParams params{
        5, 1, 0, 0, 7, 6, pixel_size, 3, 5, KLEIDICV_BORDER_TYPE_REFLECT};
    const size_t alignment = pixel_size;
    const size_t src_row_width = params.width * params.pixel_size;
    const size_t dst_width = params.width + params.left + params.right;
    const size_t dst_row_width = dst_width * params.pixel_size;

    SCOPED_TRACE(::testing::Message()
                 << "indexed alignment test: pixel_size=" << pixel_size
                 << ", src_aligned=" << src_aligned
                 << ", dst_aligned=" << dst_aligned);

    test::Array2D<uint8_t> src_reference{src_row_width, params.height,
                                         params.src_padding, params.pixel_size};
    test::Array2D<uint8_t> actual{dst_row_width, params.height,
                                  params.dst_padding, params.pixel_size};
    test::Array2D<uint8_t> expected{dst_row_width, params.height,
                                    params.dst_padding, params.pixel_size};

    const auto border = border_value();
    fill_source(src_reference, params.pixel_size);
    actual.fill(0);
    calculate_expected(src_reference, expected, params, border.data());

    RawBuffer src = make_raw_buffer(src_reference.stride() * params.height,
                                    alignment, src_aligned, 0x3c);
    RawBuffer dst = make_raw_buffer(actual.stride() * params.height, alignment,
                                    dst_aligned, kRawPaddingValue);
    copy_active_rows_to_raw(src_reference, src.data, src_reference.stride());

    ASSERT_EQ(src_aligned, is_aligned(src.data, alignment));
    ASSERT_EQ(dst_aligned, is_aligned(dst.data, alignment));

    ASSERT_EQ(
        KLEIDICV_OK,
        kleidicv_add_padding_by_copy(
            src.data, src_reference.stride(), dst.data, actual.stride(),
            params.width, params.height, params.top, params.bottom, params.left,
            params.right, params.pixel_size, params.border_type, nullptr));

    copy_active_rows_from_raw(dst.data, actual.stride(), actual);
    EXPECT_EQ_ARRAY2D(expected, actual);
    expect_padding_untouched(dst.data, actual.stride(), dst_row_width,
                             params.height, kRawPaddingValue);
  }
};

TEST_F(AddPaddingByCopyTest, MatchesReferenceAcrossGeneratedCases) {
  const std::array<size_t, 3> widths = {1, 2,
                                        test::Options::vector_length() + 1};
  const std::array<size_t, 3> heights = {1, 3, 5};

  for (kleidicv_border_type_t border_type : kBorderTypes) {
    for (size_t pixel_size : kPixelSizes) {
      for (size_t width : widths) {
        for (size_t height : heights) {
          run_reference_test({
              width,
              height,
              1 + (height % 2),
              1 + (width % 3),
              1 + ((width + pixel_size) % 4),
              1 + ((height + pixel_size) % 5),
              pixel_size,
              (width + pixel_size) % 7,
              (height + pixel_size) % 11,
              border_type,
          });
        }
      }
    }
  }
}

TEST_F(AddPaddingByCopyTest, MatchesReferenceForIndexedCopyAlignmentPaths) {
  struct AlignmentCase {
    size_t pixel_size;
    bool src_aligned;
    bool dst_aligned;
  };

  const std::array<AlignmentCase, 12> cases = {{
      {sizeof(uint16_t), true, true},
      {sizeof(uint16_t), true, false},
      {sizeof(uint16_t), false, true},
      {sizeof(uint16_t), false, false},
      {sizeof(uint32_t), true, true},
      {sizeof(uint32_t), true, false},
      {sizeof(uint32_t), false, true},
      {sizeof(uint32_t), false, false},
      {sizeof(uint64_t), true, true},
      {sizeof(uint64_t), true, false},
      {sizeof(uint64_t), false, true},
      {sizeof(uint64_t), false, false},
  }};

  for (const auto& alignment_case : cases) {
    run_indexed_alignment_test(alignment_case.pixel_size,
                               alignment_case.src_aligned,
                               alignment_case.dst_aligned);
  }
}

TEST_F(AddPaddingByCopyTest, MatchesReferenceForBorderPresenceCombinations) {
  struct BorderCombinationCase {
    size_t top;
    size_t bottom;
    size_t left;
    size_t right;
  };

  const std::array<BorderCombinationCase, 8> combinations = {{
      {0, 0, 3, 0},
      {0, 0, 0, 4},
      {2, 0, 0, 0},
      {0, 3, 0, 0},
      {2, 3, 4, 5},
      {2, 3, 0, 0},
      {0, 0, 4, 5},
      {0, 0, 0, 0},
  }};

  const std::array<size_t, 5> pixel_sizes = {1, 2, 3, 4, 8};
  const size_t width = 6;
  const size_t height = 5;

  for (kleidicv_border_type_t border_type : kBorderTypes) {
    for (size_t combo_index = 0; combo_index < combinations.size();
         ++combo_index) {
      const auto& combination = combinations[combo_index];
      run_reference_test({
          width,
          height,
          combination.top,
          combination.bottom,
          combination.left,
          combination.right,
          pixel_sizes[combo_index % pixel_sizes.size()],
          (combo_index * 3) % 7,
          (combo_index * 5) % 11,
          border_type,
      });
    }
  }
}

TEST_F(AddPaddingByCopyTest, MatchesReferenceForSpecializedAndFallbackPaths) {
  const std::vector<AddPaddingByCopyParams> cases = {
      {3, 2, 1, 2, 2, 1, 1, 5, 7, KLEIDICV_BORDER_TYPE_CONSTANT},
      {5, 4, 2, 1, 3, 2, 3, 5, 1, KLEIDICV_BORDER_TYPE_CONSTANT},
      {4, 3, 2, 1, 3, 2, 6, 4, 5, KLEIDICV_BORDER_TYPE_CONSTANT},
      {3, 2, 1, 2, 2, 3, 12, 6, 8, KLEIDICV_BORDER_TYPE_CONSTANT},
      {2, 3, 1, 2, 2, 1, 16, 5, 7, KLEIDICV_BORDER_TYPE_CONSTANT},
      {385, 2, 1, 1, 1, 1, 8, 13, 17, KLEIDICV_BORDER_TYPE_CONSTANT},
      {1, 4, 3, 2, 5, 4, 2, 6, 9, KLEIDICV_BORDER_TYPE_REPLICATE},
      {7, 5, 1, 2, 3, 4, 1, 4, 6, KLEIDICV_BORDER_TYPE_WRAP},
      {7, 4, 2, 1, 10, 9, 3, 5, 0, KLEIDICV_BORDER_TYPE_WRAP},
      {6, 5, 1, 1, 5, 6, 4, 3, 2, KLEIDICV_BORDER_TYPE_REFLECT},
      {16, 3, 1, 1, 16, 16, 4, 2, 3, KLEIDICV_BORDER_TYPE_REFLECT},
      {16, 2, 1, 1, 16, 16, 6, 1, 4, KLEIDICV_BORDER_TYPE_REFLECT},
      {8, 2, 1, 1, 8, 8, 12, 3, 2, KLEIDICV_BORDER_TYPE_REFLECT},
      {4, 2, 1, 1, 4, 4, 16, 2, 1, KLEIDICV_BORDER_TYPE_REFLECT},
      {4, 3, 2, 1, 9, 8, 2, 0, 5, KLEIDICV_BORDER_TYPE_REFLECT},
      {6, 5, 2, 1, 4, 5, 1, 8, 4, KLEIDICV_BORDER_TYPE_REVERSE},
      {4, 3, 1, 2, 4, 6, 5, 2, 4, KLEIDICV_BORDER_TYPE_REVERSE},
      {3, 4, 1, 2, 17, 19, 1, 3, 5, KLEIDICV_BORDER_TYPE_WRAP},
      {5, 3, 2, 1, 18, 16, 4, 7, 2, KLEIDICV_BORDER_TYPE_REFLECT},
      {5, 4, 1, 3, 17, 18, 3, 1, 6, KLEIDICV_BORDER_TYPE_REVERSE},
      {1, 2, 3, 1, 12, 11, 2, 4, 3, KLEIDICV_BORDER_TYPE_REFLECT},
      {1, 3, 2, 2, 10, 9, 1, 0, 5, KLEIDICV_BORDER_TYPE_REVERSE},
  };

  for (const auto& params : cases) {
    run_reference_test(params);
  }
}

TEST_F(AddPaddingByCopyTest, MatchesReferenceForWrapFastPathLayouts) {
  const std::vector<AddPaddingByCopyParams> cases = {
      {17, 5, 1, 2, 3, 0, 1, 2, 4, KLEIDICV_BORDER_TYPE_WRAP},
      {17, 5, 1, 2, 0, 3, 1, 4, 2, KLEIDICV_BORDER_TYPE_WRAP},
      {21, 4, 2, 1, 5, 4, 1, 1, 3, KLEIDICV_BORDER_TYPE_WRAP},
      {19, 3, 1, 1, 7, 6, 2, 5, 0, KLEIDICV_BORDER_TYPE_WRAP},
      {13, 4, 1, 2, 3, 4, 3, 2, 1, KLEIDICV_BORDER_TYPE_WRAP},
      {11, 3, 2, 1, 4, 5, 6, 3, 2, KLEIDICV_BORDER_TYPE_WRAP},
      {9, 3, 1, 2, 2, 3, 12, 4, 1, KLEIDICV_BORDER_TYPE_WRAP},
      {7, 4, 1, 1, 1, 2, 8, 0, 5, KLEIDICV_BORDER_TYPE_WRAP},
  };

  for (const auto& params : cases) {
    run_reference_test(params);
  }
}

TEST_F(AddPaddingByCopyTest, MatchesReferenceForFunctionCoverageLoopShapes) {
  const std::vector<AddPaddingByCopyParams> cases = {
      {32, 2, 1, 0, 0, 0, 1, 3, 5, KLEIDICV_BORDER_TYPE_CONSTANT},
      {16, 2, 1, 0, 0, 0, 6, 1, 2, KLEIDICV_BORDER_TYPE_CONSTANT},
      {5, 2, 1, 0, 0, 0, 12, 2, 3, KLEIDICV_BORDER_TYPE_CONSTANT},
      {96, 2, 1, 0, 96, 0, 1, 3, 5, KLEIDICV_BORDER_TYPE_REFLECT},
      {48, 2, 1, 0, 48, 0, 2, 3, 5, KLEIDICV_BORDER_TYPE_REFLECT},
      {24, 2, 1, 0, 24, 0, 4, 3, 5, KLEIDICV_BORDER_TYPE_REFLECT},
      {8, 2, 1, 0, 8, 0, 8, 3, 5, KLEIDICV_BORDER_TYPE_REFLECT},
      {32, 2, 1, 0, 32, 0, 3, 3, 5, KLEIDICV_BORDER_TYPE_REFLECT},
      {9, 2, 1, 0, 9, 0, 6, 3, 5, KLEIDICV_BORDER_TYPE_REFLECT},
      {5, 2, 1, 0, 5, 0, 12, 3, 5, KLEIDICV_BORDER_TYPE_REFLECT},
  };

  for (const auto& params : cases) {
    run_reference_test(params);
  }
}

TEST_F(AddPaddingByCopyTest, MatchesReferenceForLargeVerticalPaddingModes) {
  const std::vector<AddPaddingByCopyParams> cases = {
      {7, 3, 19, 17, 2, 3, 1, 4, 5, KLEIDICV_BORDER_TYPE_WRAP},
      {8, 2, 21, 18, 6, 5, 4, 2, 3, KLEIDICV_BORDER_TYPE_REFLECT},
      {9, 2, 22, 19, 6, 5, 2, 1, 4, KLEIDICV_BORDER_TYPE_REVERSE},
  };

  for (const auto& params : cases) {
    run_reference_test(params);
  }
}

TEST_F(AddPaddingByCopyTest, MatchesReferenceForSinglePixelDimensions) {
  const std::vector<AddPaddingByCopyParams> cases = {
      {1, 1, 11, 10, 9, 8, 2, 4, 3, KLEIDICV_BORDER_TYPE_WRAP},
      {1, 4, 9, 8, 13, 12, 4, 1, 2, KLEIDICV_BORDER_TYPE_REFLECT},
      {5, 1, 7, 9, 4, 5, 3, 2, 1, KLEIDICV_BORDER_TYPE_REVERSE},
      {1, 1, 12, 11, 10, 9, 1, 0, 5, KLEIDICV_BORDER_TYPE_REVERSE},
  };

  for (const auto& params : cases) {
    run_reference_test(params);
  }
}

TEST_F(AddPaddingByCopyTest, AllowsZeroSizedConstantImages) {
  const auto border = border_value();

  {
    const AddPaddingByCopyParams params{
        0, 3, 2, 1, 4, 3, 3, 0, 5, KLEIDICV_BORDER_TYPE_CONSTANT};
    test::Array2D<uint8_t> src{1, params.height, 0, 1};
    test::Array2D<uint8_t> actual{
        (params.left + params.right) * params.pixel_size,
        params.height + params.top + params.bottom, params.dst_padding,
        params.pixel_size};
    test::Array2D<uint8_t> expected = actual;

    actual.fill(0xa5);
    calculate_expected(src, expected, params, border.data());

    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_add_padding_by_copy(
                  src.data(), src.stride(), actual.data(), actual.stride(),
                  params.width, params.height, params.top, params.bottom,
                  params.left, params.right, params.pixel_size,
                  params.border_type, border.data()));
    EXPECT_EQ_ARRAY2D(expected, actual);
  }

  {
    const AddPaddingByCopyParams params{
        4, 0, 2, 3, 1, 2, 4, 0, 3, KLEIDICV_BORDER_TYPE_CONSTANT};
    test::Array2D<uint8_t> src{params.width * params.pixel_size, 1,
                               params.src_padding, params.pixel_size};
    test::Array2D<uint8_t> actual{
        (params.width + params.left + params.right) * params.pixel_size,
        params.top + params.bottom, params.dst_padding, params.pixel_size};
    test::Array2D<uint8_t> expected = actual;

    actual.fill(0xa5);
    calculate_expected(src, expected, params, border.data());

    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_add_padding_by_copy(
                  src.data(), src.stride(), actual.data(), actual.stride(),
                  params.width, params.height, params.top, params.bottom,
                  params.left, params.right, params.pixel_size,
                  params.border_type, border.data()));
    EXPECT_EQ_ARRAY2D(expected, actual);
  }
}

TEST_F(AddPaddingByCopyTest, ReturnsErrorForInvalidArguments) {
  test::Array2D<uint8_t> src{8, 4, 3, 2};
  test::Array2D<uint8_t> dst{16, 7, 5, 2};
  const auto border = border_value();

  test::test_null_args(kleidicv_add_padding_by_copy, src.data(), src.stride(),
                       dst.data(), dst.stride(), 4U, 4U, 1U, 2U, 1U, 2U, 2U,
                       KLEIDICV_BORDER_TYPE_CONSTANT, border.data());

  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 4, 4, 1, 2,
                1, 2, 2, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 4, 4, 1, 2,
                1, 2, 0, KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 4, 4, 1, 2,
                1, 2, 2, KLEIDICV_BORDER_TYPE_TRANSPARENT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 0, 4, 1, 2,
                1, 2, 2, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr));

  EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
            kleidicv_add_padding_by_copy(src.data(), src.stride(), dst.data(),
                                         dst.stride(), 4, 0, 1, 2, 1, 2, 2,
                                         KLEIDICV_BORDER_TYPE_WRAP, nullptr));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 1, 1, 0, 0,
                KLEIDICV_MAX_IMAGE_PIXELS, 1, 1, KLEIDICV_BORDER_TYPE_CONSTANT,
                border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 1, 1,
                KLEIDICV_MAX_IMAGE_PIXELS, 1, 0, 0, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(),
                KLEIDICV_MAX_IMAGE_PIXELS - 29, 1, 0, 0, 10, 20, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 1,
                KLEIDICV_MAX_IMAGE_PIXELS - 29, 10, 20, 0, 0, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 1, 1,
                KLEIDICV_MAX_IMAGE_PIXELS - 1, 0, KLEIDICV_MAX_IMAGE_PIXELS - 1,
                0, 1, KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 1, 1, 0,
                KLEIDICV_MAX_IMAGE_PIXELS - 1, 0, KLEIDICV_MAX_IMAGE_PIXELS - 1,
                1, KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(),
                KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 0, 0, 0, 0, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(),
                KLEIDICV_MAX_IMAGE_PIXELS, 1, 1, 0, 0, 0, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(),
                KLEIDICV_MAX_IMAGE_PIXELS / 2 + 1, 1, 0, 1, 0, 0, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 1,
                KLEIDICV_MAX_IMAGE_PIXELS, 0, 0, 1, 0, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(),
                KLEIDICV_MAX_IMAGE_PIXELS, KLEIDICV_MAX_IMAGE_PIXELS, 0, 0, 0,
                0, 1, KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(),
                std::numeric_limits<size_t>::max(), 1, 0, 0, 1, 0, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 1, 1, 0, 0,
                std::numeric_limits<size_t>::max(), 0, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(),
                std::numeric_limits<size_t>::max() - 1, 1, 0, 0, 1, 1, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 1, 1,
                std::numeric_limits<size_t>::max(), 0, 0, 0, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_add_padding_by_copy(
                src.data(), src.stride(), dst.data(), dst.stride(), 1,
                std::numeric_limits<size_t>::max() - 1, 1, 1, 0, 0, 1,
                KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));
}

#ifdef KLEIDICV_ALLOCATION_TESTS
TEST_F(AddPaddingByCopyTest, ReturnsRangeWhenHeapAllocationFails) {
  const auto border = border_value();
  constexpr size_t kIndexedBorderElements =
      kleidicv::kAddPaddingByCopyPreparedBorderInlineWords + 1;
  constexpr size_t kIndexedLeftPadding = kIndexedBorderElements / 2;
  constexpr size_t kIndexedRightPadding =
      kIndexedBorderElements - kIndexedLeftPadding;

  {
    test::Array2D<uint8_t> src{3100, 2};
    test::Array2D<uint8_t> dst{3102, 2};
    MockMallocToFail::enable();
    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              kleidicv_add_padding_by_copy(
                  src.data(), src.stride(), dst.data(), dst.stride(), 3100, 2,
                  0, 0, 1, 1, 1, KLEIDICV_BORDER_TYPE_CONSTANT, border.data()));
    MockMallocToFail::disable();
  }

  {
    test::Array2D<uint8_t> src{3, 2};
    test::Array2D<uint8_t> dst{3 + kIndexedLeftPadding + kIndexedRightPadding,
                               2};
    MockMallocToFail::enable();
    EXPECT_EQ(KLEIDICV_ERROR_RANGE,
              kleidicv_add_padding_by_copy(
                  src.data(), src.stride(), dst.data(), dst.stride(), 3, 2, 0,
                  0, kIndexedLeftPadding, kIndexedRightPadding, 1,
                  KLEIDICV_BORDER_TYPE_WRAP, nullptr));
    MockMallocToFail::disable();
  }
}
#endif

}  // namespace
