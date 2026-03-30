// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

class ResizeToQuarterTest final {
 public:
  explicit ResizeToQuarterTest(size_t channels, size_t padding)
      : channels_(channels), padding_(padding) {}

  void execute_test(size_t src_width, size_t src_height, size_t dst_width,
                    size_t dst_height) const {
    size_t src_logical_width = channels_ * src_width;
    size_t out_logical_width = channels_ * dst_width;

    test::Array2D<uint8_t> source{src_logical_width, src_height, padding_,
                                  channels_};
    test::Array2D<uint8_t> actual{out_logical_width, dst_height, padding_,
                                  channels_};
    actual.fill(42);
    test::Array2D<uint8_t> expected{out_logical_width, dst_height, padding_,
                                    channels_};

    // Set bottom-row and right-column to 0xFF, the rest to element position in
    // buffer
    for (size_t h = 0; h < src_height; ++h) {
      for (size_t w = 0; w < src_logical_width; ++w) {
        if (h >= src_height - 1 || w >= src_logical_width - channels_) {
          *source.at(h, w) = 0xFF;
        } else {
          *source.at(h, w) = (h * src_logical_width + w) % 0xFF;
        }
      }
    }

    calculate_expected(source, expected);

    ASSERT_EQ(KLEIDICV_OK, kleidicv_resize_linear_u8(
                               source.data(), source.stride(), src_width,
                               src_height, actual.data(), actual.stride(),
                               dst_width, dst_height, channels_));

    if (expected.compare_to(actual)) {
      std::cout << "source:\n";
      dump(&source);
      std::cout << "expected:\n";
      dump(&expected);
      std::cout << "actual:\n";
      dump(&actual);
    }

    EXPECT_EQ_ARRAY2D(actual, expected);
  }

 private:
  void calculate_expected(test::Array2D<uint8_t> &src,
                          test::Array2D<uint8_t> &expected) const {
    for (size_t dst_row = 0; dst_row < expected.height(); ++dst_row) {
      const size_t src_row = dst_row * 2;
      size_t src_idx = 0;

      for (size_t dst_idx = 0; dst_idx < expected.width();
           dst_idx += channels_, src_idx += 2 * channels_) {
        for (size_t channel = 0; channel < channels_; ++channel) {
          uint16_t value = *src.at(src_row, src_idx + channel) +
                           *src.at(src_row, src_idx + channels_ + channel) +
                           *src.at(src_row + 1, src_idx + channel) +
                           *src.at(src_row + 1, src_idx + channels_ + channel);

          *expected.at(dst_row, dst_idx + channel) =
              static_cast<uint8_t>((value + 2) / 4);
        }
      }
    }
  }

  size_t channels_;
  size_t padding_;
};

TEST(ResizeToQuarter, Channels1NoPadding) {
  ResizeToQuarterTest resize_test(1, 0);

  size_t src_width = 5 * test::Options::vector_lanes<uint8_t>() + 2;
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, Channels1Padding) {
  ResizeToQuarterTest resize_test(1, 1);

  size_t src_width = 5 * test::Options::vector_lanes<uint8_t>() + 2;
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, Channels2NoPadding) {
  ResizeToQuarterTest resize_test(2, 0);

  size_t src_width = 3 * test::Options::vector_lanes<uint8_t>() + 2;
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, Channels2Padding) {
  ResizeToQuarterTest resize_test(2, 1);

  size_t src_width = 3 * test::Options::vector_lanes<uint8_t>() + 2;
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, Channels3NoPadding) {
  ResizeToQuarterTest resize_test(3, 0);

  size_t src_width = 3 * test::Options::vector_lanes<uint8_t>() + 2;
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, Channels3Padding) {
  ResizeToQuarterTest resize_test(3, 1);

  size_t src_width = 3 * test::Options::vector_lanes<uint8_t>() + 2;
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, Channels4NoPadding) {
  ResizeToQuarterTest resize_test(4, 0);

  size_t src_width = 3 * test::Options::vector_lanes<uint8_t>() + 2;
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, Channels4Padding) {
  ResizeToQuarterTest resize_test(4, 1);

  size_t src_width = 3 * test::Options::vector_lanes<uint8_t>() + 2;
  // Set height to at least double 2x2 window height
  size_t src_height = 4;
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;
  resize_test.execute_test(src_width, src_height, dst_width, dst_height);
}

TEST(ResizeToQuarter, NullPointer) {
  const uint8_t src[4] = {};
  uint8_t dst[1];
  test::test_null_args(kleidicv_resize_linear_u8, src, 2, 2, 2, dst, 1, 1, 1,
                       1);
}

TEST(ResizeToQuarter, ZeroImageSize) {
  const uint8_t src[1] = {};
  uint8_t dst[1];

  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_resize_linear_u8(src, 1, 0, 1, dst, 1, 0, 1, 1));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_resize_linear_u8(src, 1, 1, 0, dst, 1, 1, 0, 1));
}
