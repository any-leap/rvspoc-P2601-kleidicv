// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/generator.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv_thread/kleidicv_thread.h"
#include "multithreading_fake.h"

// Tuple of width, height, thread count, channel count.
typedef std::tuple<size_t, size_t, size_t, size_t> P;

class ResizeThread : public testing::TestWithParam<P> {
 public:
  void test_resize_to_quarter() {
    size_t dst_width = 0, dst_height = 0, thread_count = 0, channels = 0;
    std::tie(dst_width, dst_height, thread_count, channels) = GetParam();
    check<uint8_t>(kleidicv_resize_linear_u8, kleidicv_thread_resize_linear_u8,
                   thread_count, 2 * dst_width, 2 * dst_height, channels,
                   dst_width, dst_height);
  }
  void test_resize_u8_2x2() {
    size_t src_width = 0, src_height = 0, thread_count = 0, channels = 0;
    std::tie(src_width, src_height, thread_count, channels) = GetParam();
    if (channels <= 1) {
      check<uint8_t>(kleidicv_resize_linear_u8,
                     kleidicv_thread_resize_linear_u8, thread_count, src_width,
                     src_height, channels, 2 * src_width, 2 * src_height);
    }
  }
  void test_resize_u8_4x4() {
    size_t src_width = 0, src_height = 0, thread_count = 0, channels = 0;
    std::tie(src_width, src_height, thread_count, channels) = GetParam();
    if (channels <= 1) {
      check<uint8_t>(kleidicv_resize_linear_u8,
                     kleidicv_thread_resize_linear_u8, thread_count, src_width,
                     src_height, channels, 4 * src_width, 4 * src_height);
    }
  }
  void test_resize_u8_down2() {
    size_t width = 0, height = 0, thread_count = 0, channels = 0;
    std::tie(width, height, thread_count, channels) = GetParam();
    if (channels <= 3) {
      check<uint8_t>(kleidicv_resize_linear_u8,
                     kleidicv_thread_resize_linear_u8, thread_count, 16 * width,
                     3 * height, channels, 9 * width, 2 * height);
    }
  }
  void test_resize_u8_down3() {
    size_t width = 0, height = 0, thread_count = 0, channels = 0;
    std::tie(width, height, thread_count, channels) = GetParam();
    if (channels <= 3) {
      check<uint8_t>(kleidicv_resize_linear_u8,
                     kleidicv_thread_resize_linear_u8, thread_count, 32 * width,
                     3 * height, channels, 13 * width, 2 * height);
    }
  }

  void test_resize_f32_2x2() {
    size_t src_width = 0, src_height = 0, thread_count = 0, channels = 0;
    std::tie(src_width, src_height, thread_count, channels) = GetParam();
    if (channels <= 1) {
      check<float>(kleidicv_resize_linear_f32,
                   kleidicv_thread_resize_linear_f32, thread_count, src_width,
                   src_height, channels, 2 * src_width, 2 * src_height);
    }
  }
  void test_resize_f32_4x4() {
    size_t src_width = 0, src_height = 0, thread_count = 0, channels = 0;
    std::tie(src_width, src_height, thread_count, channels) = GetParam();
    if (channels <= 1) {
      check<float>(kleidicv_resize_linear_f32,
                   kleidicv_thread_resize_linear_f32, thread_count, src_width,
                   src_height, 1, 4 * src_width, 4 * src_height);
    }
  }
  void test_resize_f32_8x8() {
    size_t src_width = 0, src_height = 0, thread_count = 0, channels = 0;
    std::tie(src_width, src_height, thread_count, channels) = GetParam();
    if (channels <= 1) {
      check<float>(kleidicv_resize_linear_f32,
                   kleidicv_thread_resize_linear_f32, thread_count, src_width,
                   src_height, 1, 8 * src_width, 8 * src_height);
    }
  }

 private:
  template <typename ScalarType, typename SingleThreadedFunc,
            typename MultithreadedFunc>
  static void check(SingleThreadedFunc single_threaded_func,
                    MultithreadedFunc multithreaded_func, size_t thread_count,
                    size_t src_width, size_t src_height,
                    [[maybe_unused]] size_t channels, size_t dst_width,
                    size_t dst_height) {
    test::Array2D<ScalarType> src(src_width * channels, src_height),
        dst_single(dst_width * channels, dst_height),
        dst_multi(dst_width * channels, dst_height);
    test::PseudoRandomNumberGenerator<ScalarType> generator;
    src.fill(generator);

    EXPECT_EQ(KLEIDICV_OK, single_threaded_func(
                               src.data(), src.stride(), src_width, src_height,
                               dst_single.data(), dst_single.stride(),
                               dst_width, dst_height, channels));
    EXPECT_EQ(KLEIDICV_OK,
              multithreaded_func(
                  src.data(), src.stride(), src_width, src_height,
                  dst_multi.data(), dst_multi.stride(), dst_width, dst_height,
                  channels, get_multithreading_fake(thread_count)));

    if (dst_single.compare_to(dst_multi)) {
      std::cout << "source:\n";
      dump(&src);
      std::cout << "dst_single:\n";
      dump(&dst_single);
      std::cout << "dst_multi:\n";
      dump(&dst_multi);
    }

    EXPECT_EQ_ARRAY2D(dst_single, dst_multi);
  }
};

TEST_P(ResizeThread, ResizeToQuarter) { test_resize_to_quarter(); }

TEST_P(ResizeThread, ResizeUint2x2) { test_resize_u8_2x2(); }

TEST_P(ResizeThread, ResizeUint4x4) { test_resize_u8_4x4(); }

TEST_P(ResizeThread, ResizeUintDown2) { test_resize_u8_down2(); }

TEST_P(ResizeThread, ResizeUintDown3) { test_resize_u8_down3(); }

TEST_P(ResizeThread, ResizeFloat2x2) { test_resize_f32_2x2(); }

TEST_P(ResizeThread, ResizeFloat4x4) { test_resize_f32_4x4(); }

TEST_P(ResizeThread, ResizeFloat8x8) { test_resize_f32_8x8(); }

// clang-format off
INSTANTIATE_TEST_SUITE_P(, ResizeThread,
                         testing::Values(P{1, 1, 1, 1}, P{ 1,  2, 1, 2},
                                         P{1, 2, 2, 1}, P{ 2,  1, 2, 2}, P{ 2,  1, 2, 3},
                                         P{2, 2, 1, 1}, P{ 1,  3, 2, 2},
                                         P{2, 3, 1, 1}, P{ 6,  4, 1, 2}, P{ 6,  4, 1, 3},
                                         P{4, 5, 2, 1}, P{ 2,  6, 3, 2}, P{ 2,  6, 3, 3}, P{ 3,  5, 2, 4},
                                         P{1, 7, 4, 1}, P{12, 34, 5, 2}, P{ 7, 17, 5, 3}, P{11, 19, 3, 4}));
// clang-format on

TEST(ResizeThreadTest, NotImplemented) {
  for (size_t channels = 1; channels <= 3; ++channels) {
    // Too small images
    uint8_t src[1] = {}, dst[1] = {};
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_thread_resize_linear_u8(src, sizeof(src), 2, 2, dst,
                                               sizeof(dst), 3, 7, channels,
                                               get_multithreading_fake(2)));
  }

  {
    // Too small images
    float src[1] = {}, dst[1] = {};
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_thread_resize_linear_f32(src, sizeof(src), 2, 2, dst,
                                                sizeof(dst), 3, 7, 1,
                                                get_multithreading_fake(2)));
  }

  {
    // Invalid channel count
    size_t channels = 2;
    std::vector<uint8_t> src(2UL * 2UL * channels);
    std::vector<uint8_t> dst(4UL * 4UL * channels);
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_thread_resize_linear_u8(
                  src.data(), sizeof(uint8_t) * channels * 2, 2, 2, dst.data(),
                  sizeof(uint8_t) * channels * 4, 4, 4, channels,
                  get_multithreading_fake(2)));
  }

  {
    // Invalid channel count
    size_t channels = 4;
    std::vector<uint8_t> src(48UL * 2UL * channels);
    std::vector<uint8_t> dst(32UL * 1UL * channels);
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_thread_resize_linear_u8(
                  src.data(), sizeof(uint8_t) * channels * 48, 48, 2,
                  dst.data(), sizeof(uint8_t) * channels * 32, 32, 1, channels,
                  get_multithreading_fake(2)));
  }

  {
    // Invalid channel count
    size_t channels = 2;
    std::vector<float> src(2UL * 2UL * channels);
    std::vector<float> dst(4UL * 4UL * channels);
    EXPECT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_thread_resize_linear_f32(
                  src.data(), sizeof(float) * channels * 2, 2, 2, dst.data(),
                  sizeof(float) * channels * 4, 4, 4, channels,
                  get_multithreading_fake(2)));
  }
}
