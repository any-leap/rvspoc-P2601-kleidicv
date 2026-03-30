// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RESIZE_RESIZE_LINEAR_H
#define KLEIDICV_RESIZE_RESIZE_LINEAR_H

#include <algorithm>
#include <array>
#include <cstddef>

#include "kleidicv/kleidicv.h"

#define KLEIDICV_RESIZE_MAX_WIDTH_OR_HEIGHT ((1ULL << 24) - 1)

namespace kleidicv {

inline bool resize_linear_u8_is_implemented(size_t src_width, size_t src_height,
                                            size_t dst_width, size_t dst_height,
                                            size_t channels) {
  if (src_width > KLEIDICV_RESIZE_MAX_WIDTH_OR_HEIGHT ||
      src_height > KLEIDICV_RESIZE_MAX_WIDTH_OR_HEIGHT ||
      dst_width > KLEIDICV_RESIZE_MAX_WIDTH_OR_HEIGHT ||
      dst_height > KLEIDICV_RESIZE_MAX_WIDTH_OR_HEIGHT) {
    return false;
  }

  if (src_width == 0 || src_height == 0) {
    return true;
  }

  if (src_width == 2 * dst_width && src_height == 2 * dst_height &&
      channels >= 1 && channels <= 4) {
    return true;
  }

  if (channels == 1) {
    const std::array<size_t, 2> implemented_ratios = {2, 4};
    if (std::any_of(implemented_ratios.begin(), implemented_ratios.end(),
                    [&](size_t ratio) {
                      return src_width * ratio == dst_width &&
                             src_height * ratio == dst_height;
                    })) {
      return true;
    }
  }

  // Downsize between horizontal ratios of 1/3 and 1/1: a minimal width is
  // needed to execute vector operations
  if (channels <= 3 && dst_width * 3 >= src_width && dst_width < src_width &&
      dst_height < src_height) {
    if (dst_width * channels < 8) {
      return false;
    }
    if ((dst_width * 2 >= src_width) && (channels != 3)) {
      return src_width * channels >= 16;
    }
    return src_width * channels >= 32;
  }

  return false;
}

inline bool resize_linear_f32_is_implemented(size_t src_width,
                                             size_t src_height,
                                             size_t dst_width,
                                             size_t dst_height,
                                             size_t channels) {
  if (src_width > KLEIDICV_RESIZE_MAX_WIDTH_OR_HEIGHT ||
      src_height > KLEIDICV_RESIZE_MAX_WIDTH_OR_HEIGHT ||
      dst_width > KLEIDICV_RESIZE_MAX_WIDTH_OR_HEIGHT ||
      dst_height > KLEIDICV_RESIZE_MAX_WIDTH_OR_HEIGHT) {
    return false;
  }

  if (channels != 1) {
    return false;
  }

  if (src_width == 0 || src_height == 0) {
    return true;
  }
  const std::array<size_t, 3> implemented_ratios = {2, 4, 8};
  return std::any_of(implemented_ratios.begin(), implemented_ratios.end(),
                     [&](size_t ratio) {
                       return src_width * ratio == dst_width &&
                              src_height * ratio == dst_height;
                     });
}

namespace neon {
kleidicv_error_t resize_to_quarter_u8(const uint8_t *src, size_t src_stride,
                                      size_t src_width, size_t src_height,
                                      uint8_t *dst, size_t dst_stride,
                                      size_t channels);
kleidicv_error_t kleidicv_resize_2x2_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride);
kleidicv_error_t kleidicv_resize_4x4_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride);
template <int kRatio, int kChannels>
kleidicv_error_t kleidicv_resize_generic_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);
kleidicv_error_t kleidicv_resize_linear_stripe_f32(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);
}  // namespace neon

namespace sve2 {
kleidicv_error_t resize_to_quarter_u8(const uint8_t *src, size_t src_stride,
                                      size_t src_width, size_t src_height,
                                      uint8_t *dst, size_t dst_stride,
                                      size_t channels);
kleidicv_error_t kleidicv_resize_2x2_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride);
kleidicv_error_t kleidicv_resize_4x4_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride);
template <int kRatio, int kChannels>
kleidicv_error_t kleidicv_resize_generic_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);
kleidicv_error_t kleidicv_resize_linear_stripe_f32(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);
}  // namespace sve2

namespace sme {
kleidicv_error_t resize_to_quarter_u8(const uint8_t *src, size_t src_stride,
                                      size_t src_width, size_t src_height,
                                      uint8_t *dst, size_t dst_stride,
                                      size_t channels);
kleidicv_error_t kleidicv_resize_2x2_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride);
kleidicv_error_t kleidicv_resize_4x4_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride);
template <int kRatio, int kChannels>
kleidicv_error_t kleidicv_resize_generic_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);
kleidicv_error_t kleidicv_resize_linear_stripe_f32(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);
}  // namespace sme

namespace sme2 {
kleidicv_error_t resize_to_quarter_u8(const uint8_t *src, size_t src_stride,
                                      size_t src_width, size_t src_height,
                                      uint8_t *dst, size_t dst_stride,
                                      size_t channels);
kleidicv_error_t kleidicv_resize_2x2_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride);
kleidicv_error_t kleidicv_resize_4x4_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride);
template <int kRatio, int kChannels>
kleidicv_error_t kleidicv_resize_generic_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);
kleidicv_error_t kleidicv_resize_linear_stripe_f32(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);
}  // namespace sme2

template <bool kUseSME>
kleidicv_error_t resize_linear_stripe_u8(const uint8_t *src, size_t src_stride,
                                         size_t src_width, size_t src_height,
                                         size_t y_begin, size_t y_end,
                                         uint8_t *dst, size_t dst_stride,
                                         size_t dst_width, size_t dst_height,
                                         size_t channels);

}  // namespace kleidicv

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Internal - not part of the public API and its direct use is not
/// supported. It is used by the multithreaded function.
extern kleidicv_error_t (*kleidicv_resize_linear_stripe_f32)(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // KLEIDICV_RESIZE_RESIZE_H
