// SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <utility>

#include "kleidicv/ctypes.h"
#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/resize/resize_linear.h"
#include "kleidicv/utils.h"

#define KLEIDICV_DEFINE_C_API_ALL(name, called_name)       \
  KLEIDICV_MULTIVERSION_C_API_WITH_SME(                    \
      name, &kleidicv::neon::called_name,                  \
      KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::called_name), \
      &kleidicv::sme::called_name, &kleidicv::sme2::called_name)

KLEIDICV_DEFINE_C_API_ALL(kleidicv_resize_2x2_stripe_u8,
                          kleidicv_resize_2x2_stripe_u8);
KLEIDICV_DEFINE_C_API_ALL(kleidicv_resize_4x4_stripe_u8,
                          kleidicv_resize_4x4_stripe_u8);

#define DEFINE_RESIZE_API_ALL_VECLEN_16_64(R, CH)                       \
  KLEIDICV_MULTIVERSION_C_API_VECLEN(                                   \
      kleidicv_resize_##CH##ch_r##R##_stripe_u8,                        \
      (&kleidicv::neon::kleidicv_resize_generic_stripe_u8<R, CH>),      \
      KLEIDICV_SVE2_IMPL_IF(                                            \
          (&kleidicv::sve2::kleidicv_resize_generic_stripe_u8<R, CH>)), \
      (&kleidicv::sme::kleidicv_resize_generic_stripe_u8<R, CH>),       \
      (&kleidicv::sme2::kleidicv_resize_generic_stripe_u8<R, CH>), 16, 64);

DEFINE_RESIZE_API_ALL_VECLEN_16_64(2, 1);
DEFINE_RESIZE_API_ALL_VECLEN_16_64(3, 1);
DEFINE_RESIZE_API_ALL_VECLEN_16_64(2, 2);
DEFINE_RESIZE_API_ALL_VECLEN_16_64(3, 2);
DEFINE_RESIZE_API_ALL_VECLEN_16_64(2, 3);
DEFINE_RESIZE_API_ALL_VECLEN_16_64(3, 3);

KLEIDICV_DEFINE_C_API_ALL(kleidicv_resize_linear_stripe_f32,
                          kleidicv_resize_linear_stripe_f32);

KLEIDICV_MULTIVERSION_C_API_WITH_SME(
    kleidicv_resize_to_quarter_u8, &kleidicv::neon::resize_to_quarter_u8,
    KLEIDICV_SVE2_IMPL_IF(&kleidicv::sve2::resize_to_quarter_u8),
    &kleidicv::sme::resize_to_quarter_u8,
    KLEIDICV_SME2_IMPL_IF(&kleidicv::sme2::resize_to_quarter_u8));

namespace kleidicv {

template <bool kUseSME, auto &kFunction, auto &kSMEFunction, typename... Args>
inline kleidicv_error_t call_resize(Args &&...args) {
  if constexpr (kUseSME) {
    return kSMEFunction(std::forward<Args>(args)...);
  }
  return kFunction(std::forward<Args>(args)...);
}

#define CALL_RESIZE(NAME, ...) \
  call_resize<kUseSME, NAME, NAME##_sme>(__VA_ARGS__)

// This function is too complex, but disable the warning for now.
// NOLINTBEGIN(readability-function-cognitive-complexity)
template <bool kUseSME>
kleidicv_error_t resize_linear_stripe_u8(const uint8_t *src, size_t src_stride,
                                         size_t src_width, size_t src_height,
                                         size_t y_begin, size_t y_end,
                                         uint8_t *dst, size_t dst_stride,
                                         size_t dst_width, size_t dst_height,
                                         size_t channels) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);

  if (src_width == 0 || src_height == 0) {
    return KLEIDICV_OK;
  }

  if (src_width == 2 * dst_width && src_height == 2 * dst_height) {
    size_t src_begin = y_begin * 2;
    size_t src_end = std::min<size_t>(src_height, y_end * 2);
    size_t dst_begin = y_begin;

    return CALL_RESIZE(kleidicv_resize_to_quarter_u8,
                       src + src_begin * src_stride, src_stride, src_width,
                       src_end - src_begin, dst + dst_begin * dst_stride,
                       dst_stride, channels);
  }

  if (channels == 1) {
    if (src_width * 2 == dst_width && src_height * 2 == dst_height) {
      return CALL_RESIZE(kleidicv_resize_2x2_stripe_u8, src, src_stride,
                         src_width, src_height, y_begin, y_end, dst,
                         dst_stride);
    }
    if (src_width * 4 == dst_width && src_height * 4 == dst_height) {
      return CALL_RESIZE(kleidicv_resize_4x4_stripe_u8, src, src_stride,
                         src_width, src_height, y_begin, y_end, dst,
                         dst_stride);
    }

    if (dst_width * 2 >= src_width) {
      return CALL_RESIZE(kleidicv_resize_1ch_r2_stripe_u8, src, src_stride,
                         src_width, src_height, y_begin, y_end, dst, dst_stride,
                         dst_width, dst_height);
    }
    return CALL_RESIZE(kleidicv_resize_1ch_r3_stripe_u8, src, src_stride,
                       src_width, src_height, y_begin, y_end, dst, dst_stride,
                       dst_width, dst_height);
  }

  if (channels == 2) {
    if (dst_width * 2 >= src_width) {
      return CALL_RESIZE(kleidicv_resize_2ch_r2_stripe_u8, src, src_stride,
                         src_width, src_height, y_begin, y_end, dst, dst_stride,
                         dst_width, dst_height);
    }
    return CALL_RESIZE(kleidicv_resize_2ch_r3_stripe_u8, src, src_stride,
                       src_width, src_height, y_begin, y_end, dst, dst_stride,
                       dst_width, dst_height);
  }

  assert(channels == 3);
  double inverse_scale =
      static_cast<double>(src_width) / static_cast<double>(dst_width);
  // Loading 3 vectors and TBL3 is faster than loading extra lanes
  // Use the r3 variant over 1/1.8
  if (inverse_scale < 1.8) {
    return CALL_RESIZE(kleidicv_resize_3ch_r2_stripe_u8, src, src_stride,
                       src_width, src_height, y_begin, y_end, dst, dst_stride,
                       dst_width, dst_height);
  }
  if (inverse_scale < 2.8) {
    return CALL_RESIZE(kleidicv_resize_3ch_r3_stripe_u8, src, src_stride,
                       src_width, src_height, y_begin, y_end, dst, dst_stride,
                       dst_width, dst_height);
  }

  // SVE variant does not handle the rightmost lanes of b and d vectors for 3
  // channel images, so if over 2.8, use the Neon variant only
  return neon::kleidicv_resize_generic_stripe_u8<3, 3>(
      src, src_stride, src_width, src_height, y_begin, y_end, dst, dst_stride,
      dst_width, dst_height);
}
// NOLINTEND(readability-function-cognitive-complexity)

// Explicit instantiation is needed for kleidicv_thread_resize_linear_u8
template kleidicv_error_t resize_linear_stripe_u8<false>(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height, size_t channels);

template <bool kUseSME>
kleidicv_error_t resize_linear_u8(const uint8_t *src, size_t src_stride,
                                  size_t src_width, size_t src_height,
                                  uint8_t *dst, size_t dst_stride,
                                  size_t dst_width, size_t dst_height,
                                  size_t channels) {
  if (!resize_linear_u8_is_implemented(src_width, src_height, dst_width,
                                       dst_height, channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  // For upsampling, process by src rows
  // For resize_generic, process by dst rows
  size_t y_end = (src_width < dst_width) ? src_height : dst_height;

  return resize_linear_stripe_u8<kUseSME>(src, src_stride, src_width,
                                          src_height, 0, y_end, dst, dst_stride,
                                          dst_width, dst_height, channels);
}

template <auto &StripeFunction>
kleidicv_error_t resize_linear_f32(const float *src, size_t src_stride,
                                   size_t src_width, size_t src_height,
                                   float *dst, size_t dst_stride,
                                   size_t dst_width, size_t dst_height,
                                   size_t channels) {
  if (!resize_linear_f32_is_implemented(src_width, src_height, dst_width,
                                        dst_height, channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  return StripeFunction(src, src_stride, src_width, src_height, 0, src_height,
                        dst, dst_stride, dst_width, dst_height);
}

#undef CALL_RESIZE

}  // namespace kleidicv

extern "C" {

kleidicv_error_t kleidicv_resize_linear_u8(const uint8_t *src,
                                           size_t src_stride, size_t src_width,
                                           size_t src_height, uint8_t *dst,
                                           size_t dst_stride, size_t dst_width,
                                           size_t dst_height, size_t channels) {
  return kleidicv::resize_linear_u8<false>(src, src_stride, src_width,
                                           src_height, dst, dst_stride,
                                           dst_width, dst_height, channels);
}

kleidicv_error_t kleidicv_resize_linear_u8_sme(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels) {
  return kleidicv::resize_linear_u8<true>(src, src_stride, src_width,
                                          src_height, dst, dst_stride,
                                          dst_width, dst_height, channels);
}

kleidicv_error_t kleidicv_resize_linear_f32(const float *src, size_t src_stride,
                                            size_t src_width, size_t src_height,
                                            float *dst, size_t dst_stride,
                                            size_t dst_width, size_t dst_height,
                                            size_t channels) {
  return kleidicv::resize_linear_f32<kleidicv_resize_linear_stripe_f32>(
      src, src_stride, src_width, src_height, dst, dst_stride, dst_width,
      dst_height, channels);
}

kleidicv_error_t kleidicv_resize_linear_f32_sme(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    float *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels) {
  return kleidicv::resize_linear_f32<kleidicv_resize_linear_stripe_f32_sme>(
      src, src_stride, src_width, src_height, dst, dst_stride, dst_width,
      dst_height, channels);
}

}  // extern "C"
