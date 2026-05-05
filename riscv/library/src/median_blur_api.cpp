// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// median_blur_* are regular functions (KLEIDICV_FILTER_OP_MEDIAN expands to
// `kleidicv_error_t name(...)`). u8 3x3 binds to the real impl; everything
// else returns NOT_IMPLEMENTED.

#include <vector>

#include "median_blur_decls.h"
#include "multichannel_helper.h"
#include "validation_helper.h"

#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"

extern "C" kleidicv_error_t kleidicv_median_blur_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels < 1 || channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (kleidicv_error_t e =
          kleidicv::riscv_validation::check_image_size(width, height))
    return e;
  if (kernel_width != kernel_height) return KLEIDICV_ERROR_RANGE;
  if (kernel_width < 3 || (kernel_width & 1u) == 0)
    return KLEIDICV_ERROR_RANGE;
  if (border_type != KLEIDICV_BORDER_TYPE_REPLICATE)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;

  // Per-plane kernel: 3×3 hits the 9-element sorting net, anything else
  // (5×5, 7×7, …) goes through the generic quickselect path.
  auto run_plane = [&](const uint8_t *s, size_t ss, uint8_t *d, size_t ds,
                       size_t w, size_t h) -> kleidicv_error_t {
    if (kernel_width == 3) {
      return kleidicv::scalar::median_blur_3x3_u8(s, ss, d, ds, w, h);
    }
    return kleidicv::scalar::median_blur_generic_u8(s, ss, d, ds, w, h,
                                                      kernel_width);
  };

  if (channels == 1) {
    return run_plane(src, src_stride, dst, dst_stride, width, height);
  }
  std::vector<uint8_t> sp_storage(width * height * channels);
  std::vector<uint8_t> dp_storage(width * height * channels);
  uint8_t *sp[4] = {nullptr, nullptr, nullptr, nullptr};
  uint8_t *dp[4] = {nullptr, nullptr, nullptr, nullptr};
  for (size_t c = 0; c < channels; ++c) {
    sp[c] = sp_storage.data() + c * width * height;
    dp[c] = dp_storage.data() + c * width * height;
  }
  for (size_t y = 0; y < height; ++y) {
    uint8_t *row[4] = {sp[0] + y * width, sp[1] + y * width,
                       sp[2] + y * width, sp[3] + y * width};
    kleidicv::riscv_mc::deinterleave_row_u8(src + y * src_stride, row, width,
                                              channels);
  }
  for (size_t c = 0; c < channels; ++c) {
    kleidicv_error_t e = run_plane(sp[c], width, dp[c], width, width, height);
    if (e != KLEIDICV_OK) return e;
  }
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *row[4] = {dp[0] + y * width, dp[1] + y * width,
                             dp[2] + y * width, dp[3] + y * width};
    kleidicv::riscv_mc::interleave_row_u8(dst + y * dst_stride, row, width,
                                            channels);
  }
  return KLEIDICV_OK;
}
extern "C" kleidicv_error_t kleidicv_median_blur_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, kleidicv_border_type_t border_type) {
  return kleidicv_median_blur_u8(src, src_stride, dst, dst_stride, width,
                                 height, channels, kernel_width, kernel_height,
                                 border_type);
}

#define DEFINE_NOT_IMPL(name, T)                                             \
  extern "C" kleidicv_error_t name(const T *, size_t, T *, size_t, size_t,   \
                                   size_t, size_t, size_t, size_t,          \
                                   kleidicv_border_type_t) {                 \
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;                                   \
  }

DEFINE_NOT_IMPL(kleidicv_median_blur_s8, int8_t)
DEFINE_NOT_IMPL(kleidicv_median_blur_s8_sme, int8_t)
DEFINE_NOT_IMPL(kleidicv_median_blur_u16, uint16_t)
DEFINE_NOT_IMPL(kleidicv_median_blur_u16_sme, uint16_t)
DEFINE_NOT_IMPL(kleidicv_median_blur_s16, int16_t)
DEFINE_NOT_IMPL(kleidicv_median_blur_s16_sme, int16_t)
DEFINE_NOT_IMPL(kleidicv_median_blur_u32, uint32_t)
DEFINE_NOT_IMPL(kleidicv_median_blur_u32_sme, uint32_t)
DEFINE_NOT_IMPL(kleidicv_median_blur_s32, int32_t)
DEFINE_NOT_IMPL(kleidicv_median_blur_s32_sme, int32_t)
DEFINE_NOT_IMPL(kleidicv_median_blur_f32, float)
DEFINE_NOT_IMPL(kleidicv_median_blur_f32_sme, float)

#undef DEFINE_NOT_IMPL
