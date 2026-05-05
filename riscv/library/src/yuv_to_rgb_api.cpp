// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Public dispatcher for kleidicv_yuv_to_rgb_u8. SPOC supports YUV444 only.

#include "dispatch.h"
#include "yuv_to_rgb_decls.h"
#include "validation_helper.h"

#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;

template <bool BGR, bool Alpha>
kleidicv_error_t do_yuv444(const uint8_t *src, size_t src_stride, uint8_t *dst,
                           size_t dst_stride, size_t width, size_t height) {
  if (active_backend() == Backend::Rvv) {
    return kleidicv::rvv::yuv444_to_rgb_u8<BGR, Alpha>(
        src, src_stride, dst, dst_stride, width, height);
  }
  return kleidicv::scalar::yuv444_to_rgb_u8<BGR, Alpha>(
      src, src_stride, dst, dst_stride, width, height);
}

}  // namespace

extern "C" kleidicv_error_t kleidicv_yuv_to_rgb_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t fmt) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (kleidicv_error_t e =
          kleidicv::riscv_validation::check_image_size(width, height))
    return e;
  constexpr unsigned kAllowed =
      static_cast<unsigned>(KLEIDICV_COLOR_CONVERSION_YUV_FMT_MASK) |
      static_cast<unsigned>(KLEIDICV_COLOR_CONVERSION_FLAG_BGR) |
      static_cast<unsigned>(KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA);
  if ((static_cast<unsigned>(fmt) & ~kAllowed) != 0u)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  unsigned base = static_cast<unsigned>(fmt) &
                  static_cast<unsigned>(KLEIDICV_COLOR_CONVERSION_YUV_FMT_MASK);
  if (base != KLEIDICV_COLOR_CONVERSION_FMT_YUV444) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  bool bgr =
      static_cast<unsigned>(fmt) & KLEIDICV_COLOR_CONVERSION_FLAG_BGR;
  bool alpha =
      static_cast<unsigned>(fmt) & KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA;
  if (bgr && alpha)
    return do_yuv444<true, true>(src, src_stride, dst, dst_stride, width,
                                 height);
  if (bgr)
    return do_yuv444<true, false>(src, src_stride, dst, dst_stride, width,
                                  height);
  if (alpha)
    return do_yuv444<false, true>(src, src_stride, dst, dst_stride, width,
                                  height);
  return do_yuv444<false, false>(src, src_stride, dst, dst_stride, width,
                                 height);
}

extern "C" kleidicv_error_t kleidicv_yuv_to_rgb_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t fmt) {
  return kleidicv_yuv_to_rgb_u8(src, src_stride, dst, dst_stride, width,
                                height, fmt);
}
