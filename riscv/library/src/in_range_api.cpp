// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "in_range_decls.h"
#include "validation_helper.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;
namespace V = kleidicv::riscv_validation;

kleidicv_error_t in_range_u8_dispatch(const uint8_t *src, size_t src_stride,
                                        uint8_t *dst, size_t dst_stride,
                                        size_t width, size_t height,
                                        uint8_t lo, uint8_t hi) {
  // u8 path goes through the elementwise helper which already validates
  // image-size + alignment; we still null-check here for symmetry.
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  return active_backend() == Backend::Rvv
             ? kleidicv::rvv::in_range_u8(src, src_stride, dst, dst_stride,
                                            width, height, lo, hi)
             : kleidicv::scalar::in_range_u8(src, src_stride, dst, dst_stride,
                                               width, height, lo, hi);
}

kleidicv_error_t in_range_f32_dispatch(const float *src, size_t src_stride,
                                         uint8_t *dst, size_t dst_stride,
                                         size_t width, size_t height, float lo,
                                         float hi) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (kleidicv_error_t e = V::check_image_size(width, height)) return e;
  if (kleidicv_error_t e = V::check_buffer_alignment<float>(src, src_stride))
    return e;
  return active_backend() == Backend::Rvv
             ? kleidicv::rvv::in_range_f32(src, src_stride, dst, dst_stride,
                                             width, height, lo, hi)
             : kleidicv::scalar::in_range_f32(src, src_stride, dst,
                                                dst_stride, width, height, lo,
                                                hi);
}
}  // namespace

extern "C" {
kleidicv_error_t (*kleidicv_in_range_u8)(const uint8_t *, size_t, uint8_t *,
                                         size_t, size_t, size_t, uint8_t,
                                         uint8_t) = in_range_u8_dispatch;
kleidicv_error_t (*kleidicv_in_range_u8_sme)(const uint8_t *, size_t,
                                             uint8_t *, size_t, size_t, size_t,
                                             uint8_t,
                                             uint8_t) = in_range_u8_dispatch;
kleidicv_error_t (*kleidicv_in_range_f32)(const float *, size_t, uint8_t *,
                                          size_t, size_t, size_t, float,
                                          float) = in_range_f32_dispatch;
kleidicv_error_t (*kleidicv_in_range_f32_sme)(
    const float *, size_t, uint8_t *, size_t, size_t, size_t, float,
    float) = in_range_f32_dispatch;
}  // extern "C"
