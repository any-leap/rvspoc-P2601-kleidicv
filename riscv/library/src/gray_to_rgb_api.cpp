// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "gray_to_rgb_decls.h"
#include "validation_helper.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;
namespace V = kleidicv::riscv_validation;

kleidicv_error_t dispatch(const uint8_t *src, size_t src_stride, uint8_t *dst,
                            size_t dst_stride, size_t width, size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (kleidicv_error_t e = V::check_image_size(width, height)) return e;
  // In-place (src == dst) needs right-to-left expansion to avoid clobbering
  // the gray bytes we still have to read; the scalar impl handles that, the
  // RVV strip-mined path doesn't. Force scalar when aliased.
  if (src == dst) {
    return kleidicv::scalar::gray_to_rgb_u8(src, src_stride, dst, dst_stride,
                                              width, height);
  }
  return active_backend() == Backend::Rvv
             ? kleidicv::rvv::gray_to_rgb_u8(src, src_stride, dst, dst_stride,
                                              width, height)
             : kleidicv::scalar::gray_to_rgb_u8(src, src_stride, dst,
                                                 dst_stride, width, height);
}
}  // namespace

extern "C" {
kleidicv_error_t (*kleidicv_gray_to_rgb_u8)(const uint8_t *, size_t, uint8_t *,
                                            size_t, size_t, size_t) = dispatch;
kleidicv_error_t (*kleidicv_gray_to_rgb_u8_sme)(const uint8_t *, size_t,
                                                uint8_t *, size_t, size_t,
                                                size_t) = dispatch;
}  // extern "C"
