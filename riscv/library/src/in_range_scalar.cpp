// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar in_range_u8: dst = (lower <= src <= upper) ? 0xFF : 0. Bounds are
// inclusive per the public docs.

#include <cstdint>

#include "kleidicv/kleidicv.h"

#include "elementwise_scalar.h"
#include "in_range_decls.h"

namespace kleidicv::scalar {

kleidicv_error_t in_range_u8(const uint8_t *src, size_t src_stride,
                             uint8_t *dst, size_t dst_stride, size_t width,
                             size_t height, uint8_t lower, uint8_t upper) {
  return unary_elementwise<uint8_t>(
      src, src_stride, dst, dst_stride, width, height,
      [lower, upper](uint8_t x) -> uint8_t {
        return (x >= lower && x <= upper) ? 0xFF : 0;
      });
}

// f32 input → u8 mask output: input/output sizes differ, so we don't reuse
// the unary_elementwise helper (it assumes a single T).
kleidicv_error_t in_range_f32(const float *src, size_t src_stride,
                              uint8_t *dst, size_t dst_stride, size_t width,
                              size_t height, float lower, float upper) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const float *row_src = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *row_dst = dst + y * dst_stride;
    for (size_t x = 0; x < width; ++x) {
      row_dst[x] = (row_src[x] >= lower && row_src[x] <= upper) ? 0xFF : 0;
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
