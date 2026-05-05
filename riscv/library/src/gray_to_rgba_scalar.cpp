// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "gray_to_rgba_decls.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {

kleidicv_error_t gray_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *rs = src + y * src_stride;
    uint8_t *rd = dst + y * dst_stride;
    for (size_t x = 0; x < width; ++x) {
      uint8_t g = rs[x];
      rd[4 * x + 0] = g;
      rd[4 * x + 1] = g;
      rd[4 * x + 2] = g;
      rd[4 * x + 3] = 0xFF;
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
