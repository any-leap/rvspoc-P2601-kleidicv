// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "gray_to_rgb_decls.h"

namespace kleidicv::scalar {

kleidicv_error_t gray_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  // The public API documents in-place support. When src and dst alias, each
  // iteration's write at rd[3x..3x+2] would clobber the gray bytes we still
  // need to read at src[x+1], src[x+2]. Processing right-to-left avoids that:
  // every read at src[x] happens before any write that could overlap byte
  // offsets ≥ x in the shared buffer.
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *row_src = src + y * src_stride;
    uint8_t *row_dst = dst + y * dst_stride;
    for (size_t x = width; x-- > 0;) {
      uint8_t g = row_src[x];
      row_dst[3 * x + 0] = g;
      row_dst[3 * x + 1] = g;
      row_dst[3 * x + 2] = g;
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
