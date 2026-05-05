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

  // The public API documents in-place support. With src == dst:
  // * within a row, writes at rd[3x..3x+2] would clobber gray bytes at
  //   src[x+1], src[x+2] — process right-to-left.
  // * across rows, writes to row 0's RGB span [0..3·src_stride) clobber
  //   row 1's gray bytes if src_stride < 3·src_stride (i.e. dst_stride >
  //   src_stride, the typical in-place case) — process rows bottom-to-top.
  // Doing both reversals: every read at (y, x) happens before any write
  // that could ever overlap the byte at src[(y, x)].
  for (size_t y = height; y-- > 0;) {
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
