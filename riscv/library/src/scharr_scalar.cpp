// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scharr 3x3 interleaved (dx,dy) — no border, output is (w-2) × (h-2).
//   Gx = [-3 0 3; -10 0 10; -3 0 3]
//   Gy = [-3 -10 -3; 0 0 0; 3 10 3]

#include <cstdint>

#include "kleidicv/kleidicv.h"

#include "scharr_decls.h"

namespace kleidicv::scalar {

kleidicv_error_t scharr_interleaved_s16_u8(const uint8_t *src, size_t src_stride,
                                           size_t src_width, size_t src_height,
                                           int16_t *dst, size_t dst_stride) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width < 3 || src_height < 3) return KLEIDICV_OK;

  size_t out_w = src_width - 2;
  for (size_t y = 1; y + 1 < src_height; ++y) {
    const uint8_t *r_top = src + (y - 1) * src_stride;
    const uint8_t *r_mid = src + y * src_stride;
    const uint8_t *r_bot = src + (y + 1) * src_stride;
    int16_t *rd = reinterpret_cast<int16_t *>(
        reinterpret_cast<uint8_t *>(dst) + (y - 1) * dst_stride);
    for (size_t x = 0; x < out_w; ++x) {
      int sLU = r_top[x], sU = r_top[x + 1], sRU = r_top[x + 2];
      int sL = r_mid[x], sR = r_mid[x + 2];
      int sLB = r_bot[x], sB = r_bot[x + 1], sRB = r_bot[x + 2];
      int dx = 3 * (sRU - sLU + sRB - sLB) + 10 * (sR - sL);
      int dy = 3 * (sLB + sRB - sLU - sRU) + 10 * (sB - sU);
      rd[2 * x + 0] = static_cast<int16_t>(dx);
      rd[2 * x + 1] = static_cast<int16_t>(dy);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
