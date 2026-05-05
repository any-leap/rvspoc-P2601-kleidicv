// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <limits>

#include "kleidicv/kleidicv.h"

#include "morphology_decls.h"

namespace kleidicv::scalar {

kleidicv_error_t morph_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          size_t kw, size_t kh, bool is_dilate) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;
  if ((kw & 1) == 0 || (kh & 1) == 0) return KLEIDICV_ERROR_RANGE;

  ptrdiff_t hkw = static_cast<ptrdiff_t>(kw / 2);
  ptrdiff_t hkh = static_cast<ptrdiff_t>(kh / 2);

  for (size_t y = 0; y < height; ++y) {
    uint8_t *rd = dst + y * dst_stride;
    for (size_t x = 0; x < width; ++x) {
      uint8_t best = is_dilate ? 0 : 255;
      for (ptrdiff_t dy = -hkh; dy <= hkh; ++dy) {
        ptrdiff_t sy = static_cast<ptrdiff_t>(y) + dy;
        if (sy < 0) sy = 0;
        if (static_cast<size_t>(sy) >= height) sy = height - 1;
        const uint8_t *row = src + static_cast<size_t>(sy) * src_stride;
        for (ptrdiff_t dx = -hkw; dx <= hkw; ++dx) {
          ptrdiff_t sx = static_cast<ptrdiff_t>(x) + dx;
          if (sx < 0) sx = 0;
          if (static_cast<size_t>(sx) >= width) sx = width - 1;
          uint8_t v = row[sx];
          best = is_dilate ? (v > best ? v : best) : (v < best ? v : best);
        }
      }
      rd[x] = best;
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
