// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// 5x5 Gaussian (kernel [1,4,6,4,1] separable) followed by 2x downsampling.
// Replicate border. Output dst_w = (src_w+1)/2, dst_h = (src_h+1)/2.

#include <cstdint>
#include <cstdlib>

#include "kleidicv/kleidicv.h"

#include "blur_and_downsample_decls.h"

namespace kleidicv::scalar {

namespace {

inline size_t clip(ptrdiff_t v, size_t n) {
  if (v < 0) return 0;
  if (static_cast<size_t>(v) >= n) return n - 1;
  return static_cast<size_t>(v);
}

}  // namespace

kleidicv_error_t blur_and_downsample_u8(const uint8_t *src, size_t src_stride,
                                        size_t src_width, size_t src_height,
                                        uint8_t *dst, size_t dst_stride) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width < 4 || src_height < 4) return KLEIDICV_ERROR_RANGE;

  const int K[5] = {1, 4, 6, 4, 1};
  size_t dst_w = (src_width + 1) / 2;
  size_t dst_h = (src_height + 1) / 2;
  for (size_t dy = 0; dy < dst_h; ++dy) {
    uint8_t *rd = dst + dy * dst_stride;
    ptrdiff_t sy = static_cast<ptrdiff_t>(dy) * 2;
    for (size_t dx = 0; dx < dst_w; ++dx) {
      ptrdiff_t sx = static_cast<ptrdiff_t>(dx) * 2;
      int acc = 0;
      for (int ky = -2; ky <= 2; ++ky) {
        size_t y = clip(sy + ky, src_height);
        const uint8_t *row = src + y * src_stride;
        int row_acc = 0;
        for (int kx = -2; kx <= 2; ++kx) {
          size_t x = clip(sx + kx, src_width);
          row_acc += K[kx + 2] * row[x];
        }
        acc += K[ky + 2] * row_acc;
      }
      // Total normalization: 16 * 16 = 256, with rounding.
      rd[dx] = static_cast<uint8_t>((acc + 128) >> 8);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
