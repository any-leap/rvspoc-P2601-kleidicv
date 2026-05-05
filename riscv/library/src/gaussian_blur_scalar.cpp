// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// 3x3 binomial Gaussian:
//   F = (1/16) * [1 2 1; 2 4 2; 1 2 1]   = (1/16) * [1,2,1] ⊗ [1,2,1]
// Replicate border. Rounding shift — matches upstream's `svrshr_x`.

#include <cstdint>
#include <cstdlib>

#include "kleidicv/kleidicv.h"

#include "gaussian_blur_decls.h"

namespace kleidicv::scalar {

kleidicv_error_t gaussian_blur_3x3_binomial_u8(const uint8_t *src,
                                               size_t src_stride, uint8_t *dst,
                                               size_t dst_stride, size_t width,
                                               size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  auto cy = [&](ptrdiff_t y) -> size_t {
    if (y < 0) return 0;
    if (static_cast<size_t>(y) >= height) return height - 1;
    return static_cast<size_t>(y);
  };
  auto cx = [&](ptrdiff_t x) -> size_t {
    if (x < 0) return 0;
    if (static_cast<size_t>(x) >= width) return width - 1;
    return static_cast<size_t>(x);
  };

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *r_top = src + cy(static_cast<ptrdiff_t>(y) - 1) * src_stride;
    const uint8_t *r_mid = src + y * src_stride;
    const uint8_t *r_bot = src + cy(static_cast<ptrdiff_t>(y) + 1) * src_stride;
    uint8_t *rd = dst + y * dst_stride;
    for (size_t x = 0; x < width; ++x) {
      size_t xL = cx(static_cast<ptrdiff_t>(x) - 1);
      size_t xR = cx(static_cast<ptrdiff_t>(x) + 1);
      // sum = src[y-1][x-1] + 2*src[y-1][x] + src[y-1][x+1]
      //     + 2*(src[y][x-1] + 2*src[y][x] + src[y][x+1])
      //     +    src[y+1][x-1] + 2*src[y+1][x] + src[y+1][x+1]
      int top = static_cast<int>(r_top[xL]) + 2 * static_cast<int>(r_top[x]) +
                static_cast<int>(r_top[xR]);
      int mid = static_cast<int>(r_mid[xL]) + 2 * static_cast<int>(r_mid[x]) +
                static_cast<int>(r_mid[xR]);
      int bot = static_cast<int>(r_bot[xL]) + 2 * static_cast<int>(r_bot[x]) +
                static_cast<int>(r_bot[xR]);
      int total = top + 2 * mid + bot;
      // Rounding-divide by 16.
      rd[x] = static_cast<uint8_t>((total + 8) >> 4);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
