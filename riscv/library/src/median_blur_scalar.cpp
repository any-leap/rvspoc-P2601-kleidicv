// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// 3x3 median, u8 only, replicate border. Uses a 9-element optimal sorting
// network so per-pixel cost is bounded and branch-free.

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "kleidicv/kleidicv.h"

#include "median_blur_decls.h"

namespace kleidicv::scalar {

namespace {

inline uint8_t cmin(uint8_t a, uint8_t b) { return a < b ? a : b; }
inline uint8_t cmax(uint8_t a, uint8_t b) { return a > b ? a : b; }
inline void sort2(uint8_t &a, uint8_t &b) {
  uint8_t lo = cmin(a, b), hi = cmax(a, b);
  a = lo;
  b = hi;
}

// 9-element median (median of 9), via a partial sorting network. The full
// sorting network is overkill — we only need the 5th-rank element. This is
// the standard "median of 9" sequence used by OpenCV's reference impl.
inline uint8_t median9(uint8_t v[9]) {
  sort2(v[1], v[2]);
  sort2(v[4], v[5]);
  sort2(v[7], v[8]);
  sort2(v[0], v[1]);
  sort2(v[3], v[4]);
  sort2(v[6], v[7]);
  sort2(v[1], v[2]);
  sort2(v[4], v[5]);
  sort2(v[7], v[8]);
  sort2(v[0], v[3]);
  sort2(v[5], v[8]);
  sort2(v[4], v[7]);
  sort2(v[3], v[6]);
  sort2(v[1], v[4]);
  sort2(v[2], v[5]);
  sort2(v[4], v[7]);
  sort2(v[4], v[2]);
  sort2(v[6], v[4]);
  sort2(v[4], v[2]);
  return v[4];
}

inline size_t clip(ptrdiff_t v, size_t n) {
  if (v < 0) return 0;
  if (static_cast<size_t>(v) >= n) return n - 1;
  return static_cast<size_t>(v);
}

}  // namespace

kleidicv_error_t median_blur_3x3_u8(const uint8_t *src, size_t src_stride,
                                    uint8_t *dst, size_t dst_stride,
                                    size_t width, size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *r_top = src + clip(static_cast<ptrdiff_t>(y) - 1, height) *
                                     src_stride;
    const uint8_t *r_mid = src + y * src_stride;
    const uint8_t *r_bot = src + clip(static_cast<ptrdiff_t>(y) + 1, height) *
                                     src_stride;
    uint8_t *rd = dst + y * dst_stride;
    for (size_t x = 0; x < width; ++x) {
      size_t xL = clip(static_cast<ptrdiff_t>(x) - 1, width);
      size_t xR = clip(static_cast<ptrdiff_t>(x) + 1, width);
      uint8_t v[9] = {r_top[xL], r_top[x], r_top[xR],
                      r_mid[xL], r_mid[x], r_mid[xR],
                      r_bot[xL], r_bot[x], r_bot[xR]};
      rd[x] = median9(v);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
