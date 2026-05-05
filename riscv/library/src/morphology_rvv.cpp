// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV rectangular dilate / erode (channels=1, replicate border, odd kernel
// dims). Bulk x-range vectorised; left/right edge columns scalar.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "kleidicv/kleidicv.h"
#include "morphology_decls.h"

namespace kleidicv::rvv {

namespace {

inline size_t cy(ptrdiff_t y, size_t h) {
  if (y < 0) return 0;
  if (static_cast<size_t>(y) >= h) return h - 1;
  return static_cast<size_t>(y);
}

inline uint8_t scalar_morph_one(const uint8_t *src, size_t stride, size_t y,
                                size_t x, size_t w, size_t h, ptrdiff_t hkw,
                                ptrdiff_t hkh, bool is_dilate) {
  uint8_t best = is_dilate ? 0 : 255;
  for (ptrdiff_t dy = -hkh; dy <= hkh; ++dy) {
    size_t sy = cy(static_cast<ptrdiff_t>(y) + dy, h);
    const uint8_t *row = src + sy * stride;
    for (ptrdiff_t dx = -hkw; dx <= hkw; ++dx) {
      ptrdiff_t sx = static_cast<ptrdiff_t>(x) + dx;
      if (sx < 0) sx = 0;
      if (static_cast<size_t>(sx) >= w) sx = w - 1;
      uint8_t v = row[sx];
      best = is_dilate ? (v > best ? v : best) : (v < best ? v : best);
    }
  }
  return best;
}

}  // namespace

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
    // Edge columns where x ± hkw can underflow / overflow: scalar.
    size_t left_edge = static_cast<size_t>(hkw);
    size_t right_edge = width > static_cast<size_t>(hkw)
                            ? width - static_cast<size_t>(hkw)
                            : 0;
    for (size_t x = 0; x < left_edge && x < width; ++x)
      rd[x] = scalar_morph_one(src, src_stride, y, x, width, height, hkw, hkh,
                               is_dilate);
    for (size_t x = right_edge; x < width; ++x)
      rd[x] = scalar_morph_one(src, src_stride, y, x, width, height, hkw, hkh,
                               is_dilate);

    if (left_edge >= right_edge) continue;
    // Bulk: x in [hkw, width-hkw). All horizontal accesses in-bounds.
    size_t vl;
    for (size_t x = left_edge; x < right_edge; x += vl) {
      vl = __riscv_vsetvl_e8m1(right_edge - x);
      // Initialise accumulator with kernel anchor row pixel column.
      vuint8m1_t acc;
      // Seed with first kernel row, first column.
      {
        size_t sy = cy(static_cast<ptrdiff_t>(y) - hkh, height);
        const uint8_t *row = src + sy * src_stride;
        acc = __riscv_vle8_v_u8m1(row + x - hkw, vl);
        for (ptrdiff_t dx = -hkw + 1; dx <= hkw; ++dx) {
          vuint8m1_t v = __riscv_vle8_v_u8m1(row + x + dx, vl);
          acc = is_dilate ? __riscv_vmaxu_vv_u8m1(acc, v, vl)
                          : __riscv_vminu_vv_u8m1(acc, v, vl);
        }
      }
      for (ptrdiff_t dy = -hkh + 1; dy <= hkh; ++dy) {
        size_t sy = cy(static_cast<ptrdiff_t>(y) + dy, height);
        const uint8_t *row = src + sy * src_stride;
        for (ptrdiff_t dx = -hkw; dx <= hkw; ++dx) {
          vuint8m1_t v = __riscv_vle8_v_u8m1(row + x + dx, vl);
          acc = is_dilate ? __riscv_vmaxu_vv_u8m1(acc, v, vl)
                          : __riscv_vminu_vv_u8m1(acc, v, vl);
        }
      }
      __riscv_vse8_v_u8m1(rd + x, acc, vl);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
