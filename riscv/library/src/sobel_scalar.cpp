// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Sobel 3x3 separable. Border policy: replicate.
//
// Horizontal sobel (dx) kernel:
//   [-1 0 1; -2 0 2; -1 0 1]  -> separable as Hx = [-1,0,1], Hy = [1,2,1]
// Vertical sobel (dy) kernel:
//   [-1 -2 -1; 0 0 0; 1 2 1]  -> separable as Hx = [1,2,1], Hy = [-1,0,1]

#include <cstdint>
#include <cstdlib>

#include "kleidicv/kleidicv.h"

#include "sobel_decls.h"

namespace kleidicv::scalar {

namespace {

inline size_t clip_y(ptrdiff_t y, size_t h) {
  if (y < 0) return 0;
  if (static_cast<size_t>(y) >= h) return h - 1;
  return static_cast<size_t>(y);
}
inline size_t clip_x(ptrdiff_t x, size_t w) {
  if (x < 0) return 0;
  if (static_cast<size_t>(x) >= w) return w - 1;
  return static_cast<size_t>(x);
}

inline uint8_t fetch(const uint8_t *src, size_t stride, ptrdiff_t y, ptrdiff_t x,
                     size_t w, size_t h) {
  return src[clip_y(y, h) * stride + clip_x(x, w)];
}

}  // namespace

kleidicv_error_t sobel_3x3_horizontal_s16_u8(const uint8_t *src,
                                             size_t src_stride, int16_t *dst,
                                             size_t dst_stride, size_t width,
                                             size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    int16_t *rd = reinterpret_cast<int16_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      // Horizontal derivative: vertical kernel [1,2,1], horizontal [-1,0,1].
      int sum = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        int wy = (dy == 0) ? 2 : 1;
        sum += wy * (-fetch(src, src_stride, static_cast<ptrdiff_t>(y) + dy,
                            static_cast<ptrdiff_t>(x) - 1, width, height) +
                     fetch(src, src_stride, static_cast<ptrdiff_t>(y) + dy,
                           static_cast<ptrdiff_t>(x) + 1, width, height));
      }
      rd[x] = static_cast<int16_t>(sum);
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t sobel_3x3_vertical_s16_u8(const uint8_t *src,
                                           size_t src_stride, int16_t *dst,
                                           size_t dst_stride, size_t width,
                                           size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    int16_t *rd = reinterpret_cast<int16_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      // Vertical derivative: vertical [-1,0,1], horizontal [1,2,1].
      int sum = 0;
      for (int dx = -1; dx <= 1; ++dx) {
        int wx = (dx == 0) ? 2 : 1;
        sum += wx * (-fetch(src, src_stride, static_cast<ptrdiff_t>(y) - 1,
                            static_cast<ptrdiff_t>(x) + dx, width, height) +
                     fetch(src, src_stride, static_cast<ptrdiff_t>(y) + 1,
                           static_cast<ptrdiff_t>(x) + dx, width, height));
      }
      rd[x] = static_cast<int16_t>(sum);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
