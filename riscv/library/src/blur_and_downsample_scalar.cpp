// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// 5x5 Gaussian (kernel [1,4,6,4,1] separable) followed by 2x downsampling.
// Output dst_w = (src_w+1)/2, dst_h = (src_h+1)/2. Three border modes:
// REPLICATE (edge clamp), REFLECT_101 / REVERSE (mirror without doubling
// edge), REFLECT (mirror with doubling). The RVV interior path is the
// same for all three; only the clip helper differs.

#include <cstdint>
#include <cstdlib>

#include "kleidicv/kleidicv.h"

#include "blur_and_downsample_decls.h"

namespace kleidicv::scalar {

namespace {

inline size_t clip_replicate(ptrdiff_t v, size_t n) {
  if (v < 0) return 0;
  if (static_cast<size_t>(v) >= n) return n - 1;
  return static_cast<size_t>(v);
}

// REFLECT_101 (a.k.a. BORDER_REVERSE): "...3 2 1 | 0 1 2 3 ... n-2 n-1 | n-2 n-3 ...".
// Edge is not duplicated.
inline size_t clip_reflect_101(ptrdiff_t v, size_t n) {
  if (n <= 1) return 0;
  ptrdiff_t r = v;
  while (r < 0 || r >= static_cast<ptrdiff_t>(n)) {
    if (r < 0) r = -r;
    else r = 2 * static_cast<ptrdiff_t>(n) - 2 - r;
  }
  return static_cast<size_t>(r);
}

// REFLECT (a.k.a. BORDER_REFLECT): "...2 1 0 | 0 1 2 3 ... n-1 | n-1 n-2 n-3 ...".
// Edge IS duplicated.
inline size_t clip_reflect(ptrdiff_t v, size_t n) {
  if (n == 0) return 0;
  ptrdiff_t r = v;
  while (r < 0 || r >= static_cast<ptrdiff_t>(n)) {
    if (r < 0) r = -r - 1;
    else r = 2 * static_cast<ptrdiff_t>(n) - 1 - r;
  }
  return static_cast<size_t>(r);
}

template <size_t (*Clip)(ptrdiff_t, size_t)>
inline kleidicv_error_t blur_and_downsample_impl(const uint8_t *src,
                                                   size_t src_stride,
                                                   size_t src_width,
                                                   size_t src_height,
                                                   uint8_t *dst,
                                                   size_t dst_stride) {
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
        size_t y = Clip(sy + ky, src_height);
        const uint8_t *row = src + y * src_stride;
        int row_acc = 0;
        for (int kx = -2; kx <= 2; ++kx) {
          size_t x = Clip(sx + kx, src_width);
          row_acc += K[kx + 2] * row[x];
        }
        acc += K[ky + 2] * row_acc;
      }
      rd[dx] = static_cast<uint8_t>((acc + 128) >> 8);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace

kleidicv_error_t blur_and_downsample_u8(const uint8_t *src, size_t src_stride,
                                        size_t src_width, size_t src_height,
                                        uint8_t *dst, size_t dst_stride) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width < 4 || src_height < 4) return KLEIDICV_ERROR_RANGE;
  return blur_and_downsample_impl<clip_replicate>(src, src_stride, src_width,
                                                    src_height, dst,
                                                    dst_stride);
}

kleidicv_error_t blur_and_downsample_u8_reflect_101(const uint8_t *src,
                                                      size_t src_stride,
                                                      size_t src_width,
                                                      size_t src_height,
                                                      uint8_t *dst,
                                                      size_t dst_stride) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width < 4 || src_height < 4) return KLEIDICV_ERROR_RANGE;
  return blur_and_downsample_impl<clip_reflect_101>(src, src_stride, src_width,
                                                      src_height, dst,
                                                      dst_stride);
}

kleidicv_error_t blur_and_downsample_u8_reflect(const uint8_t *src,
                                                  size_t src_stride,
                                                  size_t src_width,
                                                  size_t src_height,
                                                  uint8_t *dst,
                                                  size_t dst_stride) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width < 4 || src_height < 4) return KLEIDICV_ERROR_RANGE;
  return blur_and_downsample_impl<clip_reflect>(src, src_stride, src_width,
                                                  src_height, dst, dst_stride);
}

}  // namespace kleidicv::scalar
