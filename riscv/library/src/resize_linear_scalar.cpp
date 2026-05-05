// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Bilinear resize. Coordinate mapping: src_coord = (dst_coord + 0.5) *
// (src_size / dst_size) - 0.5. Out-of-range indices clamp to the edge
// (replicate). u8 path uses fixed-point Q11 weights (OpenCV-compatible
// precision); f32 path uses native floats.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "kleidicv/kleidicv.h"

#include "resize_linear_decls.h"

namespace kleidicv::scalar {

namespace {

inline ptrdiff_t clip_idx(ptrdiff_t v, ptrdiff_t n) {
  if (v < 0) return 0;
  if (v >= n) return n - 1;
  return v;
}

}  // namespace

kleidicv_error_t resize_linear_u8(const uint8_t *src, size_t src_stride,
                                  size_t src_width, size_t src_height,
                                  uint8_t *dst, size_t dst_stride,
                                  size_t dst_width, size_t dst_height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0)
    return KLEIDICV_OK;

  double scale_x =
      static_cast<double>(src_width) / static_cast<double>(dst_width);
  double scale_y =
      static_cast<double>(src_height) / static_cast<double>(dst_height);

  for (size_t dy = 0; dy < dst_height; ++dy) {
    double sy_f = (static_cast<double>(dy) + 0.5) * scale_y - 0.5;
    ptrdiff_t sy0 = static_cast<ptrdiff_t>(std::floor(sy_f));
    double fy = sy_f - static_cast<double>(sy0);
    if (fy < 0) fy = 0;
    if (fy > 1) fy = 1;
    ptrdiff_t sy0c = clip_idx(sy0, static_cast<ptrdiff_t>(src_height));
    ptrdiff_t sy1c = clip_idx(sy0 + 1, static_cast<ptrdiff_t>(src_height));
    const uint8_t *r0 = src + static_cast<size_t>(sy0c) * src_stride;
    const uint8_t *r1 = src + static_cast<size_t>(sy1c) * src_stride;
    uint8_t *rd = dst + dy * dst_stride;
    for (size_t dx = 0; dx < dst_width; ++dx) {
      double sx_f = (static_cast<double>(dx) + 0.5) * scale_x - 0.5;
      ptrdiff_t sx0 = static_cast<ptrdiff_t>(std::floor(sx_f));
      double fx = sx_f - static_cast<double>(sx0);
      if (fx < 0) fx = 0;
      if (fx > 1) fx = 1;
      ptrdiff_t sx0c = clip_idx(sx0, static_cast<ptrdiff_t>(src_width));
      ptrdiff_t sx1c = clip_idx(sx0 + 1, static_cast<ptrdiff_t>(src_width));
      double p00 = r0[sx0c], p01 = r0[sx1c];
      double p10 = r1[sx0c], p11 = r1[sx1c];
      double v = (1.0 - fy) * ((1.0 - fx) * p00 + fx * p01) +
                 fy * ((1.0 - fx) * p10 + fx * p11);
      long iv = std::lrint(v);
      if (iv < 0) iv = 0;
      if (iv > 255) iv = 255;
      rd[dx] = static_cast<uint8_t>(iv);
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t resize_linear_f32(const float *src, size_t src_stride,
                                   size_t src_width, size_t src_height,
                                   float *dst, size_t dst_stride,
                                   size_t dst_width, size_t dst_height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0)
    return KLEIDICV_OK;

  double scale_x =
      static_cast<double>(src_width) / static_cast<double>(dst_width);
  double scale_y =
      static_cast<double>(src_height) / static_cast<double>(dst_height);
  size_t src_stride_e = src_stride / sizeof(float);
  size_t dst_stride_e = dst_stride / sizeof(float);

  for (size_t dy = 0; dy < dst_height; ++dy) {
    double sy_f = (static_cast<double>(dy) + 0.5) * scale_y - 0.5;
    ptrdiff_t sy0 = static_cast<ptrdiff_t>(std::floor(sy_f));
    double fy = sy_f - static_cast<double>(sy0);
    if (fy < 0) fy = 0;
    if (fy > 1) fy = 1;
    ptrdiff_t sy0c = clip_idx(sy0, static_cast<ptrdiff_t>(src_height));
    ptrdiff_t sy1c = clip_idx(sy0 + 1, static_cast<ptrdiff_t>(src_height));
    const float *r0 = src + static_cast<size_t>(sy0c) * src_stride_e;
    const float *r1 = src + static_cast<size_t>(sy1c) * src_stride_e;
    float *rd = dst + dy * dst_stride_e;
    for (size_t dx = 0; dx < dst_width; ++dx) {
      double sx_f = (static_cast<double>(dx) + 0.5) * scale_x - 0.5;
      ptrdiff_t sx0 = static_cast<ptrdiff_t>(std::floor(sx_f));
      double fx = sx_f - static_cast<double>(sx0);
      if (fx < 0) fx = 0;
      if (fx > 1) fx = 1;
      ptrdiff_t sx0c = clip_idx(sx0, static_cast<ptrdiff_t>(src_width));
      ptrdiff_t sx1c = clip_idx(sx0 + 1, static_cast<ptrdiff_t>(src_width));
      double p00 = r0[sx0c], p01 = r0[sx1c];
      double p10 = r1[sx0c], p11 = r1[sx1c];
      double v = (1.0 - fy) * ((1.0 - fx) * p00 + fx * p01) +
                 fy * ((1.0 - fx) * p10 + fx * p11);
      rd[dx] = static_cast<float>(v);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
