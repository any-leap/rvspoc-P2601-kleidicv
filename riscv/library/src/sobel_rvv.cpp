// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV Sobel 3x3 (separable, replicate border). The two-pass nature is folded
// into one inline computation per output pixel — for a 3x3 kernel the
// intermediate buffer would cost more than it saves at LMUL=1. Edge columns
// (x=0 and x=width-1) use the scalar path because the RVV strip-mine assumes
// loadable x-1 and x+1 neighbours.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"
#include "sobel_decls.h"

namespace kleidicv::rvv {

namespace {

inline size_t clip_row(size_t y, ptrdiff_t dy, size_t h) {
  ptrdiff_t r = static_cast<ptrdiff_t>(y) + dy;
  if (r < 0) return 0;
  if (static_cast<size_t>(r) >= h) return h - 1;
  return static_cast<size_t>(r);
}

inline int16_t scalar_sobel_h(const uint8_t *src, size_t stride, size_t y,
                              size_t x, size_t w, size_t h) {
  size_t yt = clip_row(y, -1, h), yb = clip_row(y, 1, h);
  size_t xL = x == 0 ? 0 : x - 1;
  size_t xR = x == w - 1 ? w - 1 : x + 1;
  int top = static_cast<int>(src[yt * stride + xR]) -
            static_cast<int>(src[yt * stride + xL]);
  int mid = static_cast<int>(src[y * stride + xR]) -
            static_cast<int>(src[y * stride + xL]);
  int bot = static_cast<int>(src[yb * stride + xR]) -
            static_cast<int>(src[yb * stride + xL]);
  return static_cast<int16_t>(top + 2 * mid + bot);
}

inline int16_t scalar_sobel_v(const uint8_t *src, size_t stride, size_t y,
                              size_t x, size_t w, size_t h) {
  size_t yt = clip_row(y, -1, h), yb = clip_row(y, 1, h);
  size_t xL = x == 0 ? 0 : x - 1;
  size_t xR = x == w - 1 ? w - 1 : x + 1;
  int htop = static_cast<int>(src[yt * stride + xL]) +
             2 * static_cast<int>(src[yt * stride + x]) +
             static_cast<int>(src[yt * stride + xR]);
  int hbot = static_cast<int>(src[yb * stride + xL]) +
             2 * static_cast<int>(src[yb * stride + x]) +
             static_cast<int>(src[yb * stride + xR]);
  return static_cast<int16_t>(hbot - htop);
}

}  // namespace

kleidicv_error_t sobel_3x3_horizontal_s16_u8(const uint8_t *src,
                                             size_t src_stride, int16_t *dst,
                                             size_t dst_stride, size_t width,
                                             size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *r_top = src + clip_row(y, -1, height) * src_stride;
    const uint8_t *r_mid = src + y * src_stride;
    const uint8_t *r_bot = src + clip_row(y, 1, height) * src_stride;
    int16_t *rd = reinterpret_cast<int16_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);

    if (width <= 2) {
      for (size_t x = 0; x < width; ++x)
        rd[x] = scalar_sobel_h(src, src_stride, y, x, width, height);
      continue;
    }
    rd[0] = scalar_sobel_h(src, src_stride, y, 0, width, height);

    size_t vl;
    for (size_t x = 1; x + 1 < width; x += vl) {
      vl = __riscv_vsetvl_e8m1(width - 1 - x);
      vuint8m1_t lt = __riscv_vle8_v_u8m1(r_top + x - 1, vl);
      vuint8m1_t rt = __riscv_vle8_v_u8m1(r_top + x + 1, vl);
      vuint8m1_t lm = __riscv_vle8_v_u8m1(r_mid + x - 1, vl);
      vuint8m1_t rm = __riscv_vle8_v_u8m1(r_mid + x + 1, vl);
      vuint8m1_t lb = __riscv_vle8_v_u8m1(r_bot + x - 1, vl);
      vuint8m1_t rb = __riscv_vle8_v_u8m1(r_bot + x + 1, vl);
      // (right - left) per row, widening; reinterpret to i16.
      vint16m2_t dt = __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vwsubu_vv_u16m2(rt, lt, vl));
      vint16m2_t dm = __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vwsubu_vv_u16m2(rm, lm, vl));
      vint16m2_t db = __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vwsubu_vv_u16m2(rb, lb, vl));
      vint16m2_t two_m = __riscv_vsll_vx_i16m2(dm, 1, vl);
      vint16m2_t s = __riscv_vadd_vv_i16m2(dt, two_m, vl);
      s = __riscv_vadd_vv_i16m2(s, db, vl);
      __riscv_vse16_v_i16m2(rd + x, s, vl);
    }

    rd[width - 1] =
        scalar_sobel_h(src, src_stride, y, width - 1, width, height);
  }
  return KLEIDICV_OK;
}

kleidicv_error_t sobel_3x3_vertical_s16_u8(const uint8_t *src, size_t src_stride,
                                           int16_t *dst, size_t dst_stride,
                                           size_t width, size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *r_top = src + clip_row(y, -1, height) * src_stride;
    const uint8_t *r_bot = src + clip_row(y, 1, height) * src_stride;
    int16_t *rd = reinterpret_cast<int16_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);

    if (width <= 2) {
      for (size_t x = 0; x < width; ++x)
        rd[x] = scalar_sobel_v(src, src_stride, y, x, width, height);
      continue;
    }
    rd[0] = scalar_sobel_v(src, src_stride, y, 0, width, height);

    size_t vl;
    for (size_t x = 1; x + 1 < width; x += vl) {
      vl = __riscv_vsetvl_e8m1(width - 1 - x);
      vuint8m1_t lt = __riscv_vle8_v_u8m1(r_top + x - 1, vl);
      vuint8m1_t mt = __riscv_vle8_v_u8m1(r_top + x, vl);
      vuint8m1_t rt = __riscv_vle8_v_u8m1(r_top + x + 1, vl);
      vuint8m1_t lb = __riscv_vle8_v_u8m1(r_bot + x - 1, vl);
      vuint8m1_t mb = __riscv_vle8_v_u8m1(r_bot + x, vl);
      vuint8m1_t rb = __riscv_vle8_v_u8m1(r_bot + x + 1, vl);
      // hsmooth = left + 2*mid + right; max = 4*255 = 1020 fits in u16.
      vuint16m2_t htop = __riscv_vwaddu_vv_u16m2(lt, rt, vl);
      htop = __riscv_vwmaccu_vx_u16m2(htop, 2, mt, vl);
      vuint16m2_t hbot = __riscv_vwaddu_vv_u16m2(lb, rb, vl);
      hbot = __riscv_vwmaccu_vx_u16m2(hbot, 2, mb, vl);
      // Reinterpret to i16: 1020 fits in i16 cleanly.
      vint16m2_t s = __riscv_vsub_vv_i16m2(
          __riscv_vreinterpret_v_u16m2_i16m2(hbot),
          __riscv_vreinterpret_v_u16m2_i16m2(htop), vl);
      __riscv_vse16_v_i16m2(rd + x, s, vl);
    }

    rd[width - 1] =
        scalar_sobel_v(src, src_stride, y, width - 1, width, height);
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
