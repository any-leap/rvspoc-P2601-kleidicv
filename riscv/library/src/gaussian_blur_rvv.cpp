// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV 3x3 binomial Gaussian. Same shape as sobel_vertical's horizontal
// smoothing pass — load 3 rows × 3 columns, compute the separable [1,2,1]
// horizontal smooth twice, combine with [1,2,1] vertically, then rounding
// shift right by 4. Borders use scalar fallback.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "gaussian_blur_decls.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

namespace {

inline size_t cy(ptrdiff_t y, size_t h) {
  if (y < 0) return 0;
  if (static_cast<size_t>(y) >= h) return h - 1;
  return static_cast<size_t>(y);
}

inline uint8_t scalar_g3(const uint8_t *src, size_t stride, size_t y, size_t x,
                         size_t w, size_t h) {
  size_t yt = cy(static_cast<ptrdiff_t>(y) - 1, h);
  size_t yb = cy(static_cast<ptrdiff_t>(y) + 1, h);
  size_t xL = x == 0 ? 0 : x - 1;
  size_t xR = x == w - 1 ? w - 1 : x + 1;
  int top = static_cast<int>(src[yt * stride + xL]) +
            2 * static_cast<int>(src[yt * stride + x]) +
            static_cast<int>(src[yt * stride + xR]);
  int mid = static_cast<int>(src[y * stride + xL]) +
            2 * static_cast<int>(src[y * stride + x]) +
            static_cast<int>(src[y * stride + xR]);
  int bot = static_cast<int>(src[yb * stride + xL]) +
            2 * static_cast<int>(src[yb * stride + x]) +
            static_cast<int>(src[yb * stride + xR]);
  int total = top + 2 * mid + bot;
  return static_cast<uint8_t>((total + 8) >> 4);
}

}  // namespace

kleidicv_error_t gaussian_blur_3x3_binomial_u8(const uint8_t *src,
                                               size_t src_stride, uint8_t *dst,
                                               size_t dst_stride, size_t width,
                                               size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *r_top = src + cy(static_cast<ptrdiff_t>(y) - 1, height) *
                                     src_stride;
    const uint8_t *r_mid = src + y * src_stride;
    const uint8_t *r_bot = src + cy(static_cast<ptrdiff_t>(y) + 1, height) *
                                     src_stride;
    uint8_t *rd = dst + y * dst_stride;

    if (width <= 2) {
      for (size_t x = 0; x < width; ++x)
        rd[x] = scalar_g3(src, src_stride, y, x, width, height);
      continue;
    }
    rd[0] = scalar_g3(src, src_stride, y, 0, width, height);

    size_t vl;
    for (size_t x = 1; x + 1 < width; x += vl) {
      vl = __riscv_vsetvl_e8m1(width - 1 - x);
      // Horizontal smooth [1,2,1] per row → produces u16 (max 4*255 = 1020).
      vuint8m1_t lt = __riscv_vle8_v_u8m1(r_top + x - 1, vl);
      vuint8m1_t mt = __riscv_vle8_v_u8m1(r_top + x, vl);
      vuint8m1_t rt = __riscv_vle8_v_u8m1(r_top + x + 1, vl);
      vuint8m1_t lm = __riscv_vle8_v_u8m1(r_mid + x - 1, vl);
      vuint8m1_t mm = __riscv_vle8_v_u8m1(r_mid + x, vl);
      vuint8m1_t rm = __riscv_vle8_v_u8m1(r_mid + x + 1, vl);
      vuint8m1_t lb = __riscv_vle8_v_u8m1(r_bot + x - 1, vl);
      vuint8m1_t mb = __riscv_vle8_v_u8m1(r_bot + x, vl);
      vuint8m1_t rb = __riscv_vle8_v_u8m1(r_bot + x + 1, vl);

      vuint16m2_t htop = __riscv_vwaddu_vv_u16m2(lt, rt, vl);
      htop = __riscv_vwmaccu_vx_u16m2(htop, 2, mt, vl);
      vuint16m2_t hmid = __riscv_vwaddu_vv_u16m2(lm, rm, vl);
      hmid = __riscv_vwmaccu_vx_u16m2(hmid, 2, mm, vl);
      vuint16m2_t hbot = __riscv_vwaddu_vv_u16m2(lb, rb, vl);
      hbot = __riscv_vwmaccu_vx_u16m2(hbot, 2, mb, vl);

      // Vertical [1,2,1]: total = htop + 2*hmid + hbot. Max ≈ 4*1020 = 4080
      // (fits in u16 with room to spare).
      vuint16m2_t total = __riscv_vadd_vv_u16m2(htop, hbot, vl);
      total = __riscv_vmacc_vx_u16m2(total, 2, hmid, vl);
      // Rounding-narrow by 4 bits with vnclipu. Set rounding mode to RNU
      // (round to nearest, ties up — matches our (n+8)>>4 scalar path).
      vuint8m1_t out = __riscv_vnclipu_wx_u8m1(total, 4, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_u8m1(rd + x, out, vl);
    }

    rd[width - 1] = scalar_g3(src, src_stride, y, width - 1, width, height);
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
