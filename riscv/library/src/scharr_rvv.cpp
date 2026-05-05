// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV Scharr 3x3 interleaved. Per output pixel: 8 of the 9 source samples
// contribute (the centre cancels in both dx and dy). We compute three
// widening-subtract pairs per row, then combine with scalar weights 3 and 10
// using vmul/vmacc, finishing with a vsseg2e16 that interleaves [dx,dy].

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"
#include "scharr_decls.h"

namespace kleidicv::rvv {

kleidicv_error_t scharr_interleaved_s16_u8(const uint8_t *src, size_t src_stride,
                                           size_t src_width, size_t src_height,
                                           int16_t *dst, size_t dst_stride) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width < 3 || src_height < 3) return KLEIDICV_OK;

  size_t out_w = src_width - 2;
  for (size_t y = 1; y + 1 < src_height; ++y) {
    const uint8_t *r_top = src + (y - 1) * src_stride;
    const uint8_t *r_mid = src + y * src_stride;
    const uint8_t *r_bot = src + (y + 1) * src_stride;
    int16_t *rd = reinterpret_cast<int16_t *>(
        reinterpret_cast<uint8_t *>(dst) + (y - 1) * dst_stride);
    size_t vl;
    for (size_t x = 0; x < out_w; x += vl) {
      vl = __riscv_vsetvl_e8m1(out_w - x);
      vuint8m1_t sLU = __riscv_vle8_v_u8m1(r_top + x, vl);
      vuint8m1_t sU  = __riscv_vle8_v_u8m1(r_top + x + 1, vl);
      vuint8m1_t sRU = __riscv_vle8_v_u8m1(r_top + x + 2, vl);
      vuint8m1_t sL  = __riscv_vle8_v_u8m1(r_mid + x, vl);
      vuint8m1_t sR  = __riscv_vle8_v_u8m1(r_mid + x + 2, vl);
      vuint8m1_t sLB = __riscv_vle8_v_u8m1(r_bot + x, vl);
      vuint8m1_t sB  = __riscv_vle8_v_u8m1(r_bot + x + 1, vl);
      vuint8m1_t sRB = __riscv_vle8_v_u8m1(r_bot + x + 2, vl);

      // Widening unsigned subtract reinterpreted as i16 (values in [-255,255]).
      auto wsub_s16 = [&](vuint8m1_t a, vuint8m1_t b) {
        return __riscv_vreinterpret_v_u16m2_i16m2(
            __riscv_vwsubu_vv_u16m2(a, b, vl));
      };

      // dx = 3*(sRU - sLU + sRB - sLB) + 10*(sR - sL)
      vint16m2_t d_top = wsub_s16(sRU, sLU);
      vint16m2_t d_bot = wsub_s16(sRB, sLB);
      vint16m2_t d_mid = wsub_s16(sR, sL);
      vint16m2_t dx = __riscv_vmul_vx_i16m2(d_top, 3, vl);
      dx = __riscv_vmacc_vx_i16m2(dx, 3, d_bot, vl);
      dx = __riscv_vmacc_vx_i16m2(dx, 10, d_mid, vl);

      // dy = 3*(sLB - sLU + sRB - sRU) + 10*(sB - sU)
      vint16m2_t dyL = wsub_s16(sLB, sLU);
      vint16m2_t dyR = wsub_s16(sRB, sRU);
      vint16m2_t dyM = wsub_s16(sB, sU);
      vint16m2_t dy = __riscv_vmul_vx_i16m2(dyL, 3, vl);
      dy = __riscv_vmacc_vx_i16m2(dy, 3, dyR, vl);
      dy = __riscv_vmacc_vx_i16m2(dy, 10, dyM, vl);

      vint16m2x2_t pair = __riscv_vcreate_v_i16m2x2(dx, dy);
      __riscv_vsseg2e16_v_i16m2x2(rd + 2 * x, pair, vl);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
