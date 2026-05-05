// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// gray → RGBA via vsseg4e8 with alpha lane = 0xFF.

#include <riscv_vector.h>

#include "gray_to_rgba_decls.h"

namespace kleidicv::rvv {

kleidicv_error_t gray_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *rs = src + y * src_stride;
    uint8_t *rd = dst + y * dst_stride;
    size_t vl;
    for (size_t x = 0; x < width; x += vl) {
      vl = __riscv_vsetvl_e8m1(width - x);
      vuint8m1_t g = __riscv_vle8_v_u8m1(rs + x, vl);
      vuint8m1_t a = __riscv_vmv_v_x_u8m1(0xFF, vl);
      vuint8m1x4_t quad = __riscv_vcreate_v_u8m1x4(g, g, g, a);
      __riscv_vsseg4e8_v_u8m1x4(rd + 4 * x, quad, vl);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
