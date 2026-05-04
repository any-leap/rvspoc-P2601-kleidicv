// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV implementation of gray_to_rgb_u8.
//
// Single-shot interleaved store via vsseg3e8 — gcc 14+ exposes the segment
// store intrinsic and the matching tuple type vuint8m1x3_t. Earlier history
// of this file used three strided stores (vsse8) as a workaround for gcc
// 13.3 which lacked these intrinsics; the dev image now ships gcc 14.2.

#include <riscv_vector.h>

#include "gray_to_rgb_decls.h"

namespace kleidicv::rvv {

kleidicv_error_t gray_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *row_src = src + y * src_stride;
    uint8_t *row_dst = dst + y * dst_stride;
    size_t vl;
    for (size_t x = 0; x < width; x += vl) {
      vl = __riscv_vsetvl_e8m1(width - x);
      vuint8m1_t g = __riscv_vle8_v_u8m1(row_src + x, vl);
      vuint8m1x3_t triple = __riscv_vcreate_v_u8m1x3(g, g, g);
      __riscv_vsseg3e8_v_u8m1x3(row_dst + 3 * x, triple, vl);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
