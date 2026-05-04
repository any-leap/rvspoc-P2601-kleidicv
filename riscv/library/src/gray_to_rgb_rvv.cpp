// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV implementation of gray_to_rgb_u8.
//
// The natural RVV idiom for interleaved-channel stores is the segment store
// (vsseg3e8.v), which writes a vl-wide R/G/B triplet in one shot. gcc 13.3
// does not yet expose the segment intrinsics (added post-v1.0 in gcc 14+),
// so we use three strided stores at byte offsets 0/1/2 with stride 3 — one
// strided store per output channel. For gray_to_rgb the three channels share
// the same value, so all three strided stores write the same vector. When the
// dev image moves to gcc 14 (see riscv/docker/Dockerfile), this should be
// rewritten to a single vsseg3e8.

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
      uint8_t *base = row_dst + 3 * x;
      __riscv_vsse8_v_u8m1(base + 0, 3, g, vl);
      __riscv_vsse8_v_u8m1(base + 1, 3, g, vl);
      __riscv_vsse8_v_u8m1(base + 2, 3, g, vl);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
