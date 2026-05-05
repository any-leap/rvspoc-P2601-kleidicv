// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV in_range_u8: two unsigned compares ANDed; merge 0xFF/0.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "elementwise_rvv.h"
#include "in_range_decls.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

kleidicv_error_t in_range_u8(const uint8_t *src, size_t src_stride,
                             uint8_t *dst, size_t dst_stride, size_t width,
                             size_t height, uint8_t lower, uint8_t upper) {
  return unary_elementwise<uint8_t>(
      src, src_stride, dst, dst_stride, width, height,
      [lower, upper](auto v, size_t vl) {
        // x >= lower  <=>  !(x < lower)  ; we use vmsgeu when available.
        vbool8_t ge_lo = __riscv_vmsgeu_vx_u8m1_b8(v, lower, vl);
        vbool8_t le_hi = __riscv_vmsleu_vx_u8m1_b8(v, upper, vl);
        vbool8_t in = __riscv_vmand_mm_b8(ge_lo, le_hi, vl);
        vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
        return __riscv_vmerge_vxm_u8m1(zeros, 0xFF, in, vl);
      });
}

// f32 src → u8 mask dst. Strip-mine at e32m1 (vlen/32 lanes); the matching
// u8 fractional-LMUL type for the same lane count is u8mf4.
kleidicv_error_t in_range_f32(const float *src, size_t src_stride,
                              uint8_t *dst, size_t dst_stride, size_t width,
                              size_t height, float lower, float upper) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const float *row_src = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *row_dst = dst + y * dst_stride;
    size_t vl;
    for (size_t x = 0; x < width; x += vl) {
      vl = __riscv_vsetvl_e32m1(width - x);
      vfloat32m1_t v = __riscv_vle32_v_f32m1(row_src + x, vl);
      vbool32_t ge_lo = __riscv_vmfge_vf_f32m1_b32(v, lower, vl);
      vbool32_t le_hi = __riscv_vmfle_vf_f32m1_b32(v, upper, vl);
      vbool32_t in = __riscv_vmand_mm_b32(ge_lo, le_hi, vl);
      vuint8mf4_t zeros = __riscv_vmv_v_x_u8mf4(0, vl);
      vuint8mf4_t result = __riscv_vmerge_vxm_u8mf4(zeros, 0xFF, in, vl);
      __riscv_vse8_v_u8mf4(row_dst + x, result, vl);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
