// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV compare_equal_u8 / compare_greater_u8: vmseq / vmsgtu produces a mask;
// merge 0xFF where the mask is set, else 0.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "compare_decls.h"
#include "elementwise_rvv.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

namespace {

inline vuint8m1_t mask_to_byte(vbool8_t mask, size_t vl) {
  vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
  return __riscv_vmerge_vxm_u8m1(zeros, 0xFF, mask, vl);
}

}  // namespace

kleidicv_error_t compare_equal_u8(const uint8_t *src_a, size_t sa,
                                  const uint8_t *src_b, size_t sb, uint8_t *dst,
                                  size_t sd, size_t w, size_t h) {
  return binary_elementwise<uint8_t>(
      src_a, sa, src_b, sb, dst, sd, w, h, [](auto a, auto b, size_t vl) {
        return mask_to_byte(__riscv_vmseq_vv_u8m1_b8(a, b, vl), vl);
      });
}

kleidicv_error_t compare_greater_u8(const uint8_t *src_a, size_t sa,
                                    const uint8_t *src_b, size_t sb,
                                    uint8_t *dst, size_t sd, size_t w,
                                    size_t h) {
  return binary_elementwise<uint8_t>(
      src_a, sa, src_b, sb, dst, sd, w, h, [](auto a, auto b, size_t vl) {
        return mask_to_byte(__riscv_vmsgtu_vv_u8m1_b8(a, b, vl), vl);
      });
}

}  // namespace kleidicv::rvv
