// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV threshold_binary_u8: vmsgtu (mask = src > threshold) → vmerge to pick
// `value` where mask is set, 0 otherwise.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "elementwise_rvv.h"
#include "kleidicv/kleidicv.h"
#include "threshold_binary_decls.h"

namespace kleidicv::rvv {

kleidicv_error_t threshold_binary_u8(const uint8_t *src, size_t src_stride,
                                     uint8_t *dst, size_t dst_stride,
                                     size_t width, size_t height,
                                     uint8_t threshold, uint8_t value) {
  return unary_elementwise<uint8_t>(
      src, src_stride, dst, dst_stride, width, height,
      [threshold, value](auto v, size_t vl) {
        vbool8_t mask = __riscv_vmsgtu_vx_u8m1_b8(v, threshold, vl);
        vuint8m1_t zeros = __riscv_vmv_v_x_u8m1(0, vl);
        return __riscv_vmerge_vxm_u8m1(zeros, value, mask, vl);
      });
}

}  // namespace kleidicv::rvv
