// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV: |a| via max(a, vssub(0,a)) — saturated negation handles INT16_MIN
// overflow. sat add via vsadd. compare > threshold; merge.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "add_abs_with_threshold_decls.h"
#include "elementwise_rvv.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

kleidicv_error_t saturating_add_abs_with_threshold_s16(
    const int16_t *src_a, size_t sa, const int16_t *src_b, size_t sb,
    int16_t *dst, size_t sd, size_t w, size_t h, int16_t threshold) {
  return binary_elementwise<int16_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [threshold](auto a, auto b, size_t vl) {
        // Saturated abs(x) = max(x, sat(0 - x)). Saturated negation handles
        // INT16_MIN, whose true negation overflows.
        vint16m1_t zero = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t aa =
            __riscv_vmax_vv_i16m1(a, __riscv_vssub_vv_i16m1(zero, a, vl), vl);
        vint16m1_t bb =
            __riscv_vmax_vv_i16m1(b, __riscv_vssub_vv_i16m1(zero, b, vl), vl);
        vint16m1_t sum = __riscv_vsadd_vv_i16m1(aa, bb, vl);
        vbool16_t mask = __riscv_vmsgt_vx_i16m1_b16(sum, threshold, vl);
        return __riscv_vmerge_vvm_i16m1(zero, sum, mask, vl);
      });
}

}  // namespace kleidicv::rvv
