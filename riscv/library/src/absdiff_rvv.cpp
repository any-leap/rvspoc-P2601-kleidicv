// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV 1.0 implementation of saturating_absdiff.
//
// Strategy
// ========
// Unsigned (u8, u16): |a - b| = vmaxu(a,b) - vminu(a,b) — no saturation
// possible in the unsigned domain.
//
// Signed (s8, s16, s32): the difference can be twice the input range and abs
// can overflow the input type. Widen to the next signed size, subtract there,
// take abs as max(d, -d), then narrow with saturation via vnclip — which
// clamps to [INT_MIN, INT_MAX] of the destination width. Because the absolute
// value is non-negative, the lower bound is never hit; only the upper bound
// matters.
//
// Row-walk and strip-mine live in elementwise_rvv.h. The signed lambdas widen
// internally and narrow back to T at the end, so the input/output element
// width matches what the helper loads/stores.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "absdiff_decls.h"
#include "elementwise_rvv.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

kleidicv_error_t saturating_absdiff_u8(const uint8_t *src_a, size_t sa,
                                       const uint8_t *src_b, size_t sb,
                                       uint8_t *dst, size_t sd, size_t w,
                                       size_t h) {
  return binary_elementwise<uint8_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](auto a, auto b, size_t vl) {
        auto vmax = __riscv_vmaxu_vv_u8m1(a, b, vl);
        auto vmin = __riscv_vminu_vv_u8m1(a, b, vl);
        return __riscv_vsub_vv_u8m1(vmax, vmin, vl);
      });
}

kleidicv_error_t saturating_absdiff_u16(const uint16_t *src_a, size_t sa,
                                        const uint16_t *src_b, size_t sb,
                                        uint16_t *dst, size_t sd, size_t w,
                                        size_t h) {
  return binary_elementwise<uint16_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](auto a, auto b, size_t vl) {
        auto vmax = __riscv_vmaxu_vv_u16m1(a, b, vl);
        auto vmin = __riscv_vminu_vv_u16m1(a, b, vl);
        return __riscv_vsub_vv_u16m1(vmax, vmin, vl);
      });
}

kleidicv_error_t saturating_absdiff_s8(const int8_t *src_a, size_t sa,
                                       const int8_t *src_b, size_t sb,
                                       int8_t *dst, size_t sd, size_t w,
                                       size_t h) {
  return binary_elementwise<int8_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](auto a, auto b, size_t vl) {
        vint16m2_t diff = __riscv_vwsub_vv_i16m2(a, b, vl);
        vint16m2_t neg = __riscv_vneg_v_i16m2(diff, vl);
        vint16m2_t adiff = __riscv_vmax_vv_i16m2(diff, neg, vl);
        return __riscv_vnclip_wx_i8m1(adiff, 0, __RISCV_VXRM_RNU, vl);
      });
}

kleidicv_error_t saturating_absdiff_s16(const int16_t *src_a, size_t sa,
                                        const int16_t *src_b, size_t sb,
                                        int16_t *dst, size_t sd, size_t w,
                                        size_t h) {
  return binary_elementwise<int16_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](auto a, auto b, size_t vl) {
        vint32m2_t diff = __riscv_vwsub_vv_i32m2(a, b, vl);
        vint32m2_t neg = __riscv_vneg_v_i32m2(diff, vl);
        vint32m2_t adiff = __riscv_vmax_vv_i32m2(diff, neg, vl);
        return __riscv_vnclip_wx_i16m1(adiff, 0, __RISCV_VXRM_RNU, vl);
      });
}

kleidicv_error_t saturating_absdiff_s32(const int32_t *src_a, size_t sa,
                                        const int32_t *src_b, size_t sb,
                                        int32_t *dst, size_t sd, size_t w,
                                        size_t h) {
  return binary_elementwise<int32_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](auto a, auto b, size_t vl) {
        vint64m2_t diff = __riscv_vwsub_vv_i64m2(a, b, vl);
        vint64m2_t neg = __riscv_vneg_v_i64m2(diff, vl);
        vint64m2_t adiff = __riscv_vmax_vv_i64m2(diff, neg, vl);
        return __riscv_vnclip_wx_i32m1(adiff, 0, __RISCV_VXRM_RNU, vl);
      });
}

}  // namespace kleidicv::rvv
