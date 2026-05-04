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
// LMUL choice: the widening path uses input LMUL=1 widening to LMUL=2; a
// single VL register-group fits comfortably even at vlen=128. The unsigned
// path also uses LMUL=1 for symmetry. Phase 4 may revisit for throughput.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "absdiff_decls.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

namespace {

// Walks each row, calling `kernel(dst_row, a_row, b_row, width)` per row.
template <typename T, typename Kernel>
kleidicv_error_t for_each_row(const T *src_a, size_t src_a_stride,
                              const T *src_b, size_t src_b_stride, T *dst,
                              size_t dst_stride, size_t width, size_t height,
                              Kernel kernel) {
  if (!src_a || !src_b || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const T *row_a =
        reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(src_a) +
                                    y * src_a_stride);
    const T *row_b =
        reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(src_b) +
                                    y * src_b_stride);
    T *row_dst = reinterpret_cast<T *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    kernel(row_dst, row_a, row_b, width);
  }
  return KLEIDICV_OK;
}

}  // namespace

kleidicv_error_t saturating_absdiff_u8(const uint8_t *src_a, size_t sa,
                                       const uint8_t *src_b, size_t sb,
                                       uint8_t *dst, size_t sd, size_t w,
                                       size_t h) {
  return for_each_row<uint8_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](uint8_t *d, const uint8_t *a, const uint8_t *b, size_t n) {
        size_t vl;
        for (size_t x = 0; x < n; x += vl) {
          vl = __riscv_vsetvl_e8m1(n - x);
          vuint8m1_t va = __riscv_vle8_v_u8m1(a + x, vl);
          vuint8m1_t vb = __riscv_vle8_v_u8m1(b + x, vl);
          vuint8m1_t vmax = __riscv_vmaxu_vv_u8m1(va, vb, vl);
          vuint8m1_t vmin = __riscv_vminu_vv_u8m1(va, vb, vl);
          vuint8m1_t vd = __riscv_vsub_vv_u8m1(vmax, vmin, vl);
          __riscv_vse8_v_u8m1(d + x, vd, vl);
        }
      });
}

kleidicv_error_t saturating_absdiff_u16(const uint16_t *src_a, size_t sa,
                                        const uint16_t *src_b, size_t sb,
                                        uint16_t *dst, size_t sd, size_t w,
                                        size_t h) {
  return for_each_row<uint16_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](uint16_t *d, const uint16_t *a, const uint16_t *b, size_t n) {
        size_t vl;
        for (size_t x = 0; x < n; x += vl) {
          vl = __riscv_vsetvl_e16m1(n - x);
          vuint16m1_t va = __riscv_vle16_v_u16m1(a + x, vl);
          vuint16m1_t vb = __riscv_vle16_v_u16m1(b + x, vl);
          vuint16m1_t vmax = __riscv_vmaxu_vv_u16m1(va, vb, vl);
          vuint16m1_t vmin = __riscv_vminu_vv_u16m1(va, vb, vl);
          vuint16m1_t vd = __riscv_vsub_vv_u16m1(vmax, vmin, vl);
          __riscv_vse16_v_u16m1(d + x, vd, vl);
        }
      });
}

kleidicv_error_t saturating_absdiff_s8(const int8_t *src_a, size_t sa,
                                       const int8_t *src_b, size_t sb,
                                       int8_t *dst, size_t sd, size_t w,
                                       size_t h) {
  return for_each_row<int8_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](int8_t *d, const int8_t *a, const int8_t *b, size_t n) {
        size_t vl;
        for (size_t x = 0; x < n; x += vl) {
          vl = __riscv_vsetvl_e8m1(n - x);
          vint8m1_t va = __riscv_vle8_v_i8m1(a + x, vl);
          vint8m1_t vb = __riscv_vle8_v_i8m1(b + x, vl);
          // Widen via vwsub (signed widening subtract): i8m1 - i8m1 -> i16m2.
          vint16m2_t diff = __riscv_vwsub_vv_i16m2(va, vb, vl);
          vint16m2_t neg = __riscv_vneg_v_i16m2(diff, vl);
          vint16m2_t adiff = __riscv_vmax_vv_i16m2(diff, neg, vl);
          // Narrow with signed saturation; non-negative so only INT8_MAX side
          // matters.
          vint8m1_t vd = __riscv_vnclip_wx_i8m1(adiff, 0, vl);
          __riscv_vse8_v_i8m1(d + x, vd, vl);
        }
      });
}

kleidicv_error_t saturating_absdiff_s16(const int16_t *src_a, size_t sa,
                                        const int16_t *src_b, size_t sb,
                                        int16_t *dst, size_t sd, size_t w,
                                        size_t h) {
  return for_each_row<int16_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](int16_t *d, const int16_t *a, const int16_t *b, size_t n) {
        size_t vl;
        for (size_t x = 0; x < n; x += vl) {
          vl = __riscv_vsetvl_e16m1(n - x);
          vint16m1_t va = __riscv_vle16_v_i16m1(a + x, vl);
          vint16m1_t vb = __riscv_vle16_v_i16m1(b + x, vl);
          vint32m2_t diff = __riscv_vwsub_vv_i32m2(va, vb, vl);
          vint32m2_t neg = __riscv_vneg_v_i32m2(diff, vl);
          vint32m2_t adiff = __riscv_vmax_vv_i32m2(diff, neg, vl);
          vint16m1_t vd =
              __riscv_vnclip_wx_i16m1(adiff, 0, vl);
          __riscv_vse16_v_i16m1(d + x, vd, vl);
        }
      });
}

kleidicv_error_t saturating_absdiff_s32(const int32_t *src_a, size_t sa,
                                        const int32_t *src_b, size_t sb,
                                        int32_t *dst, size_t sd, size_t w,
                                        size_t h) {
  return for_each_row<int32_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](int32_t *d, const int32_t *a, const int32_t *b, size_t n) {
        size_t vl;
        for (size_t x = 0; x < n; x += vl) {
          vl = __riscv_vsetvl_e32m1(n - x);
          vint32m1_t va = __riscv_vle32_v_i32m1(a + x, vl);
          vint32m1_t vb = __riscv_vle32_v_i32m1(b + x, vl);
          vint64m2_t diff = __riscv_vwsub_vv_i64m2(va, vb, vl);
          vint64m2_t neg = __riscv_vneg_v_i64m2(diff, vl);
          vint64m2_t adiff = __riscv_vmax_vv_i64m2(diff, neg, vl);
          vint32m1_t vd =
              __riscv_vnclip_wx_i32m1(adiff, 0, vl);
          __riscv_vse32_v_i32m1(d + x, vd, vl);
        }
      });
}

}  // namespace kleidicv::rvv
