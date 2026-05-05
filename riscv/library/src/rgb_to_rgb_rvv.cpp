// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV RGB-family permutations: vlsegN load → rebuild tuple in target order →
// vssegM store. RVV vector types are sizeless and can't live in arrays, so
// the index-based reorder is done with a switch helper that returns the
// requested tuple element.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"
#include "rgb_to_rgb_decls.h"

namespace kleidicv::rvv {

namespace {

inline vuint8m1_t pick3(vuint8m1x3_t in, int idx) {
  switch (idx) {
    case 0: return __riscv_vget_v_u8m1x3_u8m1(in, 0);
    case 1: return __riscv_vget_v_u8m1x3_u8m1(in, 1);
    default: return __riscv_vget_v_u8m1x3_u8m1(in, 2);
  }
}
inline vuint8m1_t pick4(vuint8m1x4_t in, int idx) {
  switch (idx) {
    case 0: return __riscv_vget_v_u8m1x4_u8m1(in, 0);
    case 1: return __riscv_vget_v_u8m1x4_u8m1(in, 1);
    case 2: return __riscv_vget_v_u8m1x4_u8m1(in, 2);
    default: return __riscv_vget_v_u8m1x4_u8m1(in, 3);
  }
}

inline void row_loop_3in_3out(const uint8_t *rs, uint8_t *rd, size_t width,
                              int i0, int i1, int i2) {
  size_t vl;
  for (size_t x = 0; x < width; x += vl) {
    vl = __riscv_vsetvl_e8m1(width - x);
    vuint8m1x3_t in = __riscv_vlseg3e8_v_u8m1x3(rs + 3 * x, vl);
    vuint8m1x3_t out =
        __riscv_vcreate_v_u8m1x3(pick3(in, i0), pick3(in, i1), pick3(in, i2));
    __riscv_vsseg3e8_v_u8m1x3(rd + 3 * x, out, vl);
  }
}

inline void row_loop_4in_4out(const uint8_t *rs, uint8_t *rd, size_t width,
                              int i0, int i1, int i2, int i3) {
  size_t vl;
  for (size_t x = 0; x < width; x += vl) {
    vl = __riscv_vsetvl_e8m1(width - x);
    vuint8m1x4_t in = __riscv_vlseg4e8_v_u8m1x4(rs + 4 * x, vl);
    vuint8m1x4_t out = __riscv_vcreate_v_u8m1x4(
        pick4(in, i0), pick4(in, i1), pick4(in, i2), pick4(in, i3));
    __riscv_vsseg4e8_v_u8m1x4(rd + 4 * x, out, vl);
  }
}

inline void row_loop_3in_4out(const uint8_t *rs, uint8_t *rd, size_t width,
                              int i0, int i1, int i2) {
  size_t vl;
  for (size_t x = 0; x < width; x += vl) {
    vl = __riscv_vsetvl_e8m1(width - x);
    vuint8m1x3_t in = __riscv_vlseg3e8_v_u8m1x3(rs + 3 * x, vl);
    vuint8m1_t alpha = __riscv_vmv_v_x_u8m1(0xFF, vl);
    vuint8m1x4_t out = __riscv_vcreate_v_u8m1x4(
        pick3(in, i0), pick3(in, i1), pick3(in, i2), alpha);
    __riscv_vsseg4e8_v_u8m1x4(rd + 4 * x, out, vl);
  }
}

inline void row_loop_4in_3out(const uint8_t *rs, uint8_t *rd, size_t width,
                              int i0, int i1, int i2) {
  size_t vl;
  for (size_t x = 0; x < width; x += vl) {
    vl = __riscv_vsetvl_e8m1(width - x);
    vuint8m1x4_t in = __riscv_vlseg4e8_v_u8m1x4(rs + 4 * x, vl);
    vuint8m1x3_t out =
        __riscv_vcreate_v_u8m1x3(pick4(in, i0), pick4(in, i1), pick4(in, i2));
    __riscv_vsseg3e8_v_u8m1x3(rd + 3 * x, out, vl);
  }
}

#define ROW_DRIVER(BODY)                                                      \
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;                       \
  if (width == 0 || height == 0) return KLEIDICV_OK;                          \
  for (size_t y = 0; y < height; ++y) {                                       \
    const uint8_t *rs = src + y * src_stride;                                 \
    uint8_t *rd = dst + y * dst_stride;                                       \
    BODY;                                                                     \
  }                                                                           \
  return KLEIDICV_OK

}  // namespace

kleidicv_error_t rgb_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  ROW_DRIVER(row_loop_3in_3out(rs, rd, width, 2, 1, 0));
}
kleidicv_error_t rgb_to_rgb_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  ROW_DRIVER(row_loop_3in_3out(rs, rd, width, 0, 1, 2));
}
kleidicv_error_t rgba_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  ROW_DRIVER(row_loop_4in_4out(rs, rd, width, 2, 1, 0, 3));
}
kleidicv_error_t rgba_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  ROW_DRIVER(row_loop_4in_4out(rs, rd, width, 0, 1, 2, 3));
}
kleidicv_error_t rgb_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  ROW_DRIVER(row_loop_3in_4out(rs, rd, width, 2, 1, 0));
}
kleidicv_error_t rgb_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  ROW_DRIVER(row_loop_3in_4out(rs, rd, width, 0, 1, 2));
}
kleidicv_error_t rgba_to_bgr_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  ROW_DRIVER(row_loop_4in_3out(rs, rd, width, 2, 1, 0));
}
kleidicv_error_t rgba_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  ROW_DRIVER(row_loop_4in_3out(rs, rd, width, 0, 1, 2));
}

#undef ROW_DRIVER

}  // namespace kleidicv::rvv
