// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// remap with int16 integer-coordinate mapxy (no interpolation, just per-pixel
// pickup). Channels=1, REPLICATE/CONSTANT borders. Other remap variants
// (s16point5 fractional, f32 floating coords) remain undefined — SPOC scope.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "dispatch.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;

template <typename T>
kleidicv_error_t do_remap_s16_scalar(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels, const int16_t *mapxy, size_t mapxy_stride,
    kleidicv_border_type_t border_type, const T *border_value) {
  if (!src || !dst || !mapxy) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels != 1) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0)
    return KLEIDICV_OK;
  bool replicate = border_type == KLEIDICV_BORDER_TYPE_REPLICATE;
  bool constant = border_type == KLEIDICV_BORDER_TYPE_CONSTANT;
  if (!replicate && !constant) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  T fill = constant && border_value ? *border_value : T{0};

  size_t src_stride_e = src_stride / sizeof(T);
  size_t dst_stride_e = dst_stride / sizeof(T);
  size_t map_stride_e = mapxy_stride / sizeof(int16_t);

  for (size_t dy = 0; dy < dst_height; ++dy) {
    const int16_t *map_row = mapxy + dy * map_stride_e;
    T *drow = dst + dy * dst_stride_e;
    for (size_t dx = 0; dx < dst_width; ++dx) {
      int sx = map_row[2 * dx + 0];
      int sy = map_row[2 * dx + 1];
      bool oob = sx < 0 || sy < 0 ||
                 static_cast<size_t>(sx) >= src_width ||
                 static_cast<size_t>(sy) >= src_height;
      if (oob) {
        if (constant) {
          drow[dx] = fill;
          continue;
        }
        if (sx < 0) sx = 0;
        if (sy < 0) sy = 0;
        if (static_cast<size_t>(sx) >= src_width)
          sx = static_cast<int>(src_width) - 1;
        if (static_cast<size_t>(sy) >= src_height)
          sy = static_cast<int>(src_height) - 1;
      }
      drow[dx] = src[static_cast<size_t>(sy) * src_stride_e +
                     static_cast<size_t>(sx)];
    }
  }
  return KLEIDICV_OK;
}

// RVV path: per-row strip-mine with vlseg2e16 (loads sx/sy pairs), clamp to
// valid range, compute byte offset, gather with vluxei32, then merge constant
// fill where the original index was out-of-bounds.
kleidicv_error_t remap_s16_u8_rvv(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const int16_t *mapxy, size_t mapxy_stride, bool constant_border,
    uint8_t fill) {
  const size_t map_stride_e = mapxy_stride / sizeof(int16_t);
  const int32_t W = static_cast<int32_t>(src_width);
  const int32_t H = static_cast<int32_t>(src_height);
  const uint32_t Sstride = static_cast<uint32_t>(src_stride);

  for (size_t dy = 0; dy < dst_height; ++dy) {
    const int16_t *map_row = mapxy + dy * map_stride_e;
    uint8_t *drow = dst + dy * dst_stride;
    size_t dx = 0;
    while (dx < dst_width) {
      size_t vl = __riscv_vsetvl_e8m1(dst_width - dx);
      // Segmented load of (sx, sy) interleaved s16 pairs.
      vint16m2x2_t pair = __riscv_vlseg2e16_v_i16m2x2(map_row + 2 * dx, vl);
      vint16m2_t sx16 = __riscv_vget_v_i16m2x2_i16m2(pair, 0);
      vint16m2_t sy16 = __riscv_vget_v_i16m2x2_i16m2(pair, 1);
      // Widen to s32.
      vint32m4_t sx = __riscv_vsext_vf2_i32m4(sx16, vl);
      vint32m4_t sy = __riscv_vsext_vf2_i32m4(sy16, vl);

      // Out-of-bounds mask: sx<0 or sy<0 or sx>=W or sy>=H.
      vbool8_t m_neg_x = __riscv_vmslt_vx_i32m4_b8(sx, 0, vl);
      vbool8_t m_neg_y = __riscv_vmslt_vx_i32m4_b8(sy, 0, vl);
      vbool8_t m_oob_x = __riscv_vmsge_vx_i32m4_b8(sx, W, vl);
      vbool8_t m_oob_y = __riscv_vmsge_vx_i32m4_b8(sy, H, vl);
      vbool8_t m_oob = __riscv_vmor_mm_b8(m_neg_x, m_neg_y, vl);
      m_oob = __riscv_vmor_mm_b8(m_oob, m_oob_x, vl);
      m_oob = __riscv_vmor_mm_b8(m_oob, m_oob_y, vl);

      // Clamp to [0, W-1] / [0, H-1].
      vint32m4_t sx_c = __riscv_vmax_vx_i32m4(sx, 0, vl);
      sx_c = __riscv_vmin_vx_i32m4(sx_c, W - 1, vl);
      vint32m4_t sy_c = __riscv_vmax_vx_i32m4(sy, 0, vl);
      sy_c = __riscv_vmin_vx_i32m4(sy_c, H - 1, vl);

      // byte_off = sy_c * src_stride + sx_c (u8 → 1 byte/pixel).
      vuint32m4_t off =
          __riscv_vmul_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(sy_c),
                                 Sstride, vl);
      off = __riscv_vadd_vv_u32m4(off, __riscv_vreinterpret_v_i32m4_u32m4(sx_c),
                                   vl);

      vuint8m1_t pix = __riscv_vluxei32_v_u8m1(src, off, vl);
      if (constant_border) {
        pix = __riscv_vmerge_vxm_u8m1(pix, fill, m_oob, vl);
      }
      __riscv_vse8_v_u8m1(drow + dx, pix, vl);
      dx += vl;
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t remap_s16_u16_rvv(
    const uint16_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint16_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const int16_t *mapxy, size_t mapxy_stride, bool constant_border,
    uint16_t fill) {
  const size_t map_stride_e = mapxy_stride / sizeof(int16_t);
  const size_t dst_stride_e = dst_stride / sizeof(uint16_t);
  const int32_t W = static_cast<int32_t>(src_width);
  const int32_t H = static_cast<int32_t>(src_height);
  const uint32_t Sstride = static_cast<uint32_t>(src_stride);

  for (size_t dy = 0; dy < dst_height; ++dy) {
    const int16_t *map_row = mapxy + dy * map_stride_e;
    uint16_t *drow = dst + dy * dst_stride_e;
    size_t dx = 0;
    while (dx < dst_width) {
      // u16 output → SEW=16/m2 for output, SEW=32/m4 for indices (same VLMAX).
      size_t vl = __riscv_vsetvl_e16m2(dst_width - dx);
      vint16m2x2_t pair = __riscv_vlseg2e16_v_i16m2x2(map_row + 2 * dx, vl);
      vint16m2_t sx16 = __riscv_vget_v_i16m2x2_i16m2(pair, 0);
      vint16m2_t sy16 = __riscv_vget_v_i16m2x2_i16m2(pair, 1);
      vint32m4_t sx = __riscv_vsext_vf2_i32m4(sx16, vl);
      vint32m4_t sy = __riscv_vsext_vf2_i32m4(sy16, vl);

      vbool8_t m_neg_x = __riscv_vmslt_vx_i32m4_b8(sx, 0, vl);
      vbool8_t m_neg_y = __riscv_vmslt_vx_i32m4_b8(sy, 0, vl);
      vbool8_t m_oob_x = __riscv_vmsge_vx_i32m4_b8(sx, W, vl);
      vbool8_t m_oob_y = __riscv_vmsge_vx_i32m4_b8(sy, H, vl);
      vbool8_t m_oob = __riscv_vmor_mm_b8(m_neg_x, m_neg_y, vl);
      m_oob = __riscv_vmor_mm_b8(m_oob, m_oob_x, vl);
      m_oob = __riscv_vmor_mm_b8(m_oob, m_oob_y, vl);

      vint32m4_t sx_c = __riscv_vmax_vx_i32m4(sx, 0, vl);
      sx_c = __riscv_vmin_vx_i32m4(sx_c, W - 1, vl);
      vint32m4_t sy_c = __riscv_vmax_vx_i32m4(sy, 0, vl);
      sy_c = __riscv_vmin_vx_i32m4(sy_c, H - 1, vl);

      // byte_off = sy_c * src_stride + sx_c * 2 (u16).
      vuint32m4_t off =
          __riscv_vmul_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(sy_c),
                                 Sstride, vl);
      vuint32m4_t sx_bytes = __riscv_vsll_vx_u32m4(
          __riscv_vreinterpret_v_i32m4_u32m4(sx_c), 1, vl);
      off = __riscv_vadd_vv_u32m4(off, sx_bytes, vl);

      vuint16m2_t pix = __riscv_vluxei32_v_u16m2(src, off, vl);
      if (constant_border) {
        pix = __riscv_vmerge_vxm_u16m2(pix, fill, m_oob, vl);
      }
      __riscv_vse16_v_u16m2(drow + dx, pix, vl);
      dx += vl;
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t remap_s16_u8_impl(const uint8_t *src, size_t src_stride,
                                   size_t src_width, size_t src_height,
                                   uint8_t *dst, size_t dst_stride,
                                   size_t dst_width, size_t dst_height,
                                   size_t channels, const int16_t *mapxy,
                                   size_t mapxy_stride,
                                   kleidicv_border_type_t border_type,
                                   const uint8_t *border_value) {
  if (!src || !dst || !mapxy) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels != 1) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0)
    return KLEIDICV_OK;
  bool replicate = border_type == KLEIDICV_BORDER_TYPE_REPLICATE;
  bool constant = border_type == KLEIDICV_BORDER_TYPE_CONSTANT;
  if (!replicate && !constant) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (active_backend() == Backend::Rvv) {
    uint8_t fill = constant && border_value ? *border_value : 0;
    return remap_s16_u8_rvv(src, src_stride, src_width, src_height, dst,
                             dst_stride, dst_width, dst_height, mapxy,
                             mapxy_stride, constant, fill);
  }
  return do_remap_s16_scalar<uint8_t>(src, src_stride, src_width, src_height,
                                       dst, dst_stride, dst_width, dst_height,
                                       channels, mapxy, mapxy_stride,
                                       border_type, border_value);
}

kleidicv_error_t remap_s16_u16_impl(const uint16_t *src, size_t src_stride,
                                    size_t src_width, size_t src_height,
                                    uint16_t *dst, size_t dst_stride,
                                    size_t dst_width, size_t dst_height,
                                    size_t channels, const int16_t *mapxy,
                                    size_t mapxy_stride,
                                    kleidicv_border_type_t border_type,
                                    const uint16_t *border_value) {
  if (!src || !dst || !mapxy) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels != 1) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0)
    return KLEIDICV_OK;
  bool replicate = border_type == KLEIDICV_BORDER_TYPE_REPLICATE;
  bool constant = border_type == KLEIDICV_BORDER_TYPE_CONSTANT;
  if (!replicate && !constant) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (active_backend() == Backend::Rvv) {
    uint16_t fill = constant && border_value ? *border_value : 0;
    return remap_s16_u16_rvv(src, src_stride, src_width, src_height, dst,
                              dst_stride, dst_width, dst_height, mapxy,
                              mapxy_stride, constant, fill);
  }
  return do_remap_s16_scalar<uint16_t>(src, src_stride, src_width, src_height,
                                        dst, dst_stride, dst_width, dst_height,
                                        channels, mapxy, mapxy_stride,
                                        border_type, border_value);
}

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_remap_s16_u8)(
    const uint8_t *, size_t, size_t, size_t, uint8_t *, size_t, size_t, size_t,
    size_t, const int16_t *, size_t, kleidicv_border_type_t,
    const uint8_t *) = remap_s16_u8_impl;
kleidicv_error_t (*kleidicv_remap_s16_u8_sme)(
    const uint8_t *, size_t, size_t, size_t, uint8_t *, size_t, size_t, size_t,
    size_t, const int16_t *, size_t, kleidicv_border_type_t,
    const uint8_t *) = remap_s16_u8_impl;
kleidicv_error_t (*kleidicv_remap_s16_u16)(
    const uint16_t *, size_t, size_t, size_t, uint16_t *, size_t, size_t,
    size_t, size_t, const int16_t *, size_t, kleidicv_border_type_t,
    const uint16_t *) = remap_s16_u16_impl;
kleidicv_error_t (*kleidicv_remap_s16_u16_sme)(
    const uint16_t *, size_t, size_t, size_t, uint16_t *, size_t, size_t,
    size_t, size_t, const int16_t *, size_t, kleidicv_border_type_t,
    const uint16_t *) = remap_s16_u16_impl;

}  // extern "C"
