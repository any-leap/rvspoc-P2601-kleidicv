// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// f32 ↔ {u8,s8} via fp/int convert + narrow/widen. Strip-mine on the
// f32m1 SEW=32 ratio so the matching int LMUL is mf4 (8b) for the byte side.
//
// Saturation: f32→int uses vfcvt (rounds per frm) producing i32; the wide
// value is then narrowed via vnclip[u] which saturates at each step. We do
// two narrowing steps (32→16 then 16→8) so the final result clamps to the
// destination type's range.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "float_conv_decls.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

namespace {

#define ROW_DRIVER_F32_TO(DT, BODY)                                            \
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;                       \
  if (width == 0 || height == 0) return KLEIDICV_OK;                          \
  for (size_t y = 0; y < height; ++y) {                                       \
    const float *rs = reinterpret_cast<const float *>(                        \
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);             \
    DT *rd = reinterpret_cast<DT *>(reinterpret_cast<uint8_t *>(dst) +        \
                                    y * dst_stride);                          \
    size_t vl;                                                                \
    for (size_t x = 0; x < width; x += vl) {                                  \
      vl = __riscv_vsetvl_e32m1(width - x);                                   \
      vfloat32m1_t v = __riscv_vle32_v_f32m1(rs + x, vl);                    \
      BODY                                                                    \
    }                                                                         \
  }                                                                           \
  return KLEIDICV_OK;

#define ROW_DRIVER_INT_TO_F32(ST, BODY)                                        \
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;                       \
  if (width == 0 || height == 0) return KLEIDICV_OK;                          \
  for (size_t y = 0; y < height; ++y) {                                       \
    const ST *rs = reinterpret_cast<const ST *>(                              \
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);             \
    float *rd = reinterpret_cast<float *>(                                    \
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);                   \
    size_t vl;                                                                \
    for (size_t x = 0; x < width; x += vl) {                                  \
      vl = __riscv_vsetvl_e32m1(width - x);                                   \
      BODY                                                                    \
    }                                                                         \
  }                                                                           \
  return KLEIDICV_OK;

}  // namespace

kleidicv_error_t f32_to_u8(const float *src, size_t src_stride, uint8_t *dst,
                           size_t dst_stride, size_t width, size_t height) {
  ROW_DRIVER_F32_TO(uint8_t, {
    // NaN must convert to 0 per the public API. vfmax/vfmin propagate NaN
    // (NaN survives min/max), so detect NaN with vmfne(self, self) and
    // replace with 0 BEFORE the fp clamp + vfcvt. After the NaN→0 step,
    // ±Inf clamps to [0, 255] cleanly via the same min/max.
    vbool32_t nan_mask = __riscv_vmfne_vv_f32m1_b32(v, v, vl);
    v = __riscv_vfmerge_vfm_f32m1(v, 0.0F, nan_mask, vl);
    v = __riscv_vfmax_vf_f32m1(v, 0.0f, vl);
    v = __riscv_vfmin_vf_f32m1(v, 255.0f, vl);
    vuint32m1_t i32 = __riscv_vfcvt_xu_f_v_u32m1(v, vl);
    vuint16mf2_t i16 = __riscv_vncvt_x_x_w_u16mf2(i32, vl);
    vuint8mf4_t out = __riscv_vncvt_x_x_w_u8mf4(i16, vl);
    __riscv_vse8_v_u8mf4(rd + x, out, vl);
  })
}

kleidicv_error_t f32_to_s8(const float *src, size_t src_stride, int8_t *dst,
                           size_t dst_stride, size_t width, size_t height) {
  ROW_DRIVER_F32_TO(int8_t, {
    // Same NaN→0 handling as f32_to_u8 above.
    vbool32_t nan_mask = __riscv_vmfne_vv_f32m1_b32(v, v, vl);
    v = __riscv_vfmerge_vfm_f32m1(v, 0.0F, nan_mask, vl);
    v = __riscv_vfmax_vf_f32m1(v, -128.0f, vl);
    v = __riscv_vfmin_vf_f32m1(v, 127.0f, vl);
    vint32m1_t i32 = __riscv_vfcvt_x_f_v_i32m1(v, vl);
    vint16mf2_t i16 = __riscv_vncvt_x_x_w_i16mf2(i32, vl);
    vint8mf4_t out = __riscv_vncvt_x_x_w_i8mf4(i16, vl);
    __riscv_vse8_v_i8mf4(rd + x, out, vl);
  })
}

kleidicv_error_t u8_to_f32(const uint8_t *src, size_t src_stride, float *dst,
                           size_t dst_stride, size_t width, size_t height) {
  ROW_DRIVER_INT_TO_F32(uint8_t, {
    vuint8mf4_t u8 = __riscv_vle8_v_u8mf4(rs + x, vl);
    vuint16mf2_t u16 = __riscv_vzext_vf2_u16mf2(u8, vl);
    vuint32m1_t u32 = __riscv_vzext_vf2_u32m1(u16, vl);
    vfloat32m1_t vf = __riscv_vfcvt_f_xu_v_f32m1(u32, vl);
    __riscv_vse32_v_f32m1(rd + x, vf, vl);
  })
}

kleidicv_error_t s8_to_f32(const int8_t *src, size_t src_stride, float *dst,
                           size_t dst_stride, size_t width, size_t height) {
  ROW_DRIVER_INT_TO_F32(int8_t, {
    vint8mf4_t i8 = __riscv_vle8_v_i8mf4(rs + x, vl);
    vint16mf2_t i16 = __riscv_vsext_vf2_i16mf2(i8, vl);
    vint32m1_t i32 = __riscv_vsext_vf2_i32m1(i16, vl);
    vfloat32m1_t vf = __riscv_vfcvt_f_x_v_f32m1(i32, vl);
    __riscv_vse32_v_f32m1(rd + x, vf, vl);
  })
}

}  // namespace kleidicv::rvv
