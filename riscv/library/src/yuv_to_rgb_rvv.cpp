// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV BT.601-7 YUV444→RGB. Same shape as rgb_to_yuv: load YUV, widen to s32,
// compute R/G/B in s32 with scalar-MAC, narrow to u8 with saturation, then
// vssegN store to interleaved RGB(A) / BGR(A).

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"
#include "yuv_to_rgb_decls.h"

namespace kleidicv::rvv {

namespace {

constexpr int kWeightScale = 14;
constexpr int kRound = 1 << (kWeightScale - 1);
constexpr int kVRWeight = 18678;
constexpr int kUGWeight = -6472;
constexpr int kVGWeight = -9519;
constexpr int kUBWeight = 33292;

inline vint32m4_t widen_u8_to_s32(vuint8m1_t v, size_t vl) {
  vuint16m2_t u16 = __riscv_vzext_vf2_u16m2(v, vl);
  vuint32m4_t u32 = __riscv_vzext_vf2_u32m4(u16, vl);
  return __riscv_vreinterpret_v_u32m4_i32m4(u32);
}

inline vuint8m1_t narrow_s32_to_u8_sat(vint32m4_t v, size_t vl) {
  vint32m4_t v_nn = __riscv_vmax_vx_i32m4(v, 0, vl);
  vuint32m4_t u32 = __riscv_vreinterpret_v_i32m4_u32m4(v_nn);
  vuint16m2_t u16 = __riscv_vnclipu_wx_u16m2(u32, 0, __RISCV_VXRM_RNU, vl);
  return __riscv_vnclipu_wx_u8m1(u16, 0, __RISCV_VXRM_RNU, vl);
}

}  // namespace

template <bool BGR, bool kAlpha>
kleidicv_error_t yuv444_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;
  constexpr size_t out_chan = kAlpha ? 4 : 3;

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *rs = src + y * src_stride;
    uint8_t *rd = dst + y * dst_stride;
    size_t vl;
    for (size_t x = 0; x < width; x += vl) {
      vl = __riscv_vsetvl_e8m1(width - x);
      vuint8m1x3_t in = __riscv_vlseg3e8_v_u8m1x3(rs + 3 * x, vl);
      vuint8m1_t y8 = __riscv_vget_v_u8m1x3_u8m1(in, 0);
      vuint8m1_t u8 = __riscv_vget_v_u8m1x3_u8m1(in, 1);
      vuint8m1_t v8 = __riscv_vget_v_u8m1x3_u8m1(in, 2);

      vint32m4_t Y = widen_u8_to_s32(y8, vl);
      vint32m4_t U = __riscv_vsub_vx_i32m4(widen_u8_to_s32(u8, vl), 128, vl);
      vint32m4_t V = __riscv_vsub_vx_i32m4(widen_u8_to_s32(v8, vl), 128, vl);

      // R = Y + ((V * kVRWeight + kRound) >> 14)
      vint32m4_t Rterm = __riscv_vmul_vx_i32m4(V, kVRWeight, vl);
      Rterm = __riscv_vadd_vx_i32m4(Rterm, kRound, vl);
      vint32m4_t R = __riscv_vadd_vv_i32m4(
          Y, __riscv_vsra_vx_i32m4(Rterm, kWeightScale, vl), vl);

      // G = Y + ((U*kUG + V*kVG + kRound) >> 14)
      vint32m4_t Gterm = __riscv_vmul_vx_i32m4(U, kUGWeight, vl);
      Gterm = __riscv_vmacc_vx_i32m4(Gterm, kVGWeight, V, vl);
      Gterm = __riscv_vadd_vx_i32m4(Gterm, kRound, vl);
      vint32m4_t G = __riscv_vadd_vv_i32m4(
          Y, __riscv_vsra_vx_i32m4(Gterm, kWeightScale, vl), vl);

      // B = Y + ((U*kUB + kRound) >> 14)
      vint32m4_t Bterm = __riscv_vmul_vx_i32m4(U, kUBWeight, vl);
      Bterm = __riscv_vadd_vx_i32m4(Bterm, kRound, vl);
      vint32m4_t B = __riscv_vadd_vv_i32m4(
          Y, __riscv_vsra_vx_i32m4(Bterm, kWeightScale, vl), vl);

      vuint8m1_t r_byte = narrow_s32_to_u8_sat(R, vl);
      vuint8m1_t g_byte = narrow_s32_to_u8_sat(G, vl);
      vuint8m1_t b_byte = narrow_s32_to_u8_sat(B, vl);

      if constexpr (kAlpha) {
        vuint8m1_t a = __riscv_vmv_v_x_u8m1(0xFF, vl);
        if constexpr (BGR) {
          vuint8m1x4_t out =
              __riscv_vcreate_v_u8m1x4(b_byte, g_byte, r_byte, a);
          __riscv_vsseg4e8_v_u8m1x4(rd + out_chan * x, out, vl);
        } else {
          vuint8m1x4_t out =
              __riscv_vcreate_v_u8m1x4(r_byte, g_byte, b_byte, a);
          __riscv_vsseg4e8_v_u8m1x4(rd + out_chan * x, out, vl);
        }
      } else {
        if constexpr (BGR) {
          vuint8m1x3_t out =
              __riscv_vcreate_v_u8m1x3(b_byte, g_byte, r_byte);
          __riscv_vsseg3e8_v_u8m1x3(rd + out_chan * x, out, vl);
        } else {
          vuint8m1x3_t out =
              __riscv_vcreate_v_u8m1x3(r_byte, g_byte, b_byte);
          __riscv_vsseg3e8_v_u8m1x3(rd + out_chan * x, out, vl);
        }
      }
    }
  }
  return KLEIDICV_OK;
}

template kleidicv_error_t yuv444_to_rgb_u8<false, false>(
    const uint8_t *, size_t, uint8_t *, size_t, size_t, size_t);
template kleidicv_error_t yuv444_to_rgb_u8<false, true>(const uint8_t *, size_t,
                                                        uint8_t *, size_t,
                                                        size_t, size_t);
template kleidicv_error_t yuv444_to_rgb_u8<true, false>(const uint8_t *, size_t,
                                                        uint8_t *, size_t,
                                                        size_t, size_t);
template kleidicv_error_t yuv444_to_rgb_u8<true, true>(const uint8_t *, size_t,
                                                       uint8_t *, size_t,
                                                       size_t, size_t);

}  // namespace kleidicv::rvv
