// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV BT.601-7 RGB→YUV444 with 14-bit fixed-point coefficients.
//
// Pipeline (per strip): vlsegN load → widen each channel u8→s32 → scalar-MAC
// to compute Y_q in s32 → rounding shift → recover (B-Y), (R-Y) in s32 →
// scalar-MAC to compute U_q, V_q with the +128*2^14 bias → rounding shift →
// clamp to [0,255] → vnclipu twice (s32→u16→u8) → vsseg3e8 store.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"
#include "rgb_to_yuv_decls.h"

namespace kleidicv::rvv {

namespace {

constexpr int kRYWeight = 4899;
constexpr int kGYWeight = 9617;
constexpr int kBYWeight = 1868;
constexpr int kBUWeight = 8061;
constexpr int kRVWeight = 14369;
constexpr int kWeightScale = 14;
constexpr int kHalf = 128 << kWeightScale;
constexpr int kRound = 1 << (kWeightScale - 1);

inline vint32m4_t widen_u8_to_s32(vuint8m1_t v, size_t vl) {
  vuint16m2_t u16 = __riscv_vzext_vf2_u16m2(v, vl);
  vuint32m4_t u32 = __riscv_vzext_vf2_u32m4(u16, vl);
  return __riscv_vreinterpret_v_u32m4_i32m4(u32);
}

inline vuint8m1_t narrow_s32_to_u8_sat(vint32m4_t v, size_t vl) {
  // Clamp negatives to 0; positives reinterpret to u32 unchanged.
  vint32m4_t v_nn = __riscv_vmax_vx_i32m4(v, 0, vl);
  vuint32m4_t u32 = __riscv_vreinterpret_v_i32m4_u32m4(v_nn);
  vuint16m2_t u16 = __riscv_vnclipu_wx_u16m2(u32, 0, __RISCV_VXRM_RNU, vl);
  return __riscv_vnclipu_wx_u8m1(u16, 0, __RISCV_VXRM_RNU, vl);
}

}  // namespace

template <bool BGR, bool kAlpha>
kleidicv_error_t rgb_to_yuv444_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;
  constexpr size_t in_chan = kAlpha ? 4 : 3;

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *rs = src + y * src_stride;
    uint8_t *rd = dst + y * dst_stride;
    size_t vl;
    for (size_t x = 0; x < width; x += vl) {
      vl = __riscv_vsetvl_e8m1(width - x);
      vuint8m1_t r8, g8, b8;
      if constexpr (kAlpha) {
        vuint8m1x4_t in = __riscv_vlseg4e8_v_u8m1x4(rs + in_chan * x, vl);
        if constexpr (BGR) {
          b8 = __riscv_vget_v_u8m1x4_u8m1(in, 0);
          g8 = __riscv_vget_v_u8m1x4_u8m1(in, 1);
          r8 = __riscv_vget_v_u8m1x4_u8m1(in, 2);
        } else {
          r8 = __riscv_vget_v_u8m1x4_u8m1(in, 0);
          g8 = __riscv_vget_v_u8m1x4_u8m1(in, 1);
          b8 = __riscv_vget_v_u8m1x4_u8m1(in, 2);
        }
      } else {
        vuint8m1x3_t in = __riscv_vlseg3e8_v_u8m1x3(rs + in_chan * x, vl);
        if constexpr (BGR) {
          b8 = __riscv_vget_v_u8m1x3_u8m1(in, 0);
          g8 = __riscv_vget_v_u8m1x3_u8m1(in, 1);
          r8 = __riscv_vget_v_u8m1x3_u8m1(in, 2);
        } else {
          r8 = __riscv_vget_v_u8m1x3_u8m1(in, 0);
          g8 = __riscv_vget_v_u8m1x3_u8m1(in, 1);
          b8 = __riscv_vget_v_u8m1x3_u8m1(in, 2);
        }
      }
      vint32m4_t R = widen_u8_to_s32(r8, vl);
      vint32m4_t G = widen_u8_to_s32(g8, vl);
      vint32m4_t B = widen_u8_to_s32(b8, vl);

      // Y_q = R*kRY + G*kGY + B*kBY  (using scalar mac)
      vint32m4_t Yq = __riscv_vmul_vx_i32m4(R, kRYWeight, vl);
      Yq = __riscv_vmacc_vx_i32m4(Yq, kGYWeight, G, vl);
      Yq = __riscv_vmacc_vx_i32m4(Yq, kBYWeight, B, vl);
      vint32m4_t Y = __riscv_vsra_vx_i32m4(
          __riscv_vadd_vx_i32m4(Yq, kRound, vl), kWeightScale, vl);

      // BY = B - Y  (s32, range ~[-255, 255])
      vint32m4_t BY = __riscv_vsub_vv_i32m4(B, Y, vl);
      vint32m4_t Uq = __riscv_vmul_vx_i32m4(BY, kBUWeight, vl);
      Uq = __riscv_vadd_vx_i32m4(Uq, kHalf, vl);
      vint32m4_t U = __riscv_vsra_vx_i32m4(
          __riscv_vadd_vx_i32m4(Uq, kRound, vl), kWeightScale, vl);

      vint32m4_t RY = __riscv_vsub_vv_i32m4(R, Y, vl);
      vint32m4_t Vq = __riscv_vmul_vx_i32m4(RY, kRVWeight, vl);
      Vq = __riscv_vadd_vx_i32m4(Vq, kHalf, vl);
      vint32m4_t V = __riscv_vsra_vx_i32m4(
          __riscv_vadd_vx_i32m4(Vq, kRound, vl), kWeightScale, vl);

      vuint8m1_t y_byte = narrow_s32_to_u8_sat(Y, vl);
      vuint8m1_t u_byte = narrow_s32_to_u8_sat(U, vl);
      vuint8m1_t v_byte = narrow_s32_to_u8_sat(V, vl);

      vuint8m1x3_t out =
          __riscv_vcreate_v_u8m1x3(y_byte, u_byte, v_byte);
      __riscv_vsseg3e8_v_u8m1x3(rd + 3 * x, out, vl);
    }
  }
  return KLEIDICV_OK;
}

template kleidicv_error_t rgb_to_yuv444_u8<false, false>(
    const uint8_t *, size_t, uint8_t *, size_t, size_t, size_t);
template kleidicv_error_t rgb_to_yuv444_u8<false, true>(const uint8_t *, size_t,
                                                        uint8_t *, size_t,
                                                        size_t, size_t);
template kleidicv_error_t rgb_to_yuv444_u8<true, false>(const uint8_t *, size_t,
                                                        uint8_t *, size_t,
                                                        size_t, size_t);
template kleidicv_error_t rgb_to_yuv444_u8<true, true>(const uint8_t *, size_t,
                                                       uint8_t *, size_t,
                                                       size_t, size_t);

}  // namespace kleidicv::rvv
