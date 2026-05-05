// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV scale_u8 / scale_f32. u8 path widens to f32 lanes (LMUL=4), does
// fmul+fadd, clips to [0,255] in fp, then narrows back. f32 path is a single
// fmul+fadd at LMUL=1.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "elementwise_rvv.h"
#include "kleidicv/kleidicv.h"
#include "scale_decls.h"

namespace kleidicv::rvv {

kleidicv_error_t scale_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          double scale, double shift) {
  float fs = static_cast<float>(scale);
  float fsh = static_cast<float>(shift);
  return unary_elementwise<uint8_t>(
      src, src_stride, dst, dst_stride, width, height,
      [fs, fsh](auto v, size_t vl) {
        // u8m1 → u16m2 → u32m4 → f32m4 (4× widen total).
        vuint16m2_t u16 = __riscv_vzext_vf2_u16m2(v, vl);
        vuint32m4_t u32 = __riscv_vzext_vf2_u32m4(u16, vl);
        vfloat32m4_t f = __riscv_vfcvt_f_xu_v_f32m4(u32, vl);
        f = __riscv_vfmul_vf_f32m4(f, fs, vl);
        f = __riscv_vfadd_vf_f32m4(f, fsh, vl);
        // Clip in fp so the integer narrow doesn't need to saturate.
        f = __riscv_vfmax_vf_f32m4(f, 0.0f, vl);
        f = __riscv_vfmin_vf_f32m4(f, 255.0f, vl);
        vuint32m4_t u32r = __riscv_vfcvt_xu_f_v_u32m4(f, vl);
        vuint16m2_t u16r = __riscv_vncvt_x_x_w_u16m2(u32r, vl);
        return __riscv_vncvt_x_x_w_u8m1(u16r, vl);
      });
}

kleidicv_error_t scale_f32(const float *src, size_t src_stride, float *dst,
                           size_t dst_stride, size_t width, size_t height,
                           double scale, double shift) {
  float fs = static_cast<float>(scale);
  float fsh = static_cast<float>(shift);
  return unary_elementwise<float>(
      src, src_stride, dst, dst_stride, width, height,
      [fs, fsh](auto v, size_t vl) {
        auto r = __riscv_vfmul_vf_f32m1(v, fs, vl);
        return __riscv_vfadd_vf_f32m1(r, fsh, vl);
      });
}

}  // namespace kleidicv::rvv
