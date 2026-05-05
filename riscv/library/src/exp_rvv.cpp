// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV exp_f32. Algorithm from upstream's SVE2 exp:
//   exp(x) = 2^n * poly(r),  where  x = ln2*n + r,  r ∈ [-ln2/2, ln2/2]
// 1. round-to-int via the magic-shift trick: z = x*invLn2 + 0x1.8p23, then
//    n = z - 0x1.8p23.
// 2. r = x - n*ln2 (split as Hi/Lo for accuracy).
// 3. poly(r) = Horner over kPoly coefficients.
// 4. 2^n via bit manipulation of z's mantissa.
// "Short path" only — out-of-range inputs return whatever this produces, no
// special-cased over/underflow handling. Public docs don't promise it.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "elementwise_rvv.h"
#include "exp_decls.h"
#include "kleidicv/arithmetics/exp_constants.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

kleidicv_error_t exp_f32(const float *src, size_t src_stride, float *dst,
                         size_t dst_stride, size_t width, size_t height) {
  using namespace kleidicv::exp_f32;
  return unary_elementwise<float>(
      src, src_stride, dst, dst_stride, width, height,
      [](auto x, size_t vl) {
        // z = x * kInvLn2 + kShift  (round-to-int via magic shift)
        vfloat32m1_t shift_v = __riscv_vfmv_v_f_f32m1(kShift, vl);
        vfloat32m1_t z = __riscv_vfmacc_vf_f32m1(shift_v, kInvLn2, x, vl);

        // n = z - kShift (still as float, integer-valued)
        vfloat32m1_t n = __riscv_vfsub_vf_f32m1(z, kShift, vl);

        // r = x - n*kLn2Hi - n*kLn2Lo  (range-reduced)
        vfloat32m1_t r = __riscv_vfnmsac_vf_f32m1(x, kLn2Hi, n, vl);
        r = __riscv_vfnmsac_vf_f32m1(r, kLn2Lo, n, vl);

        // poly = ((((P[0]*r + P[1])*r + P[2])*r + P[3])*r + P[4])*r*r + r + 1
        // Implemented as 6 chained mlas matching upstream sequence.
        vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(kPoly[0], vl);
        vfloat32m1_t c1 = __riscv_vfmv_v_f_f32m1(kPoly[1], vl);
        p = __riscv_vfmadd_vv_f32m1(p, r, c1, vl);  // p = p*r + P[1]
        vfloat32m1_t c2 = __riscv_vfmv_v_f_f32m1(kPoly[2], vl);
        p = __riscv_vfmadd_vv_f32m1(p, r, c2, vl);
        vfloat32m1_t c3 = __riscv_vfmv_v_f_f32m1(kPoly[3], vl);
        p = __riscv_vfmadd_vv_f32m1(p, r, c3, vl);
        vfloat32m1_t c4 = __riscv_vfmv_v_f_f32m1(kPoly[4], vl);
        p = __riscv_vfmadd_vv_f32m1(p, r, c4, vl);
        vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
        p = __riscv_vfmadd_vv_f32m1(p, r, one, vl);
        p = __riscv_vfmadd_vv_f32m1(p, r, one, vl);

        // 2^n via bit manipulation: bit_pattern(z) << 23 carries n*2^23 in
        // the low bits; adding 0x3f800000 (= 1.0f bit pattern) yields 2^n.
        vuint32m1_t z_u = __riscv_vreinterpret_v_f32m1_u32m1(z);
        vuint32m1_t e = __riscv_vsll_vx_u32m1(z_u, 23, vl);
        vuint32m1_t scale_u =
            __riscv_vadd_vx_u32m1(e, 0x3f800000U, vl);
        vfloat32m1_t scale_f =
            __riscv_vreinterpret_v_u32m1_f32m1(scale_u);
        return __riscv_vfmul_vv_f32m1(scale_f, p, vl);
      });
}

}  // namespace kleidicv::rvv
