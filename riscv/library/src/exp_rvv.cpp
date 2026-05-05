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
// Adopts upstream's "short path + specialcase" structure: when every lane
// has |n| ≤ 126 the cheap bit-manipulated 2^n works; otherwise fall through
// to a specialcase that splits 2^n = s1·s2 so f32 overflow during the
// scale step matches expf semantics for large |x|.

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
        vfloat32m1_t fast = __riscv_vfmul_vv_f32m1(scale_f, p, vl);

        // Specialcase: |n| > 126 lanes need 2^n split as s1·s2 so the
        // f32 multiplication does not silently overflow / underflow.
        // We compute the specialcase unconditionally and merge it into
        // `fast` only on the masked lanes — cheaper than predicating the
        // whole tail in RVV.
        vfloat32m1_t abs_n = __riscv_vfabs_v_f32m1(n, vl);
        vbool32_t need_special =
            __riscv_vmfgt_vf_f32m1_b32(abs_n, 126.0F, vl);

        // b = (n <= 0) ? 0x83000000 : 0
        vbool32_t n_nonpos = __riscv_vmfle_vf_f32m1_b32(n, 0.0F, vl);
        vuint32m1_t zero_u = __riscv_vmv_v_x_u32m1(0u, vl);
        vuint32m1_t b = __riscv_vmerge_vxm_u32m1(zero_u, 0x83000000U,
                                                   n_nonpos, vl);
        vfloat32m1_t s1 = __riscv_vreinterpret_v_u32m1_f32m1(
            __riscv_vadd_vx_u32m1(b, 0x7f000000U, vl));
        vfloat32m1_t s2 = __riscv_vreinterpret_v_u32m1_f32m1(
            __riscv_vsub_vv_u32m1(e, b, vl));

        // For |n| > 192: clamp to ±inf / 0 by returning s1*s1 directly
        // (= 2^254 ≈ inf for n>0 / 2^-254 ≈ 0 for n<0).
        vbool32_t n_huge = __riscv_vmfgt_vf_f32m1_b32(abs_n, 192.0F, vl);
        vfloat32m1_t s1s1 = __riscv_vfmul_vv_f32m1(s1, s1, vl);
        vfloat32m1_t s1ps2 = __riscv_vfmul_vv_f32m1(
            s1, __riscv_vfmul_vv_f32m1(p, s2, vl), vl);
        vfloat32m1_t special =
            __riscv_vmerge_vvm_f32m1(s1ps2, s1s1, n_huge, vl);

        return __riscv_vmerge_vvm_f32m1(fast, special, need_special, vl);
      });
}

}  // namespace kleidicv::rvv
