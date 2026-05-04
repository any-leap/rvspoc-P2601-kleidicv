// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV implementation of sum_f32. Uses the widening reduce-sum
// (vfwredusum_vs_f32m1_f64m1) so the running accumulator is held in f64 and
// retains precision across large inputs — this matches the upstream Neon
// path's policy of widening to double when summing f32.

#include <riscv_vector.h>

#include "sum_decls.h"

namespace kleidicv::rvv {

kleidicv_error_t sum_f32(const float *src, size_t src_stride, size_t width,
                         size_t height, float *sum) {
  if (!src || !sum) return KLEIDICV_ERROR_NULL_POINTER;

  // f64m1 scalar-carrying accumulator: only element 0 matters, but the type
  // has to be a vector register for vfwredusum. Initialise it to 0.0.
  vfloat64m1_t acc = __riscv_vfmv_v_f_f64m1(0.0, 1);

  if (width > 0 && height > 0) {
    for (size_t y = 0; y < height; ++y) {
      const float *row = reinterpret_cast<const float *>(
          reinterpret_cast<const uint8_t *>(src) + y * src_stride);
      size_t vl;
      for (size_t x = 0; x < width; x += vl) {
        vl = __riscv_vsetvl_e32m1(width - x);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(row + x, vl);
        // vfwredusum widens each f32 lane to f64, then folds them all into
        // the scalar in element-0 of `acc`, returning the new accumulator.
        acc = __riscv_vfwredusum_vs_f32m1_f64m1(v, acc, vl);
      }
    }
  }

  double total = __riscv_vfmv_f_s_f64m1_f64(acc);
  *sum = static_cast<float>(total);
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
