// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar oracle for exp_f32: just std::exp. RVV path uses the upstream
// polynomial; results agree to within poly approximation tolerance.

#include <cmath>
#include <cstddef>

#include "kleidicv/kleidicv.h"

#include "elementwise_scalar.h"
#include "exp_decls.h"

namespace kleidicv::scalar {

kleidicv_error_t exp_f32(const float *src, size_t src_stride, float *dst,
                         size_t dst_stride, size_t width, size_t height) {
  return unary_elementwise<float>(src, src_stride, dst, dst_stride, width,
                                  height,
                                  [](float v) { return std::exp(v); });
}

}  // namespace kleidicv::scalar
