// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// dst = (sat(|a| + |b|) > threshold) ? sat(|a| + |b|) : 0
// Note |INT16_MIN| overflows; per the saturating contract that is INT16_MAX.

#include <cstdint>
#include <cstdlib>
#include <limits>

#include "kleidicv/kleidicv.h"

#include "add_abs_with_threshold_decls.h"
#include "elementwise_scalar.h"

namespace kleidicv::scalar {

kleidicv_error_t saturating_add_abs_with_threshold_s16(
    const int16_t *src_a, size_t src_a_stride, const int16_t *src_b,
    size_t src_b_stride, int16_t *dst, size_t dst_stride, size_t width,
    size_t height, int16_t threshold) {
  return binary_elementwise<int16_t>(
      src_a, src_a_stride, src_b, src_b_stride, dst, dst_stride, width, height,
      [threshold](int16_t a, int16_t b) -> int16_t {
        int32_t aa = std::abs(static_cast<int32_t>(a));
        int32_t bb = std::abs(static_cast<int32_t>(b));
        int32_t s = aa + bb;
        constexpr int32_t kMax = std::numeric_limits<int16_t>::max();
        if (s > kMax) s = kMax;
        return s > threshold ? static_cast<int16_t>(s) : int16_t{0};
      });
}

}  // namespace kleidicv::scalar
