// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar threshold_binary: dst = (src > threshold) ? value : 0.

#include <cstdint>

#include "kleidicv/kleidicv.h"

#include "elementwise_scalar.h"
#include "threshold_binary_decls.h"

namespace kleidicv::scalar {

kleidicv_error_t threshold_binary_u8(const uint8_t *src, size_t src_stride,
                                     uint8_t *dst, size_t dst_stride,
                                     size_t width, size_t height,
                                     uint8_t threshold, uint8_t value) {
  return unary_elementwise<uint8_t>(
      src, src_stride, dst, dst_stride, width, height,
      [threshold, value](uint8_t x) -> uint8_t {
        return x > threshold ? value : uint8_t{0};
      });
}

}  // namespace kleidicv::scalar
