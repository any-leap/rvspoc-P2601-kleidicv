// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar reference for bitwise_and on u8 buffers.

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

#include "bitwise_and_decls.h"
#include "elementwise_scalar.h"

namespace kleidicv::scalar {

kleidicv_error_t bitwise_and(const uint8_t *src_a, size_t src_a_stride,
                             const uint8_t *src_b, size_t src_b_stride,
                             uint8_t *dst, size_t dst_stride, size_t width,
                             size_t height) {
  return binary_elementwise<uint8_t>(
      src_a, src_a_stride, src_b, src_b_stride, dst, dst_stride, width, height,
      [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a & b); });
}

}  // namespace kleidicv::scalar
