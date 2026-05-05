// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar reference for compare_equal_u8 / compare_greater_u8.
// Output is 0xFF where the predicate holds, 0 otherwise.

#include <cstdint>

#include "kleidicv/kleidicv.h"

#include "compare_decls.h"
#include "elementwise_scalar.h"

namespace kleidicv::scalar {

kleidicv_error_t compare_equal_u8(const uint8_t *src_a, size_t sa,
                                  const uint8_t *src_b, size_t sb, uint8_t *dst,
                                  size_t sd, size_t w, size_t h) {
  return binary_elementwise<uint8_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](uint8_t a, uint8_t b) -> uint8_t { return a == b ? 0xFF : 0; });
}

kleidicv_error_t compare_greater_u8(const uint8_t *src_a, size_t sa,
                                    const uint8_t *src_b, size_t sb,
                                    uint8_t *dst, size_t sd, size_t w,
                                    size_t h) {
  return binary_elementwise<uint8_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](uint8_t a, uint8_t b) -> uint8_t { return a > b ? 0xFF : 0; });
}

}  // namespace kleidicv::scalar
