// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV bitwise_and. Element width is irrelevant; e8m1 keeps it symmetric with
// the rest of the family. Phase 4 may bump to a wider SEW for throughput.

#include <cstddef>
#include <cstdint>

#include "bitwise_and_decls.h"
#include "elementwise_rvv.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

kleidicv_error_t bitwise_and(const uint8_t *src_a, size_t sa,
                             const uint8_t *src_b, size_t sb, uint8_t *dst,
                             size_t sd, size_t w, size_t h) {
  return binary_elementwise<uint8_t>(
      src_a, sa, src_b, sb, dst, sd, w, h,
      [](auto a, auto b, size_t vl) {
        return __riscv_vand_vv_u8m1(a, b, vl);
      });
}

}  // namespace kleidicv::rvv
