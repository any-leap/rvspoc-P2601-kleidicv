// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_BITWISE_AND_DECLS_H
#define KLEIDICV_RISCV_BITWISE_AND_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t bitwise_and(const uint8_t *src_a, size_t src_a_stride,
                             const uint8_t *src_b, size_t src_b_stride,
                             uint8_t *dst, size_t dst_stride, size_t width,
                             size_t height);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t bitwise_and(const uint8_t *, size_t, const uint8_t *, size_t,
                             uint8_t *, size_t, size_t, size_t);
}  // namespace kleidicv::rvv

#endif
