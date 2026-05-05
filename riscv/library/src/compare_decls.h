// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_COMPARE_DECLS_H
#define KLEIDICV_RISCV_COMPARE_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t compare_equal_u8(const uint8_t *src_a, size_t sa,
                                  const uint8_t *src_b, size_t sb, uint8_t *dst,
                                  size_t sd, size_t w, size_t h);
kleidicv_error_t compare_greater_u8(const uint8_t *src_a, size_t sa,
                                    const uint8_t *src_b, size_t sb,
                                    uint8_t *dst, size_t sd, size_t w,
                                    size_t h);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t compare_equal_u8(const uint8_t *src_a, size_t sa,
                                  const uint8_t *src_b, size_t sb, uint8_t *dst,
                                  size_t sd, size_t w, size_t h);
kleidicv_error_t compare_greater_u8(const uint8_t *src_a, size_t sa,
                                    const uint8_t *src_b, size_t sb,
                                    uint8_t *dst, size_t sd, size_t w,
                                    size_t h);
}  // namespace kleidicv::rvv

#endif
