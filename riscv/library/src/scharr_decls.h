// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_SCHARR_DECLS_H
#define KLEIDICV_RISCV_SCHARR_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t scharr_interleaved_s16_u8(const uint8_t *src, size_t src_stride,
                                           size_t src_width, size_t src_height,
                                           int16_t *dst, size_t dst_stride);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t scharr_interleaved_s16_u8(const uint8_t *src, size_t src_stride,
                                           size_t src_width, size_t src_height,
                                           int16_t *dst, size_t dst_stride);
}  // namespace kleidicv::rvv

#endif
