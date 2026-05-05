// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Public API takes a `channels` parameter; SPOC supports channels==1 only.
// Multi-channel inputs return KLEIDICV_ERROR_NOT_IMPLEMENTED.

#ifndef KLEIDICV_RISCV_SOBEL_DECLS_H
#define KLEIDICV_RISCV_SOBEL_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t sobel_3x3_horizontal_s16_u8(const uint8_t *src,
                                             size_t src_stride, int16_t *dst,
                                             size_t dst_stride, size_t width,
                                             size_t height);
kleidicv_error_t sobel_3x3_vertical_s16_u8(const uint8_t *src,
                                           size_t src_stride, int16_t *dst,
                                           size_t dst_stride, size_t width,
                                           size_t height);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t sobel_3x3_horizontal_s16_u8(const uint8_t *src,
                                             size_t src_stride, int16_t *dst,
                                             size_t dst_stride, size_t width,
                                             size_t height);
kleidicv_error_t sobel_3x3_vertical_s16_u8(const uint8_t *src, size_t src_stride,
                                           int16_t *dst, size_t dst_stride,
                                           size_t width, size_t height);
}  // namespace kleidicv::rvv

#endif
