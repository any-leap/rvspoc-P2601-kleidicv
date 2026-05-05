// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// SPOC limits: kernel_size = 5, channels = 1, BORDER_TYPE_REPLICATE only —
// matches the only specialisation upstream's `separable_filter_2d_sc.h`
// provides. Other configurations return KLEIDICV_ERROR_NOT_IMPLEMENTED.

#ifndef KLEIDICV_RISCV_SEPARABLE_FILTER_2D_DECLS_H
#define KLEIDICV_RISCV_SEPARABLE_FILTER_2D_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t separable_filter_2d_5x5_u8(const uint8_t *src, size_t src_stride,
                                            uint8_t *dst, size_t dst_stride,
                                            size_t width, size_t height,
                                            const uint8_t *kernel_x,
                                            const uint8_t *kernel_y);
kleidicv_error_t separable_filter_2d_5x5_u16(const uint16_t *src,
                                             size_t src_stride, uint16_t *dst,
                                             size_t dst_stride, size_t width,
                                             size_t height,
                                             const uint16_t *kernel_x,
                                             const uint16_t *kernel_y);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t separable_filter_2d_5x5_u8(const uint8_t *src, size_t src_stride,
                                            uint8_t *dst, size_t dst_stride,
                                            size_t width, size_t height,
                                            const uint8_t *kernel_x,
                                            const uint8_t *kernel_y);
kleidicv_error_t separable_filter_2d_5x5_u16(const uint16_t *src,
                                             size_t src_stride, uint16_t *dst,
                                             size_t dst_stride, size_t width,
                                             size_t height,
                                             const uint16_t *kernel_x,
                                             const uint16_t *kernel_y);
}  // namespace kleidicv::rvv

#endif
