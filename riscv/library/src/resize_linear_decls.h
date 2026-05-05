// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Bilinear resize, channels=1 only. Coordinate mapping follows OpenCV's area
// convention: src_x = (dx + 0.5) * scale_x - 0.5.

#ifndef KLEIDICV_RISCV_RESIZE_LINEAR_DECLS_H
#define KLEIDICV_RISCV_RESIZE_LINEAR_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t resize_linear_u8(const uint8_t *src, size_t src_stride,
                                  size_t src_width, size_t src_height,
                                  uint8_t *dst, size_t dst_stride,
                                  size_t dst_width, size_t dst_height);
kleidicv_error_t resize_linear_f32(const float *src, size_t src_stride,
                                   size_t src_width, size_t src_height,
                                   float *dst, size_t dst_stride,
                                   size_t dst_width, size_t dst_height);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t resize_linear_u8(const uint8_t *src, size_t src_stride,
                                  size_t src_width, size_t src_height,
                                  uint8_t *dst, size_t dst_stride,
                                  size_t dst_width, size_t dst_height);
kleidicv_error_t resize_linear_f32(const float *src, size_t src_stride,
                                   size_t src_width, size_t src_height,
                                   float *dst, size_t dst_stride,
                                   size_t dst_width, size_t dst_height);
}  // namespace kleidicv::rvv

#endif
