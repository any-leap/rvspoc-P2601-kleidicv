// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// channels==1 hits the optimised RVV / scalar kernel below directly;
// channels∈{2,3,4} go through the multichannel wrap in sobel_api.cpp
// (vlsegN deinterleave → channels=1 kernel per plane → vssegN reinterleave).

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
