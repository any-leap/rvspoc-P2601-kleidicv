// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Public API also exposes scale_u8_f16 (uint8_t -> float16_t). Skipped here:
// the _Float16 storage type adds toolchain plumbing without exercising any new
// RVV pattern, and Bucket A already covers the f32-lane logic via scale_f32.

#ifndef KLEIDICV_RISCV_SCALE_DECLS_H
#define KLEIDICV_RISCV_SCALE_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t scale_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          double scale, double shift);
kleidicv_error_t scale_f32(const float *src, size_t src_stride, float *dst,
                           size_t dst_stride, size_t width, size_t height,
                           double scale, double shift);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t scale_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          double scale, double shift);
kleidicv_error_t scale_f32(const float *src, size_t src_stride, float *dst,
                           size_t dst_stride, size_t width, size_t height,
                           double scale, double shift);
}  // namespace kleidicv::rvv

#endif
