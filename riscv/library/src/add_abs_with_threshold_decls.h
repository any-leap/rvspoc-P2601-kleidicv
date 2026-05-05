// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_ADD_ABS_WITH_THRESHOLD_DECLS_H
#define KLEIDICV_RISCV_ADD_ABS_WITH_THRESHOLD_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t saturating_add_abs_with_threshold_s16(
    const int16_t *src_a, size_t src_a_stride, const int16_t *src_b,
    size_t src_b_stride, int16_t *dst, size_t dst_stride, size_t width,
    size_t height, int16_t threshold);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t saturating_add_abs_with_threshold_s16(
    const int16_t *src_a, size_t src_a_stride, const int16_t *src_b,
    size_t src_b_stride, int16_t *dst, size_t dst_stride, size_t width,
    size_t height, int16_t threshold);
}  // namespace kleidicv::rvv

#endif
