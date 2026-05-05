// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_MEDIAN_BLUR_DECLS_H
#define KLEIDICV_RISCV_MEDIAN_BLUR_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t median_blur_3x3_u8(const uint8_t *src, size_t src_stride,
                                    uint8_t *dst, size_t dst_stride,
                                    size_t width, size_t height);
}  // namespace kleidicv::scalar

#endif
