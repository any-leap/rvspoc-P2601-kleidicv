// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Rectangular dilate / erode. Replicate border, channels=1, anchor at kernel
// centre, iterations >= 1.

#ifndef KLEIDICV_RISCV_MORPHOLOGY_DECLS_H
#define KLEIDICV_RISCV_MORPHOLOGY_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t morph_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          size_t kw, size_t kh, bool is_dilate);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t morph_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          size_t kw, size_t kh, bool is_dilate);
}  // namespace kleidicv::rvv

#endif
