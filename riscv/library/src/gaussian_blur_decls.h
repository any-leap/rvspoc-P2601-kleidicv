// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// SPOC scope: 3x3 binomial only (kernel [1,2,1]⊗[1,2,1] / 16). Other kernel
// sizes / non-zero sigmas return KLEIDICV_ERROR_NOT_IMPLEMENTED.

#ifndef KLEIDICV_RISCV_GAUSSIAN_BLUR_DECLS_H
#define KLEIDICV_RISCV_GAUSSIAN_BLUR_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t gaussian_blur_3x3_binomial_u8(const uint8_t *src,
                                               size_t src_stride, uint8_t *dst,
                                               size_t dst_stride, size_t width,
                                               size_t height);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t gaussian_blur_3x3_binomial_u8(const uint8_t *src,
                                               size_t src_stride, uint8_t *dst,
                                               size_t dst_stride, size_t width,
                                               size_t height);
}  // namespace kleidicv::rvv

#endif
