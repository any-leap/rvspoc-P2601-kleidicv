// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// 3×3 zero-sigma binomial uses the dedicated fast path; other kernel sizes
// (odd ≥ 3) and non-zero sigmas go through `gaussian_blur_generic_u8` which
// builds float Gaussian coefficients on the fly and runs a separable
// scalar convolution.

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

// Generic Gaussian for kernel_size ∈ {odd ≥ 3} and arbitrary sigma. sigma=0
// uses OpenCV's default: sigma = 0.3*((ks-1)*0.5 - 1) + 0.8.
kleidicv_error_t gaussian_blur_generic_u8(const uint8_t *src, size_t src_stride,
                                          uint8_t *dst, size_t dst_stride,
                                          size_t width, size_t height,
                                          size_t kernel_width,
                                          size_t kernel_height, float sigma_x,
                                          float sigma_y);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t gaussian_blur_3x3_binomial_u8(const uint8_t *src,
                                               size_t src_stride, uint8_t *dst,
                                               size_t dst_stride, size_t width,
                                               size_t height);
}  // namespace kleidicv::rvv

#endif
