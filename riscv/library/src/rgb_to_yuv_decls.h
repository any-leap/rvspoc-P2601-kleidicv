// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_RGB_TO_YUV_DECLS_H
#define KLEIDICV_RISCV_RGB_TO_YUV_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
// kAlpha=true means src has 4 channels (RGBA/BGRA); the alpha is dropped on
// the way to YUV444's interleaved 3-channel layout.
template <bool BGR, bool kAlpha>
kleidicv_error_t rgb_to_yuv444_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
template <bool BGR, bool kAlpha>
kleidicv_error_t rgb_to_yuv444_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height);
}  // namespace kleidicv::rvv

#endif
