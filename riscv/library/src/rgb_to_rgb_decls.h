// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_RGB_TO_RGB_DECLS_H
#define KLEIDICV_RISCV_RGB_TO_RGB_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

#define DECL_RGB_FN(name)                                                     \
  kleidicv_error_t name(const uint8_t *src, size_t src_stride, uint8_t *dst, \
                        size_t dst_stride, size_t width, size_t height)

namespace kleidicv::scalar {
DECL_RGB_FN(rgb_to_bgr_u8);
DECL_RGB_FN(rgb_to_rgb_u8);
DECL_RGB_FN(rgba_to_bgra_u8);
DECL_RGB_FN(rgba_to_rgba_u8);
DECL_RGB_FN(rgb_to_bgra_u8);
DECL_RGB_FN(rgb_to_rgba_u8);
DECL_RGB_FN(rgba_to_bgr_u8);
DECL_RGB_FN(rgba_to_rgb_u8);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
DECL_RGB_FN(rgb_to_bgr_u8);
DECL_RGB_FN(rgb_to_rgb_u8);
DECL_RGB_FN(rgba_to_bgra_u8);
DECL_RGB_FN(rgba_to_rgba_u8);
DECL_RGB_FN(rgb_to_bgra_u8);
DECL_RGB_FN(rgb_to_rgba_u8);
DECL_RGB_FN(rgba_to_bgr_u8);
DECL_RGB_FN(rgba_to_rgb_u8);
}  // namespace kleidicv::rvv

#undef DECL_RGB_FN

#endif
