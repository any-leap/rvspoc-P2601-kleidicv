// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar reference for RGB-family channel reorders. Each variant is just a
// per-pixel byte permute (sometimes with a constant 0xFF alpha).

#include <cstdint>

#include "kleidicv/kleidicv.h"

#include "rgb_to_rgb_decls.h"

namespace kleidicv::scalar {

#define HEAD                                                              \
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;                  \
  if (width == 0 || height == 0) return KLEIDICV_OK;                     \
  for (size_t y = 0; y < height; ++y) {                                  \
    const uint8_t *rs = src + y * src_stride;                            \
    uint8_t *rd = dst + y * dst_stride;                                  \
    for (size_t x = 0; x < width; ++x)

#define TAIL                                                              \
  }                                                                       \
  return KLEIDICV_OK;

kleidicv_error_t rgb_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  HEAD {
    rd[3 * x + 0] = rs[3 * x + 2];
    rd[3 * x + 1] = rs[3 * x + 1];
    rd[3 * x + 2] = rs[3 * x + 0];
  }
  TAIL
}

kleidicv_error_t rgb_to_rgb_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  HEAD {
    rd[3 * x + 0] = rs[3 * x + 0];
    rd[3 * x + 1] = rs[3 * x + 1];
    rd[3 * x + 2] = rs[3 * x + 2];
  }
  TAIL
}

kleidicv_error_t rgba_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  HEAD {
    rd[4 * x + 0] = rs[4 * x + 2];
    rd[4 * x + 1] = rs[4 * x + 1];
    rd[4 * x + 2] = rs[4 * x + 0];
    rd[4 * x + 3] = rs[4 * x + 3];
  }
  TAIL
}

kleidicv_error_t rgba_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  HEAD {
    rd[4 * x + 0] = rs[4 * x + 0];
    rd[4 * x + 1] = rs[4 * x + 1];
    rd[4 * x + 2] = rs[4 * x + 2];
    rd[4 * x + 3] = rs[4 * x + 3];
  }
  TAIL
}

kleidicv_error_t rgb_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  HEAD {
    rd[4 * x + 0] = rs[3 * x + 2];
    rd[4 * x + 1] = rs[3 * x + 1];
    rd[4 * x + 2] = rs[3 * x + 0];
    rd[4 * x + 3] = 0xFF;
  }
  TAIL
}

kleidicv_error_t rgb_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  HEAD {
    rd[4 * x + 0] = rs[3 * x + 0];
    rd[4 * x + 1] = rs[3 * x + 1];
    rd[4 * x + 2] = rs[3 * x + 2];
    rd[4 * x + 3] = 0xFF;
  }
  TAIL
}

kleidicv_error_t rgba_to_bgr_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  HEAD {
    rd[3 * x + 0] = rs[4 * x + 2];
    rd[3 * x + 1] = rs[4 * x + 1];
    rd[3 * x + 2] = rs[4 * x + 0];
  }
  TAIL
}

kleidicv_error_t rgba_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  HEAD {
    rd[3 * x + 0] = rs[4 * x + 0];
    rd[3 * x + 1] = rs[4 * x + 1];
    rd[3 * x + 2] = rs[4 * x + 2];
  }
  TAIL
}

#undef HEAD
#undef TAIL

}  // namespace kleidicv::scalar
