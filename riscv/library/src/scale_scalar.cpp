// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar reference for scale_u8 / scale_f32. Mirrors upstream's `scale_value`:
// cast to float, fma, lrintf-round, saturate.

#include <cmath>
#include <cstdint>
#include <limits>

#include "kleidicv/kleidicv.h"

#include "elementwise_scalar.h"
#include "scale_decls.h"

namespace kleidicv::scalar {

kleidicv_error_t scale_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          double scale, double shift) {
  float fs = static_cast<float>(scale);
  float fsh = static_cast<float>(shift);
  return unary_elementwise<uint8_t>(
      src, src_stride, dst, dst_stride, width, height,
      [fs, fsh](uint8_t v) -> uint8_t {
        long iv = std::lrintf(static_cast<float>(v) * fs + fsh);
        if (iv < 0) return 0;
        if (iv > 255) return 255;
        return static_cast<uint8_t>(iv);
      });
}

kleidicv_error_t scale_f32(const float *src, size_t src_stride, float *dst,
                           size_t dst_stride, size_t width, size_t height,
                           double scale, double shift) {
  float fs = static_cast<float>(scale);
  float fsh = static_cast<float>(shift);
  return unary_elementwise<float>(
      src, src_stride, dst, dst_stride, width, height,
      [fs, fsh](float v) { return v * fs + fsh; });
}

}  // namespace kleidicv::scalar
