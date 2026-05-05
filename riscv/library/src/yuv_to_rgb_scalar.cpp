// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// BT.601-7 inverse YUV444→RGB with 14-bit coefficients matching upstream.

#include <cstdint>

#include "kleidicv/kleidicv.h"

#include "yuv_to_rgb_decls.h"

namespace kleidicv::scalar {

namespace {

constexpr int kWeightScale = 14;
constexpr int kRound = 1 << (kWeightScale - 1);
constexpr int kVRWeight = 18678;
constexpr int kUGWeight = -6472;
constexpr int kVGWeight = -9519;
constexpr int kUBWeight = 33292;

inline uint8_t sat_u8(int v) {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return static_cast<uint8_t>(v);
}

}  // namespace

template <bool BGR, bool kAlpha>
kleidicv_error_t yuv444_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;
  constexpr size_t out_chan = kAlpha ? 4 : 3;

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *rs = src + y * src_stride;
    uint8_t *rd = dst + y * dst_stride;
    for (size_t x = 0; x < width; ++x) {
      int Y = rs[3 * x + 0];
      int U = rs[3 * x + 1] - 128;
      int V = rs[3 * x + 2] - 128;
      int R = Y + ((V * kVRWeight + kRound) >> kWeightScale);
      int G = Y + ((U * kUGWeight + V * kVGWeight + kRound) >> kWeightScale);
      int B = Y + ((U * kUBWeight + kRound) >> kWeightScale);
      if constexpr (BGR) {
        rd[out_chan * x + 0] = sat_u8(B);
        rd[out_chan * x + 1] = sat_u8(G);
        rd[out_chan * x + 2] = sat_u8(R);
      } else {
        rd[out_chan * x + 0] = sat_u8(R);
        rd[out_chan * x + 1] = sat_u8(G);
        rd[out_chan * x + 2] = sat_u8(B);
      }
      if constexpr (kAlpha) {
        rd[out_chan * x + 3] = 0xFF;
      }
    }
  }
  return KLEIDICV_OK;
}

template kleidicv_error_t yuv444_to_rgb_u8<false, false>(
    const uint8_t *, size_t, uint8_t *, size_t, size_t, size_t);
template kleidicv_error_t yuv444_to_rgb_u8<false, true>(const uint8_t *, size_t,
                                                        uint8_t *, size_t,
                                                        size_t, size_t);
template kleidicv_error_t yuv444_to_rgb_u8<true, false>(const uint8_t *, size_t,
                                                        uint8_t *, size_t,
                                                        size_t, size_t);
template kleidicv_error_t yuv444_to_rgb_u8<true, true>(const uint8_t *, size_t,
                                                       uint8_t *, size_t,
                                                       size_t, size_t);

}  // namespace kleidicv::scalar
