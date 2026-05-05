// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// BT.601-7 RGB→YUV444 with 14-bit fixed-point coefficients matching upstream.

#include <cstdint>

#include "kleidicv/kleidicv.h"

#include "rgb_to_yuv_decls.h"

namespace kleidicv::scalar {

namespace {

constexpr int kRYWeight = 4899;
constexpr int kGYWeight = 9617;
constexpr int kBYWeight = 1868;
constexpr int kBUWeight = 8061;
constexpr int kRVWeight = 14369;
constexpr int kWeightScale = 14;
constexpr int kHalf = 128 << kWeightScale;
constexpr int kRound = 1 << (kWeightScale - 1);

inline uint8_t sat_u8(int v) {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return static_cast<uint8_t>(v);
}

}  // namespace

template <bool BGR, bool kAlpha>
kleidicv_error_t rgb_to_yuv444_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;
  constexpr int in_chan = kAlpha ? 4 : 3;
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *rs = src + y * src_stride;
    uint8_t *rd = dst + y * dst_stride;
    for (size_t x = 0; x < width; ++x) {
      int R, G, B;
      if constexpr (BGR) {
        B = rs[in_chan * x + 0];
        G = rs[in_chan * x + 1];
        R = rs[in_chan * x + 2];
      } else {
        R = rs[in_chan * x + 0];
        G = rs[in_chan * x + 1];
        B = rs[in_chan * x + 2];
      }
      int y_q = R * kRYWeight + G * kGYWeight + B * kBYWeight;
      int Y = (y_q + kRound) >> kWeightScale;
      int u_q = (B - Y) * kBUWeight + kHalf;
      int U = (u_q + kRound) >> kWeightScale;
      int v_q = (R - Y) * kRVWeight + kHalf;
      int V = (v_q + kRound) >> kWeightScale;
      rd[3 * x + 0] = sat_u8(Y);
      rd[3 * x + 1] = sat_u8(U);
      rd[3 * x + 2] = sat_u8(V);
    }
  }
  return KLEIDICV_OK;
}

template kleidicv_error_t rgb_to_yuv444_u8<false, false>(
    const uint8_t *, size_t, uint8_t *, size_t, size_t, size_t);
template kleidicv_error_t rgb_to_yuv444_u8<false, true>(const uint8_t *, size_t,
                                                        uint8_t *, size_t,
                                                        size_t, size_t);
template kleidicv_error_t rgb_to_yuv444_u8<true, false>(const uint8_t *, size_t,
                                                        uint8_t *, size_t,
                                                        size_t, size_t);
template kleidicv_error_t rgb_to_yuv444_u8<true, true>(const uint8_t *, size_t,
                                                       uint8_t *, size_t,
                                                       size_t, size_t);

}  // namespace kleidicv::scalar
