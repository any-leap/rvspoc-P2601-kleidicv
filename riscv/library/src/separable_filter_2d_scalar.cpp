// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// 5x5 separable filter (replicate border, channels=1). Matches upstream's
// saturating contract:
//   - vertical pass:   intermediate[y][x] = saturating_add over i of
//                        kernel_y[i] * src[clip(y+i-2)][x]   (u16 acc)
//   - horizontal pass: dst[y][x] = clamp_to_T(
//                        sum over i of kernel_x[i] * intermediate[y][clip(x+i-2)])

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

#include "kleidicv/kleidicv.h"

#include "separable_filter_2d_decls.h"

namespace kleidicv::scalar {

namespace {

inline size_t clip(ptrdiff_t v, size_t n) {
  if (v < 0) return 0;
  if (static_cast<size_t>(v) >= n) return n - 1;
  return static_cast<size_t>(v);
}

template <typename T>
inline T sat_add(T a, T b) {
  using W = uint64_t;
  W s = static_cast<W>(a) + static_cast<W>(b);
  W kMax = std::numeric_limits<T>::max();
  return static_cast<T>(s > kMax ? kMax : s);
}

}  // namespace

kleidicv_error_t separable_filter_2d_5x5_u8(const uint8_t *src, size_t src_stride,
                                            uint8_t *dst, size_t dst_stride,
                                            size_t width, size_t height,
                                            const uint8_t *kernel_x,
                                            const uint8_t *kernel_y) {
  if (!src || !dst || !kernel_x || !kernel_y)
    return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  std::vector<uint16_t> intermediate(width * height);
  // Vertical pass
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      uint16_t acc = 0;
      for (int i = 0; i < 5; ++i) {
        size_t sy =
            clip(static_cast<ptrdiff_t>(y) + i - 2, height);
        uint16_t prod = static_cast<uint16_t>(kernel_y[i]) *
                        static_cast<uint16_t>(src[sy * src_stride + x]);
        acc = sat_add<uint16_t>(acc, prod);
      }
      intermediate[y * width + x] = acc;
    }
  }
  // Horizontal pass
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      uint32_t acc = 0;
      for (int i = 0; i < 5; ++i) {
        size_t sx =
            clip(static_cast<ptrdiff_t>(x) + i - 2, width);
        acc += static_cast<uint32_t>(kernel_x[i]) *
               static_cast<uint32_t>(intermediate[y * width + sx]);
      }
      dst[y * dst_stride + x] =
          static_cast<uint8_t>(acc > 255U ? 255U : acc);
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t separable_filter_2d_5x5_u16(const uint16_t *src,
                                             size_t src_stride, uint16_t *dst,
                                             size_t dst_stride, size_t width,
                                             size_t height,
                                             const uint16_t *kernel_x,
                                             const uint16_t *kernel_y) {
  if (!src || !dst || !kernel_x || !kernel_y)
    return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  std::vector<uint32_t> intermediate(width * height);
  size_t src_stride_elems = src_stride / sizeof(uint16_t);
  size_t dst_stride_elems = dst_stride / sizeof(uint16_t);
  // Vertical pass
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      uint32_t acc = 0;
      for (int i = 0; i < 5; ++i) {
        size_t sy =
            clip(static_cast<ptrdiff_t>(y) + i - 2, height);
        uint32_t prod = static_cast<uint32_t>(kernel_y[i]) *
                        static_cast<uint32_t>(src[sy * src_stride_elems + x]);
        acc = sat_add<uint32_t>(acc, prod);
      }
      intermediate[y * width + x] = acc;
    }
  }
  // Horizontal pass
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      uint64_t acc = 0;
      for (int i = 0; i < 5; ++i) {
        size_t sx =
            clip(static_cast<ptrdiff_t>(x) + i - 2, width);
        acc += static_cast<uint64_t>(kernel_x[i]) *
               static_cast<uint64_t>(intermediate[y * width + sx]);
      }
      dst[y * dst_stride_elems + x] =
          static_cast<uint16_t>(acc > 65535U ? 65535U : acc);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
