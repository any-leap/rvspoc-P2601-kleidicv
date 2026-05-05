// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar reference for float_conv. f32→int rounds to nearest (lrintf, banker's
// rounding by default) and saturates; int→f32 is exact (8-bit values fit
// losslessly in f32).

#include <cmath>
#include <cstdint>
#include <limits>

#include "kleidicv/kleidicv.h"

#include "float_conv_decls.h"

namespace kleidicv::scalar {

#define HEAD_F32_TO(DT)                                                     \
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;                    \
  if (width == 0 || height == 0) return KLEIDICV_OK;                       \
  for (size_t y = 0; y < height; ++y) {                                    \
    const float *rs = reinterpret_cast<const float *>(                     \
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);          \
    DT *rd = reinterpret_cast<DT *>(reinterpret_cast<uint8_t *>(dst) +     \
                                    y * dst_stride);                       \
    for (size_t x = 0; x < width; ++x)

#define HEAD_INT_TO_F32(ST)                                                 \
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;                    \
  if (width == 0 || height == 0) return KLEIDICV_OK;                       \
  for (size_t y = 0; y < height; ++y) {                                    \
    const ST *rs = reinterpret_cast<const ST *>(                           \
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);          \
    float *rd = reinterpret_cast<float *>(                                  \
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);                \
    for (size_t x = 0; x < width; ++x)

#define TAIL                                                               \
  }                                                                        \
  return KLEIDICV_OK;

kleidicv_error_t f32_to_u8(const float *src, size_t src_stride, uint8_t *dst,
                           size_t dst_stride, size_t width, size_t height) {
  HEAD_F32_TO(uint8_t) {
    long iv = std::lrintf(rs[x]);
    if (iv < 0) rd[x] = 0;
    else if (iv > 255) rd[x] = 255;
    else rd[x] = static_cast<uint8_t>(iv);
  }
  TAIL
}

kleidicv_error_t f32_to_s8(const float *src, size_t src_stride, int8_t *dst,
                           size_t dst_stride, size_t width, size_t height) {
  HEAD_F32_TO(int8_t) {
    long iv = std::lrintf(rs[x]);
    if (iv < std::numeric_limits<int8_t>::min())
      rd[x] = std::numeric_limits<int8_t>::min();
    else if (iv > std::numeric_limits<int8_t>::max())
      rd[x] = std::numeric_limits<int8_t>::max();
    else
      rd[x] = static_cast<int8_t>(iv);
  }
  TAIL
}

kleidicv_error_t u8_to_f32(const uint8_t *src, size_t src_stride, float *dst,
                           size_t dst_stride, size_t width, size_t height) {
  HEAD_INT_TO_F32(uint8_t) { rd[x] = static_cast<float>(rs[x]); }
  TAIL
}

kleidicv_error_t s8_to_f32(const int8_t *src, size_t src_stride, float *dst,
                           size_t dst_stride, size_t width, size_t height) {
  HEAD_INT_TO_F32(int8_t) { rd[x] = static_cast<float>(rs[x]); }
  TAIL
}

#undef HEAD_F32_TO
#undef HEAD_INT_TO_F32
#undef TAIL

}  // namespace kleidicv::scalar
