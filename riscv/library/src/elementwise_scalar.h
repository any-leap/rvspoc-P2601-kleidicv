// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared row-walker for scalar elementwise binary ops on image-like 2D
// buffers. The per-op .cpp supplies a per-element functor; this header owns
// the null-pointer guard, empty-rect short-circuit, stride math, and the two
// nested loops.

#ifndef KLEIDICV_RISCV_ELEMENTWISE_SCALAR_H
#define KLEIDICV_RISCV_ELEMENTWISE_SCALAR_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {

// `Op(T, T) -> T` is invoked per element.
template <typename T, typename Op>
inline kleidicv_error_t binary_elementwise(const T *src_a, size_t src_a_stride,
                                           const T *src_b, size_t src_b_stride,
                                           T *dst, size_t dst_stride,
                                           size_t width, size_t height,
                                           Op op) {
  if (!src_a || !src_b || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const T *row_a = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src_a) + y * src_a_stride);
    const T *row_b = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src_b) + y * src_b_stride);
    T *row_dst = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(dst) +
                                       y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      row_dst[x] = op(row_a[x], row_b[x]);
    }
  }
  return KLEIDICV_OK;
}

// `Op(T) -> T` is invoked per element. Single source.
template <typename T, typename Op>
inline kleidicv_error_t unary_elementwise(const T *src, size_t src_stride,
                                          T *dst, size_t dst_stride,
                                          size_t width, size_t height, Op op) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const T *row_src = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    T *row_dst = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(dst) +
                                       y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      row_dst[x] = op(row_src[x]);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar

#endif
