// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar reference implementation of saturating_absdiff. Always built; serves
// both as the fallback when the host lacks RVV and as the bit-exact oracle the
// RVV implementation in absdiff_rvv.cpp is verified against.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kleidicv/kleidicv.h"

#include "absdiff_decls.h"

namespace kleidicv::scalar {

template <typename T>
static inline T saturating_absdiff_one(T a, T b) {
  using U = std::make_unsigned_t<T>;
  U ua = static_cast<U>(a);
  U ub = static_cast<U>(b);
  U diff = (a > b) ? static_cast<U>(ua - ub) : static_cast<U>(ub - ua);
  if constexpr (std::is_unsigned_v<T>) {
    return static_cast<T>(diff);
  } else {
    constexpr U kMax = static_cast<U>(std::numeric_limits<T>::max());
    return static_cast<T>(diff > kMax ? kMax : diff);
  }
}

template <typename T>
kleidicv_error_t saturating_absdiff(const T *src_a, size_t src_a_stride,
                                    const T *src_b, size_t src_b_stride, T *dst,
                                    size_t dst_stride, size_t width,
                                    size_t height) {
  if (!src_a || !src_b || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const T *row_a =
        reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(src_a) +
                                    y * src_a_stride);
    const T *row_b =
        reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(src_b) +
                                    y * src_b_stride);
    T *row_dst = reinterpret_cast<T *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      row_dst[x] = saturating_absdiff_one<T>(row_a[x], row_b[x]);
    }
  }
  return KLEIDICV_OK;
}

// Explicit instantiations so the function-pointer take in absdiff_api.cpp can
// resolve without seeing the template definition.
template kleidicv_error_t saturating_absdiff(const uint8_t *, size_t,
                                             const uint8_t *, size_t,
                                             uint8_t *, size_t, size_t, size_t);
template kleidicv_error_t saturating_absdiff(const int8_t *, size_t,
                                             const int8_t *, size_t, int8_t *,
                                             size_t, size_t, size_t);
template kleidicv_error_t saturating_absdiff(const uint16_t *, size_t,
                                             const uint16_t *, size_t,
                                             uint16_t *, size_t, size_t,
                                             size_t);
template kleidicv_error_t saturating_absdiff(const int16_t *, size_t,
                                             const int16_t *, size_t, int16_t *,
                                             size_t, size_t, size_t);
template kleidicv_error_t saturating_absdiff(const int32_t *, size_t,
                                             const int32_t *, size_t, int32_t *,
                                             size_t, size_t, size_t);

}  // namespace kleidicv::scalar
