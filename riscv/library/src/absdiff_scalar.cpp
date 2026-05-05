// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar reference for saturating_absdiff. Doubles as the bit-exact oracle
// for the RVV path.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kleidicv/kleidicv.h"

#include "absdiff_decls.h"
#include "elementwise_scalar.h"

namespace kleidicv::scalar {

namespace {

template <typename T>
inline T saturating_absdiff_one(T a, T b) {
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

}  // namespace

template <typename T>
kleidicv_error_t saturating_absdiff(const T *src_a, size_t src_a_stride,
                                    const T *src_b, size_t src_b_stride, T *dst,
                                    size_t dst_stride, size_t width,
                                    size_t height) {
  return binary_elementwise<T>(
      src_a, src_a_stride, src_b, src_b_stride, dst, dst_stride, width, height,
      [](T a, T b) { return saturating_absdiff_one<T>(a, b); });
}

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
