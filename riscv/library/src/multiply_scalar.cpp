// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// dst = sat(a * b). The `scale` parameter is ignored to match upstream's
// current TODO state — see multiply_decls.h.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kleidicv/kleidicv.h"

#include "elementwise_scalar.h"
#include "multiply_decls.h"

namespace kleidicv::scalar {

namespace {

template <typename T>
inline T saturating_mul_one(T a, T b) {
  using Wide = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
  Wide w = static_cast<Wide>(a) * static_cast<Wide>(b);
  constexpr Wide kMax = static_cast<Wide>(std::numeric_limits<T>::max());
  if constexpr (std::is_signed_v<T>) {
    constexpr Wide kMin = static_cast<Wide>(std::numeric_limits<T>::min());
    if (w > kMax) return std::numeric_limits<T>::max();
    if (w < kMin) return std::numeric_limits<T>::min();
  } else {
    if (w > kMax) return std::numeric_limits<T>::max();
  }
  return static_cast<T>(w);
}

}  // namespace

template <typename T>
kleidicv_error_t saturating_multiply(const T *src_a, size_t src_a_stride,
                                     const T *src_b, size_t src_b_stride,
                                     T *dst, size_t dst_stride, size_t width,
                                     size_t height, double scale) {
  (void)scale;
  return binary_elementwise<T>(
      src_a, src_a_stride, src_b, src_b_stride, dst, dst_stride, width, height,
      [](T a, T b) { return saturating_mul_one<T>(a, b); });
}

#define INST(T)                                                              \
  template kleidicv_error_t saturating_multiply(                             \
      const T *, size_t, const T *, size_t, T *, size_t, size_t, size_t,    \
      double)
INST(uint8_t);
INST(int8_t);
INST(uint16_t);
INST(int16_t);
INST(int32_t);
#undef INST

}  // namespace kleidicv::scalar
