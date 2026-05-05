// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar reference for saturating_add. Doubles as the bit-exact oracle for the
// RVV path.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kleidicv/kleidicv.h"

#include "add_decls.h"
#include "elementwise_scalar.h"

namespace kleidicv::scalar {

namespace {

template <typename T>
inline T saturating_add_one(T a, T b) {
  if constexpr (sizeof(T) == 8) {
    if constexpr (std::is_unsigned_v<T>) {
      T s = static_cast<T>(a + b);
      return s < a ? std::numeric_limits<T>::max() : s;
    } else {
      // Signed 64: overflow iff sign(a)==sign(b) and sign(a+b)!=sign(a).
      using U = std::make_unsigned_t<T>;
      U sum = static_cast<U>(static_cast<U>(a) + static_cast<U>(b));
      T r = static_cast<T>(sum);
      bool sa = a < 0;
      bool sb = b < 0;
      if (sa == sb && (r < 0) != sa) {
        return sa ? std::numeric_limits<T>::min()
                  : std::numeric_limits<T>::max();
      }
      return r;
    }
  } else {
    using L = std::conditional_t<std::is_signed_v<T>, long long,
                                 unsigned long long>;
    L wide = static_cast<L>(a) + static_cast<L>(b);
    constexpr L kMax = static_cast<L>(std::numeric_limits<T>::max());
    constexpr L kMin = static_cast<L>(std::numeric_limits<T>::min());
    if (wide > kMax) return std::numeric_limits<T>::max();
    if (wide < kMin) return std::numeric_limits<T>::min();
    return static_cast<T>(wide);
  }
}

}  // namespace

template <typename T>
kleidicv_error_t saturating_add(const T *src_a, size_t src_a_stride,
                                const T *src_b, size_t src_b_stride, T *dst,
                                size_t dst_stride, size_t width,
                                size_t height) {
  return binary_elementwise<T>(
      src_a, src_a_stride, src_b, src_b_stride, dst, dst_stride, width, height,
      [](T a, T b) { return saturating_add_one<T>(a, b); });
}

#define INST(T)                                                              \
  template kleidicv_error_t saturating_add(const T *, size_t, const T *,     \
                                           size_t, T *, size_t, size_t, size_t)
INST(uint8_t);
INST(int8_t);
INST(uint16_t);
INST(int16_t);
INST(uint32_t);
INST(int32_t);
INST(uint64_t);
INST(int64_t);
#undef INST

}  // namespace kleidicv::scalar
