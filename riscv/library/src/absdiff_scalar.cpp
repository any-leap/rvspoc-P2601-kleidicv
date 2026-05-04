// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scalar reference implementation of saturating_absdiff for the RISC-V
// backend. RVV-accelerated variants will replace the inner loop in Phase 3;
// this file proves the build pipeline (toolchain, config.h, public-header
// inclusion, extern "C" linkage) end-to-end with a single operator.
//
// Public ABI note: KleidiCV exposes its C entry points as global function
// *pointers* (see KLEIDICV_API_DECLARATION in kleidicv.h), initialised at
// load time by a resolver. This file emits those pointers directly because
// there is only one backend (scalar) on RISC-V right now; a real dispatcher
// is unnecessary until RVV impls land in Phase 3.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "kleidicv/kleidicv.h"

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
static kleidicv_error_t saturating_absdiff_impl(
    const T *src_a, size_t src_a_stride, const T *src_b, size_t src_b_stride,
    T *dst, size_t dst_stride, size_t width, size_t height) {
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

}  // namespace kleidicv::scalar

// extern "C" function-pointer definitions matching KLEIDICV_API_DECLARATION
// (see kleidicv/kleidicv.h). Both the default name and the `_sme` alias point
// at the same scalar implementation while RISC-V has no SME equivalent.
#define DEFINE_ABSDIFF_C_API(suffix, T)                              \
  kleidicv_error_t (*kleidicv_saturating_absdiff_##suffix)(          \
      const T *src_a, size_t src_a_stride, const T *src_b,           \
      size_t src_b_stride, T *dst, size_t dst_stride, size_t width,  \
      size_t height) = &kleidicv::scalar::saturating_absdiff_impl<T>;\
  kleidicv_error_t (*kleidicv_saturating_absdiff_##suffix##_sme)(    \
      const T *src_a, size_t src_a_stride, const T *src_b,           \
      size_t src_b_stride, T *dst, size_t dst_stride, size_t width,  \
      size_t height) = &kleidicv::scalar::saturating_absdiff_impl<T>

extern "C" {
DEFINE_ABSDIFF_C_API(u8, uint8_t);
DEFINE_ABSDIFF_C_API(s8, int8_t);
DEFINE_ABSDIFF_C_API(u16, uint16_t);
DEFINE_ABSDIFF_C_API(s16, int16_t);
DEFINE_ABSDIFF_C_API(s32, int32_t);
}  // extern "C"
