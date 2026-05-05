// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>

#include "kleidicv/kleidicv.h"

#include "min_max_decls.h"

namespace kleidicv::scalar {

template <typename T>
kleidicv_error_t min_max(const T *src, size_t src_stride, size_t width,
                         size_t height, T *min_out, T *max_out) {
  if (!src) return KLEIDICV_ERROR_NULL_POINTER;
  if (!min_out && !max_out) return KLEIDICV_OK;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  T mn = std::numeric_limits<T>::max();
  T mx = std::numeric_limits<T>::min();
  for (size_t y = 0; y < height; ++y) {
    const T *rs = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    for (size_t x = 0; x < width; ++x) {
      T v = rs[x];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  if (min_out) *min_out = mn;
  if (max_out) *max_out = mx;
  return KLEIDICV_OK;
}

#define INST(T)                                                                \
  template kleidicv_error_t min_max(const T *, size_t, size_t, size_t, T *, T *)
INST(uint8_t);
INST(int8_t);
INST(uint16_t);
INST(int16_t);
INST(int32_t);
#undef INST

}  // namespace kleidicv::scalar
