// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "sum_decls.h"

namespace kleidicv::scalar {

kleidicv_error_t sum_f32(const float *src, size_t src_stride, size_t width,
                         size_t height, float *sum) {
  if (!src || !sum) return KLEIDICV_ERROR_NULL_POINTER;
  // Match upstream behaviour: f64 accumulator to mitigate precision loss when
  // summing many f32 values; final cast to f32 on store.
  double acc = 0.0;
  if (width > 0 && height > 0) {
    for (size_t y = 0; y < height; ++y) {
      const float *row = reinterpret_cast<const float *>(
          reinterpret_cast<const uint8_t *>(src) + y * src_stride);
      for (size_t x = 0; x < width; ++x) acc += static_cast<double>(row[x]);
    }
  }
  *sum = static_cast<float>(acc);
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
