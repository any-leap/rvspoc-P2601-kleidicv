// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared API-entry validation helpers, mirroring the upstream
// CHECK_POINTER_AND_STRIDE / CHECK_IMAGE_SIZE / alignment macros from
// kleidicv/utils.h. Used by the riscv backend's public C entry points so
// every op rejects the same misuses (oversized images, misaligned strides
// or pointers) that the upstream NEON/SVE backends reject.

#ifndef KLEIDICV_RISCV_VALIDATION_HELPER_H
#define KLEIDICV_RISCV_VALIDATION_HELPER_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::riscv_validation {

// width * height must not overflow size_t and must not exceed
// KLEIDICV_MAX_IMAGE_PIXELS.
inline kleidicv_error_t check_image_size(size_t width, size_t height) {
  size_t pixels = 0;
  if (__builtin_mul_overflow(width, height, &pixels))
    return KLEIDICV_ERROR_RANGE;
  if (pixels > KLEIDICV_MAX_IMAGE_PIXELS) return KLEIDICV_ERROR_RANGE;
  return KLEIDICV_OK;
}

// stride is in bytes. For element types with alignment > 1, stride must be
// a multiple of sizeof(T) AND the buffer pointer must be T-aligned.
template <typename T>
inline kleidicv_error_t check_buffer_alignment(const void *ptr,
                                                  size_t stride_bytes) {
  constexpr size_t a = alignof(T);
  if constexpr (a > 1) {
    if ((stride_bytes % sizeof(T)) != 0) return KLEIDICV_ERROR_ALIGNMENT;
    if ((reinterpret_cast<uintptr_t>(ptr) & (a - 1)) != 0)
      return KLEIDICV_ERROR_ALIGNMENT;
  }
  (void)ptr;
  (void)stride_bytes;
  return KLEIDICV_OK;
}

}  // namespace kleidicv::riscv_validation

#endif
