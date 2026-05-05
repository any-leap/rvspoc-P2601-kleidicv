// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Generic byte-wise split/merge. Works for any element_size and any number of
// channels — the RVV path specialises {2,3,4} × {1,2,4,8} for throughput, this
// path is the catch-all.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kleidicv/kleidicv.h"

#include "split_merge_decls.h"

namespace kleidicv::scalar {

kleidicv_error_t split(const void *src, size_t src_stride, void **dsts,
                       const size_t *dst_strides, size_t width, size_t height,
                       size_t channels, size_t element_size) {
  if (!src || !dsts || !dst_strides) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels == 0 || element_size == 0) return KLEIDICV_ERROR_RANGE;
  for (size_t c = 0; c < channels; ++c)
    if (!dsts[c]) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *rs =
        static_cast<const uint8_t *>(src) + y * src_stride;
    for (size_t x = 0; x < width; ++x) {
      for (size_t c = 0; c < channels; ++c) {
        uint8_t *rd = static_cast<uint8_t *>(dsts[c]) + y * dst_strides[c];
        std::memcpy(rd + x * element_size,
                    rs + (x * channels + c) * element_size, element_size);
      }
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t merge(const void **srcs, const size_t *src_strides, void *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t element_size) {
  if (!srcs || !src_strides || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels == 0 || element_size == 0) return KLEIDICV_ERROR_RANGE;
  for (size_t c = 0; c < channels; ++c)
    if (!srcs[c]) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  for (size_t y = 0; y < height; ++y) {
    uint8_t *rd = static_cast<uint8_t *>(dst) + y * dst_stride;
    for (size_t x = 0; x < width; ++x) {
      for (size_t c = 0; c < channels; ++c) {
        const uint8_t *rs =
            static_cast<const uint8_t *>(srcs[c]) + y * src_strides[c];
        std::memcpy(rd + (x * channels + c) * element_size,
                    rs + x * element_size, element_size);
      }
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
