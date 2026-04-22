// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TRANSFORM_ADD_PADDING_BY_COPY_H
#define KLEIDICV_TRANSFORM_ADD_PADDING_BY_COPY_H

#include <cstddef>

#include "kleidicv/kleidicv.h"
#include "kleidicv/utils.h"

namespace kleidicv {

struct AddPaddingByCopyPreparedState;

using AddPaddingByCopyBuildBorderBufferFn = AddPaddingByCopyPreparedState(
    const void *src, size_t src_stride, void *dst, size_t dst_stride,
    size_t src_width, size_t src_height, size_t top_padding,
    size_t bottom_padding, size_t left_padding, size_t right_padding,
    size_t pixel_size, kleidicv_border_type_t border_type,
    const void *border_value);

inline bool add_padding_by_copy_is_implemented(
    size_t src_width, size_t src_height, size_t pixel_size,
    kleidicv_border_type_t border_type) {
  if (pixel_size == 0) {
    return false;
  }

  if (border_type != KLEIDICV_BORDER_TYPE_CONSTANT &&
      border_type != KLEIDICV_BORDER_TYPE_REPLICATE &&
      border_type != KLEIDICV_BORDER_TYPE_REFLECT &&
      border_type != KLEIDICV_BORDER_TYPE_WRAP &&
      border_type != KLEIDICV_BORDER_TYPE_REVERSE) {
    return false;
  }

  if (border_type != KLEIDICV_BORDER_TYPE_CONSTANT &&
      (src_width == 0 || src_height == 0)) {
    return false;
  }

  return true;
}

}  // namespace kleidicv

extern "C" {
// For internal use only. See instead kleidicv_add_padding_by_copy.
// Add padding across a horizontal stripe of the destination image. The stripe
// is defined by the range (dst_y_begin, dst_y_end].
KLEIDICV_API_DECLARATION(
    kleidicv_add_padding_by_copy_stripe,
    const kleidicv::AddPaddingByCopyPreparedState *prepared_state,
    size_t dst_y_begin, size_t dst_y_end);

// For internal use only. This helper validates the operation once, resolves the
// branch-free stripe strategy, and materializes any prepared storage needed by
// that strategy.
extern kleidicv::AddPaddingByCopyBuildBorderBufferFn
    *kleidicv_build_add_padding_by_copy_border_buffer;
}

namespace kleidicv {

namespace neon {

AddPaddingByCopyPreparedState build_add_padding_by_copy_border_buffer(
    const void *src, size_t src_stride, void *dst, size_t dst_stride,
    size_t src_width, size_t src_height, size_t top_padding,
    size_t bottom_padding, size_t left_padding, size_t right_padding,
    size_t pixel_size, kleidicv_border_type_t border_type,
    const void *border_value);

kleidicv_error_t add_padding_by_copy_stripe(
    const AddPaddingByCopyPreparedState *prepared_state, size_t dst_y_begin,
    size_t dst_y_end);

}  // namespace neon

}  // namespace kleidicv

#endif  // KLEIDICV_TRANSFORM_ADD_PADDING_BY_COPY_H
