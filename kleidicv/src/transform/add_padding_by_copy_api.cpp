// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "add_padding_by_copy_internal.h"
#include "kleidicv/dispatch.h"
#include "kleidicv/kleidicv.h"

KLEIDICV_MULTIVERSION_C_API_WITH_SME(
    kleidicv_add_padding_by_copy_stripe,
    &kleidicv::neon::add_padding_by_copy_stripe, nullptr, nullptr, nullptr);

KLEIDICV_MULTIVERSION_C_API_WITH_SME(
    kleidicv_build_add_padding_by_copy_border_buffer,
    &kleidicv::neon::build_add_padding_by_copy_border_buffer, nullptr, nullptr,
    nullptr);

namespace kleidicv {

static kleidicv_error_t add_padding_by_copy(
    const void *src, size_t src_stride, void *dst, size_t dst_stride,
    size_t src_width, size_t src_height, size_t top_padding,
    size_t bottom_padding, size_t left_padding, size_t right_padding,
    size_t pixel_size, kleidicv_border_type_t border_type,
    const void *border_value) {
  auto prepared_state = kleidicv_build_add_padding_by_copy_border_buffer(
      src, src_stride, dst, dst_stride, src_width, src_height, top_padding,
      bottom_padding, left_padding, right_padding, pixel_size, border_type,
      border_value);
  if (prepared_state.error != KLEIDICV_OK) {
    return prepared_state.error;
  }

  return kleidicv_add_padding_by_copy_stripe(&prepared_state, 0,
                                             prepared_state.dst_height);
}

}  // namespace kleidicv

extern "C" {

kleidicv_error_t kleidicv_add_padding_by_copy(
    const void *src, size_t src_stride, void *dst, size_t dst_stride,
    size_t src_width, size_t src_height, size_t top_padding,
    size_t bottom_padding, size_t left_padding, size_t right_padding,
    size_t pixel_size, kleidicv_border_type_t border_type,
    const void *border_value) {
  return kleidicv::add_padding_by_copy(
      src, src_stride, dst, dst_stride, src_width, src_height, top_padding,
      bottom_padding, left_padding, right_padding, pixel_size, border_type,
      border_value);
}

}  // extern "C"
