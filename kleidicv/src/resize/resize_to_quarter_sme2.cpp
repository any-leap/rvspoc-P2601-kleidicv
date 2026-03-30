// SPDX-FileCopyrightText: 2025 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/resize/resize_linear.h"
#include "resize_to_quarter_sc.h"

namespace kleidicv::sme2 {

KLEIDICV_LOCALLY_STREAMING KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
resize_to_quarter_u8(const uint8_t *src, size_t src_stride, size_t src_width,
                     size_t src_height, uint8_t *dst, size_t dst_stride,
                     size_t channels) {
  return resize_to_quarter_u8_sc(src, src_stride, src_width, src_height, dst,
                                 dst_stride, channels);
}

}  // namespace kleidicv::sme2
