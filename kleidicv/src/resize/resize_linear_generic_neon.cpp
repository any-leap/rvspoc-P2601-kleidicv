// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <variant>

#include "kleidicv/ctypes.h"
#include "kleidicv/resize/resize_linear.h"
#include "resize_linear_generic_u8_neon.h"

namespace kleidicv::neon {

// ratio: number of vectors to load and resize to 1 vector
// supported ratio values: 2 and 3
// supported channel counts: 1, 2 and 3
template <int kRatio, int kChannels>
kleidicv_error_t kleidicv_resize_generic_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst,  // NOLINT
    size_t dst_stride, size_t dst_width, size_t dst_height) {
  resize_linear_generic_u8::RowInterpolationConstantsGenerator<kRatio,
                                                               kChannels>
      generator{src_width, dst_width};

  auto row_interpolation_constants_variant = generator();

  if (auto *err =
          std::get_if<kleidicv_error_t>(&row_interpolation_constants_variant)) {
    return *err;
  }
  auto &row_interpolation_constants =
      *std::get_if<resize_linear_generic_u8::RowInterpolationConstants>(
          &row_interpolation_constants_variant);

  if constexpr (kChannels == 3 && kRatio == 3) {
    double inverse_scale =
        static_cast<double>(src_width) / static_cast<double>(dst_width);
    if (inverse_scale >= 2.8) {
      // Rightmost lanes of b and d vectors need to be set by scalar code, as
      // the table lookup overindexes the src registers
      resize_linear_generic_u8::ResizeGenericU8Operation<kRatio, kChannels,
                                                         true>
          operation(src, src_stride, src_width, src_height, y_begin, y_end, dst,
                    dst_stride, dst_height);
      operation.process_rows(row_interpolation_constants);
      return KLEIDICV_OK;
    }
  }

  resize_linear_generic_u8::ResizeGenericU8Operation<kRatio, kChannels>
      operation(src, src_stride, src_width, src_height, y_begin, y_end, dst,
                dst_stride, dst_height);
  operation.process_rows(row_interpolation_constants);

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(ratio, channels)               \
  template kleidicv_error_t                                          \
  kleidicv_resize_generic_stripe_u8<ratio, channels>(                \
      const uint8_t *src, size_t src_stride, size_t src_width,       \
      size_t src_height, size_t y_begin, size_t y_end, uint8_t *dst, \
      size_t dst_stride, size_t dst_width, size_t dst_height)

KLEIDICV_INSTANTIATE_TEMPLATE(1, 1);
KLEIDICV_INSTANTIATE_TEMPLATE(1, 2);
KLEIDICV_INSTANTIATE_TEMPLATE(1, 3);
KLEIDICV_INSTANTIATE_TEMPLATE(2, 1);
KLEIDICV_INSTANTIATE_TEMPLATE(2, 2);
KLEIDICV_INSTANTIATE_TEMPLATE(2, 3);
KLEIDICV_INSTANTIATE_TEMPLATE(3, 1);
KLEIDICV_INSTANTIATE_TEMPLATE(3, 2);
KLEIDICV_INSTANTIATE_TEMPLATE(3, 3);

}  // namespace kleidicv::neon
