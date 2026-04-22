// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SRC_TRANSFORM_ADD_PADDING_BY_COPY_INTERNAL_H
#define KLEIDICV_SRC_TRANSFORM_ADD_PADDING_BY_COPY_INTERNAL_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/config.h"
#include "kleidicv/containers/small_buffer.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/transform/add_padding_by_copy.h"

namespace kleidicv {

// Keep the common constant-row case on the stack to avoid heap allocation for
// typical image widths.
constexpr size_t kAddPaddingByCopyPreparedBorderInlineWords =
    (3072 + sizeof(size_t) - 1) / sizeof(size_t);

using AddPaddingByCopyPreparedBorderBuffer =
    SmallBuffer<size_t, kAddPaddingByCopyPreparedBorderInlineWords>;

enum class AddPaddingByCopyHorizontalMode : uint8_t {
  kConstant,
  kReplicate,
  kWrapFast,
  kReflectFast,
  kReverseFast,
  kIndexed,
};

using AddPaddingByCopyRepeatRowFn = void (*)(const uint8_t *value_bytes,
                                             size_t element_size, uint8_t *dst,
                                             size_t row_size);
using AddPaddingByCopyReverseCopyFn = void (*)(const uint8_t *src_last,
                                               size_t pixel_size, uint8_t *dst,
                                               size_t row_size);
using AddPaddingByCopyIndexedCopyFn = void (*)(const uint8_t *src, uint8_t *dst,
                                               const size_t *indices,
                                               size_t count, size_t pixel_size);

struct AddPaddingByCopyPreparedState {
  kleidicv_error_t error;
  const uint8_t *src;
  uint8_t *dst;
  size_t src_stride;
  size_t dst_stride;
  size_t src_width;
  size_t src_height;
  size_t top_padding;
  size_t bottom_padding;
  size_t left_padding;
  size_t right_padding;
  size_t pixel_size;
  size_t dst_width;
  size_t dst_height;
  size_t row_size;
  size_t left_size;
  size_t inner_size;
  size_t right_size;
  size_t inner_offset;
  size_t right_offset;
  size_t horizontal_data_offset_words;
  size_t top_map_offset_words;
  size_t bottom_map_offset_words;
  size_t horizontal_data_words;
  size_t top_map_words;
  size_t bottom_map_words;
  kleidicv_border_type_t border_type;
  AddPaddingByCopyHorizontalMode horizontal_mode;
  AddPaddingByCopyRepeatRowFn repeat_row_fn;
  AddPaddingByCopyReverseCopyFn reverse_copy_fn;
  AddPaddingByCopyIndexedCopyFn indexed_copy_fn;
  AddPaddingByCopyPreparedBorderBuffer storage;

  AddPaddingByCopyPreparedState()
      : error(KLEIDICV_OK),
        src(nullptr),
        dst(nullptr),
        src_stride(0),
        dst_stride(0),
        src_width(0),
        src_height(0),
        top_padding(0),
        bottom_padding(0),
        left_padding(0),
        right_padding(0),
        pixel_size(0),
        dst_width(0),
        dst_height(0),
        row_size(0),
        left_size(0),
        inner_size(0),
        right_size(0),
        inner_offset(0),
        right_offset(0),
        horizontal_data_offset_words(0),
        top_map_offset_words(0),
        bottom_map_offset_words(0),
        horizontal_data_words(0),
        top_map_words(0),
        bottom_map_words(0),
        border_type(KLEIDICV_BORDER_TYPE_CONSTANT),
        horizontal_mode(AddPaddingByCopyHorizontalMode::kConstant),
        repeat_row_fn(nullptr),
        reverse_copy_fn(nullptr),
        indexed_copy_fn(nullptr),
        storage(0) {}

  explicit AddPaddingByCopyPreparedState(size_t storage_words)
      : error(KLEIDICV_OK),
        src(nullptr),
        dst(nullptr),
        src_stride(0),
        dst_stride(0),
        src_width(0),
        src_height(0),
        top_padding(0),
        bottom_padding(0),
        left_padding(0),
        right_padding(0),
        pixel_size(0),
        dst_width(0),
        dst_height(0),
        row_size(0),
        left_size(0),
        inner_size(0),
        right_size(0),
        inner_offset(0),
        right_offset(0),
        horizontal_data_offset_words(0),
        top_map_offset_words(0),
        bottom_map_offset_words(0),
        horizontal_data_words(0),
        top_map_words(0),
        bottom_map_words(0),
        border_type(KLEIDICV_BORDER_TYPE_CONSTANT),
        horizontal_mode(AddPaddingByCopyHorizontalMode::kConstant),
        repeat_row_fn(nullptr),
        reverse_copy_fn(nullptr),
        indexed_copy_fn(nullptr),
        storage(storage_words) {}

  AddPaddingByCopyPreparedState(const AddPaddingByCopyPreparedState &) = delete;
  AddPaddingByCopyPreparedState &operator=(
      const AddPaddingByCopyPreparedState &) = delete;
  AddPaddingByCopyPreparedState(AddPaddingByCopyPreparedState &&) noexcept =
      default;
  AddPaddingByCopyPreparedState &operator=(AddPaddingByCopyPreparedState &&) =
      delete;

  const uint8_t *constant_row() const {
    return reinterpret_cast<const uint8_t *>(storage.get() +
                                             horizontal_data_offset_words);
  }

  const size_t *horizontal_indices() const {
    return storage.get() + horizontal_data_offset_words;
  }

  const size_t *top_source_rows() const {
    return storage.get() + top_map_offset_words;
  }

  const size_t *bottom_source_rows() const {
    return storage.get() + bottom_map_offset_words;
  }
};

inline AddPaddingByCopyPreparedState add_padding_by_copy_error_state(
    kleidicv_error_t error) {
  AddPaddingByCopyPreparedState state;
  state.error = error;
  return state;
}

}  // namespace kleidicv

#endif  // KLEIDICV_SRC_TRANSFORM_ADD_PADDING_BY_COPY_INTERNAL_H
