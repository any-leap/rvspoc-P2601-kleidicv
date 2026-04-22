// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "add_padding_by_copy_internal.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

// -----------------------------------------------------------------------------
// Common helpers
// -----------------------------------------------------------------------------

// Byte copy helper used by the scalar paths and prepared-state setup.
inline void copy_bytes(const uint8_t *src, uint8_t *dst, size_t size) {
  std::memcpy(static_cast<void *>(dst), static_cast<const void *>(src), size);
}

// Load a scalar from byte-addressed storage without assuming alignment.
template <typename T>
inline T load_unaligned(const uint8_t *src) {
  T value;
  copy_bytes(src, reinterpret_cast<uint8_t *>(&value), sizeof(T));
  return value;
}

template <typename PixelType>
inline bool is_pointer_aligned(const void *ptr) {
  return !is_misaligned<PixelType>(reinterpret_cast<uintptr_t>(ptr));
}

inline size_t add_padding_by_copy_bytes_to_words(size_t size_in_bytes) {
  return (size_in_bytes + sizeof(size_t) - 1) / sizeof(size_t);
}

inline kleidicv_error_t checked_add(size_t lhs, size_t rhs, size_t *dst) {
  return __builtin_add_overflow(lhs, rhs, dst) ? KLEIDICV_ERROR_RANGE
                                               : KLEIDICV_OK;
}

inline kleidicv_error_t checked_mul(size_t lhs, size_t rhs, size_t *dst) {
  return __builtin_mul_overflow(lhs, rhs, dst) ? KLEIDICV_ERROR_RANGE
                                               : KLEIDICV_OK;
}

inline kleidicv_error_t check_image_size_value(size_t width, size_t height) {
  size_t image_size = 0;
  if (__builtin_mul_overflow(width, height, &image_size)) {
    return KLEIDICV_ERROR_RANGE;
  }

  if (image_size > KLEIDICV_MAX_IMAGE_PIXELS) {
    return KLEIDICV_ERROR_RANGE;
  }

  return KLEIDICV_OK;
}

// -----------------------------------------------------------------------------
// Shared loop helpers
// -----------------------------------------------------------------------------

// Shared unroll pattern for contiguous vector-sized byte ranges.
template <typename QuadFn, typename PairFn, typename SingleFn, typename TailFn>
inline void run_unrolled_vector_loop(size_t row_size, size_t step,
                                     QuadFn quad_fn, PairFn pair_fn,
                                     SingleFn single_fn, TailFn tail_fn) {
  LoopUnroll2 loop{row_size, step};
  loop.unroll_four_times(quad_fn);
  loop.unroll_twice(pair_fn);
  loop.unroll_once(single_fn);
  loop.remaining(tail_fn);
}

// Shared unroll pattern for 3-channel interleaved vector stores.
template <typename TwiceFn, typename SingleFn, typename TailFn>
inline void run_unrolled_interleaved_loop(size_t row_size, size_t step,
                                          TwiceFn twice_fn, SingleFn single_fn,
                                          TailFn tail_fn) {
  LoopUnroll2 loop{row_size, step};
  loop.unroll_twice(twice_fn);
  loop.unroll_once(single_fn);
  loop.remaining(tail_fn);
}

// -----------------------------------------------------------------------------
// Repeated-value helpers
// -----------------------------------------------------------------------------

// Scalar fallback for filling a row with the same pixel.
inline void repeat_pixel_across_row_scalar(const uint8_t *value_bytes,
                                           size_t element_size, uint8_t *dst,
                                           size_t row_size) {
  for (size_t offset = 0; offset < row_size; offset += element_size) {
    copy_bytes(value_bytes, dst + offset, element_size);
  }
}

// Vector path for single-channel rows.
template <typename ScalarType>
inline void repeat_pixel_across_row_single_channel(const uint8_t *value_bytes,
                                                   uint8_t *dst,
                                                   size_t row_size) {
  using VectorType = typename VecTraits<ScalarType>::VectorType;
  using Vector2Type = typename VecTraits<ScalarType>::Vector2Type;
  using Vector4Type = typename VecTraits<ScalarType>::Vector4Type;

  const ScalarType value = load_unaligned<ScalarType>(value_bytes);
  const VectorType repeated_value = vdupq_n(value);
  Vector4Type repeated_value_quad;
  Vector2Type repeated_value_pair;
  repeated_value_quad.val[0] = repeated_value;
  repeated_value_quad.val[1] = repeated_value;
  repeated_value_quad.val[2] = repeated_value;
  repeated_value_quad.val[3] = repeated_value;
  repeated_value_pair.val[0] = repeated_value;
  repeated_value_pair.val[1] = repeated_value;

  auto store_repeated_quad = [&](size_t index) {
    VecTraits<ScalarType>::store(repeated_value_quad,
                                 reinterpret_cast<ScalarType *>(dst + index));
  };
  auto store_repeated_pair = [&](size_t index) {
    VecTraits<ScalarType>::store(repeated_value_pair,
                                 reinterpret_cast<ScalarType *>(dst + index));
  };
  auto store_repeated_vector = [&](size_t index) {
    VecTraits<ScalarType>::store(repeated_value,
                                 reinterpret_cast<ScalarType *>(dst + index));
  };

  run_unrolled_vector_loop(
      row_size, kVectorLength, store_repeated_quad, store_repeated_pair,
      store_repeated_vector, [&](size_t index, size_t length) {
        repeat_pixel_across_row_scalar(value_bytes, sizeof(ScalarType),
                                       dst + index, length - index);
      });
}

// Vector path for interleaved 3-channel rows.
template <typename ScalarType>
inline void repeat_pixel_across_row_interleaved_3(const uint8_t *value_bytes,
                                                  uint8_t *dst,
                                                  size_t row_size) {
  using Vector3Type = typename VecTraits<ScalarType>::Vector3Type;

  Vector3Type vector_value;
  vector_value.val[0] = vdupq_n(load_unaligned<ScalarType>(value_bytes));
  vector_value.val[1] =
      vdupq_n(load_unaligned<ScalarType>(value_bytes + sizeof(ScalarType)));
  vector_value.val[2] = vdupq_n(
      load_unaligned<ScalarType>(value_bytes + (2 * sizeof(ScalarType))));

  constexpr size_t kPixelSize = 3 * sizeof(ScalarType);
  constexpr size_t kVectorStoreSize = 3 * kVectorLength;

  auto store = [&](size_t index) {
    vst3q(reinterpret_cast<ScalarType *>(dst + index), vector_value);
  };
  auto store_twice = [&](size_t index) {
    store(index);
    store(index + kVectorStoreSize);
  };

  run_unrolled_interleaved_loop(row_size, kVectorStoreSize, store_twice, store,
                                [&](size_t index, size_t length) {
                                  repeat_pixel_across_row_scalar(
                                      value_bytes, kPixelSize, dst + index,
                                      length - index);
                                });
}

inline void repeat_pixel_across_row_dispatch(const uint8_t *value_bytes,
                                             size_t element_size, uint8_t *dst,
                                             size_t row_size) {
  switch (element_size) {
    case sizeof(uint8_t):
      repeat_pixel_across_row_single_channel<uint8_t>(value_bytes, dst,
                                                      row_size);
      break;
    case sizeof(uint16_t):
      repeat_pixel_across_row_single_channel<uint16_t>(value_bytes, dst,
                                                       row_size);
      break;
    case sizeof(uint32_t):
      repeat_pixel_across_row_single_channel<uint32_t>(value_bytes, dst,
                                                       row_size);
      break;
    case sizeof(uint64_t):
      repeat_pixel_across_row_single_channel<uint64_t>(value_bytes, dst,
                                                       row_size);
      break;
    case 3 * sizeof(uint8_t):
      repeat_pixel_across_row_interleaved_3<uint8_t>(value_bytes, dst,
                                                     row_size);
      break;
    case 3 * sizeof(uint16_t):
      repeat_pixel_across_row_interleaved_3<uint16_t>(value_bytes, dst,
                                                      row_size);
      break;
    case 3 * sizeof(uint32_t):
      repeat_pixel_across_row_interleaved_3<uint32_t>(value_bytes, dst,
                                                      row_size);
      break;
    default:
      repeat_pixel_across_row_scalar(value_bytes, element_size, dst, row_size);
      break;
  }
}

// -----------------------------------------------------------------------------
// Reversed-copy helpers
// -----------------------------------------------------------------------------

// Reverse vector lane order using 64-bit lane reversal plus half swap.
template <typename PixelType>
inline typename VecTraits<PixelType>::VectorType reverse_vector_lanes(
    typename VecTraits<PixelType>::VectorType vector_value) {
  const auto reversed_vector_64 = vrev64q(vector_value);
  return vcombine(vget_high(reversed_vector_64), vget_low(reversed_vector_64));
}

// Reverse pixel order in an interleaved 3-channel vector block.
template <typename PixelType>
inline typename VecTraits<PixelType>::Vector3Type reverse_vector3_lanes(
    typename VecTraits<PixelType>::Vector3Type vector_value) {
  vector_value.val[0] = reverse_vector_lanes<PixelType>(vector_value.val[0]);
  vector_value.val[1] = reverse_vector_lanes<PixelType>(vector_value.val[1]);
  vector_value.val[2] = reverse_vector_lanes<PixelType>(vector_value.val[2]);
  return vector_value;
}

// Scalar reverse-copy fallback.
inline void copy_reversed_pixels_scalar(const uint8_t *src_last, uint8_t *dst,
                                        size_t pixel_size, size_t row_size) {
  for (size_t offset = 0; offset < row_size; offset += pixel_size) {
    copy_bytes(src_last - static_cast<ptrdiff_t>(offset), dst + offset,
               pixel_size);
  }
}

// Vector reverse-copy path for packed pixels.
template <typename PixelType>
inline void copy_reversed_pixels_packed(const uint8_t *src_last, uint8_t *dst,
                                        size_t row_size) {
  using VectorType = typename VecTraits<PixelType>::VectorType;
  constexpr size_t kStep = kVectorLength;

  auto copy_block = [&](size_t index) {
    VectorType src_vector;
    VecTraits<PixelType>::load(
        reinterpret_cast<const PixelType *>(src_last + sizeof(PixelType) -
                                            index - kStep),
        src_vector);
    VecTraits<PixelType>::store(reverse_vector_lanes<PixelType>(src_vector),
                                reinterpret_cast<PixelType *>(dst + index));
  };
  auto copy_block_pair = [&](size_t index) {
    copy_block(index);
    copy_block(index + kStep);
  };
  auto copy_block_quad = [&](size_t index) {
    copy_block(index);
    copy_block(index + kStep);
    copy_block(index + (2 * kStep));
    copy_block(index + (3 * kStep));
  };

  run_unrolled_vector_loop(row_size, kStep, copy_block_quad, copy_block_pair,
                           copy_block, [&](size_t index, size_t length) {
                             copy_reversed_pixels_scalar(
                                 src_last - static_cast<ptrdiff_t>(index),
                                 dst + index, sizeof(PixelType),
                                 length - index);
                           });
}

// Vector reverse-copy path for interleaved 3-channel pixels.
template <typename PixelType>
inline void copy_reversed_pixels_interleaved_3(const uint8_t *src_last,
                                               uint8_t *dst, size_t row_size) {
  using Vector3Type = typename VecTraits<PixelType>::Vector3Type;

  constexpr size_t kPixelStride = 3;
  constexpr size_t kPixelSize = kPixelStride * sizeof(PixelType);
  constexpr size_t kVectorStoreSize = 3 * kVectorLength;

  auto copy_block = [&](size_t index) {
    const Vector3Type src_vector = vld3q(reinterpret_cast<const PixelType *>(
        src_last + kPixelSize - index - kVectorStoreSize));
    vst3q(reinterpret_cast<PixelType *>(dst + index),
          reverse_vector3_lanes<PixelType>(src_vector));
  };
  auto copy_block_twice = [&](size_t index) {
    copy_block(index);
    copy_block(index + kVectorStoreSize);
  };

  run_unrolled_interleaved_loop(row_size, kVectorStoreSize, copy_block_twice,
                                copy_block, [&](size_t index, size_t length) {
                                  copy_reversed_pixels_scalar(
                                      src_last - static_cast<ptrdiff_t>(index),
                                      dst + index, kPixelSize, length - index);
                                });
}

inline void copy_reversed_pixels_dispatch(const uint8_t *src_last,
                                          size_t pixel_size, uint8_t *dst,
                                          size_t row_size) {
  switch (pixel_size) {
    case sizeof(uint8_t):
      copy_reversed_pixels_packed<uint8_t>(src_last, dst, row_size);
      break;
    case sizeof(uint16_t):
      copy_reversed_pixels_packed<uint16_t>(src_last, dst, row_size);
      break;
    case sizeof(uint32_t):
      copy_reversed_pixels_packed<uint32_t>(src_last, dst, row_size);
      break;
    case sizeof(uint64_t):
      copy_reversed_pixels_packed<uint64_t>(src_last, dst, row_size);
      break;
    case 3 * sizeof(uint8_t):
      copy_reversed_pixels_interleaved_3<uint8_t>(src_last, dst, row_size);
      break;
    case 3 * sizeof(uint16_t):
      copy_reversed_pixels_interleaved_3<uint16_t>(src_last, dst, row_size);
      break;
    case 3 * sizeof(uint32_t):
      copy_reversed_pixels_interleaved_3<uint32_t>(src_last, dst, row_size);
      break;
    default:
      copy_reversed_pixels_scalar(src_last, dst, pixel_size, row_size);
      break;
  }
}

// -----------------------------------------------------------------------------
// Indexed-copy helpers
// -----------------------------------------------------------------------------

// Typed indexed copy used when the source and destination rows are aligned.
template <typename PixelType>
inline void copy_indexed_pixels_typed(const uint8_t *src, uint8_t *dst,
                                      const size_t *indices, size_t count,
                                      size_t /*pixel_size*/) {
  auto *typed_dst = reinterpret_cast<PixelType *>(dst);
  auto *typed_src = reinterpret_cast<const PixelType *>(src);

  for (size_t i = 0; i < count; ++i) {
    typed_dst[i] = typed_src[indices[i]];
  }
}

// Byte-wise indexed copy fallback.
inline void copy_indexed_pixels_scalar(const uint8_t *src, uint8_t *dst,
                                       const size_t *indices, size_t count,
                                       size_t pixel_size) {
  size_t dst_offset = 0;
  for (size_t i = 0; i < count; ++i, dst_offset += pixel_size) {
    copy_bytes(src + indices[i] * pixel_size, dst + dst_offset, pixel_size);
  }
}

template <typename PixelType>
inline bool are_rows_aligned(const void *ptr, size_t stride) {
  return is_pointer_aligned<PixelType>(ptr) &&
         !is_misaligned<PixelType>(stride);
}

inline AddPaddingByCopyIndexedCopyFn select_indexed_copy_dispatch(
    size_t pixel_size, const uint8_t *src, size_t src_stride,
    const uint8_t *dst, size_t dst_stride) {
  switch (pixel_size) {
    case sizeof(uint8_t):
      return &copy_indexed_pixels_typed<uint8_t>;
    case sizeof(uint16_t):
      if (are_rows_aligned<uint16_t>(src, src_stride) &&
          are_rows_aligned<uint16_t>(dst, dst_stride)) {
        return &copy_indexed_pixels_typed<uint16_t>;
      }
      break;
    case sizeof(uint32_t):
      if (are_rows_aligned<uint32_t>(src, src_stride) &&
          are_rows_aligned<uint32_t>(dst, dst_stride)) {
        return &copy_indexed_pixels_typed<uint32_t>;
      }
      break;
    case sizeof(uint64_t):
      if (are_rows_aligned<uint64_t>(src, src_stride) &&
          are_rows_aligned<uint64_t>(dst, dst_stride)) {
        return &copy_indexed_pixels_typed<uint64_t>;
      }
      break;
    default:
      break;
  }

  return &copy_indexed_pixels_scalar;
}

// -----------------------------------------------------------------------------
// Border mapping helpers
// -----------------------------------------------------------------------------

inline size_t replicate_index(ptrdiff_t position, size_t length) {
  return position < 0 ? 0 : length - 1;
}

inline size_t wrap_index(ptrdiff_t position, size_t length) {
  const ptrdiff_t signed_length = static_cast<ptrdiff_t>(length);
  ptrdiff_t wrapped = position % signed_length;
  if (wrapped < 0) {
    wrapped += signed_length;
  }
  return static_cast<size_t>(wrapped);
}

inline size_t reflect_index(ptrdiff_t position, size_t length) {
  if (length == 1) {
    return 0;
  }

  const ptrdiff_t period = static_cast<ptrdiff_t>(length << 1);
  ptrdiff_t reflected = position % period;
  if (reflected < 0) {
    reflected += period;
  }
  if (reflected >= static_cast<ptrdiff_t>(length)) {
    reflected = period - reflected - 1;
  }
  return static_cast<size_t>(reflected);
}

inline size_t reverse_index(ptrdiff_t position, size_t length) {
  if (length == 1) {
    return 0;
  }

  const ptrdiff_t period = static_cast<ptrdiff_t>((length - 1) << 1);
  ptrdiff_t reflected = position % period;
  if (reflected < 0) {
    reflected += period;
  }
  if (reflected >= static_cast<ptrdiff_t>(length)) {
    reflected = period - reflected;
  }
  return static_cast<size_t>(reflected);
}

template <typename MapFn>
inline void build_coordinate_sequence(size_t *dst, size_t count,
                                      ptrdiff_t start, size_t length,
                                      MapFn map_fn) {
  for (size_t i = 0; i < count; ++i) {
    dst[i] = map_fn(start + static_cast<ptrdiff_t>(i), length);
  }
}

template <typename MapFn>
inline void build_horizontal_border_indices(size_t *border_indices, size_t left,
                                            size_t right, size_t width,
                                            MapFn map_fn) {
  build_coordinate_sequence(border_indices, left, -static_cast<ptrdiff_t>(left),
                            width, map_fn);
  build_coordinate_sequence(border_indices + left, right,
                            static_cast<ptrdiff_t>(width), width, map_fn);
}

template <typename MapFn>
inline void build_vertical_border_rows(size_t *top_rows, size_t top_padding,
                                       size_t *bottom_rows,
                                       size_t bottom_padding, size_t height,
                                       MapFn map_fn) {
  build_coordinate_sequence(top_rows, top_padding,
                            -static_cast<ptrdiff_t>(top_padding), height,
                            map_fn);
  build_coordinate_sequence(bottom_rows, bottom_padding,
                            static_cast<ptrdiff_t>(height), height, map_fn);
}

inline void build_horizontal_indices(size_t *border_indices, size_t left,
                                     size_t right, size_t width,
                                     kleidicv_border_type_t border_type) {
  switch (border_type) {
    case KLEIDICV_BORDER_TYPE_WRAP:
      build_horizontal_border_indices(border_indices, left, right, width,
                                      wrap_index);
      return;
    case KLEIDICV_BORDER_TYPE_REFLECT:
      build_horizontal_border_indices(border_indices, left, right, width,
                                      reflect_index);
      return;
    case KLEIDICV_BORDER_TYPE_REVERSE:
      build_horizontal_border_indices(border_indices, left, right, width,
                                      reverse_index);
      return;
    default:
      build_horizontal_border_indices(border_indices, left, right, width,
                                      replicate_index);
      return;
  }
}

inline void build_vertical_source_rows(size_t *top_rows, size_t top_padding,
                                       size_t *bottom_rows,
                                       size_t bottom_padding, size_t height,
                                       kleidicv_border_type_t border_type) {
  switch (border_type) {
    case KLEIDICV_BORDER_TYPE_WRAP:
      build_vertical_border_rows(top_rows, top_padding, bottom_rows,
                                 bottom_padding, height, wrap_index);
      return;
    case KLEIDICV_BORDER_TYPE_REFLECT:
      build_vertical_border_rows(top_rows, top_padding, bottom_rows,
                                 bottom_padding, height, reflect_index);
      return;
    case KLEIDICV_BORDER_TYPE_REVERSE:
      build_vertical_border_rows(top_rows, top_padding, bottom_rows,
                                 bottom_padding, height, reverse_index);
      return;
    default:
      build_vertical_border_rows(top_rows, top_padding, bottom_rows,
                                 bottom_padding, height, replicate_index);
      return;
  }
}

inline bool use_wrap_fast_path(size_t src_width, size_t left_padding,
                               size_t right_padding,
                               kleidicv_border_type_t border_type) {
  return border_type == KLEIDICV_BORDER_TYPE_WRAP &&
         left_padding <= src_width && right_padding <= src_width;
}

inline bool use_reflect_fast_path(size_t src_width, size_t left_padding,
                                  size_t right_padding,
                                  kleidicv_border_type_t border_type) {
  return border_type == KLEIDICV_BORDER_TYPE_REFLECT && src_width > 1 &&
         left_padding <= src_width && right_padding <= src_width;
}

inline bool use_reverse_fast_path(size_t src_width, size_t left_padding,
                                  size_t right_padding,
                                  kleidicv_border_type_t border_type) {
  return border_type == KLEIDICV_BORDER_TYPE_REVERSE && src_width > 1 &&
         left_padding < src_width && right_padding < src_width;
}

inline AddPaddingByCopyHorizontalMode resolve_horizontal_mode(
    size_t src_width, size_t left_padding, size_t right_padding,
    kleidicv_border_type_t border_type) {
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT) {
    return AddPaddingByCopyHorizontalMode::kConstant;
  }
  if (border_type == KLEIDICV_BORDER_TYPE_REPLICATE) {
    return AddPaddingByCopyHorizontalMode::kReplicate;
  }
  if (use_wrap_fast_path(src_width, left_padding, right_padding, border_type)) {
    return AddPaddingByCopyHorizontalMode::kWrapFast;
  }
  if (use_reflect_fast_path(src_width, left_padding, right_padding,
                            border_type)) {
    return AddPaddingByCopyHorizontalMode::kReflectFast;
  }
  if (use_reverse_fast_path(src_width, left_padding, right_padding,
                            border_type)) {
    return AddPaddingByCopyHorizontalMode::kReverseFast;
  }
  return AddPaddingByCopyHorizontalMode::kIndexed;
}

inline bool mode_uses_vertical_maps(AddPaddingByCopyHorizontalMode mode) {
  return mode == AddPaddingByCopyHorizontalMode::kWrapFast ||
         mode == AddPaddingByCopyHorizontalMode::kReflectFast ||
         mode == AddPaddingByCopyHorizontalMode::kReverseFast ||
         mode == AddPaddingByCopyHorizontalMode::kIndexed;
}

// -----------------------------------------------------------------------------
// Prepared-state builder
// -----------------------------------------------------------------------------
// NOLINTBEGIN(readability-function-cognitive-complexity)
AddPaddingByCopyPreparedState build_add_padding_by_copy_border_buffer(
    const void *src_void, size_t src_stride, void *dst_void, size_t dst_stride,
    size_t src_width, size_t src_height, size_t top_padding,
    size_t bottom_padding, size_t left_padding, size_t right_padding,
    size_t pixel_size, kleidicv_border_type_t border_type,
    const void *border_value) {
  const auto *src = reinterpret_cast<const uint8_t *>(src_void);
  auto *dst = reinterpret_cast<uint8_t *>(dst_void);

  if (!add_padding_by_copy_is_implemented(src_width, src_height, pixel_size,
                                          border_type)) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_NOT_IMPLEMENTED);
  }

  if (kleidicv_error_t ptr_stride_err =
          check_pointer_and_stride(src, src_stride, src_height)) {
    return add_padding_by_copy_error_state(ptr_stride_err);
  }
  if (any_null(dst)) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_NULL_POINTER);
  }
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT && any_null(border_value)) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_NULL_POINTER);
  }

  constexpr size_t kMaxImageDimension = KLEIDICV_MAX_IMAGE_PIXELS;
  if (src_width > kMaxImageDimension || src_height > kMaxImageDimension ||
      top_padding > kMaxImageDimension || bottom_padding > kMaxImageDimension ||
      left_padding > kMaxImageDimension || right_padding > kMaxImageDimension) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_RANGE);
  }

  if (kleidicv_error_t err = check_image_size_value(src_width, src_height)) {
    return add_padding_by_copy_error_state(err);
  }

  size_t horizontal_padding = 0;
  size_t vertical_padding = 0;
  if (checked_add(left_padding, right_padding, &horizontal_padding) ||
      checked_add(top_padding, bottom_padding, &vertical_padding)) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_RANGE);
  }
  if (horizontal_padding > kMaxImageDimension ||
      vertical_padding > kMaxImageDimension) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_RANGE);
  }
  if (src_width > kMaxImageDimension - horizontal_padding ||
      src_height > kMaxImageDimension - vertical_padding) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_RANGE);
  }

  size_t dst_width = 0;
  size_t dst_height = 0;
  if (checked_add(src_width, horizontal_padding, &dst_width) ||
      checked_add(src_height, vertical_padding, &dst_height)) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_RANGE);
  }
  if (kleidicv_error_t err = check_image_size_value(dst_width, dst_height)) {
    return add_padding_by_copy_error_state(err);
  }

  size_t row_size = 0;
  size_t left_size = 0;
  size_t inner_size = 0;
  size_t right_size = 0;
  size_t right_offset = 0;
  if (checked_mul(dst_width, pixel_size, &row_size) ||
      checked_mul(left_padding, pixel_size, &left_size) ||
      checked_mul(src_width, pixel_size, &inner_size) ||
      checked_mul(right_padding, pixel_size, &right_size) ||
      checked_add(left_size, inner_size, &right_offset)) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_RANGE);
  }

  const AddPaddingByCopyHorizontalMode horizontal_mode =
      resolve_horizontal_mode(src_width, left_padding, right_padding,
                              border_type);
  size_t horizontal_data_words = 0;
  if (horizontal_mode == AddPaddingByCopyHorizontalMode::kConstant) {
    horizontal_data_words = add_padding_by_copy_bytes_to_words(row_size);
  } else if (horizontal_mode == AddPaddingByCopyHorizontalMode::kIndexed) {
    horizontal_data_words = horizontal_padding;
  }
  const size_t top_map_words =
      mode_uses_vertical_maps(horizontal_mode) ? top_padding : 0;
  const size_t bottom_map_words =
      mode_uses_vertical_maps(horizontal_mode) ? bottom_padding : 0;

  size_t storage_words = 0;
  if (checked_add(horizontal_data_words, top_map_words, &storage_words) ||
      checked_add(storage_words, bottom_map_words, &storage_words)) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_RANGE);
  }

  AddPaddingByCopyPreparedState prepared_state(storage_words);
  if (storage_words > 0 && prepared_state.storage.get() == nullptr) {
    return add_padding_by_copy_error_state(KLEIDICV_ERROR_RANGE);
  }

  prepared_state.src = src;
  prepared_state.dst = dst;
  prepared_state.src_stride = src_stride;
  prepared_state.dst_stride = dst_stride;
  prepared_state.src_width = src_width;
  prepared_state.src_height = src_height;
  prepared_state.top_padding = top_padding;
  prepared_state.bottom_padding = bottom_padding;
  prepared_state.left_padding = left_padding;
  prepared_state.right_padding = right_padding;
  prepared_state.pixel_size = pixel_size;
  prepared_state.dst_width = dst_width;
  prepared_state.dst_height = dst_height;
  prepared_state.row_size = row_size;
  prepared_state.left_size = left_size;
  prepared_state.inner_size = inner_size;
  prepared_state.right_size = right_size;
  prepared_state.inner_offset = left_size;
  prepared_state.right_offset = right_offset;
  prepared_state.horizontal_data_offset_words = 0;
  prepared_state.top_map_offset_words = horizontal_data_words;
  prepared_state.bottom_map_offset_words =
      horizontal_data_words + top_map_words;
  prepared_state.horizontal_data_words = horizontal_data_words;
  prepared_state.top_map_words = top_map_words;
  prepared_state.bottom_map_words = bottom_map_words;
  prepared_state.border_type = border_type;
  prepared_state.horizontal_mode = horizontal_mode;
  prepared_state.repeat_row_fn = &repeat_pixel_across_row_dispatch;
  prepared_state.reverse_copy_fn = &copy_reversed_pixels_dispatch;
  prepared_state.indexed_copy_fn = select_indexed_copy_dispatch(
      pixel_size, src, src_stride, dst, dst_stride);

  if (horizontal_mode == AddPaddingByCopyHorizontalMode::kConstant) {
    prepared_state.repeat_row_fn(
        reinterpret_cast<const uint8_t *>(border_value), pixel_size,
        reinterpret_cast<uint8_t *>(prepared_state.storage.get()), row_size);
  } else if (horizontal_mode == AddPaddingByCopyHorizontalMode::kIndexed) {
    build_horizontal_indices(prepared_state.storage.get() +
                                 prepared_state.horizontal_data_offset_words,
                             left_padding, right_padding, src_width,
                             border_type);
  }

  if (mode_uses_vertical_maps(horizontal_mode)) {
    build_vertical_source_rows(
        prepared_state.storage.get() + prepared_state.top_map_offset_words,
        top_padding,
        prepared_state.storage.get() + prepared_state.bottom_map_offset_words,
        bottom_padding, src_height, border_type);
  }

  return prepared_state;
}
// NOLINTEND(readability-function-cognitive-complexity)

// -----------------------------------------------------------------------------
// Row helpers
// -----------------------------------------------------------------------------

inline void copy_constant_row(const AddPaddingByCopyPreparedState &state,
                              uint8_t *dst_row) {
  copy_bytes(state.constant_row(), dst_row, state.row_size);
}

inline void fill_constant_body_row(const AddPaddingByCopyPreparedState &state,
                                   const uint8_t *src_row, uint8_t *dst_row) {
  const uint8_t *constant_row = state.constant_row();
  copy_bytes(constant_row, dst_row, state.left_size);
  copy_bytes(src_row, dst_row + state.inner_offset, state.inner_size);
  copy_bytes(constant_row + state.right_offset, dst_row + state.right_offset,
             state.right_size);
}

inline void fill_replicate_row(const AddPaddingByCopyPreparedState &state,
                               const uint8_t *src_row, uint8_t *dst_row) {
  state.repeat_row_fn(src_row, state.pixel_size, dst_row, state.left_size);
  copy_bytes(src_row, dst_row + state.inner_offset, state.inner_size);
  state.repeat_row_fn(src_row + state.inner_size - state.pixel_size,
                      state.pixel_size, dst_row + state.right_offset,
                      state.right_size);
}

inline void fill_wrap_row(const AddPaddingByCopyPreparedState &state,
                          const uint8_t *src_row, uint8_t *dst_row) {
  copy_bytes(src_row + state.inner_size - state.left_size, dst_row,
             state.left_size);
  copy_bytes(src_row, dst_row + state.inner_offset, state.inner_size);
  copy_bytes(src_row, dst_row + state.right_offset, state.right_size);
}

inline void fill_reflect_row(const AddPaddingByCopyPreparedState &state,
                             const uint8_t *src_row, uint8_t *dst_row) {
  const uint8_t *left_src_last =
      src_row + (state.left_size == 0 ? 0 : state.left_size - state.pixel_size);
  const uint8_t *right_src_last =
      src_row +
      (state.right_size == 0 ? 0 : state.inner_size - state.pixel_size);
  state.reverse_copy_fn(left_src_last, state.pixel_size, dst_row,
                        state.left_size);
  copy_bytes(src_row, dst_row + state.inner_offset, state.inner_size);
  state.reverse_copy_fn(right_src_last, state.pixel_size,
                        dst_row + state.right_offset, state.right_size);
}

inline void fill_reverse_row(const AddPaddingByCopyPreparedState &state,
                             const uint8_t *src_row, uint8_t *dst_row) {
  const size_t two_pixel_size = state.pixel_size + state.pixel_size;
  const uint8_t *left_src_last =
      src_row + (state.left_size == 0 ? 0 : state.left_size);
  const uint8_t *right_src_last =
      src_row + (state.right_size == 0 ? 0 : state.inner_size - two_pixel_size);
  state.reverse_copy_fn(left_src_last, state.pixel_size, dst_row,
                        state.left_size);
  copy_bytes(src_row, dst_row + state.inner_offset, state.inner_size);
  state.reverse_copy_fn(right_src_last, state.pixel_size,
                        dst_row + state.right_offset, state.right_size);
}

inline void fill_indexed_row(const AddPaddingByCopyPreparedState &state,
                             const uint8_t *src_row, uint8_t *dst_row) {
  const size_t *border_indices = state.horizontal_indices();
  copy_bytes(src_row, dst_row + state.inner_offset, state.inner_size);
  state.indexed_copy_fn(src_row, dst_row, border_indices, state.left_padding,
                        state.pixel_size);
  state.indexed_copy_fn(src_row, dst_row + state.right_offset,
                        border_indices + state.left_padding,
                        state.right_padding, state.pixel_size);
}

// -----------------------------------------------------------------------------
// Stripe execution helpers
// -----------------------------------------------------------------------------

// Execute a stripe whose top and bottom regions use precomputed source rows.
template <typename RowFn>
inline void execute_vertical_mapped_stripe(
    const AddPaddingByCopyPreparedState &state, size_t dst_y_begin,
    size_t dst_y_end, RowFn row_fn) {
  const size_t body_begin =
      std::min(std::max(dst_y_begin, state.top_padding), dst_y_end);
  const size_t body_end = std::max(
      body_begin, std::min(dst_y_end, state.top_padding + state.src_height));
  const size_t *top_rows = state.top_source_rows();
  const size_t *bottom_rows = state.bottom_source_rows();
  const size_t bottom_base = state.top_padding + state.src_height;

  uint8_t *dst_row = state.dst + dst_y_begin * state.dst_stride;
  for (size_t dst_y = dst_y_begin; dst_y < body_begin;
       ++dst_y, dst_row += state.dst_stride) {
    row_fn(state.src + top_rows[dst_y] * state.src_stride, dst_row);
  }

  if (body_begin < body_end) {
    const uint8_t *src_row =
        state.src + (body_begin - state.top_padding) * state.src_stride;
    for (size_t dst_y = body_begin; dst_y < body_end;
         ++dst_y, dst_row += state.dst_stride, src_row += state.src_stride) {
      row_fn(src_row, dst_row);
    }
  }

  for (size_t dst_y = body_end; dst_y < dst_y_end;
       ++dst_y, dst_row += state.dst_stride) {
    row_fn(state.src + bottom_rows[dst_y - bottom_base] * state.src_stride,
           dst_row);
  }
}

inline void execute_constant_stripe(const AddPaddingByCopyPreparedState &state,
                                    size_t dst_y_begin, size_t dst_y_end) {
  const size_t body_begin =
      std::min(std::max(dst_y_begin, state.top_padding), dst_y_end);
  const size_t body_end = std::max(
      body_begin, std::min(dst_y_end, state.top_padding + state.src_height));

  uint8_t *dst_row = state.dst + dst_y_begin * state.dst_stride;
  for (size_t dst_y = dst_y_begin; dst_y < body_begin;
       ++dst_y, dst_row += state.dst_stride) {
    copy_constant_row(state, dst_row);
  }

  if (state.inner_size == 0) {
    for (size_t dst_y = body_begin; dst_y < dst_y_end;
         ++dst_y, dst_row += state.dst_stride) {
      copy_constant_row(state, dst_row);
    }
    return;
  }

  if (body_begin < body_end) {
    const uint8_t *src_row =
        state.src + (body_begin - state.top_padding) * state.src_stride;
    for (size_t dst_y = body_begin; dst_y < body_end;
         ++dst_y, dst_row += state.dst_stride, src_row += state.src_stride) {
      fill_constant_body_row(state, src_row, dst_row);
    }
  }

  for (size_t dst_y = body_end; dst_y < dst_y_end;
       ++dst_y, dst_row += state.dst_stride) {
    copy_constant_row(state, dst_row);
  }
}

inline void execute_replicate_stripe(const AddPaddingByCopyPreparedState &state,
                                     size_t dst_y_begin, size_t dst_y_end) {
  const size_t body_begin =
      std::min(std::max(dst_y_begin, state.top_padding), dst_y_end);
  const size_t body_end = std::max(
      body_begin, std::min(dst_y_end, state.top_padding + state.src_height));
  const uint8_t *top_src_row = state.src;
  const uint8_t *bottom_src_row =
      state.src + (state.src_height - 1) * state.src_stride;

  uint8_t *dst_row = state.dst + dst_y_begin * state.dst_stride;
  for (size_t dst_y = dst_y_begin; dst_y < body_begin;
       ++dst_y, dst_row += state.dst_stride) {
    fill_replicate_row(state, top_src_row, dst_row);
  }

  if (body_begin < body_end) {
    const uint8_t *src_row =
        state.src + (body_begin - state.top_padding) * state.src_stride;
    for (size_t dst_y = body_begin; dst_y < body_end;
         ++dst_y, dst_row += state.dst_stride, src_row += state.src_stride) {
      fill_replicate_row(state, src_row, dst_row);
    }
  }

  for (size_t dst_y = body_end; dst_y < dst_y_end;
       ++dst_y, dst_row += state.dst_stride) {
    fill_replicate_row(state, bottom_src_row, dst_row);
  }
}

inline void execute_wrap_fast_stripe(const AddPaddingByCopyPreparedState &state,
                                     size_t dst_y_begin, size_t dst_y_end) {
  execute_vertical_mapped_stripe(state, dst_y_begin, dst_y_end,
                                 [&](const uint8_t *src_row, uint8_t *dst_row) {
                                   fill_wrap_row(state, src_row, dst_row);
                                 });
}

inline void execute_reflect_fast_stripe(
    const AddPaddingByCopyPreparedState &state, size_t dst_y_begin,
    size_t dst_y_end) {
  execute_vertical_mapped_stripe(state, dst_y_begin, dst_y_end,
                                 [&](const uint8_t *src_row, uint8_t *dst_row) {
                                   fill_reflect_row(state, src_row, dst_row);
                                 });
}

inline void execute_reverse_fast_stripe(
    const AddPaddingByCopyPreparedState &state, size_t dst_y_begin,
    size_t dst_y_end) {
  execute_vertical_mapped_stripe(state, dst_y_begin, dst_y_end,
                                 [&](const uint8_t *src_row, uint8_t *dst_row) {
                                   fill_reverse_row(state, src_row, dst_row);
                                 });
}

inline void execute_indexed_stripe(const AddPaddingByCopyPreparedState &state,
                                   size_t dst_y_begin, size_t dst_y_end) {
  execute_vertical_mapped_stripe(state, dst_y_begin, dst_y_end,
                                 [&](const uint8_t *src_row, uint8_t *dst_row) {
                                   fill_indexed_row(state, src_row, dst_row);
                                 });
}

// -----------------------------------------------------------------------------
// Public entry point
// -----------------------------------------------------------------------------

kleidicv_error_t add_padding_by_copy_stripe(
    const AddPaddingByCopyPreparedState *prepared_state, size_t dst_y_begin,
    size_t dst_y_end) {
  if (prepared_state == nullptr) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }
  if (prepared_state->error != KLEIDICV_OK) {
    return prepared_state->error;
  }
  if (dst_y_begin > dst_y_end || dst_y_end > prepared_state->dst_height) {
    return KLEIDICV_ERROR_RANGE;
  }
  if (dst_y_begin == dst_y_end) {
    return KLEIDICV_OK;
  }

  switch (prepared_state->horizontal_mode) {
    case AddPaddingByCopyHorizontalMode::kConstant:
      execute_constant_stripe(*prepared_state, dst_y_begin, dst_y_end);
      break;
    case AddPaddingByCopyHorizontalMode::kReplicate:
      execute_replicate_stripe(*prepared_state, dst_y_begin, dst_y_end);
      break;
    case AddPaddingByCopyHorizontalMode::kWrapFast:
      execute_wrap_fast_stripe(*prepared_state, dst_y_begin, dst_y_end);
      break;
    case AddPaddingByCopyHorizontalMode::kReflectFast:
      execute_reflect_fast_stripe(*prepared_state, dst_y_begin, dst_y_end);
      break;
    case AddPaddingByCopyHorizontalMode::kReverseFast:
      execute_reverse_fast_stripe(*prepared_state, dst_y_begin, dst_y_end);
      break;
    case AddPaddingByCopyHorizontalMode::kIndexed:
      execute_indexed_stripe(*prepared_state, dst_y_begin, dst_y_end);
      break;
  }

  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
