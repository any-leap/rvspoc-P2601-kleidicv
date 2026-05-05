// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "split_merge_decls.h"
#include "validation_helper.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;
namespace V = kleidicv::riscv_validation;

kleidicv_error_t split_dispatch(const void *src, size_t src_stride,
                                  void **dst_planes, const size_t *dst_strides,
                                  size_t width, size_t height, size_t channels,
                                  size_t element_size) {
  if (!src || !dst_planes || !dst_strides)
    return KLEIDICV_ERROR_NULL_POINTER;
  // channels < 2 is an invalid argument (a 0/1-channel split is meaningless);
  // upstream returns KLEIDICV_ERROR_RANGE for that case. channels > 4 is
  // unsupported scope and returns NOT_IMPLEMENTED. element_size must be in
  // {1, 2, 4, 8} per the public API; anything else is RANGE.
  if (channels < 2) return KLEIDICV_ERROR_RANGE;
  if (channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (element_size != 1 && element_size != 2 && element_size != 4 &&
      element_size != 8)
    return KLEIDICV_ERROR_RANGE;
  if (kleidicv_error_t e = V::check_image_size(width, height)) return e;
  // Element-size-aligned access matters when the kernel uses vlsegN/vssegN at
  // SEW > 8 — stride must be a multiple of element_size and the buffer
  // pointers must be element_size-aligned.
  if (element_size > 1) {
    if ((src_stride % element_size) != 0) return KLEIDICV_ERROR_ALIGNMENT;
    if ((reinterpret_cast<uintptr_t>(src) & (element_size - 1)) != 0)
      return KLEIDICV_ERROR_ALIGNMENT;
    for (size_t c = 0; c < channels; ++c) {
      if (!dst_planes[c]) return KLEIDICV_ERROR_NULL_POINTER;
      if ((dst_strides[c] % element_size) != 0)
        return KLEIDICV_ERROR_ALIGNMENT;
      if ((reinterpret_cast<uintptr_t>(dst_planes[c]) & (element_size - 1)) !=
          0)
        return KLEIDICV_ERROR_ALIGNMENT;
    }
  } else {
    for (size_t c = 0; c < channels; ++c)
      if (!dst_planes[c]) return KLEIDICV_ERROR_NULL_POINTER;
  }
  return active_backend() == Backend::Rvv
             ? kleidicv::rvv::split(src, src_stride, dst_planes, dst_strides,
                                      width, height, channels, element_size)
             : kleidicv::scalar::split(src, src_stride, dst_planes,
                                         dst_strides, width, height, channels,
                                         element_size);
}

kleidicv_error_t merge_dispatch(const void **src_planes,
                                  const size_t *src_strides, void *dst,
                                  size_t dst_stride, size_t width,
                                  size_t height, size_t channels,
                                  size_t element_size) {
  if (!src_planes || !src_strides || !dst)
    return KLEIDICV_ERROR_NULL_POINTER;
  // Reject unsupported channels / element_size BEFORE iterating src_planes,
  // Same semantics as split_dispatch: invalid channels is RANGE,
  // out-of-scope channels is NOT_IMPLEMENTED.
  if (channels < 2) return KLEIDICV_ERROR_RANGE;
  if (channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (element_size != 1 && element_size != 2 && element_size != 4 &&
      element_size != 8)
    return KLEIDICV_ERROR_RANGE;
  if (kleidicv_error_t e = V::check_image_size(width, height)) return e;
  if (element_size > 1) {
    if ((dst_stride % element_size) != 0) return KLEIDICV_ERROR_ALIGNMENT;
    if ((reinterpret_cast<uintptr_t>(dst) & (element_size - 1)) != 0)
      return KLEIDICV_ERROR_ALIGNMENT;
    for (size_t c = 0; c < channels; ++c) {
      if (!src_planes[c]) return KLEIDICV_ERROR_NULL_POINTER;
      if ((src_strides[c] % element_size) != 0)
        return KLEIDICV_ERROR_ALIGNMENT;
      if ((reinterpret_cast<uintptr_t>(src_planes[c]) & (element_size - 1)) !=
          0)
        return KLEIDICV_ERROR_ALIGNMENT;
    }
  } else {
    for (size_t c = 0; c < channels; ++c)
      if (!src_planes[c]) return KLEIDICV_ERROR_NULL_POINTER;
  }
  return active_backend() == Backend::Rvv
             ? kleidicv::rvv::merge(src_planes, src_strides, dst, dst_stride,
                                      width, height, channels, element_size)
             : kleidicv::scalar::merge(src_planes, src_strides, dst,
                                         dst_stride, width, height, channels,
                                         element_size);
}
}  // namespace

extern "C" {
kleidicv_error_t (*kleidicv_split)(const void *, size_t, void **,
                                   const size_t *, size_t, size_t, size_t,
                                   size_t) = split_dispatch;
kleidicv_error_t (*kleidicv_merge)(const void **, const size_t *, void *,
                                   size_t, size_t, size_t, size_t,
                                   size_t) = merge_dispatch;
}
