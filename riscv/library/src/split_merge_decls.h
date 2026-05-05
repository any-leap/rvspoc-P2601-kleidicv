// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_SPLIT_MERGE_DECLS_H
#define KLEIDICV_RISCV_SPLIT_MERGE_DECLS_H

#include <cstddef>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t split(const void *src, size_t src_stride, void **dsts,
                       const size_t *dst_strides, size_t width, size_t height,
                       size_t channels, size_t element_size);
kleidicv_error_t merge(const void **srcs, const size_t *src_strides, void *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t element_size);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t split(const void *src, size_t src_stride, void **dsts,
                       const size_t *dst_strides, size_t width, size_t height,
                       size_t channels, size_t element_size);
kleidicv_error_t merge(const void **srcs, const size_t *src_strides, void *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t element_size);
}  // namespace kleidicv::rvv

#endif
