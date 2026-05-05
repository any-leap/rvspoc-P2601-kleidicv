// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_SUB_DECLS_H
#define KLEIDICV_RISCV_SUB_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
template <typename T>
kleidicv_error_t saturating_sub(const T *src_a, size_t src_a_stride,
                                const T *src_b, size_t src_b_stride, T *dst,
                                size_t dst_stride, size_t width, size_t height);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
#define DECL(suffix, T)                                                       \
  kleidicv_error_t saturating_sub_##suffix(const T *, size_t, const T *,      \
                                           size_t, T *, size_t, size_t, size_t)
DECL(u8, uint8_t);
DECL(s8, int8_t);
DECL(u16, uint16_t);
DECL(s16, int16_t);
DECL(u32, uint32_t);
DECL(s32, int32_t);
DECL(u64, uint64_t);
DECL(s64, int64_t);
#undef DECL
}  // namespace kleidicv::rvv

#endif
