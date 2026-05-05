// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_MIN_MAX_DECLS_H
#define KLEIDICV_RISCV_MIN_MAX_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
template <typename T>
kleidicv_error_t min_max(const T *src, size_t src_stride, size_t width,
                         size_t height, T *min_out, T *max_out);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
#define DECL(suffix, T)                                                     \
  kleidicv_error_t min_max_##suffix(const T *, size_t, size_t, size_t, T *, \
                                    T *)
DECL(u8, uint8_t);
DECL(s8, int8_t);
DECL(u16, uint16_t);
DECL(s16, int16_t);
DECL(s32, int32_t);
#undef DECL
}  // namespace kleidicv::rvv

#endif
