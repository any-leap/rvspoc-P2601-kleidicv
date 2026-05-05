// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV 1.0 saturating_add via vsadd / vsaddu — single instruction, LMUL=1.
// Row-walk and strip-mine live in elementwise_rvv.h.

#include <cstddef>
#include <cstdint>

#include "add_decls.h"
#include "elementwise_rvv.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

#define DEFINE_UNSIGNED(suffix, T, SEW)                                        \
  kleidicv_error_t saturating_add_##suffix(                                    \
      const T *src_a, size_t sa, const T *src_b, size_t sb, T *dst, size_t sd, \
      size_t w, size_t h) {                                                    \
    return binary_elementwise<T>(                                              \
        src_a, sa, src_b, sb, dst, sd, w, h,                                   \
        [](auto a, auto b, size_t vl) {        \
          return __riscv_vsaddu_vv_u##SEW##m1(a, b, vl);                       \
        });                                                                    \
  }

#define DEFINE_SIGNED(suffix, T, SEW)                                          \
  kleidicv_error_t saturating_add_##suffix(                                    \
      const T *src_a, size_t sa, const T *src_b, size_t sb, T *dst, size_t sd, \
      size_t w, size_t h) {                                                    \
    return binary_elementwise<T>(                                              \
        src_a, sa, src_b, sb, dst, sd, w, h,                                   \
        [](auto a, auto b, size_t vl) {        \
          return __riscv_vsadd_vv_i##SEW##m1(a, b, vl);                        \
        });                                                                    \
  }

DEFINE_UNSIGNED(u8, uint8_t, 8)
DEFINE_SIGNED(s8, int8_t, 8)
DEFINE_UNSIGNED(u16, uint16_t, 16)
DEFINE_SIGNED(s16, int16_t, 16)
DEFINE_UNSIGNED(u32, uint32_t, 32)
DEFINE_SIGNED(s32, int32_t, 32)
DEFINE_UNSIGNED(u64, uint64_t, 64)
DEFINE_SIGNED(s64, int64_t, 64)

#undef DEFINE_UNSIGNED
#undef DEFINE_SIGNED

}  // namespace kleidicv::rvv
