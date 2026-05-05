// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV saturating_multiply: widening-multiply (vwmul/vwmulu) to 2*SEW, then
// narrow-with-saturation (vnclip/vnclipu). The `scale` parameter is ignored
// to match upstream's current behaviour.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "elementwise_rvv.h"
#include "kleidicv/kleidicv.h"
#include "multiply_decls.h"

namespace kleidicv::rvv {

#define DEFINE_UNSIGNED(suffix, T, SEW, SEW2)                                  \
  kleidicv_error_t saturating_multiply_##suffix(                               \
      const T *src_a, size_t sa, const T *src_b, size_t sb, T *dst, size_t sd, \
      size_t w, size_t h, double scale) {                                      \
    (void)scale;                                                               \
    return binary_elementwise<T>(                                              \
        src_a, sa, src_b, sb, dst, sd, w, h,                                   \
        [](auto a, auto b, size_t vl) {                                        \
          auto wide = __riscv_vwmulu_vv_u##SEW2##m2(a, b, vl);                 \
          return __riscv_vnclipu_wx_u##SEW##m1(wide, 0, __RISCV_VXRM_RNU, vl); \
        });                                                                    \
  }

#define DEFINE_SIGNED(suffix, T, SEW, SEW2)                                    \
  kleidicv_error_t saturating_multiply_##suffix(                               \
      const T *src_a, size_t sa, const T *src_b, size_t sb, T *dst, size_t sd, \
      size_t w, size_t h, double scale) {                                      \
    (void)scale;                                                               \
    return binary_elementwise<T>(                                              \
        src_a, sa, src_b, sb, dst, sd, w, h,                                   \
        [](auto a, auto b, size_t vl) {                                        \
          auto wide = __riscv_vwmul_vv_i##SEW2##m2(a, b, vl);                  \
          return __riscv_vnclip_wx_i##SEW##m1(wide, 0, __RISCV_VXRM_RNU, vl);  \
        });                                                                    \
  }

DEFINE_UNSIGNED(u8, uint8_t, 8, 16)
DEFINE_SIGNED(s8, int8_t, 8, 16)
DEFINE_UNSIGNED(u16, uint16_t, 16, 32)
DEFINE_SIGNED(s16, int16_t, 16, 32)
DEFINE_SIGNED(s32, int32_t, 32, 64)

#undef DEFINE_UNSIGNED
#undef DEFINE_SIGNED

}  // namespace kleidicv::rvv
