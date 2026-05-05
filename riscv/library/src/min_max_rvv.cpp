// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV min_max via vredmin/vredmax (vredminu/vredmaxu for unsigned). Each row
// folds into a running scalar pair held in a 1-lane LMUL=1 reduction
// destination; tail and per-strip semantics are handled by passing the
// running scalar back in as the seed for the next reduction.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "kleidicv/kleidicv.h"
#include "min_max_decls.h"

namespace kleidicv::rvv {

#define MIN_MAX_UNSIGNED(suffix, T, SEW)                                       \
  kleidicv_error_t min_max_##suffix(const T *src, size_t src_stride,           \
                                    size_t width, size_t height, T *min_out,  \
                                    T *max_out) {                              \
    if (!src) return KLEIDICV_ERROR_NULL_POINTER;                              \
    if (!min_out && !max_out) return KLEIDICV_OK;                              \
    if (width == 0 || height == 0) return KLEIDICV_OK;                         \
    T running_min = std::numeric_limits<T>::max();                             \
    T running_max = std::numeric_limits<T>::min();                             \
    for (size_t y = 0; y < height; ++y) {                                      \
      const T *rs = reinterpret_cast<const T *>(                               \
          reinterpret_cast<const uint8_t *>(src) + y * src_stride);            \
      size_t vl;                                                               \
      for (size_t x = 0; x < width; x += vl) {                                 \
        vl = __riscv_vsetvl_e##SEW##m1(width - x);                             \
        auto v = __riscv_vle##SEW##_v_u##SEW##m1(rs + x, vl);                  \
        auto seed_mn = __riscv_vmv_s_x_u##SEW##m1(running_min, vl);            \
        auto seed_mx = __riscv_vmv_s_x_u##SEW##m1(running_max, vl);            \
        auto rmn = __riscv_vredminu_vs_u##SEW##m1_u##SEW##m1(v, seed_mn, vl); \
        auto rmx = __riscv_vredmaxu_vs_u##SEW##m1_u##SEW##m1(v, seed_mx, vl); \
        running_min = __riscv_vmv_x_s_u##SEW##m1_u##SEW(rmn);                  \
        running_max = __riscv_vmv_x_s_u##SEW##m1_u##SEW(rmx);                  \
      }                                                                        \
    }                                                                          \
    if (min_out) *min_out = running_min;                                       \
    if (max_out) *max_out = running_max;                                       \
    return KLEIDICV_OK;                                                        \
  }

#define MIN_MAX_SIGNED(suffix, T, SEW)                                         \
  kleidicv_error_t min_max_##suffix(const T *src, size_t src_stride,           \
                                    size_t width, size_t height, T *min_out,  \
                                    T *max_out) {                              \
    if (!src) return KLEIDICV_ERROR_NULL_POINTER;                              \
    if (!min_out && !max_out) return KLEIDICV_OK;                              \
    if (width == 0 || height == 0) return KLEIDICV_OK;                         \
    T running_min = std::numeric_limits<T>::max();                             \
    T running_max = std::numeric_limits<T>::min();                             \
    for (size_t y = 0; y < height; ++y) {                                      \
      const T *rs = reinterpret_cast<const T *>(                               \
          reinterpret_cast<const uint8_t *>(src) + y * src_stride);            \
      size_t vl;                                                               \
      for (size_t x = 0; x < width; x += vl) {                                 \
        vl = __riscv_vsetvl_e##SEW##m1(width - x);                             \
        auto v = __riscv_vle##SEW##_v_i##SEW##m1(rs + x, vl);                  \
        auto seed_mn = __riscv_vmv_s_x_i##SEW##m1(running_min, vl);            \
        auto seed_mx = __riscv_vmv_s_x_i##SEW##m1(running_max, vl);            \
        auto rmn = __riscv_vredmin_vs_i##SEW##m1_i##SEW##m1(v, seed_mn, vl);  \
        auto rmx = __riscv_vredmax_vs_i##SEW##m1_i##SEW##m1(v, seed_mx, vl);  \
        running_min = __riscv_vmv_x_s_i##SEW##m1_i##SEW(rmn);                  \
        running_max = __riscv_vmv_x_s_i##SEW##m1_i##SEW(rmx);                  \
      }                                                                        \
    }                                                                          \
    if (min_out) *min_out = running_min;                                       \
    if (max_out) *max_out = running_max;                                       \
    return KLEIDICV_OK;                                                        \
  }

MIN_MAX_UNSIGNED(u8, uint8_t, 8)
MIN_MAX_SIGNED(s8, int8_t, 8)
MIN_MAX_UNSIGNED(u16, uint16_t, 16)
MIN_MAX_SIGNED(s16, int16_t, 16)
MIN_MAX_SIGNED(s32, int32_t, 32)

#undef MIN_MAX_UNSIGNED
#undef MIN_MAX_SIGNED

}  // namespace kleidicv::rvv
