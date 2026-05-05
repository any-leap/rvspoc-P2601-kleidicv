// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared row-walker + strip-miner for RVV elementwise binary ops at LMUL=1.
//
// The per-op .cpp passes a generic-lambda `Op(va, vb, vl) -> vd` whose
// argument and return types are the LMUL=1 vector type for `T`. Loads, stores
// and vsetvl are owned here; the lambda may use any combination of widening,
// masking, narrowing internally as long as input/output element widths match.
//
// LMUL is fixed at m1 to mirror the hand-written code we are replacing. A
// future Phase 4 throughput pass can introduce LMUL=2/4 variants by adding
// traits specialisations and a second strip-miner.

#ifndef KLEIDICV_RISCV_ELEMENTWISE_RVV_H
#define KLEIDICV_RISCV_ELEMENTWISE_RVV_H

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::rvv {

template <typename T>
struct rvv_traits;

// VEC_PFX is the C++ vector-type spelling (e.g. "uint" in vuint8m1_t).
// INTR_PFX is the intrinsic-name spelling (e.g. "u" in __riscv_vle8_v_u8m1).
#define KLEIDICV_RVV_TRAITS(T, VEC_PFX, INTR_PFX, SEW)                   \
  template <>                                                            \
  struct rvv_traits<T> {                                                 \
    using vec_t = v##VEC_PFX##SEW##m1_t;                                 \
    static inline size_t vsetvl(size_t n) {                              \
      return __riscv_vsetvl_e##SEW##m1(n);                               \
    }                                                                    \
    static inline vec_t load(const T *p, size_t vl) {                    \
      return __riscv_vle##SEW##_v_##INTR_PFX##SEW##m1(p, vl);            \
    }                                                                    \
    static inline void store(T *p, vec_t v, size_t vl) {                 \
      __riscv_vse##SEW##_v_##INTR_PFX##SEW##m1(p, v, vl);                \
    }                                                                    \
  }

KLEIDICV_RVV_TRAITS(uint8_t, uint, u, 8);
KLEIDICV_RVV_TRAITS(int8_t, int, i, 8);
KLEIDICV_RVV_TRAITS(uint16_t, uint, u, 16);
KLEIDICV_RVV_TRAITS(int16_t, int, i, 16);
KLEIDICV_RVV_TRAITS(uint32_t, uint, u, 32);
KLEIDICV_RVV_TRAITS(int32_t, int, i, 32);
KLEIDICV_RVV_TRAITS(uint64_t, uint, u, 64);
KLEIDICV_RVV_TRAITS(int64_t, int, i, 64);

// f32: load/store map onto vfloat32m1_t / vle32_v_f32m1 / vse32_v_f32m1.
template <>
struct rvv_traits<float> {
  using vec_t = vfloat32m1_t;
  static inline size_t vsetvl(size_t n) { return __riscv_vsetvl_e32m1(n); }
  static inline vec_t load(const float *p, size_t vl) {
    return __riscv_vle32_v_f32m1(p, vl);
  }
  static inline void store(float *p, vec_t v, size_t vl) {
    __riscv_vse32_v_f32m1(p, v, vl);
  }
};

#undef KLEIDICV_RVV_TRAITS

// `Op(vec_t, vec_t, size_t vl) -> vec_t` is invoked per strip.
template <typename T, typename Op>
inline kleidicv_error_t binary_elementwise(const T *src_a, size_t src_a_stride,
                                           const T *src_b, size_t src_b_stride,
                                           T *dst, size_t dst_stride,
                                           size_t width, size_t height,
                                           Op op) {
  if (!src_a || !src_b || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  using R = rvv_traits<T>;
  for (size_t y = 0; y < height; ++y) {
    const T *row_a = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src_a) + y * src_a_stride);
    const T *row_b = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src_b) + y * src_b_stride);
    T *row_dst = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(dst) +
                                       y * dst_stride);
    size_t vl;
    for (size_t x = 0; x < width; x += vl) {
      vl = R::vsetvl(width - x);
      auto va = R::load(row_a + x, vl);
      auto vb = R::load(row_b + x, vl);
      auto vd = op(va, vb, vl);
      R::store(row_dst + x, vd, vl);
    }
  }
  return KLEIDICV_OK;
}

// `Op(vec_t, size_t vl) -> vec_t` is invoked per strip. Single source.
template <typename T, typename Op>
inline kleidicv_error_t unary_elementwise(const T *src, size_t src_stride,
                                          T *dst, size_t dst_stride,
                                          size_t width, size_t height, Op op) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  using R = rvv_traits<T>;
  for (size_t y = 0; y < height; ++y) {
    const T *row_src = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    T *row_dst = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(dst) +
                                       y * dst_stride);
    size_t vl;
    for (size_t x = 0; x < width; x += vl) {
      vl = R::vsetvl(width - x);
      auto v = R::load(row_src + x, vl);
      auto vd = op(v, vl);
      R::store(row_dst + x, vd, vl);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv

#endif
