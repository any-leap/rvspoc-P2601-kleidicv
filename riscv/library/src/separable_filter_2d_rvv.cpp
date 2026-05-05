// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV 5x5 separable filter (channels=1, replicate border). Saturating contract
// matches upstream:
//   - vertical:   u16/u32 saturating-add of widening multiplies
//   - horizontal: u32/u64 plain add, then saturating narrow

#include <riscv_vector.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kleidicv/kleidicv.h"
#include "separable_filter_2d_decls.h"

namespace kleidicv::scalar {
kleidicv_error_t separable_filter_2d_5x5_u8(const uint8_t *, size_t, uint8_t *,
                                            size_t, size_t, size_t,
                                            const uint8_t *, const uint8_t *);
kleidicv_error_t separable_filter_2d_5x5_u16(const uint16_t *, size_t,
                                             uint16_t *, size_t, size_t,
                                             size_t, const uint16_t *,
                                             const uint16_t *);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {

namespace {

inline size_t clip(ptrdiff_t v, size_t n) {
  if (v < 0) return 0;
  if (static_cast<size_t>(v) >= n) return n - 1;
  return static_cast<size_t>(v);
}

}  // namespace

kleidicv_error_t separable_filter_2d_5x5_u8(const uint8_t *src, size_t src_stride,
                                            uint8_t *dst, size_t dst_stride,
                                            size_t width, size_t height,
                                            const uint8_t *kernel_x,
                                            const uint8_t *kernel_y) {
  if (!src || !dst || !kernel_x || !kernel_y)
    return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;
  if (width < 5) {
    // Tiny rows: fall back to scalar (replicate borders dominate).
    return kleidicv::scalar::separable_filter_2d_5x5_u8(
        src, src_stride, dst, dst_stride, width, height, kernel_x, kernel_y);
  }

  std::vector<uint16_t> intermediate(width * height);

  // Vertical pass: for each output row y, accumulate into intermediate[y].
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *r[5];
    for (int i = 0; i < 5; ++i)
      r[i] = src + clip(static_cast<ptrdiff_t>(y) + i - 2, height) * src_stride;
    uint16_t *out = intermediate.data() + y * width;
    size_t vl;
    for (size_t x = 0; x < width; x += vl) {
      vl = __riscv_vsetvl_e8m1(width - x);
      vuint16m2_t acc = __riscv_vwmulu_vx_u16m2(
          __riscv_vle8_v_u8m1(r[0] + x, vl), kernel_y[0], vl);
      for (int i = 1; i < 5; ++i) {
        vuint16m2_t prod = __riscv_vwmulu_vx_u16m2(
            __riscv_vle8_v_u8m1(r[i] + x, vl), kernel_y[i], vl);
        acc = __riscv_vsaddu_vv_u16m2(acc, prod, vl);
      }
      __riscv_vse16_v_u16m2(out + x, acc, vl);
    }
  }

  // Horizontal pass: bulk for x in [2, width-3]; scalar at the edges.
  auto scalar_horiz = [&](size_t y, size_t x) {
    const uint16_t *row = intermediate.data() + y * width;
    uint32_t acc = 0;
    for (int i = 0; i < 5; ++i) {
      size_t sx =
          clip(static_cast<ptrdiff_t>(x) + i - 2, width);
      acc += static_cast<uint32_t>(kernel_x[i]) *
             static_cast<uint32_t>(row[sx]);
    }
    dst[y * dst_stride + x] =
        static_cast<uint8_t>(acc > 255U ? 255U : acc);
  };

  for (size_t y = 0; y < height; ++y) {
    const uint16_t *row = intermediate.data() + y * width;
    uint8_t *out = dst + y * dst_stride;
    scalar_horiz(y, 0);
    scalar_horiz(y, 1);
    if (width > 2) scalar_horiz(y, width - 1);
    if (width > 3) scalar_horiz(y, width - 2);
    size_t vl;
    for (size_t x = 2; x + 2 < width; x += vl) {
      vl = __riscv_vsetvl_e8m1(width - 2 - x);
      vuint32m4_t acc = __riscv_vwmulu_vx_u32m4(
          __riscv_vle16_v_u16m2(row + x - 2, vl), kernel_x[0], vl);
      for (int i = 1; i < 5; ++i) {
        // Plain accumulate u32 (no saturation in horizontal acc).
        vuint32m4_t prod = __riscv_vwmulu_vx_u32m4(
            __riscv_vle16_v_u16m2(row + x - 2 + i, vl), kernel_x[i], vl);
        acc = __riscv_vadd_vv_u32m4(acc, prod, vl);
      }
      // Saturating narrow u32 -> u16, then u16 -> u8.
      vuint16m2_t u16 = __riscv_vnclipu_wx_u16m2(acc, 0, __RISCV_VXRM_RNU, vl);
      vuint8m1_t u8 = __riscv_vnclipu_wx_u8m1(u16, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_u8m1(out + x, u8, vl);
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t separable_filter_2d_5x5_u16(const uint16_t *src,
                                             size_t src_stride, uint16_t *dst,
                                             size_t dst_stride, size_t width,
                                             size_t height,
                                             const uint16_t *kernel_x,
                                             const uint16_t *kernel_y) {
  if (!src || !dst || !kernel_x || !kernel_y)
    return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;
  if (width < 5) {
    return kleidicv::scalar::separable_filter_2d_5x5_u16(
        src, src_stride, dst, dst_stride, width, height, kernel_x, kernel_y);
  }

  std::vector<uint32_t> intermediate(width * height);
  size_t src_stride_elems = src_stride / sizeof(uint16_t);
  size_t dst_stride_elems = dst_stride / sizeof(uint16_t);

  // Vertical pass
  for (size_t y = 0; y < height; ++y) {
    const uint16_t *r[5];
    for (int i = 0; i < 5; ++i)
      r[i] = src + clip(static_cast<ptrdiff_t>(y) + i - 2, height) *
                       src_stride_elems;
    uint32_t *out = intermediate.data() + y * width;
    size_t vl;
    for (size_t x = 0; x < width; x += vl) {
      vl = __riscv_vsetvl_e16m2(width - x);
      vuint32m4_t acc = __riscv_vwmulu_vx_u32m4(
          __riscv_vle16_v_u16m2(r[0] + x, vl), kernel_y[0], vl);
      for (int i = 1; i < 5; ++i) {
        vuint32m4_t prod = __riscv_vwmulu_vx_u32m4(
            __riscv_vle16_v_u16m2(r[i] + x, vl), kernel_y[i], vl);
        acc = __riscv_vsaddu_vv_u32m4(acc, prod, vl);
      }
      __riscv_vse32_v_u32m4(out + x, acc, vl);
    }
  }

  // Horizontal pass: scalar for x in {0,1,w-2,w-1}, RVV bulk in between.
  auto scalar_horiz = [&](size_t y, size_t x) {
    const uint32_t *row = intermediate.data() + y * width;
    uint64_t acc = 0;
    for (int i = 0; i < 5; ++i) {
      size_t sx =
          clip(static_cast<ptrdiff_t>(x) + i - 2, width);
      acc += static_cast<uint64_t>(kernel_x[i]) *
             static_cast<uint64_t>(row[sx]);
    }
    dst[y * dst_stride_elems + x] =
        static_cast<uint16_t>(acc > 65535U ? 65535U : acc);
  };

  for (size_t y = 0; y < height; ++y) {
    const uint32_t *row = intermediate.data() + y * width;
    uint16_t *out = dst + y * dst_stride_elems;
    scalar_horiz(y, 0);
    scalar_horiz(y, 1);
    if (width > 2) scalar_horiz(y, width - 1);
    if (width > 3) scalar_horiz(y, width - 2);
    size_t vl;
    for (size_t x = 2; x + 2 < width; x += vl) {
      vl = __riscv_vsetvl_e16m2(width - 2 - x);
      vuint64m8_t acc = __riscv_vwmulu_vx_u64m8(
          __riscv_vle32_v_u32m4(row + x - 2, vl), kernel_x[0], vl);
      for (int i = 1; i < 5; ++i) {
        vuint64m8_t prod = __riscv_vwmulu_vx_u64m8(
            __riscv_vle32_v_u32m4(row + x - 2 + i, vl), kernel_x[i], vl);
        acc = __riscv_vadd_vv_u64m8(acc, prod, vl);
      }
      vuint32m4_t u32 = __riscv_vnclipu_wx_u32m4(acc, 0, __RISCV_VXRM_RNU, vl);
      vuint16m2_t u16 = __riscv_vnclipu_wx_u16m2(u32, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse16_v_u16m2(out + x, u16, vl);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
