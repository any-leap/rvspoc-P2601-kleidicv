// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// kleidicv_transpose / kleidicv_rotate are declared as function pointers in
// the public header. We bind them at static-init to a dispatch shim that
// picks an RVV strided-store specialisation for SEW-natural pixel sizes
// (1/2/4/8 bytes) and falls back to the scalar memcpy loop for pixel_size
// ∈ {3, 6} or other unusual sizes.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "dispatch.h"
#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;

// ---- scalar reference ----

kleidicv_error_t transpose_scalar(const uint8_t *src, size_t src_stride,
                                   uint8_t *dst, size_t dst_stride,
                                   size_t src_width, size_t src_height,
                                   size_t pixel_size) {
  for (size_t y = 0; y < src_height; ++y) {
    for (size_t x = 0; x < src_width; ++x) {
      std::memcpy(dst + x * dst_stride + y * pixel_size,
                  src + y * src_stride + x * pixel_size, pixel_size);
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t rotate_scalar(const uint8_t *src, size_t src_stride,
                                size_t width, size_t height, uint8_t *dst,
                                size_t dst_stride, int angle,
                                size_t pixel_size) {
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      size_t dx, dy;
      switch (angle) {
        case 90:
          dx = height - 1 - y;
          dy = x;
          break;
        case 180:
          dx = width - 1 - x;
          dy = height - 1 - y;
          break;
        default:  // 270
          dx = y;
          dy = width - 1 - x;
          break;
      }
      std::memcpy(dst + dy * dst_stride + dx * pixel_size,
                  src + y * src_stride + x * pixel_size, pixel_size);
    }
  }
  return KLEIDICV_OK;
}

// ---- RVV specialisations for pixel_size ∈ {1, 2, 4, 8} ----
//
// Each row of source is loaded unit-stride, then strided-stored into the
// destination so the RISC-V vector unit handles the layout swap.
// Macros keep the four SEW variants honest without runtime indirection.

#define MAKE_TRANSPOSE_RVV(SEW, MLM, SUFFIX, ELEM_T)                        \
  void transpose_rvv_##SEW(const uint8_t *src, size_t src_stride,           \
                            uint8_t *dst, size_t dst_stride,                \
                            size_t src_width, size_t src_height) {          \
    constexpr size_t P = (SEW) / 8;                                         \
    for (size_t y = 0; y < src_height; ++y) {                               \
      const ELEM_T *srow =                                                  \
          reinterpret_cast<const ELEM_T *>(src + y * src_stride);           \
      uint8_t *dcol = dst + y * P;                                          \
      size_t x = 0;                                                         \
      while (x < src_width) {                                               \
        size_t vl = __riscv_vsetvl_e##SEW##MLM(src_width - x);              \
        vuint##SEW##MLM##_t v =                                             \
            __riscv_vle##SEW##_v_u##SEW##MLM(srow + x, vl);                 \
        __riscv_vsse##SEW##_v_u##SEW##MLM(                                  \
            reinterpret_cast<ELEM_T *>(dcol + x * dst_stride),              \
            static_cast<ptrdiff_t>(dst_stride), v, vl);                     \
        x += vl;                                                            \
      }                                                                     \
    }                                                                       \
    (void)SUFFIX;                                                           \
  }

MAKE_TRANSPOSE_RVV(8, m1, "u8", uint8_t)
MAKE_TRANSPOSE_RVV(16, m1, "u16", uint16_t)
MAKE_TRANSPOSE_RVV(32, m1, "u32", uint32_t)
MAKE_TRANSPOSE_RVV(64, m1, "u64", uint64_t)
#undef MAKE_TRANSPOSE_RVV

// rotate 90: dst[dy=x][dx=height-1-y] = src[y][x].
// Source row y maps to destination column (height-1-y). Strided store with
// stride dst_stride places lane i at dst[(x+i)*dst_stride + (height-1-y)*P].
#define MAKE_ROTATE90_RVV(SEW, MLM, ELEM_T)                                 \
  void rotate90_rvv_##SEW(const uint8_t *src, size_t src_stride,            \
                           size_t width, size_t height, uint8_t *dst,       \
                           size_t dst_stride) {                             \
    constexpr size_t P = (SEW) / 8;                                         \
    for (size_t y = 0; y < height; ++y) {                                   \
      const ELEM_T *srow =                                                  \
          reinterpret_cast<const ELEM_T *>(src + y * src_stride);           \
      uint8_t *dcol = dst + (height - 1 - y) * P;                           \
      size_t x = 0;                                                         \
      while (x < width) {                                                   \
        size_t vl = __riscv_vsetvl_e##SEW##MLM(width - x);                  \
        vuint##SEW##MLM##_t v =                                             \
            __riscv_vle##SEW##_v_u##SEW##MLM(srow + x, vl);                 \
        __riscv_vsse##SEW##_v_u##SEW##MLM(                                  \
            reinterpret_cast<ELEM_T *>(dcol + x * dst_stride),              \
            static_cast<ptrdiff_t>(dst_stride), v, vl);                     \
        x += vl;                                                            \
      }                                                                     \
    }                                                                       \
  }

MAKE_ROTATE90_RVV(8, m1, uint8_t)
MAKE_ROTATE90_RVV(16, m1, uint16_t)
MAKE_ROTATE90_RVV(32, m1, uint32_t)
MAKE_ROTATE90_RVV(64, m1, uint64_t)
#undef MAKE_ROTATE90_RVV

// rotate 270: dst[dy=width-1-x][dx=y] = src[y][x].
// Vectorise over the source column index: source column x maps to
// destination row (width-1-x). For lane i, x' = x+i, so dst row index is
// (width-1-x-i). A negative-stride store starting at (width-1-x) walks
// rows backward — exactly what we want.
#define MAKE_ROTATE270_RVV(SEW, MLM, ELEM_T)                                \
  void rotate270_rvv_##SEW(const uint8_t *src, size_t src_stride,           \
                            size_t width, size_t height, uint8_t *dst,      \
                            size_t dst_stride) {                            \
    constexpr size_t P = (SEW) / 8;                                         \
    for (size_t y = 0; y < height; ++y) {                                   \
      const ELEM_T *srow =                                                  \
          reinterpret_cast<const ELEM_T *>(src + y * src_stride);           \
      uint8_t *dbase = dst + (width - 1) * dst_stride + y * P;              \
      size_t x = 0;                                                         \
      while (x < width) {                                                   \
        size_t vl = __riscv_vsetvl_e##SEW##MLM(width - x);                  \
        vuint##SEW##MLM##_t v =                                             \
            __riscv_vle##SEW##_v_u##SEW##MLM(srow + x, vl);                 \
        __riscv_vsse##SEW##_v_u##SEW##MLM(                                  \
            reinterpret_cast<ELEM_T *>(dbase - x * dst_stride),             \
            -static_cast<ptrdiff_t>(dst_stride), v, vl);                    \
        x += vl;                                                            \
      }                                                                     \
    }                                                                       \
  }

MAKE_ROTATE270_RVV(8, m1, uint8_t)
MAKE_ROTATE270_RVV(16, m1, uint16_t)
MAKE_ROTATE270_RVV(32, m1, uint32_t)
MAKE_ROTATE270_RVV(64, m1, uint64_t)
#undef MAKE_ROTATE270_RVV

// rotate 180: dst[height-1-y][width-1-x] = src[y][x].
// Same row count as input, just both axes reversed. Per source row, write
// to (height-1-y) with the data laid out backward — negative-stride store
// with stride -P starting at (width-1-x).
#define MAKE_ROTATE180_RVV(SEW, MLM, ELEM_T)                                \
  void rotate180_rvv_##SEW(const uint8_t *src, size_t src_stride,           \
                            size_t width, size_t height, uint8_t *dst,      \
                            size_t dst_stride) {                            \
    constexpr size_t P = (SEW) / 8;                                         \
    for (size_t y = 0; y < height; ++y) {                                   \
      const ELEM_T *srow =                                                  \
          reinterpret_cast<const ELEM_T *>(src + y * src_stride);           \
      ELEM_T *drow = reinterpret_cast<ELEM_T *>(                            \
          dst + (height - 1 - y) * dst_stride + (width - 1) * P);           \
      size_t x = 0;                                                         \
      while (x < width) {                                                   \
        size_t vl = __riscv_vsetvl_e##SEW##MLM(width - x);                  \
        vuint##SEW##MLM##_t v =                                             \
            __riscv_vle##SEW##_v_u##SEW##MLM(srow + x, vl);                 \
        __riscv_vsse##SEW##_v_u##SEW##MLM(                                  \
            drow - x, -static_cast<ptrdiff_t>(P), v, vl);                   \
        x += vl;                                                            \
      }                                                                     \
    }                                                                       \
  }

MAKE_ROTATE180_RVV(8, m1, uint8_t)
MAKE_ROTATE180_RVV(16, m1, uint16_t)
MAKE_ROTATE180_RVV(32, m1, uint32_t)
MAKE_ROTATE180_RVV(64, m1, uint64_t)
#undef MAKE_ROTATE180_RVV

// ---- public dispatch ----

kleidicv_error_t do_transpose(const void *src_v, size_t src_stride, void *dst_v,
                              size_t dst_stride, size_t src_width,
                              size_t src_height, size_t pixel_size) {
  if (!src_v || !dst_v) return KLEIDICV_ERROR_NULL_POINTER;
  if (pixel_size == 0) return KLEIDICV_ERROR_RANGE;
  if (src_width == 0 || src_height == 0) return KLEIDICV_OK;
  const auto *src = static_cast<const uint8_t *>(src_v);
  auto *dst = static_cast<uint8_t *>(dst_v);

  if (active_backend() == Backend::Rvv) {
    switch (pixel_size) {
      case 1:
        transpose_rvv_8(src, src_stride, dst, dst_stride, src_width,
                         src_height);
        return KLEIDICV_OK;
      case 2:
        transpose_rvv_16(src, src_stride, dst, dst_stride, src_width,
                          src_height);
        return KLEIDICV_OK;
      case 4:
        transpose_rvv_32(src, src_stride, dst, dst_stride, src_width,
                          src_height);
        return KLEIDICV_OK;
      case 8:
        transpose_rvv_64(src, src_stride, dst, dst_stride, src_width,
                          src_height);
        return KLEIDICV_OK;
      default:
        break;  // pixel_size 3, 6, ...: scalar fallback below
    }
  }
  return transpose_scalar(src, src_stride, dst, dst_stride, src_width,
                          src_height, pixel_size);
}

kleidicv_error_t do_rotate(const void *src_v, size_t src_stride, size_t width,
                            size_t height, void *dst_v, size_t dst_stride,
                            int angle, size_t pixel_size) {
  if (!src_v || !dst_v) return KLEIDICV_ERROR_NULL_POINTER;
  if (pixel_size == 0) return KLEIDICV_ERROR_RANGE;
  if (width == 0 || height == 0) return KLEIDICV_OK;
  int a = ((angle % 360) + 360) % 360;
  if (a != 90 && a != 180 && a != 270) return KLEIDICV_ERROR_RANGE;
  const auto *src = static_cast<const uint8_t *>(src_v);
  auto *dst = static_cast<uint8_t *>(dst_v);

  if (active_backend() == Backend::Rvv) {
    auto try_rvv = [&]() -> bool {
#define DISPATCH(FN_PREFIX)                                          \
  switch (pixel_size) {                                              \
    case 1:                                                          \
      FN_PREFIX##_8(src, src_stride, width, height, dst, dst_stride); \
      return true;                                                   \
    case 2:                                                          \
      FN_PREFIX##_16(src, src_stride, width, height, dst, dst_stride); \
      return true;                                                   \
    case 4:                                                          \
      FN_PREFIX##_32(src, src_stride, width, height, dst, dst_stride); \
      return true;                                                   \
    case 8:                                                          \
      FN_PREFIX##_64(src, src_stride, width, height, dst, dst_stride); \
      return true;                                                   \
    default:                                                         \
      return false;                                                  \
  }
      if (a == 90) DISPATCH(rotate90_rvv);
      if (a == 180) DISPATCH(rotate180_rvv);
      if (a == 270) DISPATCH(rotate270_rvv);
#undef DISPATCH
      return false;
    };
    if (try_rvv()) return KLEIDICV_OK;
  }
  return rotate_scalar(src, src_stride, width, height, dst, dst_stride, a,
                       pixel_size);
}

}  // namespace

extern "C" {
kleidicv_error_t (*kleidicv_transpose)(const void *, size_t, void *, size_t,
                                       size_t, size_t, size_t) = do_transpose;
kleidicv_error_t (*kleidicv_rotate)(const void *, size_t, size_t, size_t,
                                    void *, size_t, int,
                                    size_t) = do_rotate;
}
