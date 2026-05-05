// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "min_max_decls.h"
#include "validation_helper.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;
namespace V = kleidicv::riscv_validation;

template <typename T, typename Rvv>
kleidicv_error_t min_max_dispatch(const T *src, size_t src_stride, size_t width,
                                    size_t height, T *min_out, T *max_out,
                                    Rvv rvv_fn) {
  // Upstream allows either or both of min_out / max_out to be null (no-op);
  // src must be non-null and width/height must both be > 0 (RANGE otherwise).
  if (!src) return KLEIDICV_ERROR_NULL_POINTER;
  if (kleidicv_error_t e = V::check_image_size(width, height)) return e;
  if (width == 0 || height == 0) return KLEIDICV_ERROR_RANGE;
  if (kleidicv_error_t e =
          V::check_buffer_alignment<T>(src, src_stride, height))
    return e;
  return active_backend() == Backend::Rvv
             ? rvv_fn(src, src_stride, width, height, min_out, max_out)
             : kleidicv::scalar::min_max<T>(src, src_stride, width, height,
                                              min_out, max_out);
}

#define MK_DISPATCH(suffix, T)                                                 \
  kleidicv_error_t min_max_##suffix##_dispatch(                                \
      const T *src, size_t src_stride, size_t width, size_t height,            \
      T *min_out, T *max_out) {                                                \
    return min_max_dispatch<T>(src, src_stride, width, height, min_out,        \
                                 max_out, &kleidicv::rvv::min_max_##suffix);   \
  }
MK_DISPATCH(u8, uint8_t)
MK_DISPATCH(s8, int8_t)
MK_DISPATCH(u16, uint16_t)
MK_DISPATCH(s16, int16_t)
MK_DISPATCH(s32, int32_t)
#undef MK_DISPATCH
}  // namespace

extern "C" {
#define WIRE(suffix, T)                                                        \
  kleidicv_error_t (*kleidicv_min_max_##suffix)(                               \
      const T *, size_t, size_t, size_t, T *, T *) =                           \
      min_max_##suffix##_dispatch;                                             \
  kleidicv_error_t (*kleidicv_min_max_##suffix##_sme)(                         \
      const T *, size_t, size_t, size_t, T *, T *) =                           \
      min_max_##suffix##_dispatch
WIRE(u8, uint8_t);
WIRE(s8, int8_t);
WIRE(u16, uint16_t);
WIRE(s16, int16_t);
WIRE(s32, int32_t);
#undef WIRE
}
