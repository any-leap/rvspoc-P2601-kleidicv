// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "float_conv_decls.h"
#include "validation_helper.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;
namespace V = kleidicv::riscv_validation;

template <typename Src, typename Dst, auto Scalar, auto Rvv>
kleidicv_error_t fconv_dispatch(const Src *src, size_t src_stride, Dst *dst,
                                  size_t dst_stride, size_t width,
                                  size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (kleidicv_error_t e = V::check_image_size(width, height)) return e;
  if (kleidicv_error_t e =
          V::check_buffer_alignment<Src>(src, src_stride, height))
    return e;
  if (kleidicv_error_t e =
          V::check_buffer_alignment<Dst>(dst, dst_stride, height))
    return e;
  return active_backend() == Backend::Rvv
             ? Rvv(src, src_stride, dst, dst_stride, width, height)
             : Scalar(src, src_stride, dst, dst_stride, width, height);
}
}  // namespace

extern "C" {
#define WIRE(name, Src, Dst)                                                  \
  kleidicv_error_t (*kleidicv_##name)(const Src *, size_t, Dst *, size_t,    \
                                      size_t, size_t) =                       \
      fconv_dispatch<Src, Dst, &kleidicv::scalar::name,                       \
                       &kleidicv::rvv::name>;                                  \
  kleidicv_error_t (*kleidicv_##name##_sme)(const Src *, size_t, Dst *,      \
                                            size_t, size_t, size_t) =         \
      fconv_dispatch<Src, Dst, &kleidicv::scalar::name, &kleidicv::rvv::name>

WIRE(f32_to_u8, float, uint8_t);
WIRE(f32_to_s8, float, int8_t);
WIRE(u8_to_f32, uint8_t, float);
WIRE(s8_to_f32, int8_t, float);
#undef WIRE
}
