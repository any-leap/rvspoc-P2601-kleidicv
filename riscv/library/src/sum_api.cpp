// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "sum_decls.h"
#include "validation_helper.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;
namespace V = kleidicv::riscv_validation;

kleidicv_error_t sum_f32_dispatch(const float *src, size_t src_stride,
                                    size_t width, size_t height, float *out) {
  if (!out) return KLEIDICV_ERROR_NULL_POINTER;
  if (kleidicv_error_t e = V::check_image_size(width, height)) return e;
  // Empty rect: short-circuit with OK; the impl returns sum=0.
  if (width == 0 || height == 0) {
    *out = 0.0F;
    return KLEIDICV_OK;
  }
  if (!src) return KLEIDICV_ERROR_NULL_POINTER;
  if (kleidicv_error_t e = V::check_buffer_alignment<float>(src, src_stride))
    return e;
  return active_backend() == Backend::Rvv
             ? kleidicv::rvv::sum_f32(src, src_stride, width, height, out)
             : kleidicv::scalar::sum_f32(src, src_stride, width, height, out);
}
}  // namespace

extern "C" {
kleidicv_error_t (*kleidicv_sum_f32)(const float *, size_t, size_t, size_t,
                                     float *) = sum_f32_dispatch;
kleidicv_error_t (*kleidicv_sum_f32_sme)(const float *, size_t, size_t, size_t,
                                         float *) = sum_f32_dispatch;
}  // extern "C"
