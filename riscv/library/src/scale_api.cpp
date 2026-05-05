// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "scale_decls.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::select;

using FnU8 = kleidicv_error_t (*)(const uint8_t *, size_t, uint8_t *, size_t,
                                  size_t, size_t, double, double);
using FnF32 = kleidicv_error_t (*)(const float *, size_t, float *, size_t,
                                   size_t, size_t, double, double);

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_scale_u8)(const uint8_t *, size_t, uint8_t *,
                                      size_t, size_t, size_t, double, double) =
    select<FnU8>(&kleidicv::scalar::scale_u8, &kleidicv::rvv::scale_u8);

kleidicv_error_t (*kleidicv_scale_f32)(const float *, size_t, float *, size_t,
                                       size_t, size_t, double, double) =
    select<FnF32>(&kleidicv::scalar::scale_f32, &kleidicv::rvv::scale_f32);
kleidicv_error_t (*kleidicv_scale_f32_sme)(const float *, size_t, float *,
                                           size_t, size_t, size_t, double,
                                           double) =
    select<FnF32>(&kleidicv::scalar::scale_f32, &kleidicv::rvv::scale_f32);

}  // extern "C"
