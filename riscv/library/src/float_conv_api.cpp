// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "float_conv_decls.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::select;
using FnF32U8 = kleidicv_error_t (*)(const float *, size_t, uint8_t *, size_t,
                                     size_t, size_t);
using FnF32S8 = kleidicv_error_t (*)(const float *, size_t, int8_t *, size_t,
                                     size_t, size_t);
using FnU8F32 = kleidicv_error_t (*)(const uint8_t *, size_t, float *, size_t,
                                     size_t, size_t);
using FnS8F32 = kleidicv_error_t (*)(const int8_t *, size_t, float *, size_t,
                                     size_t, size_t);
}  // namespace

extern "C" {
kleidicv_error_t (*kleidicv_f32_to_u8)(const float *, size_t, uint8_t *, size_t,
                                       size_t, size_t) =
    select<FnF32U8>(&kleidicv::scalar::f32_to_u8, &kleidicv::rvv::f32_to_u8);
kleidicv_error_t (*kleidicv_f32_to_u8_sme)(const float *, size_t, uint8_t *,
                                           size_t, size_t, size_t) =
    select<FnF32U8>(&kleidicv::scalar::f32_to_u8, &kleidicv::rvv::f32_to_u8);

kleidicv_error_t (*kleidicv_f32_to_s8)(const float *, size_t, int8_t *, size_t,
                                       size_t, size_t) =
    select<FnF32S8>(&kleidicv::scalar::f32_to_s8, &kleidicv::rvv::f32_to_s8);
kleidicv_error_t (*kleidicv_f32_to_s8_sme)(const float *, size_t, int8_t *,
                                           size_t, size_t, size_t) =
    select<FnF32S8>(&kleidicv::scalar::f32_to_s8, &kleidicv::rvv::f32_to_s8);

kleidicv_error_t (*kleidicv_u8_to_f32)(const uint8_t *, size_t, float *, size_t,
                                       size_t, size_t) =
    select<FnU8F32>(&kleidicv::scalar::u8_to_f32, &kleidicv::rvv::u8_to_f32);
kleidicv_error_t (*kleidicv_u8_to_f32_sme)(const uint8_t *, size_t, float *,
                                           size_t, size_t, size_t) =
    select<FnU8F32>(&kleidicv::scalar::u8_to_f32, &kleidicv::rvv::u8_to_f32);

kleidicv_error_t (*kleidicv_s8_to_f32)(const int8_t *, size_t, float *, size_t,
                                       size_t, size_t) =
    select<FnS8F32>(&kleidicv::scalar::s8_to_f32, &kleidicv::rvv::s8_to_f32);
kleidicv_error_t (*kleidicv_s8_to_f32_sme)(const int8_t *, size_t, float *,
                                           size_t, size_t, size_t) =
    select<FnS8F32>(&kleidicv::scalar::s8_to_f32, &kleidicv::rvv::s8_to_f32);
}
