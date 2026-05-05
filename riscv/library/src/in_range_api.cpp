// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "in_range_decls.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::select;

using FnU8 = kleidicv_error_t (*)(const uint8_t *, size_t, uint8_t *, size_t,
                                  size_t, size_t, uint8_t, uint8_t);
using FnF32 = kleidicv_error_t (*)(const float *, size_t, uint8_t *, size_t,
                                   size_t, size_t, float, float);

FnU8 resolve_u8() {
  return select<FnU8>(&kleidicv::scalar::in_range_u8,
                     &kleidicv::rvv::in_range_u8);
}
FnF32 resolve_f32() {
  return select<FnF32>(&kleidicv::scalar::in_range_f32,
                      &kleidicv::rvv::in_range_f32);
}

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_in_range_u8)(const uint8_t *, size_t, uint8_t *,
                                         size_t, size_t, size_t, uint8_t,
                                         uint8_t) = resolve_u8();
kleidicv_error_t (*kleidicv_in_range_u8_sme)(const uint8_t *, size_t,
                                             uint8_t *, size_t, size_t, size_t,
                                             uint8_t, uint8_t) = resolve_u8();

kleidicv_error_t (*kleidicv_in_range_f32)(const float *, size_t, uint8_t *,
                                          size_t, size_t, size_t, float,
                                          float) = resolve_f32();
kleidicv_error_t (*kleidicv_in_range_f32_sme)(const float *, size_t, uint8_t *,
                                              size_t, size_t, size_t, float,
                                              float) = resolve_f32();

}  // extern "C"
