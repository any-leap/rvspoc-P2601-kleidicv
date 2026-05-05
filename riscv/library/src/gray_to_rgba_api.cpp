// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "gray_to_rgba_decls.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::select;
using Fn = kleidicv_error_t (*)(const uint8_t *, size_t, uint8_t *, size_t,
                                size_t, size_t);
Fn resolve() {
  return select<Fn>(&kleidicv::scalar::gray_to_rgba_u8,
                    &kleidicv::rvv::gray_to_rgba_u8);
}
}  // namespace

extern "C" {
kleidicv_error_t (*kleidicv_gray_to_rgba_u8)(const uint8_t *, size_t, uint8_t *,
                                             size_t, size_t,
                                             size_t) = resolve();
kleidicv_error_t (*kleidicv_gray_to_rgba_u8_sme)(const uint8_t *, size_t,
                                                 uint8_t *, size_t, size_t,
                                                 size_t) = resolve();
}
