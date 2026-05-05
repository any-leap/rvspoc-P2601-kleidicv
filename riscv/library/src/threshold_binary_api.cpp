// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "threshold_binary_decls.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::select;

using Fn = kleidicv_error_t (*)(const uint8_t *, size_t, uint8_t *, size_t,
                                size_t, size_t, uint8_t, uint8_t);

Fn resolve() {
  return select<Fn>(&kleidicv::scalar::threshold_binary_u8,
                    &kleidicv::rvv::threshold_binary_u8);
}

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_threshold_binary_u8)(
    const uint8_t *, size_t, uint8_t *, size_t, size_t, size_t, uint8_t,
    uint8_t) = resolve();
kleidicv_error_t (*kleidicv_threshold_binary_u8_sme)(
    const uint8_t *, size_t, uint8_t *, size_t, size_t, size_t, uint8_t,
    uint8_t) = resolve();

}  // extern "C"
