// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// extern "C" entry points for kleidicv_bitwise_and.

#include "bitwise_and_decls.h"
#include "dispatch.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::select;

using AndFn = kleidicv_error_t (*)(const uint8_t *, size_t, const uint8_t *,
                                   size_t, uint8_t *, size_t, size_t, size_t);

AndFn resolve() {
  return select<AndFn>(&kleidicv::scalar::bitwise_and,
                       &kleidicv::rvv::bitwise_and);
}

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_bitwise_and)(const uint8_t *, size_t,
                                         const uint8_t *, size_t, uint8_t *,
                                         size_t, size_t, size_t) = resolve();
kleidicv_error_t (*kleidicv_bitwise_and_sme)(const uint8_t *, size_t,
                                             const uint8_t *, size_t,
                                             uint8_t *, size_t, size_t,
                                             size_t) = resolve();

}  // extern "C"
