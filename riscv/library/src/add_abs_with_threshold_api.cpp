// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "add_abs_with_threshold_decls.h"
#include "dispatch.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::select;

using Fn = kleidicv_error_t (*)(const int16_t *, size_t, const int16_t *,
                                size_t, int16_t *, size_t, size_t, size_t,
                                int16_t);

Fn resolve() {
  return select<Fn>(&kleidicv::scalar::saturating_add_abs_with_threshold_s16,
                    &kleidicv::rvv::saturating_add_abs_with_threshold_s16);
}

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_saturating_add_abs_with_threshold_s16)(
    const int16_t *, size_t, const int16_t *, size_t, int16_t *, size_t,
    size_t, size_t, int16_t) = resolve();
kleidicv_error_t (*kleidicv_saturating_add_abs_with_threshold_s16_sme)(
    const int16_t *, size_t, const int16_t *, size_t, int16_t *, size_t,
    size_t, size_t, int16_t) = resolve();

}  // extern "C"
