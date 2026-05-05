// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "exp_decls.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::select;

using Fn = kleidicv_error_t (*)(const float *, size_t, float *, size_t,
                                size_t, size_t);

Fn resolve() {
  return select<Fn>(&kleidicv::scalar::exp_f32, &kleidicv::rvv::exp_f32);
}

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_exp_f32)(const float *, size_t, float *, size_t,
                                     size_t, size_t) = resolve();
kleidicv_error_t (*kleidicv_exp_f32_sme)(const float *, size_t, float *,
                                         size_t, size_t, size_t) = resolve();

}  // extern "C"
