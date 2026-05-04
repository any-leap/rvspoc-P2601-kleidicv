// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "sum_decls.h"

#include "kleidicv/kleidicv.h"

namespace {
using Fn = kleidicv_error_t (*)(const float *, size_t, size_t, size_t,
                                float *);
Fn resolve() {
  return kleidicv::riscv_dispatch::select<Fn>(&kleidicv::scalar::sum_f32,
                                              &kleidicv::rvv::sum_f32);
}
}  // namespace

extern "C" {
kleidicv_error_t (*kleidicv_sum_f32)(const float *, size_t, size_t, size_t,
                                     float *) = resolve();
kleidicv_error_t (*kleidicv_sum_f32_sme)(const float *, size_t, size_t, size_t,
                                         float *) = resolve();
}  // extern "C"
