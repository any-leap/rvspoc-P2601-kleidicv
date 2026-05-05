// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "multiply_decls.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::select;

template <typename T>
using Fn = kleidicv_error_t (*)(const T *, size_t, const T *, size_t, T *,
                                size_t, size_t, size_t, double);

template <typename T>
Fn<T> resolve_for(Fn<T> rvv_fn) {
  return select<Fn<T>>(&kleidicv::scalar::saturating_multiply<T>, rvv_fn);
}

}  // namespace

extern "C" {

#define WIRE(suffix, T)                                                       \
  kleidicv_error_t (*kleidicv_saturating_multiply_##suffix)(                  \
      const T *, size_t, const T *, size_t, T *, size_t, size_t, size_t,      \
      double) = resolve_for<T>(&kleidicv::rvv::saturating_multiply_##suffix); \
  kleidicv_error_t (*kleidicv_saturating_multiply_##suffix##_sme)(            \
      const T *, size_t, const T *, size_t, T *, size_t, size_t, size_t,      \
      double) = resolve_for<T>(&kleidicv::rvv::saturating_multiply_##suffix)

WIRE(u8, uint8_t);
WIRE(s8, int8_t);
WIRE(u16, uint16_t);
WIRE(s16, int16_t);
WIRE(s32, int32_t);

#undef WIRE

}  // extern "C"
