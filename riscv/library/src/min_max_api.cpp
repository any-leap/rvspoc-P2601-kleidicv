// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "min_max_decls.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::select;
template <typename T>
using Fn = kleidicv_error_t (*)(const T *, size_t, size_t, size_t, T *, T *);
template <typename T>
Fn<T> resolve_for(Fn<T> rvv) {
  return select<Fn<T>>(&kleidicv::scalar::min_max<T>, rvv);
}
}  // namespace

extern "C" {
#define WIRE(suffix, T)                                                       \
  kleidicv_error_t (*kleidicv_min_max_##suffix)(                              \
      const T *, size_t, size_t, size_t, T *, T *) =                          \
      resolve_for<T>(&kleidicv::rvv::min_max_##suffix);                       \
  kleidicv_error_t (*kleidicv_min_max_##suffix##_sme)(                        \
      const T *, size_t, size_t, size_t, T *, T *) =                          \
      resolve_for<T>(&kleidicv::rvv::min_max_##suffix)

WIRE(u8, uint8_t);
WIRE(s8, int8_t);
WIRE(u16, uint16_t);
WIRE(s16, int16_t);
WIRE(s32, int32_t);
#undef WIRE
}
