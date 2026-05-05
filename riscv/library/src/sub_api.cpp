// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// extern "C" entry points for kleidicv_saturating_sub_*.

#include "dispatch.h"
#include "sub_decls.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::select;

template <typename T>
using SubFn = kleidicv_error_t (*)(const T *, size_t, const T *, size_t, T *,
                                   size_t, size_t, size_t);

template <typename T>
SubFn<T> resolve_for(SubFn<T> rvv_fn) {
  return select<SubFn<T>>(&kleidicv::scalar::saturating_sub<T>, rvv_fn);
}

}  // namespace

extern "C" {

#define WIRE(suffix, T)                                                  \
  kleidicv_error_t (*kleidicv_saturating_sub_##suffix)(                  \
      const T *, size_t, const T *, size_t, T *, size_t, size_t,         \
      size_t) = resolve_for<T>(&kleidicv::rvv::saturating_sub_##suffix); \
  kleidicv_error_t (*kleidicv_saturating_sub_##suffix##_sme)(            \
      const T *, size_t, const T *, size_t, T *, size_t, size_t,         \
      size_t) = resolve_for<T>(&kleidicv::rvv::saturating_sub_##suffix)

WIRE(u8, uint8_t);
WIRE(s8, int8_t);
WIRE(u16, uint16_t);
WIRE(s16, int16_t);
WIRE(u32, uint32_t);
WIRE(s32, int32_t);
WIRE(u64, uint64_t);
WIRE(s64, int64_t);

#undef WIRE

}  // extern "C"
