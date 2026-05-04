// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// extern "C" entry points for the saturating_absdiff family. Each pointer is
// resolved once at static init time via the riscv_dispatch helper and then
// stays put — no per-call branch.

#include "absdiff_decls.h"
#include "dispatch.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::select;

// Function-pointer type for each public entry, used to keep the select<>
// invocation type-safe.
template <typename T>
using AbsdiffFn = kleidicv_error_t (*)(const T *, size_t, const T *, size_t,
                                       T *, size_t, size_t, size_t);

template <typename T>
AbsdiffFn<T> resolve_for(AbsdiffFn<T> rvv_fn) {
  return select<AbsdiffFn<T>>(&kleidicv::scalar::saturating_absdiff<T>, rvv_fn);
}

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_saturating_absdiff_u8)(
    const uint8_t *, size_t, const uint8_t *, size_t, uint8_t *, size_t,
    size_t, size_t) = resolve_for<uint8_t>(&kleidicv::rvv::saturating_absdiff_u8);
kleidicv_error_t (*kleidicv_saturating_absdiff_u8_sme)(
    const uint8_t *, size_t, const uint8_t *, size_t, uint8_t *, size_t,
    size_t, size_t) = resolve_for<uint8_t>(&kleidicv::rvv::saturating_absdiff_u8);

kleidicv_error_t (*kleidicv_saturating_absdiff_s8)(
    const int8_t *, size_t, const int8_t *, size_t, int8_t *, size_t, size_t,
    size_t) = resolve_for<int8_t>(&kleidicv::rvv::saturating_absdiff_s8);
kleidicv_error_t (*kleidicv_saturating_absdiff_s8_sme)(
    const int8_t *, size_t, const int8_t *, size_t, int8_t *, size_t, size_t,
    size_t) = resolve_for<int8_t>(&kleidicv::rvv::saturating_absdiff_s8);

kleidicv_error_t (*kleidicv_saturating_absdiff_u16)(
    const uint16_t *, size_t, const uint16_t *, size_t, uint16_t *, size_t,
    size_t, size_t) =
    resolve_for<uint16_t>(&kleidicv::rvv::saturating_absdiff_u16);
kleidicv_error_t (*kleidicv_saturating_absdiff_u16_sme)(
    const uint16_t *, size_t, const uint16_t *, size_t, uint16_t *, size_t,
    size_t, size_t) =
    resolve_for<uint16_t>(&kleidicv::rvv::saturating_absdiff_u16);

kleidicv_error_t (*kleidicv_saturating_absdiff_s16)(
    const int16_t *, size_t, const int16_t *, size_t, int16_t *, size_t,
    size_t, size_t) =
    resolve_for<int16_t>(&kleidicv::rvv::saturating_absdiff_s16);
kleidicv_error_t (*kleidicv_saturating_absdiff_s16_sme)(
    const int16_t *, size_t, const int16_t *, size_t, int16_t *, size_t,
    size_t, size_t) =
    resolve_for<int16_t>(&kleidicv::rvv::saturating_absdiff_s16);

kleidicv_error_t (*kleidicv_saturating_absdiff_s32)(
    const int32_t *, size_t, const int32_t *, size_t, int32_t *, size_t,
    size_t, size_t) =
    resolve_for<int32_t>(&kleidicv::rvv::saturating_absdiff_s32);
kleidicv_error_t (*kleidicv_saturating_absdiff_s32_sme)(
    const int32_t *, size_t, const int32_t *, size_t, int32_t *, size_t,
    size_t, size_t) =
    resolve_for<int32_t>(&kleidicv::rvv::saturating_absdiff_s32);

}  // extern "C"

// Public introspection helper used by tests to assert which backend was
// chosen. Not part of KleidiCV ABI; the symbol lives in our own namespace.
extern "C" const char *kleidicv_riscv_active_backend() {
  return kleidicv::riscv_dispatch::active_backend() ==
                 kleidicv::riscv_dispatch::Backend::Rvv
             ? "rvv"
             : "scalar";
}
