// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "compare_decls.h"
#include "dispatch.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::select;

using Fn = kleidicv_error_t (*)(const uint8_t *, size_t, const uint8_t *,
                                size_t, uint8_t *, size_t, size_t, size_t);

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_compare_equal_u8)(
    const uint8_t *, size_t, const uint8_t *, size_t, uint8_t *, size_t, size_t,
    size_t) = select<Fn>(&kleidicv::scalar::compare_equal_u8,
                         &kleidicv::rvv::compare_equal_u8);
kleidicv_error_t (*kleidicv_compare_equal_u8_sme)(
    const uint8_t *, size_t, const uint8_t *, size_t, uint8_t *, size_t, size_t,
    size_t) = select<Fn>(&kleidicv::scalar::compare_equal_u8,
                         &kleidicv::rvv::compare_equal_u8);

kleidicv_error_t (*kleidicv_compare_greater_u8)(
    const uint8_t *, size_t, const uint8_t *, size_t, uint8_t *, size_t, size_t,
    size_t) = select<Fn>(&kleidicv::scalar::compare_greater_u8,
                         &kleidicv::rvv::compare_greater_u8);
kleidicv_error_t (*kleidicv_compare_greater_u8_sme)(
    const uint8_t *, size_t, const uint8_t *, size_t, uint8_t *, size_t, size_t,
    size_t) = select<Fn>(&kleidicv::scalar::compare_greater_u8,
                         &kleidicv::rvv::compare_greater_u8);

}  // extern "C"
