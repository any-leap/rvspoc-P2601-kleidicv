// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "split_merge_decls.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::select;
using SplitFn = kleidicv_error_t (*)(const void *, size_t, void **,
                                     const size_t *, size_t, size_t, size_t,
                                     size_t);
using MergeFn = kleidicv_error_t (*)(const void **, const size_t *, void *,
                                     size_t, size_t, size_t, size_t, size_t);
}  // namespace

extern "C" {
kleidicv_error_t (*kleidicv_split)(const void *, size_t, void **,
                                   const size_t *, size_t, size_t, size_t,
                                   size_t) =
    select<SplitFn>(&kleidicv::scalar::split, &kleidicv::rvv::split);

kleidicv_error_t (*kleidicv_merge)(const void **, const size_t *, void *,
                                   size_t, size_t, size_t, size_t, size_t) =
    select<MergeFn>(&kleidicv::scalar::merge, &kleidicv::rvv::merge);
}
