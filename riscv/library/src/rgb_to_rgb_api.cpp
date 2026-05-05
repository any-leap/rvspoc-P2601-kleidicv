// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "rgb_to_rgb_decls.h"

#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::select;
using Fn = kleidicv_error_t (*)(const uint8_t *, size_t, uint8_t *, size_t,
                                size_t, size_t);
}  // namespace

extern "C" {

#define WIRE(name)                                                          \
  kleidicv_error_t (*kleidicv_##name)(const uint8_t *, size_t, uint8_t *,  \
                                      size_t, size_t, size_t) =            \
      select<Fn>(&kleidicv::scalar::name, &kleidicv::rvv::name);            \
  kleidicv_error_t (*kleidicv_##name##_sme)(const uint8_t *, size_t,       \
                                            uint8_t *, size_t, size_t,    \
                                            size_t) =                       \
      select<Fn>(&kleidicv::scalar::name, &kleidicv::rvv::name)

WIRE(rgb_to_bgr_u8);
WIRE(rgba_to_bgra_u8);
WIRE(rgb_to_bgra_u8);
WIRE(rgb_to_rgba_u8);
WIRE(rgba_to_bgr_u8);
WIRE(rgba_to_rgb_u8);

// rgb_to_rgb / rgba_to_rgba have no _sme alias in the public header.
kleidicv_error_t (*kleidicv_rgb_to_rgb_u8)(const uint8_t *, size_t, uint8_t *,
                                           size_t, size_t, size_t) =
    select<Fn>(&kleidicv::scalar::rgb_to_rgb_u8,
               &kleidicv::rvv::rgb_to_rgb_u8);
kleidicv_error_t (*kleidicv_rgba_to_rgba_u8)(const uint8_t *, size_t,
                                             uint8_t *, size_t, size_t,
                                             size_t) =
    select<Fn>(&kleidicv::scalar::rgba_to_rgba_u8,
               &kleidicv::rvv::rgba_to_rgba_u8);

#undef WIRE

}  // extern "C"
