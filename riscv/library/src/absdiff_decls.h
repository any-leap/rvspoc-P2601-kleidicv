// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Declarations of saturating_absdiff impls in both backends, so the api glue
// can take their addresses without dragging in template bodies.

#ifndef KLEIDICV_RISCV_ABSDIFF_DECLS_H
#define KLEIDICV_RISCV_ABSDIFF_DECLS_H

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
template <typename T>
kleidicv_error_t saturating_absdiff(const T *src_a, size_t src_a_stride,
                                    const T *src_b, size_t src_b_stride, T *dst,
                                    size_t dst_stride, size_t width,
                                    size_t height);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {
kleidicv_error_t saturating_absdiff_u8(const uint8_t *, size_t,
                                       const uint8_t *, size_t, uint8_t *,
                                       size_t, size_t, size_t);
kleidicv_error_t saturating_absdiff_s8(const int8_t *, size_t, const int8_t *,
                                       size_t, int8_t *, size_t, size_t,
                                       size_t);
kleidicv_error_t saturating_absdiff_u16(const uint16_t *, size_t,
                                        const uint16_t *, size_t, uint16_t *,
                                        size_t, size_t, size_t);
kleidicv_error_t saturating_absdiff_s16(const int16_t *, size_t,
                                        const int16_t *, size_t, int16_t *,
                                        size_t, size_t, size_t);
kleidicv_error_t saturating_absdiff_s32(const int32_t *, size_t,
                                        const int32_t *, size_t, int32_t *,
                                        size_t, size_t, size_t);
}  // namespace kleidicv::rvv

#endif
