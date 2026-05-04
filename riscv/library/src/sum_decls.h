// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RISCV_SUM_DECLS_H
#define KLEIDICV_RISCV_SUM_DECLS_H

#include <cstddef>

#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t sum_f32(const float *src, size_t src_stride, size_t width,
                         size_t height, float *sum);
}

namespace kleidicv::rvv {
kleidicv_error_t sum_f32(const float *src, size_t src_stride, size_t width,
                         size_t height, float *sum);
}

#endif
