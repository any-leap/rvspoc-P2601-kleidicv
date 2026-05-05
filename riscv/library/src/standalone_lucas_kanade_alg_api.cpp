// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>

#include "dispatch.h"
#include "kleidicv/kleidicv.h"
#include "standalone_lucas_kanade_alg_decls.h"

namespace {

using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;

kleidicv_error_t dispatch_standalone_lk_u8(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *prev_deriv_data, size_t prev_deriv_stride,
    const uint8_t *next_data, size_t next_data_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, uint8_t *status, float *err, int window_width,
    int window_height, int termination_count, double termination_epsilon,
    bool get_min_eigen_vals, float min_eigen_vals_threshold) {
  if (active_backend() == Backend::Rvv) {
    return kleidicv::rvv::standalone_lucas_kanade_alg_u8(
        prev_data, prev_data_stride, prev_deriv_data, prev_deriv_stride,
        next_data, next_data_stride, width, height, channels, prev_points,
        next_points, point_count, status, err, window_width, window_height,
        termination_count, termination_epsilon, get_min_eigen_vals,
        min_eigen_vals_threshold);
  }
  return kleidicv::scalar::standalone_lucas_kanade_alg_u8(
      prev_data, prev_data_stride, prev_deriv_data, prev_deriv_stride,
      next_data, next_data_stride, width, height, channels, prev_points,
      next_points, point_count, status, err, window_width, window_height,
      termination_count, termination_epsilon, get_min_eigen_vals,
      min_eigen_vals_threshold);
}

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_standalone_lucas_kanade_alg_u8)(
    const uint8_t *, size_t, const int16_t *, size_t, const uint8_t *, size_t,
    int, int, int, const float *, float *, size_t, uint8_t *, float *, int,
    int, int, double, bool, float) = dispatch_standalone_lk_u8;

kleidicv_error_t (*kleidicv_standalone_lucas_kanade_alg_u8_sme)(
    const uint8_t *, size_t, const int16_t *, size_t, const uint8_t *, size_t,
    int, int, int, const float *, float *, size_t, uint8_t *, float *, int,
    int, int, double, bool, float) = dispatch_standalone_lk_u8;

}  // extern "C"
