// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "standalone_lucas_kanade_alg_decls.h"
#include "optical_flow_lk_common.h"

namespace kleidicv::scalar {

kleidicv_error_t standalone_lucas_kanade_alg_u8(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *prev_deriv_data, size_t prev_deriv_stride,
    const uint8_t *next_data, size_t next_data_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, uint8_t *status, float *err, int window_width,
    int window_height, int termination_count, double termination_epsilon,
    bool get_min_eigen_vals, float min_eigen_vals_threshold) {
  using kleidicv::riscv_lk::PatchBuffer;
  using kleidicv::riscv_lk::ScalarImpl;

  if (kleidicv_error_t e = kleidicv::riscv_lk::validate_args(
          prev_data, prev_data_stride, prev_deriv_data, prev_deriv_stride,
          next_data, next_data_stride, width, height, channels, prev_points,
          next_points, point_count, window_width, window_height)) {
    return e;
  }
  if (point_count == 0) return KLEIDICV_OK;

  PatchBuffer buf(window_width, window_height, channels);
  if (!buf.valid()) return KLEIDICV_ERROR_ALLOCATION;

  return kleidicv::riscv_lk::lk_compute<ScalarImpl>(
      buf.window(), buf.scharr_window(), prev_data, prev_data_stride,
      prev_deriv_data, prev_deriv_stride, next_data, next_data_stride, width,
      height, channels, prev_points, next_points, point_count, status, err,
      window_width, window_height, termination_count, termination_epsilon,
      get_min_eigen_vals, min_eigen_vals_threshold);
}

}  // namespace kleidicv::scalar
