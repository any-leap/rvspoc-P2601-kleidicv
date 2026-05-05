// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV path for the standalone LK kernel. The two SIMD primitives
// (sample_patch_and_gradients / accumulate_mismatch_vector) vectorise the
// per-row inner loop with vsetvli LMUL=2, SEW=16. Outer per-point control
// flow + structure-tensor solve stay scalar (only ever ~tens of points per
// frame). Behaviour matches scalar bit-exactly: same fixed-point shifts,
// 64-bit accumulation, OpenCV-style FractionBits=14.

#include <riscv_vector.h>

#include <cstdint>

#include "optical_flow_lk_common.h"
#include "standalone_lucas_kanade_alg_decls.h"

namespace kleidicv::rvv {

namespace {

using kleidicv::riscv_lk::kFixedPointDescale;
using kleidicv::riscv_lk::kFractionBits;

// Widening lerp for u8→s16 patches (FractionBits-5 = 9-bit right shift,
// rounded). Returns vint16 lane-wise.
inline vint16m2_t lerp_u8_to_s16(vuint8m1_t tl, vuint8m1_t tr, vuint8m1_t bl,
                                  vuint8m1_t br, int16_t c_tl, int16_t c_tr,
                                  int16_t c_bl, int16_t c_br, size_t vl) {
  vuint16m2_t tl16 = __riscv_vzext_vf2_u16m2(tl, vl);
  vuint16m2_t tr16 = __riscv_vzext_vf2_u16m2(tr, vl);
  vuint16m2_t bl16 = __riscv_vzext_vf2_u16m2(bl, vl);
  vuint16m2_t br16 = __riscv_vzext_vf2_u16m2(br, vl);
  vint16m2_t tl_s = __riscv_vreinterpret_v_u16m2_i16m2(tl16);
  vint16m2_t tr_s = __riscv_vreinterpret_v_u16m2_i16m2(tr16);
  vint16m2_t bl_s = __riscv_vreinterpret_v_u16m2_i16m2(bl16);
  vint16m2_t br_s = __riscv_vreinterpret_v_u16m2_i16m2(br16);

  // 32-bit widening multiply-add chain for headroom (coeffs ≤ 2^14).
  vint32m4_t acc = __riscv_vwmul_vx_i32m4(tl_s, c_tl, vl);
  acc = __riscv_vwmacc_vx_i32m4(acc, c_tr, tr_s, vl);
  acc = __riscv_vwmacc_vx_i32m4(acc, c_bl, bl_s, vl);
  acc = __riscv_vwmacc_vx_i32m4(acc, c_br, br_s, vl);
  // Round-shift-right by 9, narrowing s32 → s16. Saturating clip is fine —
  // valid LK samples never reach the i16 limit.
  return __riscv_vnclip_wx_i16m2(acc, kFractionBits - 5, __RISCV_VXRM_RNU, vl);
}

// Widening lerp for s16 (Scharr) patches with FractionBits right shift.
inline vint16m2_t lerp_s16(vint16m2_t tl, vint16m2_t tr, vint16m2_t bl,
                            vint16m2_t br, int16_t c_tl, int16_t c_tr,
                            int16_t c_bl, int16_t c_br, size_t vl) {
  vint32m4_t acc = __riscv_vwmul_vx_i32m4(tl, c_tl, vl);
  acc = __riscv_vwmacc_vx_i32m4(acc, c_tr, tr, vl);
  acc = __riscv_vwmacc_vx_i32m4(acc, c_bl, bl, vl);
  acc = __riscv_vwmacc_vx_i32m4(acc, c_br, br, vl);
  return __riscv_vnclip_wx_i16m2(acc, kFractionBits, __RISCV_VXRM_RNU, vl);
}

struct RvvImpl {
  static void sample_patch_and_gradients(
      int16_t *window, int16_t *scharr_window, const uint8_t *prev_data,
      ptrdiff_t prev_data_stride, const int16_t *scharr_data,
      ptrdiff_t scharr_stride_elements, int channels, int window_corner_x,
      int window_corner_y, int window_width, int window_height,
      int16_t coeff_tl, int16_t coeff_tr, int16_t coeff_bl, int16_t coeff_br,
      float &sum_xx, float &sum_xy, float &sum_yy) {
    int64_t acc_xx = 0, acc_xy = 0, acc_yy = 0;
    const int row_elems = window_width * channels;

    for (int y = 0; y < window_height; ++y) {
      const uint8_t *prev_row0 =
          prev_data + (y + window_corner_y) * prev_data_stride +
          static_cast<ptrdiff_t>(window_corner_x) * channels;
      const uint8_t *prev_row1 = prev_row0 + prev_data_stride;
      const int16_t *sch_row0 =
          scharr_data + (y + window_corner_y) * scharr_stride_elements +
          static_cast<ptrdiff_t>(window_corner_x) * 2L * channels;
      const int16_t *sch_row1 = sch_row0 + scharr_stride_elements;
      int16_t *win_row = window + static_cast<ptrdiff_t>(y) * row_elems;
      int16_t *scw_row =
          scharr_window + static_cast<ptrdiff_t>(y) * row_elems * 2L;

      size_t i = 0;
      while (i < static_cast<size_t>(row_elems)) {
        size_t vl =
            __riscv_vsetvl_e16m2(static_cast<size_t>(row_elems) - i);

        // Patch sample.
        vuint8m1_t tl = __riscv_vle8_v_u8m1(prev_row0 + i, vl);
        vuint8m1_t tr = __riscv_vle8_v_u8m1(prev_row0 + i + channels, vl);
        vuint8m1_t bl = __riscv_vle8_v_u8m1(prev_row1 + i, vl);
        vuint8m1_t br = __riscv_vle8_v_u8m1(prev_row1 + i + channels, vl);
        vint16m2_t patch = lerp_u8_to_s16(tl, tr, bl, br, coeff_tl, coeff_tr,
                                            coeff_bl, coeff_br, vl);
        __riscv_vse16_v_i16m2(win_row + i, patch, vl);

        // Scharr (interleaved: load segmented then lerp x and y separately).
        vint16m2x2_t s_tl =
            __riscv_vlseg2e16_v_i16m2x2(sch_row0 + i * 2, vl);
        vint16m2x2_t s_tr =
            __riscv_vlseg2e16_v_i16m2x2(sch_row0 + (i + channels) * 2, vl);
        vint16m2x2_t s_bl =
            __riscv_vlseg2e16_v_i16m2x2(sch_row1 + i * 2, vl);
        vint16m2x2_t s_br =
            __riscv_vlseg2e16_v_i16m2x2(sch_row1 + (i + channels) * 2, vl);

        vint16m2_t sx = lerp_s16(__riscv_vget_v_i16m2x2_i16m2(s_tl, 0),
                                   __riscv_vget_v_i16m2x2_i16m2(s_tr, 0),
                                   __riscv_vget_v_i16m2x2_i16m2(s_bl, 0),
                                   __riscv_vget_v_i16m2x2_i16m2(s_br, 0),
                                   coeff_tl, coeff_tr, coeff_bl, coeff_br, vl);
        vint16m2_t sy = lerp_s16(__riscv_vget_v_i16m2x2_i16m2(s_tl, 1),
                                   __riscv_vget_v_i16m2x2_i16m2(s_tr, 1),
                                   __riscv_vget_v_i16m2x2_i16m2(s_bl, 1),
                                   __riscv_vget_v_i16m2x2_i16m2(s_br, 1),
                                   coeff_tl, coeff_tr, coeff_bl, coeff_br, vl);

        vint16m2x2_t sxy = __riscv_vcreate_v_i16m2x2(sx, sy);
        __riscv_vsseg2e16_v_i16m2x2(scw_row + i * 2, sxy, vl);

        // Accumulate xx/xy/yy as int64 to avoid overflow on large windows.
        // Widening multiply (s16*s16 → s32), then widening reduce-sum to s64.
        vint32m4_t xx = __riscv_vwmul_vv_i32m4(sx, sx, vl);
        vint32m4_t xy = __riscv_vwmul_vv_i32m4(sx, sy, vl);
        vint32m4_t yy = __riscv_vwmul_vv_i32m4(sy, sy, vl);

        vint64m1_t zero =
            __riscv_vmv_v_x_i64m1(0, __riscv_vsetvlmax_e64m1());
        vint64m1_t r_xx = __riscv_vwredsum_vs_i32m4_i64m1(xx, zero, vl);
        vint64m1_t r_xy = __riscv_vwredsum_vs_i32m4_i64m1(xy, zero, vl);
        vint64m1_t r_yy = __riscv_vwredsum_vs_i32m4_i64m1(yy, zero, vl);
        acc_xx += __riscv_vmv_x_s_i64m1_i64(r_xx);
        acc_xy += __riscv_vmv_x_s_i64m1_i64(r_xy);
        acc_yy += __riscv_vmv_x_s_i64m1_i64(r_yy);

        i += vl;
      }
    }
    sum_xx = static_cast<float>(acc_xx) * kFixedPointDescale;
    sum_xy = static_cast<float>(acc_xy) * kFixedPointDescale;
    sum_yy = static_cast<float>(acc_yy) * kFixedPointDescale;
  }

  static void accumulate_mismatch_vector(
      const uint8_t *next_data, ptrdiff_t next_stride, const int16_t *window,
      const int16_t *scharr_window, int channels, int window_corner_x,
      int window_corner_y, int window_width, int window_height,
      int16_t coeff_tl, int16_t coeff_tr, int16_t coeff_bl, int16_t coeff_br,
      float &sum_dx, float &sum_dy) {
    int64_t acc_x = 0, acc_y = 0;
    const int row_elems = window_width * channels;

    for (int y = 0; y < window_height; ++y) {
      const uint8_t *next_row0 =
          next_data + (y + window_corner_y) * next_stride +
          static_cast<ptrdiff_t>(window_corner_x) * channels;
      const uint8_t *next_row1 = next_row0 + next_stride;
      const int16_t *win_row = window + static_cast<ptrdiff_t>(y) * row_elems;
      const int16_t *scw_row =
          scharr_window + static_cast<ptrdiff_t>(y) * row_elems * 2L;

      size_t i = 0;
      while (i < static_cast<size_t>(row_elems)) {
        size_t vl =
            __riscv_vsetvl_e16m2(static_cast<size_t>(row_elems) - i);

        vuint8m1_t tl = __riscv_vle8_v_u8m1(next_row0 + i, vl);
        vuint8m1_t tr = __riscv_vle8_v_u8m1(next_row0 + i + channels, vl);
        vuint8m1_t bl = __riscv_vle8_v_u8m1(next_row1 + i, vl);
        vuint8m1_t br = __riscv_vle8_v_u8m1(next_row1 + i + channels, vl);
        vint16m2_t sample = lerp_u8_to_s16(tl, tr, bl, br, coeff_tl, coeff_tr,
                                             coeff_bl, coeff_br, vl);
        vint16m2_t win = __riscv_vle16_v_i16m2(win_row + i, vl);
        vint16m2_t diff = __riscv_vsub_vv_i16m2(sample, win, vl);

        vint16m2x2_t sxy =
            __riscv_vlseg2e16_v_i16m2x2(scw_row + i * 2, vl);
        vint16m2_t sx = __riscv_vget_v_i16m2x2_i16m2(sxy, 0);
        vint16m2_t sy = __riscv_vget_v_i16m2x2_i16m2(sxy, 1);

        vint32m4_t px = __riscv_vwmul_vv_i32m4(sx, diff, vl);
        vint32m4_t py = __riscv_vwmul_vv_i32m4(sy, diff, vl);

        vint64m1_t zero =
            __riscv_vmv_v_x_i64m1(0, __riscv_vsetvlmax_e64m1());
        vint64m1_t r_x = __riscv_vwredsum_vs_i32m4_i64m1(px, zero, vl);
        vint64m1_t r_y = __riscv_vwredsum_vs_i32m4_i64m1(py, zero, vl);
        acc_x += __riscv_vmv_x_s_i64m1_i64(r_x);
        acc_y += __riscv_vmv_x_s_i64m1_i64(r_y);

        i += vl;
      }
    }
    sum_dx = static_cast<float>(acc_x) * kFixedPointDescale;
    sum_dy = static_cast<float>(acc_y) * kFixedPointDescale;
  }
};

}  // namespace

kleidicv_error_t standalone_lucas_kanade_alg_u8(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *prev_deriv_data, size_t prev_deriv_stride,
    const uint8_t *next_data, size_t next_data_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, uint8_t *status, float *err, int window_width,
    int window_height, int termination_count, double termination_epsilon,
    bool get_min_eigen_vals, float min_eigen_vals_threshold) {
  using kleidicv::riscv_lk::PatchBuffer;

  if (kleidicv_error_t e = kleidicv::riscv_lk::validate_args(
          prev_data, prev_data_stride, prev_deriv_data, prev_deriv_stride,
          next_data, next_data_stride, width, height, channels, prev_points,
          next_points, point_count, window_width, window_height)) {
    return e;
  }
  if (point_count == 0) return KLEIDICV_OK;

  PatchBuffer buf(window_width, window_height, channels);
  if (!buf.valid()) return KLEIDICV_ERROR_ALLOCATION;

  return kleidicv::riscv_lk::lk_compute<RvvImpl>(
      buf.window(), buf.scharr_window(), prev_data, prev_data_stride,
      prev_deriv_data, prev_deriv_stride, next_data, next_data_stride, width,
      height, channels, prev_points, next_points, point_count, status, err,
      window_width, window_height, termination_count, termination_epsilon,
      get_min_eigen_vals, min_eigen_vals_threshold);
}

}  // namespace kleidicv::rvv
