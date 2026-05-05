// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Port of upstream `standalone_lucas_kanade_alg_common.h` for the RISC-V
// build. Drops the AArch64-only macros (KLEIDICV_TARGET_FN_ATTRS,
// KLEIDICV_STREAMING) and the SVE2 buffer dependency. Algorithm matches
// upstream so OpenCV-style fixed-point (FractionBits=14) results stay
// bit-exact across backends.

#ifndef KLEIDICV_RISCV_OPTICAL_FLOW_LK_COMMON_H
#define KLEIDICV_RISCV_OPTICAL_FLOW_LK_COMMON_H

#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::riscv_lk {

static constexpr int kFractionBits = 14;
static constexpr float kFixedPointDescale = (1.0F / (1 << 20));

template <int FractionBits>
inline int round_fixed_point(int x) {
  int half = 1 << (FractionBits - 1);
  return (x + half) >> FractionBits;
}

// Validation matching upstream.
inline kleidicv_error_t validate_args(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *scharr_data, size_t scharr_stride_bytes,
    const uint8_t *next_data, size_t next_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, int window_width, int window_height) {
  if (width <= 0 || height <= 0 || channels <= 0 || window_width <= 0 ||
      window_height <= 0) {
    return KLEIDICV_ERROR_RANGE;
  }
  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  if (!prev_data || !scharr_data || !next_data) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }
  if (point_count > 0 && (!prev_points || !next_points)) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }
  // KLEIDICV_MAX_IMAGE_PIXELS guard — reject before width*height*channels can
  // overflow size_t in the row-elements / scharr-row-bytes calculations
  // below (matches upstream test/api/test_standalone_lucas_kanade_alg.cpp's
  // INT_MAX coverage).
  {
    size_t pixels = 0;
    if (__builtin_mul_overflow(static_cast<size_t>(width),
                                 static_cast<size_t>(height), &pixels))
      return KLEIDICV_ERROR_RANGE;
    if (pixels > KLEIDICV_MAX_IMAGE_PIXELS) return KLEIDICV_ERROR_RANGE;
  }
  if ((scharr_stride_bytes % sizeof(int16_t)) != 0) {
    return KLEIDICV_ERROR_ALIGNMENT;
  }
  const size_t row_elems =
      static_cast<size_t>(width) * static_cast<size_t>(channels);
  const size_t image_row_bytes = row_elems * sizeof(uint8_t);
  if (prev_data_stride < image_row_bytes || next_stride < image_row_bytes) {
    return KLEIDICV_ERROR_RANGE;
  }
  const size_t scharr_row_bytes = row_elems * 2U * sizeof(int16_t);
  if (scharr_stride_bytes < scharr_row_bytes) {
    return KLEIDICV_ERROR_RANGE;
  }
  if (window_width > KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE ||
      window_height > KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE) {
    return KLEIDICV_ERROR_RANGE;
  }
  return KLEIDICV_OK;
}

template <int Shift>
inline std::array<int16_t, 4> get_lerp_params(float x, float y) {
  int16_t a =
      static_cast<int16_t>(rintf((1.0F - x) * (1.0F - y) * (1 << Shift)));
  int16_t b = static_cast<int16_t>(rintf(x * (1.0F - y) * (1 << Shift)));
  int16_t c = static_cast<int16_t>(rintf((1.0F - x) * y * (1 << Shift)));
  int16_t d = static_cast<int16_t>((1 << Shift) - a - b - c);
  return {a, b, c, d};
}

inline bool point_out_of_bounds(int x, int y, int width, int height,
                                int window_width, int window_height) {
  return x < -window_width || x >= width || y < -window_height || y >= height;
}

struct StructureTensor {
  float sum_scharr_xx = 0;
  float sum_scharr_xy = 0;
  float sum_scharr_yy = 0;
  float determinant = 0;
  float min_eigen_val = 0;
};

inline StructureTensor finish_structure_tensor(float xx, float xy, float yy,
                                                int window_width,
                                                int window_height) {
  StructureTensor t{};
  t.sum_scharr_xx = xx;
  t.sum_scharr_xy = xy;
  t.sum_scharr_yy = yy;
  t.determinant = xx * yy - xy * xy;
  t.min_eigen_val =
      (yy + xx - sqrtf((xx - yy) * (xx - yy) + 4.0F * xy * xy)) /
      static_cast<float>(2 * window_width * window_height);
  return t;
}

// Scalar implementation of the two SIMD primitives. The Impl template
// parameter on `compute()` lets RVV swap them out without touching the
// orchestration logic.
struct ScalarImpl {
  // Bilinear-interpolated patch + Scharr derivatives at fractional position
  // (coeff_*). Writes window/scharr_window with fixed-point samples and
  // accumulates xx/xy/yy gradient sums.
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

      for (int i = 0; i < row_elems; ++i) {
        // Patch sample (descale by FractionBits-5 → /512 of weighted sum).
        int p_tl = prev_row0[i];
        int p_tr = prev_row0[i + channels];
        int p_bl = prev_row1[i];
        int p_br = prev_row1[i + channels];
        int sum = p_tl * coeff_tl + p_tr * coeff_tr + p_bl * coeff_bl +
                  p_br * coeff_br;
        int16_t patch =
            static_cast<int16_t>(round_fixed_point<kFractionBits - 5>(sum));
        win_row[i] = patch;

        // Scharr x.
        int sx_tl = sch_row0[i * 2];
        int sx_tr = sch_row0[(i + channels) * 2];
        int sx_bl = sch_row1[i * 2];
        int sx_br = sch_row1[(i + channels) * 2];
        int sx_sum = sx_tl * coeff_tl + sx_tr * coeff_tr + sx_bl * coeff_bl +
                     sx_br * coeff_br;
        int16_t scharr_x = static_cast<int16_t>(
            round_fixed_point<kFractionBits>(sx_sum));

        // Scharr y.
        int sy_tl = sch_row0[i * 2 + 1];
        int sy_tr = sch_row0[(i + channels) * 2 + 1];
        int sy_bl = sch_row1[i * 2 + 1];
        int sy_br = sch_row1[(i + channels) * 2 + 1];
        int sy_sum = sy_tl * coeff_tl + sy_tr * coeff_tr + sy_bl * coeff_bl +
                     sy_br * coeff_br;
        int16_t scharr_y = static_cast<int16_t>(
            round_fixed_point<kFractionBits>(sy_sum));

        scw_row[i * 2] = scharr_x;
        scw_row[i * 2 + 1] = scharr_y;

        acc_xx += static_cast<int64_t>(scharr_x) * scharr_x;
        acc_xy += static_cast<int64_t>(scharr_x) * scharr_y;
        acc_yy += static_cast<int64_t>(scharr_y) * scharr_y;
      }
    }
    sum_xx = static_cast<float>(acc_xx) * kFixedPointDescale;
    sum_xy = static_cast<float>(acc_xy) * kFixedPointDescale;
    sum_yy = static_cast<float>(acc_yy) * kFixedPointDescale;
  }

  // Photometric-residual mismatch vector (b in the LK system).
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

      for (int i = 0; i < row_elems; ++i) {
        int p_tl = next_row0[i];
        int p_tr = next_row0[i + channels];
        int p_bl = next_row1[i];
        int p_br = next_row1[i + channels];
        int sum = p_tl * coeff_tl + p_tr * coeff_tr + p_bl * coeff_bl +
                  p_br * coeff_br;
        int sample = round_fixed_point<kFractionBits - 5>(sum);
        int diff = sample - win_row[i];
        int sx = scw_row[i * 2];
        int sy = scw_row[i * 2 + 1];
        acc_x += static_cast<int64_t>(sx) * diff;
        acc_y += static_cast<int64_t>(sy) * diff;
      }
    }
    sum_dx = static_cast<float>(acc_x) * kFixedPointDescale;
    sum_dy = static_cast<float>(acc_y) * kFixedPointDescale;
  }
};

// LK orchestration: per-point coarse-to-fine Newton iteration. The Impl
// type provides the two SIMD primitives.
template <typename Impl>
inline kleidicv_error_t lk_compute(
    int16_t *window, int16_t *scharr_window, const uint8_t *prev_data,
    size_t prev_data_stride, const int16_t *scharr_data,
    size_t scharr_stride_bytes, const uint8_t *next_data, size_t next_stride,
    int width, int height, int channels, const float *prev_points,
    float *next_points, size_t point_count, uint8_t *status, float *err,
    int window_width, int window_height, int termination_count,
    double termination_epsilon, bool get_min_eigen_vals,
    float min_eigen_vals_threshold) {
  const ptrdiff_t scharr_stride_elements =
      static_cast<ptrdiff_t>(scharr_stride_bytes / sizeof(int16_t));
  const float half_w = static_cast<float>(window_width - 1) * 0.5F;
  const float half_h = static_cast<float>(window_height - 1) * 0.5F;

  for (size_t pi = 0; pi < point_count; ++pi) {
    const float prev_x = prev_points[pi * 2] - half_w;
    const float prev_y = prev_points[pi * 2 + 1] - half_h;
    const int prev_xi = static_cast<int>(floorf(prev_x));
    const int prev_yi = static_cast<int>(floorf(prev_y));

    if (point_out_of_bounds(prev_xi, prev_yi, width, height, window_width,
                             window_height)) {
      if (status) status[pi] = 0;
      continue;
    }

    const auto [coeff_tl, coeff_tr, coeff_bl, coeff_br] =
        get_lerp_params<kFractionBits>(prev_x - static_cast<float>(prev_xi),
                                        prev_y - static_cast<float>(prev_yi));

    float xx = 0, xy = 0, yy = 0;
    Impl::sample_patch_and_gradients(
        window, scharr_window, prev_data,
        static_cast<ptrdiff_t>(prev_data_stride), scharr_data,
        scharr_stride_elements, channels, prev_xi, prev_yi, window_width,
        window_height, coeff_tl, coeff_tr, coeff_bl, coeff_br, xx, xy, yy);
    StructureTensor tensor =
        finish_structure_tensor(xx, xy, yy, window_width, window_height);

    if (err && get_min_eigen_vals) err[pi] = tensor.min_eigen_val;

    if (tensor.min_eigen_val < min_eigen_vals_threshold ||
        tensor.determinant < FLT_EPSILON) {
      if (status) status[pi] = 0;
      continue;
    }

    const float inv_det = 1.0F / tensor.determinant;

    float next_x = next_points[pi * 2] - half_w;
    float next_y = next_points[pi * 2 + 1] - half_h;
    float prev_vx = 0, prev_vy = 0;
    bool point_ok = true;

    for (int j = 0; j < termination_count; ++j) {
      const int next_xi = static_cast<int>(floorf(next_x));
      const int next_yi = static_cast<int>(floorf(next_y));
      if (point_out_of_bounds(next_xi, next_yi, width, height, window_width,
                               window_height)) {
        point_ok = false;
        break;
      }
      const auto [c_tl, c_tr, c_bl, c_br] = get_lerp_params<kFractionBits>(
          next_x - static_cast<float>(next_xi),
          next_y - static_cast<float>(next_yi));

      float sdx = 0, sdy = 0;
      Impl::accumulate_mismatch_vector(
          next_data, static_cast<ptrdiff_t>(next_stride), window, scharr_window,
          channels, next_xi, next_yi, window_width, window_height, c_tl, c_tr,
          c_bl, c_br, sdx, sdy);

      const float vx = (xy * sdy - yy * sdx) * inv_det;
      const float vy = (xy * sdx - xx * sdy) * inv_det;
      next_x += vx;
      next_y += vy;
      next_points[pi * 2] = next_x + half_w;
      next_points[pi * 2 + 1] = next_y + half_h;

      if (vx * vx + vy * vy <= termination_epsilon) break;

      if (j != 0 && fabsf(vx + prev_vx) < 0.01F &&
          fabsf(vy + prev_vy) < 0.01F) {
        next_points[pi * 2] -= vx * 0.5F;
        next_points[pi * 2 + 1] -= vy * 0.5F;
        break;
      }
      prev_vx = vx;
      prev_vy = vy;
    }

    if (!point_ok) {
      if (status) status[pi] = 0;
      continue;
    }

    if (!status || status[pi]) {
      float nx = next_points[pi * 2] - half_w;
      float ny = next_points[pi * 2 + 1] - half_h;
      const int nxi = static_cast<int>(floorf(nx));
      const int nyi = static_cast<int>(floorf(ny));
      if (point_out_of_bounds(nxi, nyi, width, height, window_width,
                               window_height)) {
        if (status) status[pi] = 0;
      } else if (err && !get_min_eigen_vals) {
        const auto [c_tl, c_tr, c_bl, c_br] = get_lerp_params<kFractionBits>(
            nx - static_cast<float>(nxi), ny - static_cast<float>(nyi));
        float errval = 0;
        for (int y = 0; y < window_height; ++y) {
          int16_t *win_row =
              window + static_cast<ptrdiff_t>(y) * window_width * channels;
          const uint8_t *nrow0 =
              next_data + (y + nyi) * static_cast<ptrdiff_t>(next_stride) +
              static_cast<ptrdiff_t>(nxi) * channels;
          const uint8_t *nrow1 = nrow0 + next_stride;
          for (int x = 0; x < window_width * channels; ++x) {
            int s = nrow0[x] * c_tl + nrow0[x + channels] * c_tr +
                    nrow1[x] * c_bl + nrow1[x + channels] * c_br;
            int diff = round_fixed_point<kFractionBits - 5>(s) - win_row[x];
            errval += fabsf(static_cast<float>(diff));
          }
        }
        err[pi] = errval /
                  static_cast<float>(32L * window_width * channels * window_height);
      }
    }
  }
  return KLEIDICV_OK;
}

// Allocates the two scratch buffers (window + scharr_window). Stack-allocates
// when small enough; falls back to malloc otherwise.
class PatchBuffer {
 public:
  PatchBuffer(int window_width, int window_height, int channels) {
    patch_size_ = static_cast<size_t>(window_width) *
                   static_cast<size_t>(window_height) *
                   static_cast<size_t>(channels);
    const size_t total = patch_size_ * 3U;
    if (total <= kStackElems) {
      window_ = stack_;
    } else {
      window_ = static_cast<int16_t *>(std::malloc(total * sizeof(int16_t)));
      heap_ = window_;
    }
  }
  PatchBuffer(const PatchBuffer &) = delete;
  PatchBuffer &operator=(const PatchBuffer &) = delete;
  ~PatchBuffer() {
    if (heap_) std::free(heap_);
  }
  int16_t *window() { return window_; }
  int16_t *scharr_window() { return window_ ? window_ + patch_size_ : nullptr; }
  bool valid() const { return window_ != nullptr; }

 private:
  // 21x21 single-channel default window fits on stack.
  static constexpr size_t kStackElems = 21UL * 21UL * 3UL;
  int16_t stack_[kStackElems];
  int16_t *window_ = nullptr;
  int16_t *heap_ = nullptr;
  size_t patch_size_ = 0;
};

}  // namespace kleidicv::riscv_lk

#endif  // KLEIDICV_RISCV_OPTICAL_FLOW_LK_COMMON_H
