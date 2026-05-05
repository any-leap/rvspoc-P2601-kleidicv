// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// 3x3 binomial Gaussian:
//   F = (1/16) * [1 2 1; 2 4 2; 1 2 1]   = (1/16) * [1,2,1] ⊗ [1,2,1]
// Replicate border. Rounding shift — matches upstream's `svrshr_x`.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "kleidicv/kleidicv.h"

#include "gaussian_blur_decls.h"

namespace kleidicv::scalar {

namespace {

inline size_t clip_replicate(ptrdiff_t v, size_t n) {
  if (v < 0) return 0;
  if (static_cast<size_t>(v) >= n) return n - 1;
  return static_cast<size_t>(v);
}

// Build a 1D Gaussian kernel of `ks` taps (odd) with the given sigma. If
// sigma <= 0 we fall back to OpenCV's default:
//   sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
// Coefficients are normalised to sum to 1.
void make_gaussian_kernel(size_t ks, float sigma, std::vector<float> &out) {
  out.resize(ks);
  if (sigma <= 0.0F) {
    sigma = 0.3F * (static_cast<float>(ks - 1) * 0.5F - 1.0F) + 0.8F;
  }
  const float center = static_cast<float>(ks - 1) * 0.5F;
  const float inv_2_sigma2 = 1.0F / (2.0F * sigma * sigma);
  float total = 0.0F;
  for (size_t i = 0; i < ks; ++i) {
    float d = static_cast<float>(i) - center;
    out[i] = std::exp(-d * d * inv_2_sigma2);
    total += out[i];
  }
  for (size_t i = 0; i < ks; ++i) out[i] /= total;
}

}  // namespace

kleidicv_error_t gaussian_blur_3x3_binomial_u8(const uint8_t *src,
                                               size_t src_stride, uint8_t *dst,
                                               size_t dst_stride, size_t width,
                                               size_t height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  auto cy = [&](ptrdiff_t y) -> size_t {
    if (y < 0) return 0;
    if (static_cast<size_t>(y) >= height) return height - 1;
    return static_cast<size_t>(y);
  };
  auto cx = [&](ptrdiff_t x) -> size_t {
    if (x < 0) return 0;
    if (static_cast<size_t>(x) >= width) return width - 1;
    return static_cast<size_t>(x);
  };

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *r_top = src + cy(static_cast<ptrdiff_t>(y) - 1) * src_stride;
    const uint8_t *r_mid = src + y * src_stride;
    const uint8_t *r_bot = src + cy(static_cast<ptrdiff_t>(y) + 1) * src_stride;
    uint8_t *rd = dst + y * dst_stride;
    for (size_t x = 0; x < width; ++x) {
      size_t xL = cx(static_cast<ptrdiff_t>(x) - 1);
      size_t xR = cx(static_cast<ptrdiff_t>(x) + 1);
      // sum = src[y-1][x-1] + 2*src[y-1][x] + src[y-1][x+1]
      //     + 2*(src[y][x-1] + 2*src[y][x] + src[y][x+1])
      //     +    src[y+1][x-1] + 2*src[y+1][x] + src[y+1][x+1]
      int top = static_cast<int>(r_top[xL]) + 2 * static_cast<int>(r_top[x]) +
                static_cast<int>(r_top[xR]);
      int mid = static_cast<int>(r_mid[xL]) + 2 * static_cast<int>(r_mid[x]) +
                static_cast<int>(r_mid[xR]);
      int bot = static_cast<int>(r_bot[xL]) + 2 * static_cast<int>(r_bot[x]) +
                static_cast<int>(r_bot[xR]);
      int total = top + 2 * mid + bot;
      // Rounding-divide by 16.
      rd[x] = static_cast<uint8_t>((total + 8) >> 4);
    }
  }
  return KLEIDICV_OK;
}

// Generic separable Gaussian for arbitrary odd kernel sizes and sigma.
// Two-pass: vertical → intermediate buffer → horizontal. f32 internally to
// keep the code simple; output saturates to u8.
kleidicv_error_t gaussian_blur_generic_u8(const uint8_t *src, size_t src_stride,
                                          uint8_t *dst, size_t dst_stride,
                                          size_t width, size_t height,
                                          size_t kernel_width,
                                          size_t kernel_height, float sigma_x,
                                          float sigma_y) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if ((kernel_width & 1u) == 0 || (kernel_height & 1u) == 0)
    return KLEIDICV_ERROR_RANGE;
  if (kernel_width < 3 || kernel_height < 3) return KLEIDICV_ERROR_RANGE;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  std::vector<float> kx_v, ky_v;
  make_gaussian_kernel(kernel_width, sigma_x, kx_v);
  make_gaussian_kernel(kernel_height, sigma_y, ky_v);
  const float *kx = kx_v.data();
  const float *ky = ky_v.data();
  const ptrdiff_t hx = static_cast<ptrdiff_t>(kernel_width) / 2;
  const ptrdiff_t hy = static_cast<ptrdiff_t>(kernel_height) / 2;

  // Vertical pass: tmp[y][x] = sum_{i} ky[i] * src[clip(y+i-hy)][x]
  std::vector<float> tmp(width * height);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      float acc = 0.0F;
      for (size_t i = 0; i < kernel_height; ++i) {
        size_t sy = clip_replicate(
            static_cast<ptrdiff_t>(y) + static_cast<ptrdiff_t>(i) - hy, height);
        acc += ky[i] * static_cast<float>(src[sy * src_stride + x]);
      }
      tmp[y * width + x] = acc;
    }
  }

  // Horizontal pass: dst[y][x] = sum_{i} kx[i] * tmp[y][clip(x+i-hx)]
  for (size_t y = 0; y < height; ++y) {
    uint8_t *rd = dst + y * dst_stride;
    const float *trow = tmp.data() + y * width;
    for (size_t x = 0; x < width; ++x) {
      float acc = 0.0F;
      for (size_t i = 0; i < kernel_width; ++i) {
        size_t sx = clip_replicate(
            static_cast<ptrdiff_t>(x) + static_cast<ptrdiff_t>(i) - hx, width);
        acc += kx[i] * trow[sx];
      }
      int iv = static_cast<int>(std::lround(acc));
      if (iv < 0) iv = 0;
      if (iv > 255) iv = 255;
      rd[x] = static_cast<uint8_t>(iv);
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::scalar
