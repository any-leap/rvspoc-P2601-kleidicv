// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Public API for separable_filter_2d (5×5 only) and gaussian_blur (3×3
// binomial only). Both pass-throughs accept channels∈{1,2,3,4}: channels==1
// hits the existing fast path, channels>1 deinterleaves with vlsegN, runs
// the channels=1 kernel on each plane, and reinterleaves with vssegN.

#include <cstdint>
#include <vector>

#include "dispatch.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"

#include "gaussian_blur_decls.h"
#include "multichannel_helper.h"
#include "separable_filter_2d_decls.h"

namespace {

using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;

// 5×5 u8 multi-channel wrap (kernel stride is just kernel_x/y unchanged).
template <typename Kernel>
kleidicv_error_t mc_sep5_u8(Kernel kernel, const uint8_t *src,
                              size_t src_stride, uint8_t *dst, size_t dst_stride,
                              size_t width, size_t height, size_t channels,
                              const uint8_t *kx, const uint8_t *ky) {
  std::vector<uint8_t> sp_storage(width * height * channels);
  std::vector<uint8_t> dp_storage(width * height * channels);
  uint8_t *sp[4] = {nullptr, nullptr, nullptr, nullptr};
  uint8_t *dp[4] = {nullptr, nullptr, nullptr, nullptr};
  for (size_t c = 0; c < channels; ++c) {
    sp[c] = sp_storage.data() + c * width * height;
    dp[c] = dp_storage.data() + c * width * height;
  }
  for (size_t y = 0; y < height; ++y) {
    uint8_t *row[4] = {sp[0] + y * width, sp[1] + y * width,
                       sp[2] + y * width, sp[3] + y * width};
    kleidicv::riscv_mc::deinterleave_row_u8(src + y * src_stride, row, width,
                                              channels);
  }
  for (size_t c = 0; c < channels; ++c) {
    kleidicv_error_t e =
        kernel(sp[c], width, dp[c], width, width, height, kx, ky);
    if (e != KLEIDICV_OK) return e;
  }
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *row[4] = {dp[0] + y * width, dp[1] + y * width,
                             dp[2] + y * width, dp[3] + y * width};
    kleidicv::riscv_mc::interleave_row_u8(dst + y * dst_stride, row, width,
                                            channels);
  }
  return KLEIDICV_OK;
}

template <typename Kernel>
kleidicv_error_t mc_sep5_u16(Kernel kernel, const uint16_t *src,
                               size_t src_stride, uint16_t *dst,
                               size_t dst_stride, size_t width, size_t height,
                               size_t channels, const uint16_t *kx,
                               const uint16_t *ky) {
  const size_t src_se = src_stride / sizeof(uint16_t);
  const size_t dst_se = dst_stride / sizeof(uint16_t);
  std::vector<uint16_t> sp_storage(width * height * channels);
  std::vector<uint16_t> dp_storage(width * height * channels);
  uint16_t *sp[4] = {nullptr, nullptr, nullptr, nullptr};
  uint16_t *dp[4] = {nullptr, nullptr, nullptr, nullptr};
  for (size_t c = 0; c < channels; ++c) {
    sp[c] = sp_storage.data() + c * width * height;
    dp[c] = dp_storage.data() + c * width * height;
  }
  for (size_t y = 0; y < height; ++y) {
    uint16_t *row[4] = {sp[0] + y * width, sp[1] + y * width,
                        sp[2] + y * width, sp[3] + y * width};
    kleidicv::riscv_mc::deinterleave_row_u16(src + y * src_se, row, width,
                                               channels);
  }
  for (size_t c = 0; c < channels; ++c) {
    kleidicv_error_t e = kernel(sp[c], width * sizeof(uint16_t), dp[c],
                                  width * sizeof(uint16_t), width, height,
                                  kx, ky);
    if (e != KLEIDICV_OK) return e;
  }
  for (size_t y = 0; y < height; ++y) {
    const uint16_t *row[4] = {dp[0] + y * width, dp[1] + y * width,
                              dp[2] + y * width, dp[3] + y * width};
    kleidicv::riscv_mc::interleave_row_u16(dst + y * dst_se, row, width,
                                             channels);
  }
  return KLEIDICV_OK;
}

template <typename Kernel>
kleidicv_error_t mc_gauss3_u8(Kernel kernel, const uint8_t *src,
                                size_t src_stride, uint8_t *dst,
                                size_t dst_stride, size_t width, size_t height,
                                size_t channels) {
  std::vector<uint8_t> sp_storage(width * height * channels);
  std::vector<uint8_t> dp_storage(width * height * channels);
  uint8_t *sp[4] = {nullptr, nullptr, nullptr, nullptr};
  uint8_t *dp[4] = {nullptr, nullptr, nullptr, nullptr};
  for (size_t c = 0; c < channels; ++c) {
    sp[c] = sp_storage.data() + c * width * height;
    dp[c] = dp_storage.data() + c * width * height;
  }
  for (size_t y = 0; y < height; ++y) {
    uint8_t *row[4] = {sp[0] + y * width, sp[1] + y * width,
                       sp[2] + y * width, sp[3] + y * width};
    kleidicv::riscv_mc::deinterleave_row_u8(src + y * src_stride, row, width,
                                              channels);
  }
  for (size_t c = 0; c < channels; ++c) {
    kleidicv_error_t e = kernel(sp[c], width, dp[c], width, width, height);
    if (e != KLEIDICV_OK) return e;
  }
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *row[4] = {dp[0] + y * width, dp[1] + y * width,
                             dp[2] + y * width, dp[3] + y * width};
    kleidicv::riscv_mc::interleave_row_u8(dst + y * dst_stride, row, width,
                                            channels);
  }
  return KLEIDICV_OK;
}

}  // namespace

extern "C" kleidicv_error_t kleidicv_separable_filter_2d_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint8_t *kernel_x,
    size_t kernel_width, const uint8_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type) {
  if (channels < 1 || channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (kernel_width != 5 || kernel_height != 5)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (border_type != KLEIDICV_BORDER_TYPE_REPLICATE)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  auto kernel = active_backend() == Backend::Rvv
                    ? &kleidicv::rvv::separable_filter_2d_5x5_u8
                    : &kleidicv::scalar::separable_filter_2d_5x5_u8;
  if (channels == 1) {
    return kernel(src, src_stride, dst, dst_stride, width, height, kernel_x,
                  kernel_y);
  }
  return mc_sep5_u8(kernel, src, src_stride, dst, dst_stride, width, height,
                     channels, kernel_x, kernel_y);
}
extern "C" kleidicv_error_t kleidicv_separable_filter_2d_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint8_t *kernel_x,
    size_t kernel_width, const uint8_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type) {
  return kleidicv_separable_filter_2d_u8(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_x,
      kernel_width, kernel_y, kernel_height, border_type);
}

extern "C" kleidicv_error_t kleidicv_separable_filter_2d_u16(
    const uint16_t *src, size_t src_stride, uint16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint16_t *kernel_x,
    size_t kernel_width, const uint16_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type) {
  if (channels < 1 || channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (kernel_width != 5 || kernel_height != 5)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (border_type != KLEIDICV_BORDER_TYPE_REPLICATE)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  auto kernel = active_backend() == Backend::Rvv
                    ? &kleidicv::rvv::separable_filter_2d_5x5_u16
                    : &kleidicv::scalar::separable_filter_2d_5x5_u16;
  if (channels == 1) {
    return kernel(src, src_stride, dst, dst_stride, width, height, kernel_x,
                  kernel_y);
  }
  return mc_sep5_u16(kernel, src, src_stride, dst, dst_stride, width, height,
                      channels, kernel_x, kernel_y);
}
extern "C" kleidicv_error_t kleidicv_separable_filter_2d_u16_sme(
    const uint16_t *src, size_t src_stride, uint16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint16_t *kernel_x,
    size_t kernel_width, const uint16_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type) {
  return kleidicv_separable_filter_2d_u16(
      src, src_stride, dst, dst_stride, width, height, channels, kernel_x,
      kernel_width, kernel_y, kernel_height, border_type);
}

extern "C" kleidicv_error_t kleidicv_gaussian_blur_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, float sigma_x, float sigma_y,
    kleidicv_border_type_t border_type) {
  if (channels < 1 || channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (border_type != KLEIDICV_BORDER_TYPE_REPLICATE)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (kernel_width != 3 || kernel_height != 3)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (sigma_x != 0.0f || sigma_y != 0.0f)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  auto kernel = active_backend() == Backend::Rvv
                    ? &kleidicv::rvv::gaussian_blur_3x3_binomial_u8
                    : &kleidicv::scalar::gaussian_blur_3x3_binomial_u8;
  if (channels == 1) {
    return kernel(src, src_stride, dst, dst_stride, width, height);
  }
  return mc_gauss3_u8(kernel, src, src_stride, dst, dst_stride, width, height,
                       channels);
}
extern "C" kleidicv_error_t kleidicv_gaussian_blur_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, float sigma_x, float sigma_y,
    kleidicv_border_type_t border_type) {
  return kleidicv_gaussian_blur_u8(src, src_stride, dst, dst_stride, width,
                                   height, channels, kernel_width,
                                   kernel_height, sigma_x, sigma_y,
                                   border_type);
}
