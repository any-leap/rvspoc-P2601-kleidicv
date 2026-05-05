// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// channels==1: existing fast path. channels∈{2,3,4}: deinterleave the source
// into planar scratch (vlsegN), run the channels=1 sobel kernel on each
// plane, reinterleave the s16 output (vssegN). Output layout per pixel for
// C-channel input is [s16_c0, s16_c1, …, s16_{C-1}].

#include <vector>

#include "dispatch.h"
#include "multichannel_helper.h"
#include "sobel_decls.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;

using SobelKernel = kleidicv_error_t (*)(const uint8_t *, size_t, int16_t *,
                                          size_t, size_t, size_t);

kleidicv_error_t mc_sobel_wrap(SobelKernel kernel, const uint8_t *src,
                                size_t src_stride, int16_t *dst,
                                size_t dst_stride, size_t width, size_t height,
                                size_t channels) {
  std::vector<uint8_t> src_planes(width * height * channels);
  std::vector<int16_t> dst_planes(width * height * channels);
  uint8_t *sp[4] = {nullptr, nullptr, nullptr, nullptr};
  int16_t *dp[4] = {nullptr, nullptr, nullptr, nullptr};
  for (size_t c = 0; c < channels; ++c) {
    sp[c] = src_planes.data() + c * width * height;
    dp[c] = dst_planes.data() + c * width * height;
  }
  for (size_t y = 0; y < height; ++y) {
    uint8_t *row_dst[4] = {sp[0] + y * width, sp[1] + y * width,
                           sp[2] + y * width, sp[3] + y * width};
    kleidicv::riscv_mc::deinterleave_row_u8(src + y * src_stride, row_dst,
                                              width, channels);
  }
  for (size_t c = 0; c < channels; ++c) {
    kleidicv_error_t e = kernel(sp[c], width, dp[c],
                                  width * sizeof(int16_t), width, height);
    if (e != KLEIDICV_OK) return e;
  }
  for (size_t y = 0; y < height; ++y) {
    const int16_t *row_src[4] = {dp[0] + y * width, dp[1] + y * width,
                                  dp[2] + y * width, dp[3] + y * width};
    int16_t *drow = reinterpret_cast<int16_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    kleidicv::riscv_mc::interleave_row_s16(drow, row_src, width, channels);
  }
  return KLEIDICV_OK;
}

}  // namespace

extern "C" kleidicv_error_t kleidicv_sobel_3x3_horizontal_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels) {
  if (channels < 1 || channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  SobelKernel kernel = active_backend() == Backend::Rvv
                            ? &kleidicv::rvv::sobel_3x3_horizontal_s16_u8
                            : &kleidicv::scalar::sobel_3x3_horizontal_s16_u8;
  if (channels == 1) {
    return kernel(src, src_stride, dst, dst_stride, width, height);
  }
  return mc_sobel_wrap(kernel, src, src_stride, dst, dst_stride, width, height,
                        channels);
}

extern "C" kleidicv_error_t kleidicv_sobel_3x3_horizontal_s16_u8_sme(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels) {
  return kleidicv_sobel_3x3_horizontal_s16_u8(src, src_stride, dst, dst_stride,
                                              width, height, channels);
}

extern "C" kleidicv_error_t kleidicv_sobel_3x3_vertical_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels) {
  if (channels < 1 || channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  SobelKernel kernel = active_backend() == Backend::Rvv
                            ? &kleidicv::rvv::sobel_3x3_vertical_s16_u8
                            : &kleidicv::scalar::sobel_3x3_vertical_s16_u8;
  if (channels == 1) {
    return kernel(src, src_stride, dst, dst_stride, width, height);
  }
  return mc_sobel_wrap(kernel, src, src_stride, dst, dst_stride, width, height,
                        channels);
}

extern "C" kleidicv_error_t kleidicv_sobel_3x3_vertical_s16_u8_sme(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels) {
  return kleidicv_sobel_3x3_vertical_s16_u8(src, src_stride, dst, dst_stride,
                                            width, height, channels);
}
