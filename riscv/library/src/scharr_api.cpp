// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Scharr (3×3, valid output, no border). Output per pixel for C-channel input
// is 2*C int16 values laid out [dx_c0, dy_c0, dx_c1, dy_c1, …].
//
// channels==1: existing fast path.
// channels∈{2,3,4}: deinterleave src → run channels=1 scharr per plane →
// reinterleave the per-channel (dx, dy) pairs into the multi-channel
// interleaved layout.

#include <vector>

#include "dispatch.h"
#include "multichannel_helper.h"
#include "scharr_decls.h"

#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;

using ScharrKernel = kleidicv_error_t (*)(const uint8_t *, size_t, size_t,
                                           size_t, int16_t *, size_t);

kleidicv_error_t mc_scharr_wrap(ScharrKernel kernel, const uint8_t *src,
                                  size_t src_stride, size_t width,
                                  size_t height, size_t channels, int16_t *dst,
                                  size_t dst_stride) {
  if (width < 3 || height < 3) return KLEIDICV_OK;
  const size_t out_w = width - 2;
  const size_t out_h = height - 2;
  std::vector<uint8_t> src_planes(width * height * channels);
  // Each output plane holds 2 int16 per pixel (dx, dy) → out_w * 2 * out_h.
  std::vector<int16_t> dst_planes(out_w * out_h * 2 * channels);
  uint8_t *sp[4] = {nullptr, nullptr, nullptr, nullptr};
  int16_t *dp[4] = {nullptr, nullptr, nullptr, nullptr};
  for (size_t c = 0; c < channels; ++c) {
    sp[c] = src_planes.data() + c * width * height;
    dp[c] = dst_planes.data() + c * out_w * out_h * 2;
  }
  for (size_t y = 0; y < height; ++y) {
    uint8_t *row_dst[4] = {sp[0] + y * width, sp[1] + y * width,
                           sp[2] + y * width, sp[3] + y * width};
    kleidicv::riscv_mc::deinterleave_row_u8(src + y * src_stride, row_dst,
                                              width, channels);
  }
  for (size_t c = 0; c < channels; ++c) {
    kleidicv_error_t e = kernel(sp[c], width, width, height, dp[c],
                                  out_w * 2 * sizeof(int16_t));
    if (e != KLEIDICV_OK) return e;
  }
  // Reinterleave: per output pixel, the C planes each contributed (dx, dy);
  // we want [dx_c0, dy_c0, dx_c1, dy_c1, …] per pixel.
  for (size_t y = 0; y < out_h; ++y) {
    int16_t *drow = reinterpret_cast<int16_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    const int16_t *plane_pair_rows[4] = {
        dp[0] + y * out_w * 2, dp[1] + y * out_w * 2,
        dp[2] + y * out_w * 2, dp[3] + y * out_w * 2};
    kleidicv::riscv_mc::interleave_row_s16_pairs(drow, plane_pair_rows, out_w,
                                                  channels);
  }
  return KLEIDICV_OK;
}

}  // namespace

extern "C" kleidicv_error_t kleidicv_scharr_interleaved_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride) {
  if (src_channels < 1 || src_channels > 4)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  ScharrKernel kernel = active_backend() == Backend::Rvv
                            ? &kleidicv::rvv::scharr_interleaved_s16_u8
                            : &kleidicv::scalar::scharr_interleaved_s16_u8;
  if (src_channels == 1) {
    return kernel(src, src_stride, src_width, src_height, dst, dst_stride);
  }
  return mc_scharr_wrap(kernel, src, src_stride, src_width, src_height,
                         src_channels, dst, dst_stride);
}

extern "C" kleidicv_error_t kleidicv_scharr_interleaved_s16_u8_sme(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride) {
  return kleidicv_scharr_interleaved_s16_u8(src, src_stride, src_width,
                                            src_height, src_channels, dst,
                                            dst_stride);
}
