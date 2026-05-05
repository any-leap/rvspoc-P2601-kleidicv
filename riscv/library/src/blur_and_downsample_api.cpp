// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "blur_and_downsample_decls.h"
#include "dispatch.h"
#include "multichannel_helper.h"

#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"

namespace {
using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;

using BdsKernel = kleidicv_error_t (*)(const uint8_t *, size_t, size_t, size_t,
                                        uint8_t *, size_t);

// Multi-channel: deinterleave into planar src, run channels=1
// blur_and_downsample on each plane (output dims are halved per axis), then
// reinterleave the smaller dst.
kleidicv_error_t mc_bds(BdsKernel kernel, const uint8_t *src,
                          size_t src_stride, size_t src_width,
                          size_t src_height, uint8_t *dst, size_t dst_stride,
                          size_t channels) {
  const size_t dst_w = (src_width + 1) / 2;
  const size_t dst_h = (src_height + 1) / 2;
  std::vector<uint8_t> sp_storage(src_width * src_height * channels);
  std::vector<uint8_t> dp_storage(dst_w * dst_h * channels);
  uint8_t *sp[4] = {nullptr, nullptr, nullptr, nullptr};
  uint8_t *dp[4] = {nullptr, nullptr, nullptr, nullptr};
  for (size_t c = 0; c < channels; ++c) {
    sp[c] = sp_storage.data() + c * src_width * src_height;
    dp[c] = dp_storage.data() + c * dst_w * dst_h;
  }
  for (size_t y = 0; y < src_height; ++y) {
    uint8_t *row[4] = {sp[0] + y * src_width, sp[1] + y * src_width,
                       sp[2] + y * src_width, sp[3] + y * src_width};
    kleidicv::riscv_mc::deinterleave_row_u8(src + y * src_stride, row,
                                              src_width, channels);
  }
  for (size_t c = 0; c < channels; ++c) {
    kleidicv_error_t e = kernel(sp[c], src_width, src_width, src_height,
                                  dp[c], dst_w);
    if (e != KLEIDICV_OK) return e;
  }
  for (size_t y = 0; y < dst_h; ++y) {
    const uint8_t *row[4] = {dp[0] + y * dst_w, dp[1] + y * dst_w,
                             dp[2] + y * dst_w, dp[3] + y * dst_w};
    kleidicv::riscv_mc::interleave_row_u8(dst + y * dst_stride, row, dst_w,
                                            channels);
  }
  return KLEIDICV_OK;
}

}  // namespace

extern "C" kleidicv_error_t kleidicv_blur_and_downsample_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels,
    kleidicv_border_type_t border_type) {
  if (channels < 1 || channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  // The LK pyramid build passes REVERSE (reflect_101); the pyramid pre-fills
  // border pixels with reflect_101 data, so the kernel's internal REPLICATE
  // clipping never reaches outside the valid range. Accept both border modes.
  if (border_type != KLEIDICV_BORDER_TYPE_REPLICATE &&
      border_type != KLEIDICV_BORDER_TYPE_REVERSE)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  BdsKernel kernel = active_backend() == Backend::Rvv
                          ? &kleidicv::rvv::blur_and_downsample_u8
                          : &kleidicv::scalar::blur_and_downsample_u8;
  if (channels == 1) {
    return kernel(src, src_stride, src_width, src_height, dst, dst_stride);
  }
  return mc_bds(kernel, src, src_stride, src_width, src_height, dst,
                  dst_stride, channels);
}

extern "C" kleidicv_error_t kleidicv_blur_and_downsample_u8_sme(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels,
    kleidicv_border_type_t border_type) {
  return kleidicv_blur_and_downsample_u8(src, src_stride, src_width, src_height,
                                         dst, dst_stride, channels,
                                         border_type);
}
