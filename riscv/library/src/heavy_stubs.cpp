// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// NOT_IMPLEMENTED stubs for the Bucket E + F heavyweight ops the SPOC has not
// yet ported. We only define the symbols our smoke tests reference. Other
// public-header function-pointer declarations are external linkage but
// don't need a definition unless code dereferences them.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dispatch.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"
#include "morphology_decls.h"
#include "multichannel_helper.h"

namespace {

using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;

kleidicv_error_t morph_u8_single_channel(const uint8_t *src, size_t src_stride,
                                          uint8_t *dst, size_t dst_stride,
                                          size_t width, size_t height,
                                          size_t kw, size_t kh,
                                          size_t iterations, bool is_dilate) {
  auto run = [&](const uint8_t *s, size_t ss, uint8_t *d, size_t ds) {
    return active_backend() == Backend::Rvv
               ? kleidicv::rvv::morph_u8(s, ss, d, ds, width, height, kw, kh,
                                          is_dilate)
               : kleidicv::scalar::morph_u8(s, ss, d, ds, width, height, kw,
                                             kh, is_dilate);
  };
  if (iterations == 1) return run(src, src_stride, dst, dst_stride);
  std::vector<uint8_t> scratch(width * height);
  size_t scratch_stride = width;
  kleidicv_error_t e = run(src, src_stride, dst, dst_stride);
  if (e != KLEIDICV_OK) return e;
  for (size_t i = 1; i < iterations; ++i) {
    if (i & 1) {
      e = run(dst, dst_stride, scratch.data(), scratch_stride);
    } else {
      e = run(scratch.data(), scratch_stride, dst, dst_stride);
    }
    if (e != KLEIDICV_OK) return e;
  }
  if ((iterations - 1) & 1) {
    for (size_t y = 0; y < height; ++y) {
      const uint8_t *srow = scratch.data() + y * scratch_stride;
      uint8_t *drow = dst + y * dst_stride;
      for (size_t x = 0; x < width; ++x) drow[x] = srow[x];
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t do_morph_u8(const uint8_t *src, size_t src_stride,
                             uint8_t *dst, size_t dst_stride, size_t width,
                             size_t height, size_t channels, size_t kw,
                             size_t kh, size_t anchor_x, size_t anchor_y,
                             kleidicv_border_type_t border,
                             const uint8_t * /*border_value*/,
                             size_t iterations, bool is_dilate) {
  // Round-7 fix: hoist null + image-size validation above the channel
  // dispatch so multi-channel calls reject null inputs / oversized images
  // before the deinterleave path dereferences src.
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels < 1 || channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  {
    size_t pixels = 0;
    if (__builtin_mul_overflow(width, height, &pixels) ||
        pixels > KLEIDICV_MAX_IMAGE_PIXELS)
      return KLEIDICV_ERROR_RANGE;
  }
  if (border != KLEIDICV_BORDER_TYPE_REPLICATE)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  if (iterations == 0) return KLEIDICV_OK;
  if (anchor_x != kw / 2 || anchor_y != kh / 2)
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;

  if (channels == 1) {
    return morph_u8_single_channel(src, src_stride, dst, dst_stride, width,
                                     height, kw, kh, iterations, is_dilate);
  }

  // Multi-channel: deinterleave src, run per plane, reinterleave dst.
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
    kleidicv_error_t e = morph_u8_single_channel(
        sp[c], width, dp[c], width, width, height, kw, kh, iterations,
        is_dilate);
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

kleidicv_error_t dilate_u8_dispatch(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, size_t anchor_x, size_t anchor_y,
    kleidicv_border_type_t border_type, const uint8_t *border_value,
    size_t iterations) {
  return do_morph_u8(src, src_stride, dst, dst_stride, width, height, channels,
                     kernel_width, kernel_height, anchor_x, anchor_y,
                     border_type, border_value, iterations, /*is_dilate=*/true);
}

kleidicv_error_t erode_u8_dispatch(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, size_t anchor_x, size_t anchor_y,
    kleidicv_border_type_t border_type, const uint8_t *border_value,
    size_t iterations) {
  return do_morph_u8(src, src_stride, dst, dst_stride, width, height, channels,
                     kernel_width, kernel_height, anchor_x, anchor_y,
                     border_type, border_value, iterations,
                     /*is_dilate=*/false);
}

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_dilate_u8)(
    const uint8_t *, size_t, uint8_t *, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, kleidicv_border_type_t, const uint8_t *,
    size_t) = dilate_u8_dispatch;
kleidicv_error_t (*kleidicv_dilate_u8_sme)(
    const uint8_t *, size_t, uint8_t *, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, kleidicv_border_type_t, const uint8_t *,
    size_t) = dilate_u8_dispatch;
kleidicv_error_t (*kleidicv_erode_u8)(
    const uint8_t *, size_t, uint8_t *, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, kleidicv_border_type_t, const uint8_t *,
    size_t) = erode_u8_dispatch;
kleidicv_error_t (*kleidicv_erode_u8_sme)(
    const uint8_t *, size_t, uint8_t *, size_t, size_t, size_t, size_t, size_t,
    size_t, size_t, size_t, kleidicv_border_type_t, const uint8_t *,
    size_t) = erode_u8_dispatch;

// resize_linear: real scalar bilinear, channels=1. RVV optimisation TBD.
}  // extern "C" — close so we can include header at namespace scope
#include "resize_linear_decls.h"
extern "C" {
kleidicv_error_t kleidicv_resize_linear_u8(const uint8_t *src, size_t src_stride,
                                           size_t src_width, size_t src_height,
                                           uint8_t *dst, size_t dst_stride,
                                           size_t dst_width, size_t dst_height,
                                           size_t channels) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels != 1) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  // Reject obviously-overflowing or too-tight strides up front (matches
  // upstream public-API semantics — bilinear walk past the row end is
  // KLEIDICV_ERROR_RANGE).
  if (src_height > 1 && src_stride < src_width) return KLEIDICV_ERROR_RANGE;
  if (dst_height > 1 && dst_stride < dst_width) return KLEIDICV_ERROR_RANGE;
  {
    size_t pixels = 0;
    if (__builtin_mul_overflow(src_width, src_height, &pixels) ||
        pixels > KLEIDICV_MAX_IMAGE_PIXELS)
      return KLEIDICV_ERROR_RANGE;
    if (__builtin_mul_overflow(dst_width, dst_height, &pixels) ||
        pixels > KLEIDICV_MAX_IMAGE_PIXELS)
      return KLEIDICV_ERROR_RANGE;
  }
  return active_backend() == Backend::Rvv
             ? kleidicv::rvv::resize_linear_u8(src, src_stride, src_width,
                                                src_height, dst, dst_stride,
                                                dst_width, dst_height)
             : kleidicv::scalar::resize_linear_u8(src, src_stride, src_width,
                                                   src_height, dst, dst_stride,
                                                   dst_width, dst_height);
}
kleidicv_error_t kleidicv_resize_linear_u8_sme(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels) {
  return kleidicv_resize_linear_u8(src, src_stride, src_width, src_height, dst,
                                   dst_stride, dst_width, dst_height,
                                   channels);
}
kleidicv_error_t kleidicv_resize_linear_f32(const float *src, size_t src_stride,
                                            size_t src_width, size_t src_height,
                                            float *dst, size_t dst_stride,
                                            size_t dst_width,
                                            size_t dst_height,
                                            size_t channels) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels != 1) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  // f32 strides are byte-strides per the public API; need to be a multiple
  // of sizeof(float) so the row-walk stride math is exact.
  if (src_height > 1 && (src_stride % sizeof(float)) != 0)
    return KLEIDICV_ERROR_ALIGNMENT;
  if (dst_height > 1 && (dst_stride % sizeof(float)) != 0)
    return KLEIDICV_ERROR_ALIGNMENT;
  if (src_height > 1 && src_stride < src_width * sizeof(float))
    return KLEIDICV_ERROR_RANGE;
  if (dst_height > 1 && dst_stride < dst_width * sizeof(float))
    return KLEIDICV_ERROR_RANGE;
  {
    size_t pixels = 0;
    if (__builtin_mul_overflow(src_width, src_height, &pixels) ||
        pixels > KLEIDICV_MAX_IMAGE_PIXELS)
      return KLEIDICV_ERROR_RANGE;
    if (__builtin_mul_overflow(dst_width, dst_height, &pixels) ||
        pixels > KLEIDICV_MAX_IMAGE_PIXELS)
      return KLEIDICV_ERROR_RANGE;
  }
  return active_backend() == Backend::Rvv
             ? kleidicv::rvv::resize_linear_f32(src, src_stride, src_width,
                                                 src_height, dst, dst_stride,
                                                 dst_width, dst_height)
             : kleidicv::scalar::resize_linear_f32(src, src_stride, src_width,
                                                    src_height, dst, dst_stride,
                                                    dst_width, dst_height);
}
kleidicv_error_t kleidicv_resize_linear_f32_sme(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    float *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels) {
  return kleidicv_resize_linear_f32(src, src_stride, src_width, src_height,
                                    dst, dst_stride, dst_width, dst_height,
                                    channels);
}

// warp_perspective lives in its own translation unit (warp_perspective.cpp)
// so we keep the rvv path and dispatch glue out of this file.
#if 0
kleidicv_error_t kleidicv_warp_perspective_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const float M[9], size_t channels,
    kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, const uint8_t *border_value) {
  if (!src || !dst || !M) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels != 1) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  bool replicate = border_type == KLEIDICV_BORDER_TYPE_REPLICATE;
  bool constant = border_type == KLEIDICV_BORDER_TYPE_CONSTANT;
  if (!replicate && !constant) return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  uint8_t fill = constant && border_value ? *border_value : 0;
  for (size_t dy = 0; dy < dst_height; ++dy) {
    uint8_t *drow = dst + dy * dst_stride;
    for (size_t dx = 0; dx < dst_width; ++dx) {
      float fdx = static_cast<float>(dx);
      float fdy = static_cast<float>(dy);
      float sx_p = M[0] * fdx + M[1] * fdy + M[2];
      float sy_p = M[3] * fdx + M[4] * fdy + M[5];
      float sw_p = M[6] * fdx + M[7] * fdy + M[8];
      if (sw_p == 0.0f) { drow[dx] = fill; continue; }
      float sx = sx_p / sw_p;
      float sy = sy_p / sw_p;
      if (interpolation == KLEIDICV_INTERPOLATION_NEAREST) {
        int ix = static_cast<int>(sx + 0.5f);
        int iy = static_cast<int>(sy + 0.5f);
        bool oor = ix < 0 || iy < 0 ||
                   static_cast<size_t>(ix) >= src_width ||
                   static_cast<size_t>(iy) >= src_height;
        if (oor) {
          if (constant) { drow[dx] = fill; continue; }
          if (ix < 0) ix = 0;
          if (iy < 0) iy = 0;
          if (static_cast<size_t>(ix) >= src_width)
            ix = static_cast<int>(src_width) - 1;
          if (static_cast<size_t>(iy) >= src_height)
            iy = static_cast<int>(src_height) - 1;
        }
        drow[dx] = src[static_cast<size_t>(iy) * src_stride +
                       static_cast<size_t>(ix)];
      } else {
        int ix0 = static_cast<int>(sx);
        if (sx < 0) ix0 -= 1;
        int iy0 = static_cast<int>(sy);
        if (sy < 0) iy0 -= 1;
        float fx = sx - static_cast<float>(ix0);
        float fy = sy - static_cast<float>(iy0);
        int ix1 = ix0 + 1, iy1 = iy0 + 1;
        auto fetch = [&](int x, int y) -> float {
          bool oor = x < 0 || y < 0 ||
                     static_cast<size_t>(x) >= src_width ||
                     static_cast<size_t>(y) >= src_height;
          if (oor) {
            if (constant) return static_cast<float>(fill);
            if (x < 0) x = 0;
            if (y < 0) y = 0;
            if (static_cast<size_t>(x) >= src_width)
              x = static_cast<int>(src_width) - 1;
            if (static_cast<size_t>(y) >= src_height)
              y = static_cast<int>(src_height) - 1;
          }
          return static_cast<float>(src[static_cast<size_t>(y) * src_stride +
                                        static_cast<size_t>(x)]);
        };
        float p00 = fetch(ix0, iy0), p01 = fetch(ix1, iy0);
        float p10 = fetch(ix0, iy1), p11 = fetch(ix1, iy1);
        float v = (1.0f - fy) * ((1.0f - fx) * p00 + fx * p01) +
                  fy * ((1.0f - fx) * p10 + fx * p11);
        int iv = static_cast<int>(v + 0.5f);
        if (iv < 0) iv = 0;
        if (iv > 255) iv = 255;
        drow[dx] = static_cast<uint8_t>(iv);
      }
    }
  }
  return KLEIDICV_OK;
}
kleidicv_error_t kleidicv_warp_perspective_u8_sme(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const float M[9], size_t channels,
    kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, const uint8_t *border_value) {
  return kleidicv_warp_perspective_u8(src, src_stride, src_width, src_height,
                                      dst, dst_stride, dst_width, dst_height,
                                      M, channels, interpolation, border_type,
                                      border_value);
}
#endif

}  // extern "C"
// Optical flow + Lucas-Kanade now have real implementations in
// optical_flow_pyramid.cpp and standalone_lucas_kanade_alg_*.cpp.

// ---- function-pointer NOT_IMPLEMENTED stubs ----
//
// These public symbols are declared as function pointers
// (KLEIDICV_API_DECLARATION) in the upstream header but the corresponding
// kernels are out of SPOC scope (s16point5/float remap, yuv semiplanar,
// scale-to-f16, count_nonzeros, min_max_f32, min_max_loc). The upstream
// kleidicv-benchmark binary links against every public symbol, so we
// provide bound function pointers that just return KLEIDICV_ERROR_NOT_IMPLEMENTED.
// Google-Benchmark filters or `--benchmark_filter=` should exclude these
// when running comparative numbers; otherwise they'll error per-iteration.
namespace {
template <typename Fn>
struct StubBinder { static Fn *bind() { return nullptr; } };
}

// `float16_t` is already provided by kleidicv/ctypes.h (typedef of _Float16).

namespace {

kleidicv_error_t stub_count_nonzeros_u8(const uint8_t *, size_t, size_t,
                                         size_t, size_t *) {
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}
kleidicv_error_t stub_min_max_f32(const float *, size_t, size_t, size_t,
                                   float *, float *) {
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}
kleidicv_error_t stub_min_max_loc_u8(const uint8_t *, size_t, size_t, size_t,
                                      size_t *, size_t *) {
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}
kleidicv_error_t stub_remap_f32_u8(const uint8_t *, size_t, size_t, size_t,
                                    uint8_t *, size_t, size_t, size_t, size_t,
                                    const float *, size_t, const float *,
                                    size_t, kleidicv_interpolation_type_t,
                                    kleidicv_border_type_t, const uint8_t *) {
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}
kleidicv_error_t stub_remap_f32_u16(const uint16_t *, size_t, size_t, size_t,
                                     uint16_t *, size_t, size_t, size_t,
                                     size_t, const float *, size_t,
                                     const float *, size_t,
                                     kleidicv_interpolation_type_t,
                                     kleidicv_border_type_t,
                                     const uint16_t *) {
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}
kleidicv_error_t stub_remap_s16p5_u8(const uint8_t *, size_t, size_t, size_t,
                                      uint8_t *, size_t, size_t, size_t,
                                      size_t, const int16_t *, size_t,
                                      const uint16_t *, size_t,
                                      kleidicv_border_type_t,
                                      const uint8_t *) {
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}
kleidicv_error_t stub_remap_s16p5_u16(const uint16_t *, size_t, size_t,
                                       size_t, uint16_t *, size_t, size_t,
                                       size_t, size_t, const int16_t *,
                                       size_t, const uint16_t *, size_t,
                                       kleidicv_border_type_t,
                                       const uint16_t *) {
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}
kleidicv_error_t stub_scale_u8_f16(const uint8_t *, size_t, float16_t *,
                                    size_t, size_t, size_t, double, double) {
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}
kleidicv_error_t stub_yuv_semi_to_rgb(const uint8_t *, size_t, const uint8_t *,
                                       size_t, uint8_t *, size_t, size_t,
                                       size_t, kleidicv_color_conversion_t) {
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}

}  // namespace

extern "C" {

kleidicv_error_t (*kleidicv_count_nonzeros_u8)(const uint8_t *, size_t, size_t,
                                                size_t,
                                                size_t *) = stub_count_nonzeros_u8;
kleidicv_error_t (*kleidicv_count_nonzeros_u8_sme)(
    const uint8_t *, size_t, size_t, size_t, size_t *) = stub_count_nonzeros_u8;
kleidicv_error_t (*kleidicv_min_max_f32)(const float *, size_t, size_t, size_t,
                                          float *,
                                          float *) = stub_min_max_f32;
kleidicv_error_t (*kleidicv_min_max_f32_sme)(const float *, size_t, size_t,
                                              size_t, float *,
                                              float *) = stub_min_max_f32;
kleidicv_error_t (*kleidicv_min_max_loc_u8)(const uint8_t *, size_t, size_t,
                                             size_t, size_t *,
                                             size_t *) = stub_min_max_loc_u8;
kleidicv_error_t (*kleidicv_min_max_loc_u8_sme)(
    const uint8_t *, size_t, size_t, size_t, size_t *,
    size_t *) = stub_min_max_loc_u8;
kleidicv_error_t (*kleidicv_remap_f32_u8)(
    const uint8_t *, size_t, size_t, size_t, uint8_t *, size_t, size_t,
    size_t, size_t, const float *, size_t, const float *, size_t,
    kleidicv_interpolation_type_t, kleidicv_border_type_t,
    const uint8_t *) = stub_remap_f32_u8;
kleidicv_error_t (*kleidicv_remap_f32_u8_sme)(
    const uint8_t *, size_t, size_t, size_t, uint8_t *, size_t, size_t,
    size_t, size_t, const float *, size_t, const float *, size_t,
    kleidicv_interpolation_type_t, kleidicv_border_type_t,
    const uint8_t *) = stub_remap_f32_u8;
kleidicv_error_t (*kleidicv_remap_f32_u16)(
    const uint16_t *, size_t, size_t, size_t, uint16_t *, size_t, size_t,
    size_t, size_t, const float *, size_t, const float *, size_t,
    kleidicv_interpolation_type_t, kleidicv_border_type_t,
    const uint16_t *) = stub_remap_f32_u16;
kleidicv_error_t (*kleidicv_remap_f32_u16_sme)(
    const uint16_t *, size_t, size_t, size_t, uint16_t *, size_t, size_t,
    size_t, size_t, const float *, size_t, const float *, size_t,
    kleidicv_interpolation_type_t, kleidicv_border_type_t,
    const uint16_t *) = stub_remap_f32_u16;
kleidicv_error_t (*kleidicv_remap_s16point5_u8)(
    const uint8_t *, size_t, size_t, size_t, uint8_t *, size_t, size_t,
    size_t, size_t, const int16_t *, size_t, const uint16_t *, size_t,
    kleidicv_border_type_t, const uint8_t *) = stub_remap_s16p5_u8;
kleidicv_error_t (*kleidicv_remap_s16point5_u8_sme)(
    const uint8_t *, size_t, size_t, size_t, uint8_t *, size_t, size_t,
    size_t, size_t, const int16_t *, size_t, const uint16_t *, size_t,
    kleidicv_border_type_t, const uint8_t *) = stub_remap_s16p5_u8;
kleidicv_error_t (*kleidicv_remap_s16point5_u16)(
    const uint16_t *, size_t, size_t, size_t, uint16_t *, size_t, size_t,
    size_t, size_t, const int16_t *, size_t, const uint16_t *, size_t,
    kleidicv_border_type_t, const uint16_t *) = stub_remap_s16p5_u16;
kleidicv_error_t (*kleidicv_remap_s16point5_u16_sme)(
    const uint16_t *, size_t, size_t, size_t, uint16_t *, size_t, size_t,
    size_t, size_t, const int16_t *, size_t, const uint16_t *, size_t,
    kleidicv_border_type_t, const uint16_t *) = stub_remap_s16p5_u16;
kleidicv_error_t (*kleidicv_scale_u8_f16)(
    const uint8_t *, size_t, float16_t *, size_t, size_t, size_t, double,
    double) = stub_scale_u8_f16;
kleidicv_error_t (*kleidicv_scale_u8_f16_sme)(
    const uint8_t *, size_t, float16_t *, size_t, size_t, size_t, double,
    double) = stub_scale_u8_f16;
kleidicv_error_t (*kleidicv_yuv_semiplanar_to_rgb_u8)(
    const uint8_t *, size_t, const uint8_t *, size_t, uint8_t *, size_t,
    size_t, size_t, kleidicv_color_conversion_t) = stub_yuv_semi_to_rgb;
kleidicv_error_t (*kleidicv_yuv_semiplanar_to_rgb_u8_sme)(
    const uint8_t *, size_t, const uint8_t *, size_t, uint8_t *, size_t,
    size_t, size_t, kleidicv_color_conversion_t) = stub_yuv_semi_to_rgb;

kleidicv_error_t kleidicv_rgb_to_yuv_semiplanar_u8(
    const uint8_t *, size_t, uint8_t *, size_t, uint8_t *, size_t, size_t,
    size_t, kleidicv_color_conversion_t) {
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}
kleidicv_error_t kleidicv_rgb_to_yuv_semiplanar_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *y, size_t ys, uint8_t *uv,
    size_t uvs, size_t w, size_t h, kleidicv_color_conversion_t fmt) {
  return kleidicv_rgb_to_yuv_semiplanar_u8(src, src_stride, y, ys, uv, uvs, w,
                                            h, fmt);
}

}  // extern "C"
