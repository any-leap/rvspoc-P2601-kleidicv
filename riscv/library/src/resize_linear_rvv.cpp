// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV bilinear resize, channels=1. Uses indexed-load (vluxei) to gather the
// four neighbour pixels per destination column, then float bilinear blend.
// Per-row x-axis indices and weights are precomputed once at the start of
// the call so the inner loop is a straight gather → MAC → narrow.

#include <riscv_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "resize_linear_decls.h"

namespace kleidicv::rvv {

namespace {

inline ptrdiff_t clip_idx(ptrdiff_t v, ptrdiff_t n) {
  if (v < 0) return 0;
  if (v >= n) return n - 1;
  return v;
}

// Malloc-backed scratch buffer. The build runs with `-fno-exceptions`, so
// `std::vector::resize` would call `terminate()` on OOM instead of letting
// the API return KLEIDICV_ERROR_ALLOCATION. RAII via unique_ptr<T[], free>.
template <typename T>
struct Scratch {
  std::unique_ptr<T[], void (*)(void *)> ptr{nullptr, &std::free};
  T *get() { return ptr.get(); }
  bool valid() const { return ptr != nullptr; }
};

template <typename T>
Scratch<T> alloc_scratch(size_t n) {
  Scratch<T> s;
  if (n == 0) return s;
  s.ptr.reset(static_cast<T *>(std::malloc(n * sizeof(T))));
  return s;
}

}  // namespace

kleidicv_error_t resize_linear_u8(const uint8_t *src, size_t src_stride,
                                  size_t src_width, size_t src_height,
                                  uint8_t *dst, size_t dst_stride,
                                  size_t dst_width, size_t dst_height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0)
    return KLEIDICV_OK;

  const float scale_x =
      static_cast<float>(src_width) / static_cast<float>(dst_width);
  const float scale_y =
      static_cast<float>(src_height) / static_cast<float>(dst_height);

  // Precompute per-dx index/weight tables (depend only on scale_x and dst_w).
  auto sx0c = alloc_scratch<uint32_t>(dst_width);
  auto sx1c = alloc_scratch<uint32_t>(dst_width);
  auto fx_tbl = alloc_scratch<float>(dst_width);
  if (!sx0c.valid() || !sx1c.valid() || !fx_tbl.valid())
    return KLEIDICV_ERROR_ALLOCATION;
  for (size_t dx = 0; dx < dst_width; ++dx) {
    float sx_f = (static_cast<float>(dx) + 0.5F) * scale_x - 0.5F;
    ptrdiff_t s0 = static_cast<ptrdiff_t>(std::floor(sx_f));
    float fx = sx_f - static_cast<float>(s0);
    if (fx < 0) fx = 0;
    if (fx > 1) fx = 1;
    sx0c.get()[dx] = static_cast<uint32_t>(
        clip_idx(s0, static_cast<ptrdiff_t>(src_width)));
    sx1c.get()[dx] = static_cast<uint32_t>(
        clip_idx(s0 + 1, static_cast<ptrdiff_t>(src_width)));
    fx_tbl.get()[dx] = fx;
  }

  for (size_t dy = 0; dy < dst_height; ++dy) {
    float sy_f = (static_cast<float>(dy) + 0.5F) * scale_y - 0.5F;
    ptrdiff_t sy0 = static_cast<ptrdiff_t>(std::floor(sy_f));
    float fy = sy_f - static_cast<float>(sy0);
    if (fy < 0) fy = 0;
    if (fy > 1) fy = 1;
    const ptrdiff_t H = static_cast<ptrdiff_t>(src_height);
    const uint8_t *r0 = src + static_cast<size_t>(clip_idx(sy0, H)) * src_stride;
    const uint8_t *r1 =
        src + static_cast<size_t>(clip_idx(sy0 + 1, H)) * src_stride;
    uint8_t *rd = dst + dy * dst_stride;

    size_t dx = 0;
    while (dx < dst_width) {
      size_t vl = __riscv_vsetvl_e32m4(dst_width - dx);
      // Load index + weight tables for this stripe.
      vuint32m4_t idx0 = __riscv_vle32_v_u32m4(sx0c.get() + dx, vl);
      vuint32m4_t idx1 = __riscv_vle32_v_u32m4(sx1c.get() + dx, vl);
      vfloat32m4_t fx_v = __riscv_vle32_v_f32m4(fx_tbl.get() + dx, vl);

      // Indexed u8 gather → widen → float.
      vuint8m1_t p00_u8 = __riscv_vluxei32_v_u8m1(r0, idx0, vl);
      vuint8m1_t p01_u8 = __riscv_vluxei32_v_u8m1(r0, idx1, vl);
      vuint8m1_t p10_u8 = __riscv_vluxei32_v_u8m1(r1, idx0, vl);
      vuint8m1_t p11_u8 = __riscv_vluxei32_v_u8m1(r1, idx1, vl);

      vfloat32m4_t p00 = __riscv_vfwcvt_f_xu_v_f32m4(
          __riscv_vwcvtu_x_x_v_u16m2(p00_u8, vl), vl);
      vfloat32m4_t p01 = __riscv_vfwcvt_f_xu_v_f32m4(
          __riscv_vwcvtu_x_x_v_u16m2(p01_u8, vl), vl);
      vfloat32m4_t p10 = __riscv_vfwcvt_f_xu_v_f32m4(
          __riscv_vwcvtu_x_x_v_u16m2(p10_u8, vl), vl);
      vfloat32m4_t p11 = __riscv_vfwcvt_f_xu_v_f32m4(
          __riscv_vwcvtu_x_x_v_u16m2(p11_u8, vl), vl);

      // top = p00 + fx*(p01-p00); bot = p10 + fx*(p11-p10);
      // out = top + fy*(bot-top)
      vfloat32m4_t d01 = __riscv_vfsub_vv_f32m4(p01, p00, vl);
      vfloat32m4_t top = __riscv_vfmacc_vv_f32m4(p00, fx_v, d01, vl);
      vfloat32m4_t d11 = __riscv_vfsub_vv_f32m4(p11, p10, vl);
      vfloat32m4_t bot = __riscv_vfmacc_vv_f32m4(p10, fx_v, d11, vl);
      vfloat32m4_t dtb = __riscv_vfsub_vv_f32m4(bot, top, vl);
      vfloat32m4_t out_f = __riscv_vfmacc_vf_f32m4(top, fy, dtb, vl);

      // Round, clamp [0,255], narrow → u8.
      vint32m4_t out_i = __riscv_vfcvt_x_f_v_i32m4(out_f, vl);
      vuint32m4_t out_u =
          __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmax_vx_i32m4(out_i, 0, vl));
      vuint16m2_t out_u16 =
          __riscv_vnclipu_wx_u16m2(out_u, 0, __RISCV_VXRM_RNU, vl);
      vuint8m1_t out_u8 =
          __riscv_vnclipu_wx_u8m1(out_u16, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_u8m1(rd + dx, out_u8, vl);
      dx += vl;
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t resize_linear_f32(const float *src, size_t src_stride,
                                   size_t src_width, size_t src_height,
                                   float *dst, size_t dst_stride,
                                   size_t dst_width, size_t dst_height) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0)
    return KLEIDICV_OK;

  const float scale_x =
      static_cast<float>(src_width) / static_cast<float>(dst_width);
  const float scale_y =
      static_cast<float>(src_height) / static_cast<float>(dst_height);
  const size_t src_stride_e = src_stride / sizeof(float);
  const size_t dst_stride_e = dst_stride / sizeof(float);

  auto sx0c_b = alloc_scratch<uint32_t>(dst_width);
  auto sx1c_b = alloc_scratch<uint32_t>(dst_width);
  auto fx_tbl = alloc_scratch<float>(dst_width);
  if (!sx0c_b.valid() || !sx1c_b.valid() || !fx_tbl.valid())
    return KLEIDICV_ERROR_ALLOCATION;
  for (size_t dx = 0; dx < dst_width; ++dx) {
    float sx_f = (static_cast<float>(dx) + 0.5F) * scale_x - 0.5F;
    ptrdiff_t s0 = static_cast<ptrdiff_t>(std::floor(sx_f));
    float fx = sx_f - static_cast<float>(s0);
    if (fx < 0) fx = 0;
    if (fx > 1) fx = 1;
    // f32 indices are in bytes for vluxei (byte-offset semantics).
    sx0c_b.get()[dx] = static_cast<uint32_t>(
        clip_idx(s0, static_cast<ptrdiff_t>(src_width))) *
                  static_cast<uint32_t>(sizeof(float));
    sx1c_b.get()[dx] = static_cast<uint32_t>(
        clip_idx(s0 + 1, static_cast<ptrdiff_t>(src_width))) *
                  static_cast<uint32_t>(sizeof(float));
    fx_tbl.get()[dx] = fx;
  }

  for (size_t dy = 0; dy < dst_height; ++dy) {
    float sy_f = (static_cast<float>(dy) + 0.5F) * scale_y - 0.5F;
    ptrdiff_t sy0 = static_cast<ptrdiff_t>(std::floor(sy_f));
    float fy = sy_f - static_cast<float>(sy0);
    if (fy < 0) fy = 0;
    if (fy > 1) fy = 1;
    const ptrdiff_t H = static_cast<ptrdiff_t>(src_height);
    const float *r0 = src + static_cast<size_t>(clip_idx(sy0, H)) * src_stride_e;
    const float *r1 =
        src + static_cast<size_t>(clip_idx(sy0 + 1, H)) * src_stride_e;
    float *rd = dst + dy * dst_stride_e;

    size_t dx = 0;
    while (dx < dst_width) {
      size_t vl = __riscv_vsetvl_e32m4(dst_width - dx);
      vuint32m4_t bidx0 = __riscv_vle32_v_u32m4(sx0c_b.get() + dx, vl);
      vuint32m4_t bidx1 = __riscv_vle32_v_u32m4(sx1c_b.get() + dx, vl);
      vfloat32m4_t fx_v = __riscv_vle32_v_f32m4(fx_tbl.get() + dx, vl);

      vfloat32m4_t p00 = __riscv_vluxei32_v_f32m4(r0, bidx0, vl);
      vfloat32m4_t p01 = __riscv_vluxei32_v_f32m4(r0, bidx1, vl);
      vfloat32m4_t p10 = __riscv_vluxei32_v_f32m4(r1, bidx0, vl);
      vfloat32m4_t p11 = __riscv_vluxei32_v_f32m4(r1, bidx1, vl);

      vfloat32m4_t d01 = __riscv_vfsub_vv_f32m4(p01, p00, vl);
      vfloat32m4_t top = __riscv_vfmacc_vv_f32m4(p00, fx_v, d01, vl);
      vfloat32m4_t d11 = __riscv_vfsub_vv_f32m4(p11, p10, vl);
      vfloat32m4_t bot = __riscv_vfmacc_vv_f32m4(p10, fx_v, d11, vl);
      vfloat32m4_t dtb = __riscv_vfsub_vv_f32m4(bot, top, vl);
      vfloat32m4_t out = __riscv_vfmacc_vf_f32m4(top, fy, dtb, vl);
      __riscv_vse32_v_f32m4(rd + dx, out, vl);
      dx += vl;
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
