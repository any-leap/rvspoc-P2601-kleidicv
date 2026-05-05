// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Perspective warp (3×3 homography), channels=1, NEAREST or LINEAR interp,
// REPLICATE or CONSTANT border. Per row dy, all three projective accumulators
// are linear in dx, so we precompute (M[0]·dx + …) as a vfloat32 vector and
// drive both nearest-neighbour and bilinear paths off the same x/y/w lanes.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>
#include <cmath>

#include "dispatch.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/kleidicv.h"

namespace {

using kleidicv::riscv_dispatch::active_backend;
using kleidicv::riscv_dispatch::Backend;

inline int clip_i(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

kleidicv_error_t warp_perspective_u8_scalar(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const float M[9], kleidicv_interpolation_type_t interp,
    bool constant_border, uint8_t fill) {
  for (size_t dy = 0; dy < dst_height; ++dy) {
    uint8_t *drow = dst + dy * dst_stride;
    for (size_t dx = 0; dx < dst_width; ++dx) {
      float fdx = static_cast<float>(dx);
      float fdy = static_cast<float>(dy);
      float sx_p = M[0] * fdx + M[1] * fdy + M[2];
      float sy_p = M[3] * fdx + M[4] * fdy + M[5];
      float sw_p = M[6] * fdx + M[7] * fdy + M[8];
      if (sw_p == 0.0F) {
        drow[dx] = fill;
        continue;
      }
      float sx = sx_p / sw_p;
      float sy = sy_p / sw_p;
      if (interp == KLEIDICV_INTERPOLATION_NEAREST) {
        int ix = static_cast<int>(sx + 0.5F);
        int iy = static_cast<int>(sy + 0.5F);
        bool oor = ix < 0 || iy < 0 ||
                   static_cast<size_t>(ix) >= src_width ||
                   static_cast<size_t>(iy) >= src_height;
        if (oor) {
          if (constant_border) {
            drow[dx] = fill;
            continue;
          }
          ix = clip_i(ix, 0, static_cast<int>(src_width) - 1);
          iy = clip_i(iy, 0, static_cast<int>(src_height) - 1);
        }
        drow[dx] = src[static_cast<size_t>(iy) * src_stride +
                       static_cast<size_t>(ix)];
      } else {
        int ix0 = static_cast<int>(std::floor(sx));
        int iy0 = static_cast<int>(std::floor(sy));
        float fx = sx - static_cast<float>(ix0);
        float fy = sy - static_cast<float>(iy0);
        int ix1 = ix0 + 1, iy1 = iy0 + 1;
        auto fetch = [&](int x, int y) -> float {
          bool oor = x < 0 || y < 0 ||
                     static_cast<size_t>(x) >= src_width ||
                     static_cast<size_t>(y) >= src_height;
          if (oor) {
            if (constant_border) return static_cast<float>(fill);
            x = clip_i(x, 0, static_cast<int>(src_width) - 1);
            y = clip_i(y, 0, static_cast<int>(src_height) - 1);
          }
          return static_cast<float>(src[static_cast<size_t>(y) * src_stride +
                                        static_cast<size_t>(x)]);
        };
        float p00 = fetch(ix0, iy0), p01 = fetch(ix1, iy0);
        float p10 = fetch(ix0, iy1), p11 = fetch(ix1, iy1);
        float v = (1.0F - fy) * ((1.0F - fx) * p00 + fx * p01) +
                  fy * ((1.0F - fx) * p10 + fx * p11);
        int iv = static_cast<int>(v + 0.5F);
        drow[dx] = static_cast<uint8_t>(clip_i(iv, 0, 255));
      }
    }
  }
  return KLEIDICV_OK;
}

// RVV nearest path. Vectorise over dx; sx/sy are float-derived per lane.
kleidicv_error_t warp_perspective_u8_rvv_nearest(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const float M[9], bool constant_border, uint8_t fill) {
  const int32_t W = static_cast<int32_t>(src_width);
  const int32_t H = static_cast<int32_t>(src_height);
  const uint32_t Sstride = static_cast<uint32_t>(src_stride);

  for (size_t dy = 0; dy < dst_height; ++dy) {
    const float fdy = static_cast<float>(dy);
    const float bx = M[1] * fdy + M[2];
    const float by = M[4] * fdy + M[5];
    const float bw = M[7] * fdy + M[8];
    uint8_t *drow = dst + dy * dst_stride;

    size_t dx = 0;
    while (dx < dst_width) {
      size_t vl = __riscv_vsetvl_e8m1(dst_width - dx);
      // dx_v = vid + dx (as f32).
      vuint32m4_t idv = __riscv_vid_v_u32m4(vl);
      vfloat32m4_t dx_v = __riscv_vfcvt_f_xu_v_f32m4(idv, vl);
      dx_v = __riscv_vfadd_vf_f32m4(dx_v, static_cast<float>(dx), vl);

      vfloat32m4_t sx_p = __riscv_vfmacc_vf_f32m4(
          __riscv_vfmv_v_f_f32m4(bx, vl), M[0], dx_v, vl);
      vfloat32m4_t sy_p = __riscv_vfmacc_vf_f32m4(
          __riscv_vfmv_v_f_f32m4(by, vl), M[3], dx_v, vl);
      vfloat32m4_t sw_p = __riscv_vfmacc_vf_f32m4(
          __riscv_vfmv_v_f_f32m4(bw, vl), M[6], dx_v, vl);

      // w == 0 mask (degenerate): tag as oob, the fill path handles them.
      vbool8_t m_w_zero = __riscv_vmfeq_vf_f32m4_b8(sw_p, 0.0F, vl);
      // Replace zeros with 1.0 to avoid div-by-zero in the rcp path; values
      // get masked to fill below either way.
      vfloat32m4_t sw_safe =
          __riscv_vfmerge_vfm_f32m4(sw_p, 1.0F, m_w_zero, vl);
      vfloat32m4_t sx_f = __riscv_vfdiv_vv_f32m4(sx_p, sw_safe, vl);
      vfloat32m4_t sy_f = __riscv_vfdiv_vv_f32m4(sy_p, sw_safe, vl);

      // Nearest = (int)(s + 0.5) — matches the scalar reference, which uses
      // C-cast (RTZ) on the +0.5 value. We use vfcvt RTZ for the same
      // semantics; do NOT adjust for negatives here (RTZ on the +0.5 result
      // is exactly the scalar behaviour).
      sx_f = __riscv_vfadd_vf_f32m4(sx_f, 0.5F, vl);
      sy_f = __riscv_vfadd_vf_f32m4(sy_f, 0.5F, vl);
      vint32m4_t ix = __riscv_vfcvt_rtz_x_f_v_i32m4(sx_f, vl);
      vint32m4_t iy = __riscv_vfcvt_rtz_x_f_v_i32m4(sy_f, vl);

      // Out-of-bounds mask.
      vbool8_t m_oob_lx = __riscv_vmslt_vx_i32m4_b8(ix, 0, vl);
      vbool8_t m_oob_ly = __riscv_vmslt_vx_i32m4_b8(iy, 0, vl);
      vbool8_t m_oob_hx = __riscv_vmsge_vx_i32m4_b8(ix, W, vl);
      vbool8_t m_oob_hy = __riscv_vmsge_vx_i32m4_b8(iy, H, vl);
      vbool8_t m_oob = __riscv_vmor_mm_b8(m_oob_lx, m_oob_ly, vl);
      m_oob = __riscv_vmor_mm_b8(m_oob, m_oob_hx, vl);
      m_oob = __riscv_vmor_mm_b8(m_oob, m_oob_hy, vl);
      m_oob = __riscv_vmor_mm_b8(m_oob, m_w_zero, vl);

      // Clamp.
      vint32m4_t ixc = __riscv_vmin_vx_i32m4(
          __riscv_vmax_vx_i32m4(ix, 0, vl), W - 1, vl);
      vint32m4_t iyc = __riscv_vmin_vx_i32m4(
          __riscv_vmax_vx_i32m4(iy, 0, vl), H - 1, vl);

      vuint32m4_t off = __riscv_vmul_vx_u32m4(
          __riscv_vreinterpret_v_i32m4_u32m4(iyc), Sstride, vl);
      off = __riscv_vadd_vv_u32m4(
          off, __riscv_vreinterpret_v_i32m4_u32m4(ixc), vl);
      vuint8m1_t pix = __riscv_vluxei32_v_u8m1(src, off, vl);
      if (constant_border) {
        pix = __riscv_vmerge_vxm_u8m1(pix, fill, m_oob, vl);
      }
      __riscv_vse8_v_u8m1(drow + dx, pix, vl);
      dx += vl;
    }
  }
  return KLEIDICV_OK;
}

// RVV bilinear path. Same outer math as nearest but four gathers + blend.
kleidicv_error_t warp_perspective_u8_rvv_bilinear(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const float M[9], bool constant_border, uint8_t fill) {
  const int32_t W = static_cast<int32_t>(src_width);
  const int32_t H = static_cast<int32_t>(src_height);
  const uint32_t Sstride = static_cast<uint32_t>(src_stride);
  const float fillf = static_cast<float>(fill);

  for (size_t dy = 0; dy < dst_height; ++dy) {
    const float fdy = static_cast<float>(dy);
    const float bx = M[1] * fdy + M[2];
    const float by = M[4] * fdy + M[5];
    const float bw = M[7] * fdy + M[8];
    uint8_t *drow = dst + dy * dst_stride;

    size_t dx = 0;
    while (dx < dst_width) {
      size_t vl = __riscv_vsetvl_e8m1(dst_width - dx);
      vuint32m4_t idv = __riscv_vid_v_u32m4(vl);
      vfloat32m4_t dx_v = __riscv_vfcvt_f_xu_v_f32m4(idv, vl);
      dx_v = __riscv_vfadd_vf_f32m4(dx_v, static_cast<float>(dx), vl);

      vfloat32m4_t sx_p = __riscv_vfmacc_vf_f32m4(
          __riscv_vfmv_v_f_f32m4(bx, vl), M[0], dx_v, vl);
      vfloat32m4_t sy_p = __riscv_vfmacc_vf_f32m4(
          __riscv_vfmv_v_f_f32m4(by, vl), M[3], dx_v, vl);
      vfloat32m4_t sw_p = __riscv_vfmacc_vf_f32m4(
          __riscv_vfmv_v_f_f32m4(bw, vl), M[6], dx_v, vl);

      vbool8_t m_w_zero = __riscv_vmfeq_vf_f32m4_b8(sw_p, 0.0F, vl);
      vfloat32m4_t sw_safe =
          __riscv_vfmerge_vfm_f32m4(sw_p, 1.0F, m_w_zero, vl);
      vfloat32m4_t sx = __riscv_vfdiv_vv_f32m4(sx_p, sw_safe, vl);
      vfloat32m4_t sy = __riscv_vfdiv_vv_f32m4(sy_p, sw_safe, vl);

      // floor(s) → integer; fx/fy = s - floor(s).
      vint32m4_t ix0 = __riscv_vfcvt_rtz_x_f_v_i32m4(sx, vl);
      vint32m4_t iy0 = __riscv_vfcvt_rtz_x_f_v_i32m4(sy, vl);
      // Adjust toward -inf for negative non-integer.
      vfloat32m4_t fix0 = __riscv_vfcvt_f_x_v_f32m4(ix0, vl);
      vfloat32m4_t fiy0 = __riscv_vfcvt_f_x_v_f32m4(iy0, vl);
      vbool8_t m_x_neg = __riscv_vmflt_vv_f32m4_b8(sx, fix0, vl);
      vbool8_t m_y_neg = __riscv_vmflt_vv_f32m4_b8(sy, fiy0, vl);
      ix0 = __riscv_vsub_vx_i32m4_mu(m_x_neg, ix0, ix0, 1, vl);
      iy0 = __riscv_vsub_vx_i32m4_mu(m_y_neg, iy0, iy0, 1, vl);
      vfloat32m4_t fx = __riscv_vfsub_vv_f32m4(
          sx, __riscv_vfcvt_f_x_v_f32m4(ix0, vl), vl);
      vfloat32m4_t fy = __riscv_vfsub_vv_f32m4(
          sy, __riscv_vfcvt_f_x_v_f32m4(iy0, vl), vl);
      vint32m4_t ix1 = __riscv_vadd_vx_i32m4(ix0, 1, vl);
      vint32m4_t iy1 = __riscv_vadd_vx_i32m4(iy0, 1, vl);

      auto gather = [&](vint32m4_t xi, vint32m4_t yi) -> vfloat32m4_t {
        vbool8_t oo = __riscv_vmslt_vx_i32m4_b8(xi, 0, vl);
        oo = __riscv_vmor_mm_b8(oo, __riscv_vmslt_vx_i32m4_b8(yi, 0, vl), vl);
        oo = __riscv_vmor_mm_b8(oo, __riscv_vmsge_vx_i32m4_b8(xi, W, vl), vl);
        oo = __riscv_vmor_mm_b8(oo, __riscv_vmsge_vx_i32m4_b8(yi, H, vl), vl);
        oo = __riscv_vmor_mm_b8(oo, m_w_zero, vl);
        vint32m4_t xc = __riscv_vmin_vx_i32m4(
            __riscv_vmax_vx_i32m4(xi, 0, vl), W - 1, vl);
        vint32m4_t yc = __riscv_vmin_vx_i32m4(
            __riscv_vmax_vx_i32m4(yi, 0, vl), H - 1, vl);
        vuint32m4_t off = __riscv_vmul_vx_u32m4(
            __riscv_vreinterpret_v_i32m4_u32m4(yc), Sstride, vl);
        off = __riscv_vadd_vv_u32m4(
            off, __riscv_vreinterpret_v_i32m4_u32m4(xc), vl);
        vuint8m1_t pix = __riscv_vluxei32_v_u8m1(src, off, vl);
        vfloat32m4_t pf = __riscv_vfwcvt_f_xu_v_f32m4(
            __riscv_vwcvtu_x_x_v_u16m2(pix, vl), vl);
        if (constant_border) {
          pf = __riscv_vfmerge_vfm_f32m4(pf, fillf, oo, vl);
        }
        return pf;
      };

      vfloat32m4_t p00 = gather(ix0, iy0);
      vfloat32m4_t p01 = gather(ix1, iy0);
      vfloat32m4_t p10 = gather(ix0, iy1);
      vfloat32m4_t p11 = gather(ix1, iy1);

      vfloat32m4_t d01 = __riscv_vfsub_vv_f32m4(p01, p00, vl);
      vfloat32m4_t top = __riscv_vfmacc_vv_f32m4(p00, fx, d01, vl);
      vfloat32m4_t d11 = __riscv_vfsub_vv_f32m4(p11, p10, vl);
      vfloat32m4_t bot = __riscv_vfmacc_vv_f32m4(p10, fx, d11, vl);
      vfloat32m4_t dtb = __riscv_vfsub_vv_f32m4(bot, top, vl);
      vfloat32m4_t out_f = __riscv_vfmacc_vv_f32m4(top, fy, dtb, vl);
      out_f = __riscv_vfadd_vf_f32m4(out_f, 0.5F, vl);
      vint32m4_t out_i = __riscv_vfcvt_rtz_x_f_v_i32m4(out_f, vl);
      vuint32m4_t out_u = __riscv_vreinterpret_v_i32m4_u32m4(
          __riscv_vmax_vx_i32m4(out_i, 0, vl));
      vuint16m2_t out_u16 =
          __riscv_vnclipu_wx_u16m2(out_u, 0, __RISCV_VXRM_RNU, vl);
      vuint8m1_t out_u8 =
          __riscv_vnclipu_wx_u8m1(out_u16, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_u8m1(drow + dx, out_u8, vl);
      dx += vl;
    }
  }
  return KLEIDICV_OK;
}

}  // namespace

extern "C" kleidicv_error_t kleidicv_warp_perspective_u8(
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
  if (active_backend() == Backend::Rvv) {
    if (interpolation == KLEIDICV_INTERPOLATION_NEAREST) {
      return warp_perspective_u8_rvv_nearest(src, src_stride, src_width,
                                               src_height, dst, dst_stride,
                                               dst_width, dst_height, M,
                                               constant, fill);
    }
    return warp_perspective_u8_rvv_bilinear(src, src_stride, src_width,
                                             src_height, dst, dst_stride,
                                             dst_width, dst_height, M,
                                             constant, fill);
  }
  return warp_perspective_u8_scalar(src, src_stride, src_width, src_height,
                                     dst, dst_stride, dst_width, dst_height,
                                     M, interpolation, constant, fill);
}

extern "C" kleidicv_error_t kleidicv_warp_perspective_u8_sme(
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
