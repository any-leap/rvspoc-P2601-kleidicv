// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV 5×5 binomial blur + 2× downsample. Horizontal pass produces a u16
// buffer for each of the 5 contributing source rows (vectorised over the
// interior, scalar-clip on the 2-pixel left/right margins). Vertical pass
// + downsample combines the 5 buffers with stride-2 loads and narrows
// (acc+128)>>8 back to u8. Output is (src_w+1)/2 × (src_h+1)/2.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "blur_and_downsample_decls.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv::scalar {
kleidicv_error_t blur_and_downsample_u8(const uint8_t *, size_t, size_t, size_t,
                                        uint8_t *, size_t);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {

namespace {

inline size_t clip(ptrdiff_t v, size_t n) {
  if (v < 0) return 0;
  if (static_cast<size_t>(v) >= n) return n - 1;
  return static_cast<size_t>(v);
}

// One-row horizontal blur: hbuf[x] = K∗src[y][x] with K=[1,4,6,4,1] and
// replicate clip. Output is u16 (max value 16·255 = 4080).
void horiz_blur_row_u8(const uint8_t *src_row, size_t src_width,
                        uint16_t *hbuf) {
  // Border columns 0 and 1 — replicate clip is required, do scalar.
  for (size_t x = 0; x < 2 && x < src_width; ++x) {
    int s_m2 = src_row[clip(static_cast<ptrdiff_t>(x) - 2, src_width)];
    int s_m1 = src_row[clip(static_cast<ptrdiff_t>(x) - 1, src_width)];
    int s_0 = src_row[x];
    int s_p1 = src_row[clip(static_cast<ptrdiff_t>(x) + 1, src_width)];
    int s_p2 = src_row[clip(static_cast<ptrdiff_t>(x) + 2, src_width)];
    hbuf[x] =
        static_cast<uint16_t>(s_m2 + 4 * s_m1 + 6 * s_0 + 4 * s_p1 + s_p2);
  }

  // Interior — strip-mined RVV. x ∈ [2, src_width-2), unit-stride loads at
  // offsets {-2,-1,0,+1,+2}.
  if (src_width >= 5) {
    size_t i = 2;
    const size_t end = src_width - 2;
    while (i < end) {
      size_t vl = __riscv_vsetvl_e8m1(end - i);
      vuint8m1_t v_m2 = __riscv_vle8_v_u8m1(src_row + i - 2, vl);
      vuint8m1_t v_m1 = __riscv_vle8_v_u8m1(src_row + i - 1, vl);
      vuint8m1_t v_0 = __riscv_vle8_v_u8m1(src_row + i, vl);
      vuint8m1_t v_p1 = __riscv_vle8_v_u8m1(src_row + i + 1, vl);
      vuint8m1_t v_p2 = __riscv_vle8_v_u8m1(src_row + i + 2, vl);
      // K=[1,4,6,4,1]: acc = (v_m2 + v_p2) + 4·(v_m1 + v_p1) + 6·v_0.
      vuint16m2_t acc = __riscv_vwaddu_vv_u16m2(v_m2, v_p2, vl);
      acc = __riscv_vwmaccu_vx_u16m2(acc, 4, v_m1, vl);
      acc = __riscv_vwmaccu_vx_u16m2(acc, 4, v_p1, vl);
      acc = __riscv_vwmaccu_vx_u16m2(acc, 6, v_0, vl);
      __riscv_vse16_v_u16m2(hbuf + i, acc, vl);
      i += vl;
    }
  }

  // Border columns src_width-2 and src_width-1.
  for (size_t x = (src_width >= 2 ? src_width - 2 : 0); x < src_width; ++x) {
    if (x < 2) continue;  // already handled above
    int s_m2 = src_row[clip(static_cast<ptrdiff_t>(x) - 2, src_width)];
    int s_m1 = src_row[clip(static_cast<ptrdiff_t>(x) - 1, src_width)];
    int s_0 = src_row[x];
    int s_p1 = src_row[clip(static_cast<ptrdiff_t>(x) + 1, src_width)];
    int s_p2 = src_row[clip(static_cast<ptrdiff_t>(x) + 2, src_width)];
    hbuf[x] =
        static_cast<uint16_t>(s_m2 + 4 * s_m1 + 6 * s_0 + 4 * s_p1 + s_p2);
  }
}

}  // namespace

kleidicv_error_t blur_and_downsample_u8(const uint8_t *src, size_t src_stride,
                                        size_t src_width, size_t src_height,
                                        uint8_t *dst, size_t dst_stride) {
  if (!src || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (src_width < 4 || src_height < 4) return KLEIDICV_ERROR_RANGE;

  const size_t dst_w = (src_width + 1) / 2;
  const size_t dst_h = (src_height + 1) / 2;

  // Row buffers for the 5 vertical taps. Reused in a sliding window across
  // output rows so each source row gets a horizontal pass at most once.
  std::vector<uint16_t> hbuf_storage(5 * src_width);
  uint16_t *hbufs[5] = {
      hbuf_storage.data() + 0 * src_width, hbuf_storage.data() + 1 * src_width,
      hbuf_storage.data() + 2 * src_width, hbuf_storage.data() + 3 * src_width,
      hbuf_storage.data() + 4 * src_width};
  // Track which source row each slot currently holds (-1 = empty).
  ptrdiff_t hbuf_row[5] = {-1, -1, -1, -1, -1};

  auto get_hbuf = [&](ptrdiff_t y) -> const uint16_t * {
    size_t cy = clip(y, src_height);
    size_t slot = cy % 5;
    if (hbuf_row[slot] != static_cast<ptrdiff_t>(cy)) {
      horiz_blur_row_u8(src + cy * src_stride, src_width, hbufs[slot]);
      hbuf_row[slot] = static_cast<ptrdiff_t>(cy);
    }
    return hbufs[slot];
  };

  for (size_t dy = 0; dy < dst_h; ++dy) {
    const ptrdiff_t sy = static_cast<ptrdiff_t>(dy) * 2;
    const uint16_t *r0 = get_hbuf(sy - 2);
    const uint16_t *r1 = get_hbuf(sy - 1);
    const uint16_t *r2 = get_hbuf(sy);
    const uint16_t *r3 = get_hbuf(sy + 1);
    const uint16_t *r4 = get_hbuf(sy + 2);

    uint8_t *drow = dst + dy * dst_stride;

    // Vectorise over output columns. Each lane samples src column 2·dx, so
    // strided-load with byte stride 2·sizeof(uint16_t) = 4.
    size_t dx = 0;
    while (dx < dst_w) {
      size_t vl = __riscv_vsetvl_e8m1(dst_w - dx);
      const ptrdiff_t base_sx = static_cast<ptrdiff_t>(dx) * 2;
      const ptrdiff_t stride_bytes = 2 * static_cast<ptrdiff_t>(sizeof(uint16_t));

      vuint16m2_t h0 = __riscv_vlse16_v_u16m2(r0 + base_sx, stride_bytes, vl);
      vuint16m2_t h1 = __riscv_vlse16_v_u16m2(r1 + base_sx, stride_bytes, vl);
      vuint16m2_t h2 = __riscv_vlse16_v_u16m2(r2 + base_sx, stride_bytes, vl);
      vuint16m2_t h3 = __riscv_vlse16_v_u16m2(r3 + base_sx, stride_bytes, vl);
      vuint16m2_t h4 = __riscv_vlse16_v_u16m2(r4 + base_sx, stride_bytes, vl);

      // Vertical 5-tap blur: acc = (h0+h4) + 4·(h1+h3) + 6·h2 (u32).
      vuint32m4_t acc = __riscv_vwaddu_vv_u32m4(h0, h4, vl);
      acc = __riscv_vwmaccu_vx_u32m4(acc, 4, h1, vl);
      acc = __riscv_vwmaccu_vx_u32m4(acc, 4, h3, vl);
      acc = __riscv_vwmaccu_vx_u32m4(acc, 6, h2, vl);

      // Round-narrow (acc + 128) >> 8 → u16 → saturating-narrow → u8.
      vuint16m2_t mid =
          __riscv_vnclipu_wx_u16m2(acc, 8, __RISCV_VXRM_RNU, vl);
      vuint8m1_t out =
          __riscv_vnclipu_wx_u8m1(mid, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_u8m1(drow + dx, out, vl);
      dx += vl;
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::rvv
