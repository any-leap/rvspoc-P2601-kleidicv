// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared deinterleave / reinterleave scaffold for multi-channel filters.
// channels∈{2,3,4} (KLEIDICV_MAXIMUM_CHANNEL_COUNT=4).
//
// Strategy: keep the existing channels=1 RVV kernels untouched (they're the
// hot path) and run a per-channel pass over planar scratch buffers for
// channels>1. The deinterleave/reinterleave passes themselves are vectorised
// with vlsegN/vssegN, so even the channels>1 path stays on the vector unit.

#ifndef KLEIDICV_RISCV_MULTICHANNEL_HELPER_H
#define KLEIDICV_RISCV_MULTICHANNEL_HELPER_H

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

namespace kleidicv::riscv_mc {

// Deinterleave one row of u8 with `channels` interleaved planes into separate
// planar destinations. dst[c] receives `width` u8 values for channel c.
inline void deinterleave_row_u8(const uint8_t *src, uint8_t **dst,
                                  size_t width, size_t channels) {
  size_t x = 0;
  while (x < width) {
    size_t vl = __riscv_vsetvl_e8m1(width - x);
    if (channels == 2) {
      vuint8m1x2_t v = __riscv_vlseg2e8_v_u8m1x2(src + x * 2, vl);
      __riscv_vse8_v_u8m1(dst[0] + x, __riscv_vget_v_u8m1x2_u8m1(v, 0), vl);
      __riscv_vse8_v_u8m1(dst[1] + x, __riscv_vget_v_u8m1x2_u8m1(v, 1), vl);
    } else if (channels == 3) {
      vuint8m1x3_t v = __riscv_vlseg3e8_v_u8m1x3(src + x * 3, vl);
      __riscv_vse8_v_u8m1(dst[0] + x, __riscv_vget_v_u8m1x3_u8m1(v, 0), vl);
      __riscv_vse8_v_u8m1(dst[1] + x, __riscv_vget_v_u8m1x3_u8m1(v, 1), vl);
      __riscv_vse8_v_u8m1(dst[2] + x, __riscv_vget_v_u8m1x3_u8m1(v, 2), vl);
    } else /* channels == 4 */ {
      vuint8m1x4_t v = __riscv_vlseg4e8_v_u8m1x4(src + x * 4, vl);
      __riscv_vse8_v_u8m1(dst[0] + x, __riscv_vget_v_u8m1x4_u8m1(v, 0), vl);
      __riscv_vse8_v_u8m1(dst[1] + x, __riscv_vget_v_u8m1x4_u8m1(v, 1), vl);
      __riscv_vse8_v_u8m1(dst[2] + x, __riscv_vget_v_u8m1x4_u8m1(v, 2), vl);
      __riscv_vse8_v_u8m1(dst[3] + x, __riscv_vget_v_u8m1x4_u8m1(v, 3), vl);
    }
    x += vl;
  }
}

inline void interleave_row_u8(uint8_t *dst, const uint8_t *const *src,
                                size_t width, size_t channels) {
  size_t x = 0;
  while (x < width) {
    size_t vl = __riscv_vsetvl_e8m1(width - x);
    if (channels == 2) {
      vuint8m1_t a = __riscv_vle8_v_u8m1(src[0] + x, vl);
      vuint8m1_t b = __riscv_vle8_v_u8m1(src[1] + x, vl);
      vuint8m1x2_t v = __riscv_vcreate_v_u8m1x2(a, b);
      __riscv_vsseg2e8_v_u8m1x2(dst + x * 2, v, vl);
    } else if (channels == 3) {
      vuint8m1_t a = __riscv_vle8_v_u8m1(src[0] + x, vl);
      vuint8m1_t b = __riscv_vle8_v_u8m1(src[1] + x, vl);
      vuint8m1_t c = __riscv_vle8_v_u8m1(src[2] + x, vl);
      vuint8m1x3_t v = __riscv_vcreate_v_u8m1x3(a, b, c);
      __riscv_vsseg3e8_v_u8m1x3(dst + x * 3, v, vl);
    } else /* channels == 4 */ {
      vuint8m1_t a = __riscv_vle8_v_u8m1(src[0] + x, vl);
      vuint8m1_t b = __riscv_vle8_v_u8m1(src[1] + x, vl);
      vuint8m1_t c = __riscv_vle8_v_u8m1(src[2] + x, vl);
      vuint8m1_t d = __riscv_vle8_v_u8m1(src[3] + x, vl);
      vuint8m1x4_t v = __riscv_vcreate_v_u8m1x4(a, b, c, d);
      __riscv_vsseg4e8_v_u8m1x4(dst + x * 4, v, vl);
    }
    x += vl;
  }
}

inline void interleave_row_s16(int16_t *dst, const int16_t *const *src,
                                 size_t width, size_t channels) {
  size_t x = 0;
  while (x < width) {
    size_t vl = __riscv_vsetvl_e16m1(width - x);
    if (channels == 2) {
      vint16m1_t a = __riscv_vle16_v_i16m1(src[0] + x, vl);
      vint16m1_t b = __riscv_vle16_v_i16m1(src[1] + x, vl);
      vint16m1x2_t v = __riscv_vcreate_v_i16m1x2(a, b);
      __riscv_vsseg2e16_v_i16m1x2(dst + x * 2, v, vl);
    } else if (channels == 3) {
      vint16m1_t a = __riscv_vle16_v_i16m1(src[0] + x, vl);
      vint16m1_t b = __riscv_vle16_v_i16m1(src[1] + x, vl);
      vint16m1_t c = __riscv_vle16_v_i16m1(src[2] + x, vl);
      vint16m1x3_t v = __riscv_vcreate_v_i16m1x3(a, b, c);
      __riscv_vsseg3e16_v_i16m1x3(dst + x * 3, v, vl);
    } else /* channels == 4 */ {
      vint16m1_t a = __riscv_vle16_v_i16m1(src[0] + x, vl);
      vint16m1_t b = __riscv_vle16_v_i16m1(src[1] + x, vl);
      vint16m1_t c = __riscv_vle16_v_i16m1(src[2] + x, vl);
      vint16m1_t d = __riscv_vle16_v_i16m1(src[3] + x, vl);
      vint16m1x4_t v = __riscv_vcreate_v_i16m1x4(a, b, c, d);
      __riscv_vsseg4e16_v_i16m1x4(dst + x * 4, v, vl);
    }
    x += vl;
  }
}

inline void deinterleave_row_u16(const uint16_t *src, uint16_t **dst,
                                   size_t width, size_t channels) {
  size_t x = 0;
  while (x < width) {
    size_t vl = __riscv_vsetvl_e16m1(width - x);
    if (channels == 2) {
      vuint16m1x2_t v = __riscv_vlseg2e16_v_u16m1x2(src + x * 2, vl);
      __riscv_vse16_v_u16m1(dst[0] + x, __riscv_vget_v_u16m1x2_u16m1(v, 0), vl);
      __riscv_vse16_v_u16m1(dst[1] + x, __riscv_vget_v_u16m1x2_u16m1(v, 1), vl);
    } else if (channels == 3) {
      vuint16m1x3_t v = __riscv_vlseg3e16_v_u16m1x3(src + x * 3, vl);
      __riscv_vse16_v_u16m1(dst[0] + x, __riscv_vget_v_u16m1x3_u16m1(v, 0), vl);
      __riscv_vse16_v_u16m1(dst[1] + x, __riscv_vget_v_u16m1x3_u16m1(v, 1), vl);
      __riscv_vse16_v_u16m1(dst[2] + x, __riscv_vget_v_u16m1x3_u16m1(v, 2), vl);
    } else /* channels == 4 */ {
      vuint16m1x4_t v = __riscv_vlseg4e16_v_u16m1x4(src + x * 4, vl);
      __riscv_vse16_v_u16m1(dst[0] + x, __riscv_vget_v_u16m1x4_u16m1(v, 0), vl);
      __riscv_vse16_v_u16m1(dst[1] + x, __riscv_vget_v_u16m1x4_u16m1(v, 1), vl);
      __riscv_vse16_v_u16m1(dst[2] + x, __riscv_vget_v_u16m1x4_u16m1(v, 2), vl);
      __riscv_vse16_v_u16m1(dst[3] + x, __riscv_vget_v_u16m1x4_u16m1(v, 3), vl);
    }
    x += vl;
  }
}

inline void interleave_row_u16(uint16_t *dst, const uint16_t *const *src,
                                 size_t width, size_t channels) {
  size_t x = 0;
  while (x < width) {
    size_t vl = __riscv_vsetvl_e16m1(width - x);
    if (channels == 2) {
      vuint16m1_t a = __riscv_vle16_v_u16m1(src[0] + x, vl);
      vuint16m1_t b = __riscv_vle16_v_u16m1(src[1] + x, vl);
      vuint16m1x2_t v = __riscv_vcreate_v_u16m1x2(a, b);
      __riscv_vsseg2e16_v_u16m1x2(dst + x * 2, v, vl);
    } else if (channels == 3) {
      vuint16m1_t a = __riscv_vle16_v_u16m1(src[0] + x, vl);
      vuint16m1_t b = __riscv_vle16_v_u16m1(src[1] + x, vl);
      vuint16m1_t c = __riscv_vle16_v_u16m1(src[2] + x, vl);
      vuint16m1x3_t v = __riscv_vcreate_v_u16m1x3(a, b, c);
      __riscv_vsseg3e16_v_u16m1x3(dst + x * 3, v, vl);
    } else /* channels == 4 */ {
      vuint16m1_t a = __riscv_vle16_v_u16m1(src[0] + x, vl);
      vuint16m1_t b = __riscv_vle16_v_u16m1(src[1] + x, vl);
      vuint16m1_t c = __riscv_vle16_v_u16m1(src[2] + x, vl);
      vuint16m1_t d = __riscv_vle16_v_u16m1(src[3] + x, vl);
      vuint16m1x4_t v = __riscv_vcreate_v_u16m1x4(a, b, c, d);
      __riscv_vsseg4e16_v_u16m1x4(dst + x * 4, v, vl);
    }
    x += vl;
  }
}

// Same as interleave_row_s16 but for the scharr output where each pixel
// contributes 2 derivative slots (dx, dy). With C input channels and
// 2 derivative slots per channel, the output has 2*C interleaved values per
// pixel laid out as [dxR, dyR, dxG, dyG, dxB, dyB, …]. We feed one channel at
// a time: src[c] contains 2*width int16 values (x, y, x, y, …).
inline void interleave_row_s16_pairs(int16_t *dst,
                                       const int16_t *const *pair_src,
                                       size_t width, size_t channels) {
  // Reduce to the standard interleave by treating each "pixel" as 2*channels
  // s16 values and copying the two slots per source-channel into adjacent
  // output lanes.
  for (size_t x = 0; x < width; ++x) {
    for (size_t c = 0; c < channels; ++c) {
      dst[(x * channels + c) * 2 + 0] = pair_src[c][x * 2 + 0];
      dst[(x * channels + c) * 2 + 1] = pair_src[c][x * 2 + 1];
    }
  }
}

}  // namespace kleidicv::riscv_mc

#endif
