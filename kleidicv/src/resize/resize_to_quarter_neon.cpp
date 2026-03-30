// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cstddef>

#include "kleidicv/neon.h"
#include "kleidicv/resize/resize_linear.h"

namespace kleidicv::neon {

/// Resizes source data by averaging 4 elements to one.
/// In-place operation is not supported.
///
/// Only even source dimensions are supported.
/// Supports 1-, 2-, 3- and 4-channel interleaved images.
///
/// Example of 2x2 to 1x1 conversion:
/// ```
/// | a | b | --> | (a+b+c+d)/4 |
/// | c | d |
/// ```

static inline uint8x16_t average_2x2_blocks_u8(uint8x16x2_t top,
                                               uint8x16x2_t bottom) {
  uint16x8_t top_pairs_summed_0 = vpaddlq_u8(top.val[0]);
  uint16x8_t top_pairs_summed_1 = vpaddlq_u8(top.val[1]);
  uint16x8_t bottom_pairs_summed_0 = vpaddlq_u8(bottom.val[0]);
  uint16x8_t bottom_pairs_summed_1 = vpaddlq_u8(bottom.val[1]);

  uint16x8_t result_before_averaging_0 =
      vaddq_u16(top_pairs_summed_0, bottom_pairs_summed_0);
  uint16x8_t result_before_averaging_1 =
      vaddq_u16(top_pairs_summed_1, bottom_pairs_summed_1);

  return vrshrn_high_n_u16(vrshrn_n_u16(result_before_averaging_0, 2),
                           result_before_averaging_1, 2);
}

static inline void load_and_deinterleave_2_channels(const uint8_t *src,
                                                    uint8x16x2_t &channel_0,
                                                    uint8x16x2_t &channel_1) {
  uint8x16x2_t interleaved_lo = vld2q_u8(src);
  uint8x16x2_t interleaved_hi = vld2q_u8(src + kVectorLength * 2);

  channel_0.val[0] = interleaved_lo.val[0];
  channel_0.val[1] = interleaved_hi.val[0];
  channel_1.val[0] = interleaved_lo.val[1];
  channel_1.val[1] = interleaved_hi.val[1];
}

static inline void load_and_deinterleave_4_channels(const uint8_t *src,
                                                    uint8x16x2_t &channel_0,
                                                    uint8x16x2_t &channel_1,
                                                    uint8x16x2_t &channel_2,
                                                    uint8x16x2_t &channel_3) {
  uint8x16x4_t interleaved_lo = vld4q_u8(src);
  uint8x16x4_t interleaved_hi = vld4q_u8(src + kVectorLength * 4);

  channel_0.val[0] = interleaved_lo.val[0];
  channel_0.val[1] = interleaved_hi.val[0];
  channel_1.val[0] = interleaved_lo.val[1];
  channel_1.val[1] = interleaved_hi.val[1];
  channel_2.val[0] = interleaved_lo.val[2];
  channel_2.val[1] = interleaved_hi.val[2];
  channel_3.val[0] = interleaved_lo.val[3];
  channel_3.val[1] = interleaved_hi.val[3];
}

static inline void load_and_deinterleave_3_channels(const uint8_t *src,
                                                    uint8x16x2_t &channel_0,
                                                    uint8x16x2_t &channel_1,
                                                    uint8x16x2_t &channel_2) {
  uint8x16x3_t interleaved_lo = vld3q_u8(src);
  uint8x16x3_t interleaved_hi = vld3q_u8(src + kVectorLength * 3);

  channel_0.val[0] = interleaved_lo.val[0];
  channel_0.val[1] = interleaved_hi.val[0];
  channel_1.val[0] = interleaved_lo.val[1];
  channel_1.val[1] = interleaved_hi.val[1];
  channel_2.val[0] = interleaved_lo.val[2];
  channel_2.val[1] = interleaved_hi.val[2];
}

template <int kChannel>
void resize_to_quarter_vector_path(const uint8_t *src_l, size_t src_stride,
                                   uint8_t *dst_l);

template <>
void resize_to_quarter_vector_path<1>(const uint8_t *src_l, size_t src_stride,
                                      uint8_t *dst_l) {
  using VecTraits = neon::VecTraits<uint8_t>;
  KLEIDICV_PREFETCH(src_l + 1024);
  KLEIDICV_PREFETCH(src_l + src_stride + 1024);

  uint8x16x4_t top_line, bottom_line;
  uint8x16x2_t result;
  uint8x16x2_t top_pair, bottom_pair;

  VecTraits::load(src_l, top_line);
  VecTraits::load(&src_l[src_stride], bottom_line);

  top_pair.val[0] = top_line.val[0];
  top_pair.val[1] = top_line.val[1];
  bottom_pair.val[0] = bottom_line.val[0];
  bottom_pair.val[1] = bottom_line.val[1];
  result.val[0] = average_2x2_blocks_u8(top_pair, bottom_pair);

  top_pair.val[0] = top_line.val[2];
  top_pair.val[1] = top_line.val[3];
  bottom_pair.val[0] = bottom_line.val[2];
  bottom_pair.val[1] = bottom_line.val[3];
  result.val[1] = average_2x2_blocks_u8(top_pair, bottom_pair);

  VecTraits::store(result, dst_l);
}

template <>
void resize_to_quarter_vector_path<2>(const uint8_t *src_l, size_t src_stride,
                                      uint8_t *dst_l) {
  KLEIDICV_PREFETCH(src_l + 1024);
  KLEIDICV_PREFETCH(src_l + src_stride + 1024);

  uint8x16x2_t top_ch_0, top_ch_1;
  uint8x16x2_t bottom_ch_0, bottom_ch_1;

  load_and_deinterleave_2_channels(src_l, top_ch_0, top_ch_1);
  load_and_deinterleave_2_channels(src_l + src_stride, bottom_ch_0,
                                   bottom_ch_1);

  uint8x16x2_t result = {average_2x2_blocks_u8(top_ch_0, bottom_ch_0),
                         average_2x2_blocks_u8(top_ch_1, bottom_ch_1)};
  vst2q_u8(dst_l, result);
}

template <>
void resize_to_quarter_vector_path<3>(const uint8_t *src_l, size_t src_stride,
                                      uint8_t *dst_l) {
  KLEIDICV_PREFETCH(src_l + 1024);
  KLEIDICV_PREFETCH(src_l + src_stride + 1024);

  uint8x16x2_t top_ch_0, top_ch_1, top_ch_2;
  uint8x16x2_t bottom_ch_0, bottom_ch_1, bottom_ch_2;

  load_and_deinterleave_3_channels(src_l, top_ch_0, top_ch_1, top_ch_2);
  load_and_deinterleave_3_channels(src_l + src_stride, bottom_ch_0, bottom_ch_1,
                                   bottom_ch_2);

  uint8x16x3_t result = {average_2x2_blocks_u8(top_ch_0, bottom_ch_0),
                         average_2x2_blocks_u8(top_ch_1, bottom_ch_1),
                         average_2x2_blocks_u8(top_ch_2, bottom_ch_2)};
  vst3q_u8(dst_l, result);
}

template <>
void resize_to_quarter_vector_path<4>(const uint8_t *src_l, size_t src_stride,
                                      uint8_t *dst_l) {
  KLEIDICV_PREFETCH(src_l + 1024);
  KLEIDICV_PREFETCH(src_l + src_stride + 1024);

  uint8x16x2_t top_ch_0, top_ch_1, top_ch_2, top_ch_3;
  uint8x16x2_t bottom_ch_0, bottom_ch_1, bottom_ch_2, bottom_ch_3;

  load_and_deinterleave_4_channels(src_l, top_ch_0, top_ch_1, top_ch_2,
                                   top_ch_3);
  load_and_deinterleave_4_channels(src_l + src_stride, bottom_ch_0, bottom_ch_1,
                                   bottom_ch_2, bottom_ch_3);

  uint8x16x4_t result = {average_2x2_blocks_u8(top_ch_0, bottom_ch_0),
                         average_2x2_blocks_u8(top_ch_1, bottom_ch_1),
                         average_2x2_blocks_u8(top_ch_2, bottom_ch_2),
                         average_2x2_blocks_u8(top_ch_3, bottom_ch_3)};

  vst4q_u8(dst_l, result);
}

template <int kChannels>
static kleidicv_error_t resize_to_quarter_u8_imp(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride) {
  // When channels == 1, unroll twice
  constexpr size_t kDstStep = kVectorLength * (kChannels == 1 ? 2 : kChannels);
  constexpr size_t kSrcStep = 2 * kDstStep;

  LoopUnroll2 loop{src_height, /* Process two rows */ 2};

  loop.unroll_once([&](size_t index) {
    const uint8_t *cur_src = src + index * src_stride;
    uint8_t *cur_dst = dst + (index / 2) * dst_stride;
    size_t remaining = src_width * kChannels;

    for (; remaining >= kSrcStep;
         remaining -= kSrcStep, cur_dst += kDstStep, cur_src += kSrcStep) {
      resize_to_quarter_vector_path<kChannels>(cur_src, src_stride, cur_dst);
    }

    for (; remaining > 0; remaining -= (kChannels * 2UL),
                          cur_src += (kChannels * 2UL), cur_dst += kChannels) {
      disable_loop_vectorization();
      for (size_t ch = 0; ch < kChannels; ++ch) {
        cur_dst[ch] = rounding_shift_right<uint16_t>(
            static_cast<uint16_t>(cur_src[ch]) + cur_src[ch + kChannels] +
                cur_src[src_stride + ch] + cur_src[src_stride + ch + kChannels],
            2);
      }
    }
  });

  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t resize_to_quarter_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels) {
  switch (channels) {
    case 1:
      return resize_to_quarter_u8_imp<1>(src, src_stride, src_width, src_height,
                                         dst, dst_stride);
    case 2:
      return resize_to_quarter_u8_imp<2>(src, src_stride, src_width, src_height,
                                         dst, dst_stride);
    case 3:
      return resize_to_quarter_u8_imp<3>(src, src_stride, src_width, src_height,
                                         dst, dst_stride);
    default:
      assert(channels == 4);
      return resize_to_quarter_u8_imp<4>(src, src_stride, src_width, src_height,
                                         dst, dst_stride);
  }
}

}  // namespace kleidicv::neon
