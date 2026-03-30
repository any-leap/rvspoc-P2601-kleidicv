// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RESIZE_TO_QUARTER_SC_H
#define KLEIDICV_RESIZE_TO_QUARTER_SC_H

#include <cassert>
#include <cstddef>

#include "kleidicv/sve2.h"

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

namespace KLEIDICV_TARGET_NAMESPACE {

static inline svuint16_t resize_parallel_vectors(
    svuint8_t top_row, svuint8_t bottom_row) KLEIDICV_STREAMING {
  svuint16_t result_before_averaging_b = svaddlb(top_row, bottom_row);
  svuint16_t result_before_averaging_t = svaddlt(top_row, bottom_row);
  svuint16_t result_before_averaging = svadd_x(
      svptrue_b16(), result_before_averaging_b, result_before_averaging_t);
  return svreinterpret_u16_u8(svrshrnb(result_before_averaging, 2));
}

static inline svuint8_t resize_parallel_vectors_2x(
    svuint8_t top_row_0, svuint8_t top_row_1, svuint8_t bottom_row_0,
    svuint8_t bottom_row_1) KLEIDICV_STREAMING {
  svuint16_t res0 = resize_parallel_vectors(top_row_0, bottom_row_0);
  svuint16_t res1 = resize_parallel_vectors(top_row_1, bottom_row_1);
  return svuzp1(svreinterpret_u8_u16(res0), svreinterpret_u8_u16(res1));
}

static inline svuint8_t resize_two_channels(
    svuint8_t top_row_c0, svuint8_t top_row_c1, svuint8_t bottom_row_c0,
    svuint8_t bottom_row_c1) KLEIDICV_STREAMING {
  svuint16_t res0_before_averaging_b = svaddlb(top_row_c0, bottom_row_c0);
  svuint16_t res0_before_averaging_t = svaddlt(top_row_c0, bottom_row_c0);
  svuint16_t res0_before_averaging =
      svadd_x(svptrue_b16(), res0_before_averaging_b, res0_before_averaging_t);
  svuint16_t res1_before_averaging_b = svaddlb(top_row_c1, bottom_row_c1);
  svuint16_t res1_before_averaging_t = svaddlt(top_row_c1, bottom_row_c1);
  svuint16_t res1_before_averaging =
      svadd_x(svptrue_b16(), res1_before_averaging_b, res1_before_averaging_t);
  return svrshrnt(svrshrnb(res0_before_averaging, 2), res1_before_averaging, 2);
}

// Template declaration for parallel_rows_vectors_path
template <size_t kChannels>
void parallel_rows_vectors_path(size_t length, Rows<const uint8_t> src_rows,
                                Rows<uint8_t> dst_rows) KLEIDICV_STREAMING;

// Template declaration for parallel_rows_vectors_path_2x
template <size_t kChannels>
void parallel_rows_vectors_path_2x(Rows<const uint8_t> src_rows,
                                   Rows<uint8_t> dst_rows) KLEIDICV_STREAMING;

// Template specialization for 1 channel 2x
template <>
void parallel_rows_vectors_path_2x<1>(
    Rows<const uint8_t> src_rows, Rows<uint8_t> dst_rows) KLEIDICV_STREAMING {
  svbool_t pg = VecTraits<uint8_t>::svptrue();
#if KLEIDICV_TARGET_SME2
  svcount_t pg_counter = svptrue_c8();
  auto src_top = svld1_x2(pg_counter, &src_rows.at(0)[0]);
  auto src_bottom = svld1_x2(pg_counter, &src_rows.at(1)[0]);
  svuint8_t top_row_0 = svget2(src_top, 0);
  svuint8_t top_row_1 = svget2(src_top, 1);
  svuint8_t bottom_row_0 = svget2(src_bottom, 0);
  svuint8_t bottom_row_1 = svget2(src_bottom, 1);
#else
  svuint8_t top_row_0 = svld1(pg, &src_rows.at(0)[0]);
  svuint8_t bottom_row_0 = svld1(pg, &src_rows.at(1)[0]);
  svuint8_t top_row_1 = svld1_vnum(pg, &src_rows.at(0)[0], 1);
  svuint8_t bottom_row_1 = svld1_vnum(pg, &src_rows.at(1)[0], 1);
#endif  // KLEIDICV_TARGET_SME2
  svuint8_t result = resize_parallel_vectors_2x(top_row_0, top_row_1,
                                                bottom_row_0, bottom_row_1);
  svst1(pg, &dst_rows[0], result);
}

// Template specialization for 1 channel
template <>
void parallel_rows_vectors_path<1>(size_t length, Rows<const uint8_t> src_rows,
                                   Rows<uint8_t> dst_rows) KLEIDICV_STREAMING {
  svbool_t pg_src = VecTraits<uint8_t>::svwhilelt(size_t{0}, length * 2);
  svbool_t pg_dst = VecTraits<uint16_t>::svwhilelt(size_t{0}, length);
  svuint8_t top_line = svld1(pg_src, &src_rows.at(0)[0]);
  svuint8_t bottom_line = svld1(pg_src, &src_rows.at(1)[0]);
  svuint16_t result = resize_parallel_vectors(top_line, bottom_line);
  svst1b(pg_dst, &dst_rows[0], result);
}

// Template specialization for 2 channels 2x
template <>
void parallel_rows_vectors_path_2x<2>(
    Rows<const uint8_t> src_rows, Rows<uint8_t> dst_rows) KLEIDICV_STREAMING {
  svbool_t pg = VecTraits<uint8_t>::svptrue();
#if KLEIDICV_TARGET_SME2
  svcount_t pg_counter = svptrue_c8();
  auto src_top = svld1_x4(pg_counter, &src_rows.at(0)[0]);
  auto src_bottom = svld1_x4(pg_counter, &src_rows.at(1)[0]);
  svuint8_t top_row_0 = svget4(src_top, 0);
  svuint8_t top_row_1 = svget4(src_top, 1);
  svuint8_t top_row_2 = svget4(src_top, 2);
  svuint8_t top_row_3 = svget4(src_top, 3);
  svuint8_t bottom_row_0 = svget4(src_bottom, 0);
  svuint8_t bottom_row_1 = svget4(src_bottom, 1);
  svuint8_t bottom_row_2 = svget4(src_bottom, 2);
  svuint8_t bottom_row_3 = svget4(src_bottom, 3);
#else
  svuint8_t top_row_0 = svld1(pg, &src_rows.at(0)[0]);
  svuint8_t top_row_1 = svld1_vnum(pg, &src_rows.at(0)[0], 1);
  svuint8_t top_row_2 = svld1_vnum(pg, &src_rows.at(0)[0], 2);
  svuint8_t top_row_3 = svld1_vnum(pg, &src_rows.at(0)[0], 3);
  svuint8_t bottom_row_0 = svld1(pg, &src_rows.at(1)[0]);
  svuint8_t bottom_row_1 = svld1_vnum(pg, &src_rows.at(1)[0], 1);
  svuint8_t bottom_row_2 = svld1_vnum(pg, &src_rows.at(1)[0], 2);
  svuint8_t bottom_row_3 = svld1_vnum(pg, &src_rows.at(1)[0], 3);
#endif

  svuint8_t result_lo = resize_two_channels(
      svuzp1(top_row_0, top_row_1), svuzp2(top_row_0, top_row_1),
      svuzp1(bottom_row_0, bottom_row_1), svuzp2(bottom_row_0, bottom_row_1));
  svuint8_t result_hi = resize_two_channels(
      svuzp1(top_row_2, top_row_3), svuzp2(top_row_2, top_row_3),
      svuzp1(bottom_row_2, bottom_row_3), svuzp2(bottom_row_2, bottom_row_3));

  svst1(pg, &dst_rows[0], result_lo);
  svst1_vnum(pg, &dst_rows[0], 1, result_hi);
}

// Template specialization for 2 channels
template <>
void parallel_rows_vectors_path<2>(size_t dst_length,
                                   Rows<const uint8_t> src_rows,
                                   Rows<uint8_t> dst_rows) KLEIDICV_STREAMING {
  svbool_t pg_dst = VecTraits<uint8_t>::svwhilelt(size_t{0}, dst_length * 2);
#if KLEIDICV_TARGET_SME2
  svcount_t pg_src_cnt = svwhilelt_c8(0, dst_length * 2 * 2, 2);
  auto src_top = svld1_x2(pg_src_cnt, &src_rows.at(0)[0]);
  auto src_bottom = svld1_x2(pg_src_cnt, &src_rows.at(1)[0]);
  svuint8_t top_row_0 = svget2(src_top, 0);
  svuint8_t top_row_1 = svget2(src_top, 1);
  svuint8_t bottom_row_0 = svget2(src_bottom, 0);
  svuint8_t bottom_row_1 = svget2(src_bottom, 1);
#else
  svbool_t pg_src_0, pg_src_1;
  VecTraits<uint8_t>::make_consecutive_predicates(pg_dst, pg_src_0, pg_src_1);

  svuint8_t top_row_0 = svld1(pg_src_0, &src_rows.at(0)[0]);
  svuint8_t top_row_1 = svld1_vnum(pg_src_1, &src_rows.at(0)[0], 1);
  svuint8_t bottom_row_0 = svld1(pg_src_0, &src_rows.at(1)[0]);
  svuint8_t bottom_row_1 = svld1_vnum(pg_src_1, &src_rows.at(1)[0], 1);
#endif

  svst1(pg_dst, &dst_rows[0],
        resize_two_channels(svuzp1(top_row_0, top_row_1),
                            svuzp2(top_row_0, top_row_1),
                            svuzp1(bottom_row_0, bottom_row_1),
                            svuzp2(bottom_row_0, bottom_row_1)));
}

// Template specialization for 3 channels 2x
template <>
void parallel_rows_vectors_path_2x<3>(
    Rows<const uint8_t> src_rows, Rows<uint8_t> dst_rows) KLEIDICV_STREAMING {
  svbool_t pg = VecTraits<uint8_t>::svptrue();
  const ptrdiff_t vec_stride = static_cast<ptrdiff_t>(svcntb() * 3);
  svuint8x3_t top_row_0 = svld3(pg, &src_rows.at(0)[0]);
  svuint8x3_t top_row_1 = svld3(pg, &src_rows.at(0)[vec_stride]);
  svuint8x3_t bottom_row_0 = svld3(pg, &src_rows.at(1)[0]);
  svuint8x3_t bottom_row_1 = svld3(pg, &src_rows.at(1)[vec_stride]);

  svuint8x3_t result = svcreate3(
      resize_parallel_vectors_2x(svget3(top_row_0, 0), svget3(top_row_1, 0),
                                 svget3(bottom_row_0, 0),
                                 svget3(bottom_row_1, 0)),
      resize_parallel_vectors_2x(svget3(top_row_0, 1), svget3(top_row_1, 1),
                                 svget3(bottom_row_0, 1),
                                 svget3(bottom_row_1, 1)),
      resize_parallel_vectors_2x(svget3(top_row_0, 2), svget3(top_row_1, 2),
                                 svget3(bottom_row_0, 2),
                                 svget3(bottom_row_1, 2)));
  svst3(pg, &dst_rows[0], result);
}

// Template specialization for 3 channels
template <>
void parallel_rows_vectors_path<3>(size_t length, Rows<const uint8_t> src_rows,
                                   Rows<uint8_t> dst_rows) KLEIDICV_STREAMING {
  svbool_t pg_src = VecTraits<uint8_t>::svwhilelt(size_t{0}, length * 2);
  svuint8x3_t top_row = svld3(pg_src, &src_rows.at(0)[0]);
  svuint8x3_t bottom_row = svld3(pg_src, &src_rows.at(1)[0]);
  svuint16_t result_c0 =
      resize_parallel_vectors(svget3(top_row, 0), svget3(bottom_row, 0));
  svuint16_t result_c1 =
      resize_parallel_vectors(svget3(top_row, 1), svget3(bottom_row, 1));
  svuint16_t result_c2 =
      resize_parallel_vectors(svget3(top_row, 2), svget3(bottom_row, 2));

  svuint8x3_t result =
      svcreate3(svuzp1(svreinterpret_u8_u16(result_c0), svdup_u8(0)),
                svuzp1(svreinterpret_u8_u16(result_c1), svdup_u8(0)),
                svuzp1(svreinterpret_u8_u16(result_c2), svdup_u8(0)));
  svbool_t pg_dst = VecTraits<uint8_t>::svwhilelt(size_t{0}, length);
  svst3(pg_dst, &dst_rows[0], result);
}

// Template specialization for 4 channels 2x
template <>
void parallel_rows_vectors_path_2x<4>(
    Rows<const uint8_t> src_rows, Rows<uint8_t> dst_rows) KLEIDICV_STREAMING {
  svbool_t pg = VecTraits<uint8_t>::svptrue();
  const ptrdiff_t vec_stride =
      static_cast<ptrdiff_t>(VecTraits<uint8_t>::num_lanes() * 4);
  svuint8x4_t top_row_0 = svld4(pg, &src_rows.at(0)[0]);
  svuint8x4_t top_row_1 = svld4(pg, &src_rows.at(0)[vec_stride]);
  svuint8x4_t bottom_row_0 = svld4(pg, &src_rows.at(1)[0]);
  svuint8x4_t bottom_row_1 = svld4(pg, &src_rows.at(1)[vec_stride]);

  svuint8x4_t result = svcreate4(
      resize_parallel_vectors_2x(svget4(top_row_0, 0), svget4(top_row_1, 0),
                                 svget4(bottom_row_0, 0),
                                 svget4(bottom_row_1, 0)),
      resize_parallel_vectors_2x(svget4(top_row_0, 1), svget4(top_row_1, 1),
                                 svget4(bottom_row_0, 1),
                                 svget4(bottom_row_1, 1)),
      resize_parallel_vectors_2x(svget4(top_row_0, 2), svget4(top_row_1, 2),
                                 svget4(bottom_row_0, 2),
                                 svget4(bottom_row_1, 2)),
      resize_parallel_vectors_2x(svget4(top_row_0, 3), svget4(top_row_1, 3),
                                 svget4(bottom_row_0, 3),
                                 svget4(bottom_row_1, 3)));
  svst4(pg, &dst_rows[0], result);
}

// Template specialization for 4 channels
template <>
void parallel_rows_vectors_path<4>(size_t length, Rows<const uint8_t> src_rows,
                                   Rows<uint8_t> dst_rows) KLEIDICV_STREAMING {
  svbool_t pg_src = VecTraits<uint8_t>::svwhilelt(size_t{0}, 2 * length);
  svuint8x4_t top_row = svld4(pg_src, &src_rows.at(0)[0]);
  svuint8x4_t bottom_row = svld4(pg_src, &src_rows.at(1)[0]);
  svuint16_t result_c0 =
      resize_parallel_vectors(svget4(top_row, 0), svget4(bottom_row, 0));
  svuint16_t result_c1 =
      resize_parallel_vectors(svget4(top_row, 1), svget4(bottom_row, 1));
  svuint16_t result_c2 =
      resize_parallel_vectors(svget4(top_row, 2), svget4(bottom_row, 2));
  svuint16_t result_c3 =
      resize_parallel_vectors(svget4(top_row, 3), svget4(bottom_row, 3));

  svuint8x4_t result =
      svcreate4(svuzp1(svreinterpret_u8_u16(result_c0), svdup_u8(0)),
                svuzp1(svreinterpret_u8_u16(result_c1), svdup_u8(0)),
                svuzp1(svreinterpret_u8_u16(result_c2), svdup_u8(0)),
                svuzp1(svreinterpret_u8_u16(result_c3), svdup_u8(0)));
  svbool_t pg_dst = VecTraits<uint8_t>::svwhilelt(size_t{0}, length);
  svst4(pg_dst, &dst_rows[0], result);
}

template <size_t kChannels, typename ScalarType>
void process_parallel_rows(Rows<const ScalarType> src_rows, size_t src_width,
                           Rows<ScalarType> dst_rows) KLEIDICV_STREAMING {
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<ScalarType>;

  LoopUnroll2 loop{src_width, VecTraits::num_lanes()};
  loop.unroll_twice([&](size_t index) KLEIDICV_STREAMING {
    parallel_rows_vectors_path_2x<kChannels>(src_rows.at(0, index),
                                             dst_rows.at(0, index / 2));
  });

  auto vector_path = [&](size_t index) KLEIDICV_STREAMING {
    const size_t remaining = (src_width - index) / 2;
    parallel_rows_vectors_path<kChannels>(remaining, src_rows.at(0, index),
                                          dst_rows.at(0, index / 2));
  };

  loop.unroll_once(vector_path);
  loop.remaining([&](size_t index, size_t)
                     KLEIDICV_STREAMING { vector_path(index); });
}

KLEIDICV_TARGET_FN_ATTRS static kleidicv_error_t resize_to_quarter_u8_sc(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels) KLEIDICV_STREAMING {
  Rows<const uint8_t> src_rows{src, src_stride, channels};
  Rows<uint8_t> dst_rows{dst, dst_stride, channels};
  LoopUnroll2 loop{src_height, /* Process two rows */ 2};

  // Process two rows at once.
  loop.unroll_once([&](size_t row) KLEIDICV_STREAMING {
    auto r = static_cast<ptrdiff_t>(row);
    if (channels == 1) {
      process_parallel_rows<1>(src_rows.at(r), src_width, dst_rows.at(r / 2));
    } else if (channels == 2) {
      process_parallel_rows<2>(src_rows.at(r), src_width, dst_rows.at(r / 2));
    } else if (channels == 3) {
      process_parallel_rows<3>(src_rows.at(r), src_width, dst_rows.at(r / 2));
    } else {
      assert(channels == 4);
      process_parallel_rows<4>(src_rows.at(r), src_width, dst_rows.at(r / 2));
    }
  });
  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_RESIZE_TO_QUARTER_SC_H
