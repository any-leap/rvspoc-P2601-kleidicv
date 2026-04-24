// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RESIZE_LINEAR_GENERIC_SC_H
#define KLEIDICV_RESIZE_LINEAR_GENERIC_SC_H

#include <algorithm>
#include <cstddef>
#include <memory>

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

//------------------------------------------------------
/// Generic resize for ratios 1/3 to 1/1, u8, 1channel
//------------------------------------------------------

namespace resize_generic_u8 {

// For the coordinate calculation, fixed-point format is used, for better
// performance. Fixed-point format:
// - lowest 16 bits are the fractional part, that is the kFixpBits constant
// - at interpolation, the high 8 bits are used from the fractional part
//   (this is a good compromise between accuracy and performance: because the
//   result is 8bits, the error only affects the least significant 1-2 bits, see
//   the accuracy calculation in kleidicv.h
// - to get the integer part, right shift by 16 bits, or zip/unzip/tbl etc. to
//   get the bytes needed
// - for better accuracy, rounding is needed everywhere, i.e. adding 0.5, which
//   is 1 << 15

static constexpr ptrdiff_t kFixpBits = 16;
static constexpr ptrdiff_t kFixpHalf = (1UL << (kFixpBits - 1));

// Precalc 1 item:
// Frac:       2 vectors u16
// Idx:        1 vector   u8 (left_idx)
// Src_index:  uint64 (separate array)
template <size_t kRatio>
struct PrecalcIterator {
  size_t index_;
  uint64_t *src_index_ptr_;
  const size_t kStep, kIdxFracStep;
  uint8_t *idx_ptr_;
  uint16_t *frac_ptr_;
  PrecalcIterator(size_t kStepDst, uint64_t *src_indices,
                  uint8_t *p_idx_frac) KLEIDICV_STREAMING
      : index_{0},
        src_index_ptr_{src_indices},
        kStep{kStepDst},
        kIdxFracStep{kStep * (2 + 1)},
        idx_ptr_{p_idx_frac},
        frac_ptr_{reinterpret_cast<uint16_t *>(p_idx_frac + kStep)} {}

  PrecalcIterator &operator++() KLEIDICV_STREAMING {
    ++index_;
    ++src_index_ptr_;
    idx_ptr_ += kIdxFracStep;
    frac_ptr_ += kIdxFracStep / 2;
    return *this;
  }
};

template <int kRatio, int kChannels>
class PrecalcIndicesFractions final {
 public:
  PrecalcIndicesFractions(size_t src_width, size_t dst_width,
                          ptrdiff_t kStep) KLEIDICV_STREAMING
      : src_width_{src_width},
        dst_width_{dst_width},
        n_iterations_{0},
        n_iterations_2x_{0},
        kStep_{kStep},
        precalc_src_bases_{nullptr, &std::free},
        precalc_idx_frac_{nullptr, &std::free} {}

  PrecalcIterator<kRatio> begin() const KLEIDICV_STREAMING {
    return PrecalcIterator<kRatio>(kStep_, precalc_src_bases_.get(),
                                   precalc_idx_frac_.get());
  }

  bool precalculate_indices_fractions_srcindices() KLEIDICV_STREAMING {
    if (!allocate_temp_buffers()) {
      return false;
    }

    // These starting values are not aligned to center. The center alignment
    // must be added only once. When added to a center-aligned source_x
    // value, the result will be center-aligned.
    svuint32_t vsx0b = make_vsx0(0);
    svuint32_t vsx0t = make_vsx0(1);
    svuint32_t vsx1b = make_vsx0(2 * svcntw());
    svuint32_t vsx1t = make_vsx0(2 * svcntw() + 1);
    // from each even 16bit element, take the low byte, and the high is 0
    svuint8_t vsxfrac_bottom_tbl =
        svreinterpret_u8_u16(svindex_u16(0xFF00, 0x0004));
    // from each odd 16bit element, take the low byte, and the high is 0
    svuint8_t vsxfrac_top_tbl =
        svreinterpret_u8_u16(svindex_u16(0xFF02, 0x0004));

    svuint8_t vchannels = svreinterpret_u8_u32(
        svdup_n_u32(kChannels == 4 ? 0x03020100U : 0x01000100));

    // Difference in source x coordinate, for one vector path
    const uint64_t sx_fixp_step = rounding_div(
        ((src_width_ * kStep_ / kChannels) << kFixpBits), dst_width_);
    uint64_t sx_fixp = to_src_x(0);
    const uint64_t max_src_index =
        std::max(src_width_ * kChannels - kStep_ * kRatio, 0UL);
    // For 1,2,4 channels dx can be iterated vector by vector, but not for 3
    ptrdiff_t dx = 0;
    for (auto pcit = begin(); pcit.index_ < n_iterations_;
         ++pcit, dx += kStep_ / kChannels) {
      // Repeatedly adding sx_fixp_vector_step is faster than multiplication,
      // but it accumulates fixed-point error; periodic recalibration resets
      // it. The maximum per-addition error of sx_fixp_vector_step is 0.5 / (1
      // << 16). Only the upper 8 bits of the 16-bit fractional part are used
      // for interpolation, so once the accumulated error reaches 1 / (1 <<
      // 8), it can affect later stages. This corresponds to 512 additions,
      // which is calculated by this mask.
      constexpr uint64_t kRecalibrateCycleMask = ((1 << 9) - 1);
      if ((pcit.index_ & kRecalibrateCycleMask) == 0) {
        sx_fixp = to_src_x(dx);
      }

      n_iterations_2x_ = (sx_fixp >> kFixpBits) * kChannels <= max_src_index
                             ? pcit.index_
                             : n_iterations_2x_;
      calculate_indices_fractions_srcindex(pcit, sx_fixp, vsx0b, vsx0t, vsx1b,
                                           vsx1t, vsxfrac_bottom_tbl,
                                           vsxfrac_top_tbl, vchannels);
      sx_fixp += sx_fixp_step;
    }
    return true;
  }

  bool precalculate_indices_fractions_srcindices_3ch() KLEIDICV_STREAMING {
    if (!allocate_temp_buffers()) {
      return false;
    }

    // These starting values are not aligned to center. The center alignment
    // must be added only once. When added to a center-aligned source_x
    // value, the result will be center-aligned.
    svuint32_t vsx0b_R = make_vsx0(0);
    svuint32_t vsx0t_R = make_vsx0(1);
    svuint32_t vsx1b_R = make_vsx0(2 * svcntw());
    svuint32_t vsx1t_R = make_vsx0(2 * svcntw() + 1);

    svuint32_t vsx0b_G = make_vsx0(4 * svcntw());
    svuint32_t vsx0t_G = make_vsx0(4 * svcntw() + 1);
    svuint32_t vsx1b_G = make_vsx0(6 * svcntw());
    svuint32_t vsx1t_G = make_vsx0(6 * svcntw() + 1);

    svuint32_t vsx0b_B = make_vsx0(8 * svcntw());
    svuint32_t vsx0t_B = make_vsx0(8 * svcntw() + 1);
    svuint32_t vsx1b_B = make_vsx0(10 * svcntw());
    svuint32_t vsx1t_B = make_vsx0(10 * svcntw() + 1);

    size_t kVL = svcntb();
    svuint8_t vchannels_R = svindex_u8(0, 1);
    svuint8_t vchannels_G = svindex_u8(kVL % 3, 1);
    svuint8_t vchannels_B = svindex_u8((kVL + kVL) % 3, 1);
    // Decrease by 3 while they are >= 3 --> so we get the modulo
    size_t steps = (kVL - 1) / 3;
    for (size_t i = 0; i < steps; ++i) {
      vchannels_R = svsub_n_u8_m(svcmpge_n_u8(svptrue_b8(), vchannels_R, 3),
                                 vchannels_R, 3);
      vchannels_G = svsub_n_u8_m(svcmpge_n_u8(svptrue_b8(), vchannels_G, 3),
                                 vchannels_G, 3);
      vchannels_B = svsub_n_u8_m(svcmpge_n_u8(svptrue_b8(), vchannels_B, 3),
                                 vchannels_B, 3);
    }

    // from each even 16bit element, take the low byte, and the high is 0
    svuint8_t vsxfrac_bottom_tbl =
        svreinterpret_u8_u16(svindex_u16(0xFF00, 0x0004));
    // from each odd 16bit element, take the low byte, and the high is 0
    svuint8_t vsxfrac_top_tbl =
        svreinterpret_u8_u16(svindex_u16(0xFF02, 0x0004));

    // Difference in source x coordinate, for three vector paths (one iteration
    // in this calculation)
    const uint64_t sx_fixp_step3 =
        rounding_div((src_width_ * kStep_) << kFixpBits, dst_width_);
    uint64_t sx_fixp = to_src_x(0);
    const uint64_t max_src_index =
        std::max(src_width_ * kChannels - kStep_ * kRatio, 0UL);
    ptrdiff_t dx = 0;
    auto pcit = begin();
    while (pcit.index_ < n_iterations_) {
      // Repeatedly adding sx_fixp_vector_step is faster than multiplication,
      // but it accumulates fixed-point error; periodic recalibration resets
      // it. The maximum per-addition error of sx_fixp_vector_step is 0.5 / (1
      // << 16). Only the upper 8 bits of the 16-bit fractional part are used
      // for interpolation, so once the accumulated error reaches 1 / (1 <<
      // 8), it can affect later stages. This corresponds to 512 additions,
      // but it will trigger each 3rd time, so the mask should be set to 128.
      constexpr uint64_t kRecalibrateCycleMask = ((1 << 7) - 1);
      if ((pcit.index_ & kRecalibrateCycleMask) == 0) {
        sx_fixp = to_src_x(dx);
      }

      calculate_indices_fractions_srcindex(pcit, sx_fixp, vsx0b_R, vsx0t_R,
                                           vsx1b_R, vsx1t_R, vsxfrac_bottom_tbl,
                                           vsxfrac_top_tbl, vchannels_R);
      n_iterations_2x_ = *pcit.src_index_ptr_ <= max_src_index
                             ? pcit.index_
                             : n_iterations_2x_;
      ++pcit;
      if (pcit.index_ >= n_iterations_) {
        break;
      }
      calculate_indices_fractions_srcindex(pcit, sx_fixp, vsx0b_G, vsx0t_G,
                                           vsx1b_G, vsx1t_G, vsxfrac_bottom_tbl,
                                           vsxfrac_top_tbl, vchannels_G);
      n_iterations_2x_ = *pcit.src_index_ptr_ <= max_src_index
                             ? pcit.index_
                             : n_iterations_2x_;
      ++pcit;
      if (pcit.index_ >= n_iterations_) {
        break;
      }
      calculate_indices_fractions_srcindex(pcit, sx_fixp, vsx0b_B, vsx0t_B,
                                           vsx1b_B, vsx1t_B, vsxfrac_bottom_tbl,
                                           vsxfrac_top_tbl, vchannels_B);
      n_iterations_2x_ = *pcit.src_index_ptr_ <= max_src_index
                             ? pcit.index_
                             : n_iterations_2x_;
      ++pcit;
      sx_fixp += sx_fixp_step3;
      dx += kStep_;
    }
    // For best performance and simplicity, there is a common routine that loads
    // the extra lane. This is executed in all cases, so there is no code
    // duplication, but it does not have branching there either, so performance
    // is better.
    // But when the last processed vector is not full, the last index would
    // overread the input buffer. To prevent this, set the very last index to 0.
    if (dst_width_ * kChannels < n_iterations_ * kStep_) {
      *(pcit.idx_ptr_ - 1) = 0;
    }
    return true;
  }

  size_t n_iterations() const KLEIDICV_STREAMING { return n_iterations_; }
  size_t n_iterations_2x() const KLEIDICV_STREAMING { return n_iterations_2x_; }
  uint64_t *src_bases() const KLEIDICV_STREAMING {
    return precalc_src_bases_.get();
  }
  uint8_t *idx_frac() const KLEIDICV_STREAMING {
    return precalc_idx_frac_.get();
  }

 private:
  using FreeDeleter = decltype(&std::free);

  bool allocate_temp_buffers() KLEIDICV_STREAMING {
    // Allocate a bit more so don't have to care about overindexing
    ptrdiff_t rounded_width = align_up(dst_width_ * kChannels, kStep_);
    n_iterations_ = rounded_width / kStep_;
    size_t idx_bytes = sizeof(uint8_t) * rounded_width;
    size_t xfrac_bytes = sizeof(uint16_t) * rounded_width;
    precalc_idx_frac_.reset(
        static_cast<uint8_t *>(malloc(idx_bytes + xfrac_bytes)));
    if (!precalc_idx_frac_) {
      return false;
    }
    size_t src_bases_bytes = sizeof(uint64_t) * rounded_width / kStep_;
    precalc_src_bases_.reset(static_cast<uint64_t *>(malloc(src_bases_bytes)));
    return static_cast<bool>(precalc_src_bases_);
  }

  template <typename T = uint64_t>
  static T rounding_div(uint64_t nom, uint64_t denom) KLEIDICV_STREAMING {
    return static_cast<T>((nom + denom / 2) / denom);
  }

  // Scale coordinate using this formula, so the center is aligned:
  //   source_x = (destination_x + 0.5) / scale - 0.5;
  //   plus 1/256/2 for later rounding the fractional part to 8bits
  static uint64_t aligned_scale(uint64_t x, uint64_t nom,
                                uint64_t denom) KLEIDICV_STREAMING {
    return rounding_div(((x << kFixpBits) + kFixpHalf) * nom, denom) -
           kFixpHalf + (1 << (kFixpBits - 9));
  }

  uint64_t to_src_x(uint64_t dx) const KLEIDICV_STREAMING {
    return aligned_scale(dx, src_width_, dst_width_);
  }

  // Scale destination x coordinate to source x coordinate, into fixed-point,
  // without center correction
  uint32_t scale_x(uint64_t dx) const KLEIDICV_STREAMING {
    return rounding_div<uint32_t>(((dx * src_width_) << kFixpBits), dst_width_);
  }

  svuint32_t make_vsx0(uint64_t dx) const KLEIDICV_STREAMING {
    // Creates source x coordinates starting with dx, stepping by 2
    // and finally shifted left by 8, to support the later svaddhn operation
    uint32_t sx[64];  // maximum possible vector length in u32 units
    for (size_t i = 0; i < svcntw(); ++i) {
      sx[i] = scale_x((dx + 2 * i) / kChannels) << 8;
    }
    return svld1(svptrue_b32(), sx);
  }

  void calculate_indices_fractions_srcindex(
      PrecalcIterator<kRatio> &pcit, uint64_t sx_fixp, const svuint32_t &vsx0b,
      const svuint32_t &vsx0t, const svuint32_t &vsx1b, const svuint32_t &vsx1t,
      const svuint8_t &vsxfrac_bottom_tbl, const svuint8_t &vsxfrac_top_tbl,
      [[maybe_unused]] const svuint8_t &vchannels) const KLEIDICV_STREAMING {
    // << 8: to prepare for addhn, have the fractional part in the high half
    uint32_t xfrac0 = static_cast<uint32_t>(sx_fixp & ((1 << kFixpBits) - 1))
                      << 8;
    // get the interesting part: 8+8 bits of integer and fractional part
    svuint16x2_t vsx_delta =
        svcreate2(svaddhnt_n_u32(svaddhnb_n_u32(vsx0b, xfrac0), vsx0t, xfrac0),
                  svaddhnt_n_u32(svaddhnb_n_u32(vsx1b, xfrac0), vsx1t, xfrac0));
    if constexpr (kChannels == 3) {
      // When vsx0 starts from other than zero, this offset must be subtracted
      uint16_t start{};
      svst1(svptrue_pat_b16(SV_VL1), &start, svget2(vsx_delta, 0));
      start = start & 0xFF00;
      vsx_delta =
          svcreate2(svsub_n_u16_x(svptrue_b16(), svget2(vsx_delta, 0), start),
                    svsub_n_u16_x(svptrue_b16(), svget2(vsx_delta, 1), start));
      sx_fixp += (start >> 8) << kFixpBits;
    }
    svuint8x2_t vsx_delta8 =
        svcreate2(svreinterpret_u8_u16(svget2(vsx_delta, 0)),
                  svreinterpret_u8_u16(svget2(vsx_delta, 1)));
    // left pixels' indices: integer part
    svuint8_t vsx_left_idx =
        svuzp2_u8(svget2(vsx_delta8, 0), svget2(vsx_delta8, 1));
    if constexpr (kChannels > 1) {
      if constexpr (kChannels == 3) {
        vsx_left_idx = svmul_n_u8_x(svptrue_b8(), vsx_left_idx, 3);
      } else {
        static_assert(kChannels == 2 || kChannels == 4);
        vsx_left_idx =
            svlsl_n_u8_x(svptrue_b8(), vsx_left_idx, kChannels == 4 ? 2 : 1);
      }
      vsx_left_idx = svadd_u8_x(svptrue_b8(), vsx_left_idx, vchannels);
    }

    uint64_t srcindex = (sx_fixp >> kFixpBits) * kChannels;
    if constexpr (kChannels == 3) {
      // When vsx_left_idx starts from other than zero, this offset must be
      // subtracted
      uint8_t start{};
      svst1(svptrue_pat_b8(SV_VL1), &start, vsx_left_idx);
      vsx_left_idx = svsub_n_u8_x(svptrue_b8(), vsx_left_idx, start);
      srcindex += start;
    }

    *pcit.src_index_ptr_ = srcindex;
    svst1(svptrue_b8(), pcit.idx_ptr_, vsx_left_idx);

    // fractional part is widened to 16 bits for further operations
    svuint16_t vsxfrac_b =
        svreinterpret_u16_u8(svtbl2_u8(vsx_delta8, vsxfrac_bottom_tbl));
    svuint16_t vsxfrac_t =
        svreinterpret_u16_u8(svtbl2_u8(vsx_delta8, vsxfrac_top_tbl));
    svst1(svptrue_b16(), pcit.frac_ptr_, vsxfrac_b);
    svst1_vnum(svptrue_b16(), pcit.frac_ptr_, 1, vsxfrac_t);
  }

  const size_t src_width_;
  const size_t dst_width_;
  size_t n_iterations_;
  size_t n_iterations_2x_;
  const ptrdiff_t kStep_;
  std::unique_ptr<uint64_t, FreeDeleter> precalc_src_bases_;
  std::unique_ptr<uint8_t, FreeDeleter> precalc_idx_frac_;
};

// ratio: number of vectors to load and resize to 1 vector
// - supported combinations of (ratio, channel):
// (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)
template <int kRatio, int kChannels, bool kSetRightmostLanes = false>
class ResizeGenericU8Operation final {
 public:
  ResizeGenericU8Operation(const uint8_t *src, size_t src_stride,
                           size_t src_width, size_t src_height, size_t y_begin,
                           size_t y_end,
                           uint8_t *dst,  // NOLINT
                           size_t dst_stride, size_t dst_width,
                           size_t dst_height) KLEIDICV_STREAMING
      : src_rows_{src, src_stride, kChannels},
        dst_rows_{dst, dst_stride, kChannels},
        src_width_{src_width},
        src_height_{src_height},
        y_begin_{y_begin},
        y_end_{y_end},
        dst_width_{dst_width},
        dst_height_{dst_height},
        kStep_{static_cast<ptrdiff_t>(svcntb())},
        precalc_{src_width, dst_width, kStep_} {}

  kleidicv_error_t process_rows() KLEIDICV_STREAMING {
    bool precalc_success = false;
    if constexpr (kChannels == 3) {
      precalc_success =
          precalc_.precalculate_indices_fractions_srcindices_3ch();
    } else {
      precalc_success = precalc_.precalculate_indices_fractions_srcindices();
    }
    if (!precalc_success) {
      return KLEIDICV_ERROR_ALLOCATION;
    }

    for (uint64_t dst_y = y_begin_; dst_y < y_end_; ++dst_y) {
      process_row(dst_y);
    }

    return KLEIDICV_OK;
  }

 private:
  template <typename T = uint64_t>
  static T rounding_div(uint64_t nom, uint64_t denom) KLEIDICV_STREAMING {
    return static_cast<T>((nom + denom / 2) / denom);
  }

  // Scale coordinate using this formula, so the center is aligned:
  //   source_x = (destination_x + 0.5) / scale - 0.5;
  //   plus 1/256/2 for later rounding the fractional part to 8bits
  static uint64_t aligned_scale(uint64_t x, uint64_t nom,
                                uint64_t denom) KLEIDICV_STREAMING {
    return rounding_div(((x << kFixpBits) + kFixpHalf) * nom, denom) -
           kFixpHalf + (1 << (kFixpBits - 9));
  }

  uint64_t to_src_y(uint64_t dy) const KLEIDICV_STREAMING {
    return aligned_scale(dy, src_height_, dst_height_);
  }

  static svuint16_t svshll8b(svuint8_t a) KLEIDICV_STREAMING {
    return svreinterpret_u16_u8(svtrn1(svdup_n_u8(0), a));
  }
  static svuint16_t svshll8t(svuint8_t a) KLEIDICV_STREAMING {
    return svreinterpret_u16_u8(svtrn2(svdup_n_u8(0), a));
  }

  static svuint8x2_t load8x2_u8(const uint8_t *p) KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME2
    return svld1_x2(svptrue_c8(), p);
#else
    return svcreate2(svld1(svptrue_b8(), p), svld1_vnum(svptrue_b8(), p, 1));
#endif
  }

  svuint8x2_t load8x2_while_u8(const uint8_t *p, uint64_t i,
                               uint64_t n) const KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME2
    return svld1_x2(svwhilelt_c8(i, n, 2), p);
#else
    svbool_t pg1 = svwhilelt_b8(i, n);
    svbool_t pg2 = svwhilelt_b8(i + kStep_, n);
    return svcreate2(svld1(pg1, p), svld1_vnum(pg2, p, 1));
#endif
  }

  static svuint8x3_t load8x3_u8(const uint8_t *p) KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME2
    svuint8x2_t sv2 = svld1_x2(svptrue_c8(), p);
    return svcreate3(svget2(sv2, 0), svget2(sv2, 1),
                     svld1_vnum(svptrue_b8(), p, 2));
#else
    return svcreate3(svld1(svptrue_b8(), p), svld1_vnum(svptrue_b8(), p, 1),
                     svld1_vnum(svptrue_b8(), p, 2));
#endif
  }

  svuint8x3_t load8x3_while_u8(const uint8_t *p, uint64_t i,
                               uint64_t n) const KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME2
    svcount_t pgc = svwhilelt_c8(i, n, 2);
    svbool_t pgb = svwhilelt_b8(i + 2 * kStep_, n);
    svuint8x2_t sv2 = svld1_x2(pgc, p);
    return svcreate3(svget2(sv2, 0), svget2(sv2, 1), svld1_vnum(pgb, p, 2));
#else
    svbool_t pg1 = svwhilelt_b8(i, n);
    svbool_t pg2 = svwhilelt_b8(i + kStep_, n);
    svbool_t pg3 = svwhilelt_b8(i + 2 * kStep_, n);
    return svcreate3(svld1(pg1, p), svld1_vnum(pg2, p, 1),
                     svld1_vnum(pg3, p, 2));
#endif
  }

  svuint8_t interpolate(const PrecalcIterator<kRatio> &pcit, uint16_t yfrac,
                        svuint8_t a, svuint8_t b, svuint8_t c,
                        svuint8_t d) const KLEIDICV_STREAMING {
#if KLEIDICV_TARGET_SME2
    svuint16x2_t vsxfrac = svld1_x2(svptrue_c8(), pcit.frac_ptr_);
    svuint16_t vsxfrac_b = svget2(vsxfrac, 0);
    svuint16_t vsxfrac_t = svget2(vsxfrac, 1);
#else
    svuint16_t vsxfrac_b = svld1(svptrue_b16(), pcit.frac_ptr_);
    svuint16_t vsxfrac_t = svld1_vnum(svptrue_b16(), pcit.frac_ptr_, 1);
#endif
    svuint16_t half = svdup_n_u16(128);
    svuint8_t left = svaddhnb(
        svshll8b(a), svmla_n_u16_x(svptrue_b16(), half, svsublb(c, a), yfrac));
    svuint8_t right = svaddhnb(
        svshll8b(b), svmla_n_u16_x(svptrue_b16(), half, svsublb(d, b), yfrac));
    left = svaddhnt(left, svshll8t(a),
                    svmla_n_u16_x(svptrue_b16(), half, svsublt(c, a), yfrac));
    right = svaddhnt(right, svshll8t(b),
                     svmla_n_u16_x(svptrue_b16(), half, svsublt(d, b), yfrac));

    svuint8_t res =
        svaddhnb(svshll8b(left),
                 svmla_x(svptrue_b16(), half, svsublb(right, left), vsxfrac_b));
    return svaddhnt(
        res, svshll8t(left),
        svmla_x(svptrue_b16(), half, svsublt(right, left), vsxfrac_t));
  }

  svuint8_t common_vector_path_r2(
      const PrecalcIterator<kRatio> &pcit, uint16_t yfrac, svuint8x2_t topsrc,
      svuint8x2_t bottomsrc) const KLEIDICV_STREAMING {
    svuint8_t vsx0_idx = svld1(svptrue_b8(), pcit.idx_ptr_);
    svuint8_t vsx1_idx = svadd_n_u8_x(svptrue_b8(), vsx0_idx, kChannels);
    svuint8_t a = svtbl2_u8(topsrc, vsx0_idx);
    svuint8_t b = svtbl2_u8(topsrc, vsx1_idx);
    svuint8_t c = svtbl2_u8(bottomsrc, vsx0_idx);
    svuint8_t d = svtbl2_u8(bottomsrc, vsx1_idx);
    return interpolate(pcit, yfrac, a, b, c, d);
  }

  svuint8_t vector_path_r2(const PrecalcIterator<kRatio> &pcit, uint16_t yfrac,
                           const uint8_t *src_top,
                           const uint8_t *src_bottom) const KLEIDICV_STREAMING {
    // Load 2*step elements, that's enough for 1/2 < scale < 1.0
    uint64_t src_index = *pcit.src_index_ptr_;
    svuint8x2_t topsrc = load8x2_u8(&src_top[src_index]);
    svuint8x2_t bottomsrc = load8x2_u8(&src_bottom[src_index]);
    return common_vector_path_r2(pcit, yfrac, topsrc, bottomsrc);
  }

  svuint8_t remaining_path_r2(const PrecalcIterator<kRatio> &pcit,
                              uint16_t yfrac, const uint8_t *src_top,
                              const uint8_t *src_bottom) const
      KLEIDICV_STREAMING {
    // Load 2*step elements, that's enough for 1/2 < scale < 1.0
    uint64_t src_index = *pcit.src_index_ptr_;
    svuint8x2_t topsrc = load8x2_while_u8(&src_top[src_index], src_index,
                                          src_width_ * kChannels);
    svuint8x2_t bottomsrc = load8x2_while_u8(&src_bottom[src_index], src_index,
                                             src_width_ * kChannels);
    return common_vector_path_r2(pcit, yfrac, topsrc, bottomsrc);
  }

  svuint8_t common_vector_path_r3(
      const PrecalcIterator<kRatio> &pcit, uint16_t yfrac, svuint8x3_t topsrc,
      svuint8x3_t bottomsrc, const uint8_t *src_top_ptr,
      const uint8_t *src_bottom_ptr) const KLEIDICV_STREAMING {
    svuint8_t vsx0_idx = svld1(svptrue_b8(), pcit.idx_ptr_);
    svuint8_t vsx1_idx{};
    if constexpr (kSetRightmostLanes) {
      // Make room for the last one, which is loaded separately and put together
      // using EXT, for better performance
      vsx1_idx = svinsr_n_u8(vsx0_idx, 0);
      vsx1_idx = svadd_n_u8_x(svptrue_b8(), vsx1_idx, kChannels);
    } else {
      vsx1_idx = svadd_n_u8_x(svptrue_b8(), vsx0_idx, kChannels);
    }
    svuint8_t a =
        svtbl2_u8(svcreate2(svget3(topsrc, 0), svget3(topsrc, 1)), vsx0_idx);
    svuint8_t b =
        svtbl2_u8(svcreate2(svget3(topsrc, 0), svget3(topsrc, 1)), vsx1_idx);
    svuint8_t c = svtbl2_u8(
        svcreate2(svget3(bottomsrc, 0), svget3(bottomsrc, 1)), vsx0_idx);
    svuint8_t d = svtbl2_u8(
        svcreate2(svget3(bottomsrc, 0), svget3(bottomsrc, 1)), vsx1_idx);

    vsx0_idx =
        svsub_n_u8_x(svptrue_b8(), vsx0_idx, static_cast<uint8_t>(2 * kStep_));
    vsx1_idx =
        svsub_n_u8_x(svptrue_b8(), vsx1_idx, static_cast<uint8_t>(2 * kStep_));
    a = svtbx_u8(a, svget3(topsrc, 2), vsx0_idx);
    b = svtbx_u8(b, svget3(topsrc, 2), vsx1_idx);
    c = svtbx_u8(c, svget3(bottomsrc, 2), vsx0_idx);
    d = svtbx_u8(d, svget3(bottomsrc, 2), vsx1_idx);
    if constexpr (kSetRightmostLanes) {
      svbool_t pg = svptrue_pat_b8(SV_VL1);
      ptrdiff_t last_index = std::min(
          src_width_ * kChannels - 1,
          pcit.idx_ptr_[kStep_ - 1] + *pcit.src_index_ptr_ + kChannels);
      b = svext_u8(b, svld1_u8(pg, src_top_ptr + last_index), 1UL);
      d = svext_u8(d, svld1_u8(pg, src_bottom_ptr + last_index), 1UL);
    }
    return interpolate(pcit, yfrac, a, b, c, d);
  }

  svuint8_t vector_path_r3(const PrecalcIterator<kRatio> &pcit, uint16_t yfrac,
                           const uint8_t *src_top,
                           const uint8_t *src_bottom) const KLEIDICV_STREAMING {
    // Load 3*2*step elements, that's enough for 1/3 < scale < 1.0
    uint64_t src_index = *pcit.src_index_ptr_;
    svuint8x3_t topsrc = load8x3_u8(&src_top[src_index]);
    svuint8x3_t bottomsrc = load8x3_u8(&src_bottom[src_index]);
    return common_vector_path_r3(pcit, yfrac, topsrc, bottomsrc, src_top,
                                 src_bottom);
  }

  svuint8_t remaining_path_r3(const PrecalcIterator<kRatio> &pcit,
                              uint16_t yfrac, const uint8_t *src_top,
                              const uint8_t *src_bottom) const
      KLEIDICV_STREAMING {
    // Load 3*step elements, that's enough for 1/3 < scale < 1.0
    uint64_t src_index = *pcit.src_index_ptr_;
    svuint8x3_t topsrc = load8x3_while_u8(&src_top[src_index], src_index,
                                          src_width_ * kChannels);
    svuint8x3_t bottomsrc = load8x3_while_u8(&src_bottom[src_index], src_index,
                                             src_width_ * kChannels);
    return common_vector_path_r3(pcit, yfrac, topsrc, bottomsrc, src_top,
                                 src_bottom);
  }

  void process_row(uint64_t dy) const KLEIDICV_STREAMING {
    uint64_t sy_fixp = to_src_y(dy);
    ptrdiff_t sy = static_cast<ptrdiff_t>(sy_fixp >> kFixpBits);
    const uint8_t *src_top = &src_rows_.at(sy)[0];
    const uint8_t *src_bottom = &src_rows_.at(sy + 1)[0];
    uint8_t *dst = &dst_rows_.at(static_cast<ptrdiff_t>(dy))[0];
    uint8_t *dst_end = dst + dst_width_ * kChannels;
    // Get the highest 8 bits of the fractional part
    // This is a good compromise between accuracy and performance
    // Because the result is 8bits, the error only affects the least
    // significant 1-2 bits, see the accuracy calculation in kleidicv.h
    uint16_t yfrac =
        static_cast<uint16_t>((sy_fixp - (sy << kFixpBits)) >> (kFixpBits - 8));
    auto pcit = precalc_.begin();
    while (pcit.index_ + 1 < precalc_.n_iterations_2x()) {
      svuint8_t res0, res1;
      if constexpr (kRatio == 3) {
        res0 = vector_path_r3(pcit, yfrac, src_top, src_bottom);
        ++pcit;
        res1 = vector_path_r3(pcit, yfrac, src_top, src_bottom);
        ++pcit;
      } else if constexpr (kRatio == 2) {
        res0 = vector_path_r2(pcit, yfrac, src_top, src_bottom);
        ++pcit;
        res1 = vector_path_r2(pcit, yfrac, src_top, src_bottom);
        ++pcit;
      }
#if KLEIDICV_TARGET_SME2
      svst1(svptrue_c8(), dst, svcreate2(res0, res1));
#else
      svst1(svptrue_b8(), dst, res0);
      svst1_vnum(svptrue_b8(), dst, 1, res1);
#endif  // KLEIDICV_TARGET_SME2
      dst += 2 * kStep_;
    }

    // similar to above, but only a single vector path and with predicates
    while (pcit.index_ < precalc_.n_iterations()) {
      svbool_t pgdst = svwhilelt_b8_s64(0L, dst_end - dst);
      svuint8_t res;
      if constexpr (kRatio == 2) {
        res = remaining_path_r2(pcit, yfrac, src_top, src_bottom);
      } else if constexpr (kRatio == 3) {
        res = remaining_path_r3(pcit, yfrac, src_top, src_bottom);
      }
      svst1(pgdst, dst, res);
      ++pcit;
      dst += kStep_;
    }
  }

  const Rows<const uint8_t> src_rows_;
  const Rows<uint8_t> dst_rows_;
  const size_t src_width_;
  const size_t src_height_;
  const size_t y_begin_;
  const size_t y_end_;
  const size_t dst_width_;
  const size_t dst_height_;
  const ptrdiff_t kStep_;
  PrecalcIndicesFractions<kRatio, kChannels> precalc_;
};

}  // namespace resize_generic_u8

// ratio: number of vectors to load and resize to 1 vector
// - supported combinations of (ratio, channel): (2, 1), (2, 2), (3, 1), (3,
// 2)
template <int kRatio, int kChannels>
kleidicv_error_t kleidicv_resize_generic_stripe_u8_sc(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end,
    uint8_t *dst,  // NOLINT
    size_t dst_stride, size_t dst_width, size_t dst_height) KLEIDICV_STREAMING {
  if constexpr (kChannels == 3 && kRatio == 3) {
    double inverse_scale =
        static_cast<double>(src_width) / static_cast<double>(dst_width);
    if (inverse_scale >= 2.8) {
      // Rightmost lane(s) of b and d vectors need to be set separately, as
      // the loaded src registers don't have the last pixels
      resize_generic_u8::ResizeGenericU8Operation<kRatio, kChannels, true>
          operation(src, src_stride, src_width, src_height, y_begin, y_end, dst,
                    dst_stride, dst_width, dst_height);
      return operation.process_rows();
    }
  }

  resize_generic_u8::ResizeGenericU8Operation<kRatio, kChannels> operation(
      src, src_stride, src_width, src_height, y_begin, y_end, dst, dst_stride,
      dst_width, dst_height);
  return operation.process_rows();
}

#define KLEIDICV_INSTANTIATE_TEMPLATE_SC(ratio, channels)            \
  template kleidicv_error_t                                          \
  kleidicv_resize_generic_stripe_u8_sc<ratio, channels>(             \
      const uint8_t *src, size_t src_stride, size_t src_width,       \
      size_t src_height, size_t y_begin, size_t y_end, uint8_t *dst, \
      size_t dst_stride, size_t dst_width, size_t dst_height)        \
      KLEIDICV_STREAMING

KLEIDICV_INSTANTIATE_TEMPLATE_SC(2L, 1L);
KLEIDICV_INSTANTIATE_TEMPLATE_SC(2L, 2L);
KLEIDICV_INSTANTIATE_TEMPLATE_SC(2L, 3L);
KLEIDICV_INSTANTIATE_TEMPLATE_SC(3L, 1L);
KLEIDICV_INSTANTIATE_TEMPLATE_SC(3L, 2L);
KLEIDICV_INSTANTIATE_TEMPLATE_SC(3L, 3L);

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_RESIZE_LINEAR_GENERIC_SC_H
