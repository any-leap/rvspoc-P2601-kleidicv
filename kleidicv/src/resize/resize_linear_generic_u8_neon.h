// SPDX-FileCopyrightText: 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <variant>

#include "kleidicv/ctypes.h"
#include "kleidicv/neon.h"
#include "kleidicv/utils.h"

namespace kleidicv::neon::resize_linear_generic_u8 {

//------------------------------------------------------
/// Generic resize for ratios 1/3 to 1/1, u8
//------------------------------------------------------

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
static constexpr ptrdiff_t kStep = kVectorLength / sizeof(uint8_t);
static constexpr ptrdiff_t kHalfStep = kStep / 2;

struct FullVectorInterpolationConstants {
  uint8_t idx[kStep];
  uint16_t xfrac[kStep];
  ptrdiff_t src_element_index;
};

struct HalfVectorInterpolationConstants {
  uint8_t idx[kHalfStep];
  uint16_t xfrac[kHalfStep];
  ptrdiff_t src_element_index;
  ptrdiff_t dst_element_index;
};

struct VectorPathNums {
  size_t two_x;
  size_t half;

  explicit VectorPathNums(std::pair<size_t, size_t> sizes)
      : two_x{sizes.first}, half{sizes.second} {}
};

template <typename T = int64_t>
static T rounding_div(int64_t nom, int64_t denom) {
  return static_cast<T>((nom + denom / 2) / denom);
}

// Scale coordinate using this formula, so the center is aligned:
//   source_x = (destination_x + 0.5) / scale - 0.5;
//   plus 1/256/2 for later rounding the fractional part to 8bits
static inline int64_t aligned_scale(int64_t x, int64_t nom, int64_t denom) {
  return rounding_div(((x << kFixpBits) + kFixpHalf) * nom, denom) - kFixpHalf +
         (1 << (kFixpBits - 9));
}

class RowInterpolationConstants {
 public:
  // Constructible only through create
  RowInterpolationConstants() = delete;

  static std::variant<RowInterpolationConstants, kleidicv_error_t> create(
      VectorPathNums num_of_vector_paths) {
    {
      uint8_t *allocation = static_cast<uint8_t *>(malloc(
          num_of_vector_paths.two_x * 2 *
              sizeof(FullVectorInterpolationConstants) +
          num_of_vector_paths.half * sizeof(HalfVectorInterpolationConstants)));
      if (!allocation) {
        return KLEIDICV_ERROR_ALLOCATION;
      }

      return RowInterpolationConstants{num_of_vector_paths, allocation};
    }
    return KLEIDICV_OK;
  }

  VectorPathNums num_of_vector_paths() const { return num_of_vector_paths_; }

  FullVectorInterpolationConstants *full_vector_constants_array() const {
    return full_vector_constants_array_;
  }

  HalfVectorInterpolationConstants *half_vector_constants_array() const {
    return half_vector_constants_array_;
  }

 private:
  RowInterpolationConstants(VectorPathNums num_of_vector_paths, uint8_t *buffer)
      : buffer_{buffer, &std::free},
        full_vector_constants_array_{
            reinterpret_cast<FullVectorInterpolationConstants *>(buffer)},
        half_vector_constants_array_{
            reinterpret_cast<HalfVectorInterpolationConstants *>(
                full_vector_constants_array_ +
                (num_of_vector_paths.two_x * 2))},
        num_of_vector_paths_{num_of_vector_paths} {}

  using FreeDeleter = decltype(&std::free);
  std::unique_ptr<uint8_t, FreeDeleter> buffer_;
  FullVectorInterpolationConstants *const full_vector_constants_array_;
  HalfVectorInterpolationConstants *const half_vector_constants_array_;
  const VectorPathNums num_of_vector_paths_;
};

template <int kRatio, int kChannels>
class RowInterpolationConstantsGeneratorBase {
 protected:
  RowInterpolationConstantsGeneratorBase(size_t src_width, size_t dst_width)
      : src_width_{src_width},
        dst_width_{dst_width},
        vsidx_tbl_{2, 6, 10, 14, 18, 22, 26, 30},
        vsfrac_tbl_{1,  255, 5,  255, 9,  255, 13, 255,
                    17, 255, 21, 255, 25, 255, 29, 255},
        vsx_max_{vdupq_n_s32(static_cast<int32_t>(to_src_x(dst_width_) - 1))} {}

  std::pair<size_t, size_t> calculate_num_of_vector_paths() {
    size_t two_x = ((src_width_ * kChannels) >= (sizeof(uint8x16_t) * kRatio))
                       ? ((dst_width_ * kChannels) / (2 * kStep))
                       : 0;

    size_t remaining_dx_after_2x_cycle =
        (dst_width_ * kChannels) - (two_x * 2 * kStep);
    size_t half = align_up(remaining_dx_after_2x_cycle, kHalfStep) / kHalfStep;
    return {two_x, half};
  }

  // Scale destination x coordinate to source x coordinate, into fixed-point,
  // without center correction
  int32_t scale_x(size_t dx) const {
    return rounding_div<int32_t>(
        static_cast<int64_t>((dx * src_width_) << kFixpBits),
        static_cast<int64_t>(dst_width_));
  }

  int64_t to_src_x(size_t dx) const {
    return aligned_scale(static_cast<int64_t>(dx),
                         static_cast<int64_t>(src_width_),
                         static_cast<int64_t>(dst_width_));
  }

  int64_t clamp_src_x(int64_t sx, int64_t max_sx) const {
    return std::clamp(sx, int64_t{0}, max_sx);
  }

  uint8x16_t vadd_clamped(int32x4_t a, int32x4_t b) const {
    const int32x4_t zero = vdupq_n_s32(0);
    return vreinterpretq_u8_s32(
        vminq_s32(vmaxq_s32(vaddq_s32(a, b), zero), vsx_max_));
  }

  const size_t src_width_;
  const size_t dst_width_;
  const uint8x8_t vsidx_tbl_;
  const uint8x16_t vsfrac_tbl_;
  const int32x4_t vsx_max_;
};

template <int kRatio, int kChannels>
class RowInterpolationConstantsGenerator final
    : RowInterpolationConstantsGeneratorBase<kRatio, kChannels> {
 public:
  static constexpr size_t kFullVectorSrcReadSize =
      (kRatio == 1 && kChannels > 1) ? sizeof(uint8x16x2_t)
                                     : (sizeof(uint8x16_t) * kRatio);
  using Base = RowInterpolationConstantsGeneratorBase<kRatio, kChannels>;
  RowInterpolationConstantsGenerator(size_t src_width, size_t dst_width)
      : Base{src_width, dst_width},
        // These starting values are not aligned to center. The center alignment
        // must be added only once. When added to a center-aligned source_x
        // value, the result will be center-aligned.
        vsx0_0_{Base::scale_x(0), Base::scale_x(1 / kChannels),
                Base::scale_x(2 / kChannels), Base::scale_x(3 / kChannels)},
        vsx0_1_{Base::scale_x(4 / kChannels), Base::scale_x(5 / kChannels),
                Base::scale_x(6 / kChannels), Base::scale_x(7 / kChannels)},
        vsx0_2_{Base::scale_x(8 / kChannels), Base::scale_x(9 / kChannels),
                Base::scale_x(10 / kChannels), Base::scale_x(11 / kChannels)},
        vsx0_3_{Base::scale_x(12 / kChannels), Base::scale_x(13 / kChannels),
                Base::scale_x(14 / kChannels), Base::scale_x(15 / kChannels)} {}

  std::variant<RowInterpolationConstants, kleidicv_error_t> operator()() {
    VectorPathNums v{Base::calculate_num_of_vector_paths()};
    auto row_interpolation_constants_variant =
        RowInterpolationConstants::create(v);
    if (std::holds_alternative<kleidicv_error_t>(
            row_interpolation_constants_variant)) {
      // Creation failed with some error, return with the variant as it is
      return row_interpolation_constants_variant;
    }
    auto &row_interpolation_constants = *std::get_if<RowInterpolationConstants>(
        &row_interpolation_constants_variant);

    size_t dx = 0;
    int64_t sx_fixp = 0;

    // Calculate constants for full vectors

    // Maximum source coordinate for vector path 2x
    const int64_t max_sx_2x = static_cast<int64_t>(
        (Base::src_width_ * kChannels - kFullVectorSrcReadSize) / kChannels);

    // Difference in source x coordinate for one vector path
    const int64_t sx_fixp_vector_step = rounding_div<int64_t>(
        static_cast<int64_t>((Base::src_width_ * kStep / kChannels)
                             << kFixpBits),
        static_cast<int64_t>(Base::dst_width_));

    for (size_t i = 0;
         i < row_interpolation_constants.num_of_vector_paths().two_x; ++i) {
      // Repeatedly adding sx_fixp_vector_step is faster than scaling dx to sx,
      // but it accumulates fixed-point error; periodic recalibration resets it.
      // The maximum per-addition error of sx_fixp_vector_step is 0.5 / (1 <<
      // 16). Only the upper 8 bits of the 16-bit fractional part are used for
      // interpolation, so once the accumulated error reaches 1 / (1 << 8), it
      // can affect later stages. This corresponds to 512 additions. Since two
      // additions are performed per cycle, we recalibrate every 256 cycles,
      // calculated by this mask.
      constexpr uint64_t kRecalibrateCycleMask = ((1 << 8) - 1);
      if ((i & kRecalibrateCycleMask) == 0) {
        sx_fixp = Base::to_src_x(dx);
      }

      // Pull back sx if it would overrun
      int64_t sx_candidate = sx_fixp >> kFixpBits;
      int64_t sx_base = Base::clamp_src_x(sx_candidate, max_sx_2x);
      calculate_indices_fractions_base_2x(
          row_interpolation_constants.full_vector_constants_array()[i * 2],
          sx_base, sx_fixp);
      sx_fixp += sx_fixp_vector_step;
      dx += kStep / kChannels;

      // Pull back sx if it would overrun
      sx_candidate = sx_fixp >> kFixpBits;
      sx_base = Base::clamp_src_x(sx_candidate, max_sx_2x);
      calculate_indices_fractions_base_2x(
          row_interpolation_constants
              .full_vector_constants_array()[(i * 2) + 1],
          sx_base, sx_fixp);
      sx_fixp += sx_fixp_vector_step;
      dx += kStep / kChannels;
    }

    // Calculate constants for half vectors

    sx_fixp = Base::to_src_x(dx);

    // Difference in source x coordinate for one destination pixel
    const int64_t sx_fixp_one_dst_pixel = rounding_div<int64_t>(
        static_cast<int64_t>(Base::src_width_ << kFixpBits),
        static_cast<int64_t>(Base::dst_width_));
    // Maximum source coordinate for half vector path
    const int64_t max_sx_half =
        std::max(static_cast<int64_t>(Base::src_width_ * kChannels) -
                     static_cast<int64_t>(sizeof(uint8x16_t) *
                                          std::max(1, kRatio - 1)),
                 0L) /
        kChannels;
    // Maximum destination coordinate for half vector path
    const size_t max_dx_half = Base::dst_width_ - (kHalfStep / kChannels);
    // Difference in source x coordinate for the half vector path
    const int64_t sx_fixp_half_step = rounding_div<int64_t>(
        static_cast<int64_t>((Base::src_width_ * kHalfStep / kChannels)
                             << kFixpBits),
        static_cast<int64_t>(Base::dst_width_));

    for (size_t i = 0;
         i < row_interpolation_constants.num_of_vector_paths().half; ++i) {
      // If (dx + half vector length) would overrun the buffer, pull it back
      size_t dx_pulled_back = std::min(dx, max_dx_half);
      // Pull back sx if dx was pulled back
      sx_fixp -=
          static_cast<int64_t>(dx - dx_pulled_back) * sx_fixp_one_dst_pixel;
      dx = dx_pulled_back;
      // If (sx_base + reading length) would overrun the buffer, pull sx back
      // again
      int64_t sx_candidate = sx_fixp >> kFixpBits;
      int64_t sx_base = Base::clamp_src_x(sx_candidate, max_sx_half);
      calculate_indices_fractions_base_half(
          row_interpolation_constants.half_vector_constants_array()[i], sx_base,
          sx_fixp, dx);

      dx += kHalfStep / kChannels;
      sx_fixp += sx_fixp_half_step;
    }

    return row_interpolation_constants_variant;
  }

 private:
  void calculate_indices_fractions_base_2x(
      FullVectorInterpolationConstants &constants, int64_t sx_base,
      int64_t sx_fixp) {
    int32_t xfrac0 = static_cast<int32_t>(sx_fixp - (sx_base << kFixpBits));
    int32x4_t vfrac = vdupq_n_s32(xfrac0);
    // Calculate x coordinate delta from sx_base, the integer part of source x
    uint8x16x2_t vsx_delta_lo, vsx_delta_hi;
    if constexpr (kRatio == 1) {
      vsx_delta_lo.val[0] = Base::vadd_clamped(vsx0_0_, vfrac);
      vsx_delta_lo.val[1] = Base::vadd_clamped(vsx0_1_, vfrac);
      vsx_delta_hi.val[0] = Base::vadd_clamped(vsx0_2_, vfrac);
      vsx_delta_hi.val[1] = Base::vadd_clamped(vsx0_3_, vfrac);
    } else {
      vsx_delta_lo.val[0] = vreinterpretq_u8_s32(vaddq_s32(vsx0_0_, vfrac));
      vsx_delta_lo.val[1] = vreinterpretq_u8_s32(vaddq_s32(vsx0_1_, vfrac));
      vsx_delta_hi.val[0] = vreinterpretq_u8_s32(vaddq_s32(vsx0_2_, vfrac));
      vsx_delta_hi.val[1] = vreinterpretq_u8_s32(vaddq_s32(vsx0_3_, vfrac));
    }
    uint8x8_t idx0 = vqtbl2_u8(vsx_delta_lo, Base::vsidx_tbl_);
    uint8x8_t idx1 = vqtbl2_u8(vsx_delta_hi, Base::vsidx_tbl_);
    uint8x16_t vsx0_idx = vcombine_u8(idx0, idx1);
    if constexpr (kChannels > 1) {
      vsx0_idx = vshlq_n_u8(vsx0_idx, kChannels == 4 ? 2 : 1);
      vsx0_idx =
          vaddq_u8(vsx0_idx, vreinterpretq_u8_u32(vdupq_n_u32(
                                 kChannels == 4 ? 0x03020100U : 0x01000100)));
    }
    vst1q(constants.idx, vsx0_idx);
    uint16x8x2_t vsxfrac;
    vsxfrac.val[0] =
        vreinterpretq_u16_u8(vqtbl2q_u8(vsx_delta_lo, Base::vsfrac_tbl_));
    vsxfrac.val[1] =
        vreinterpretq_u16_u8(vqtbl2q_u8(vsx_delta_hi, Base::vsfrac_tbl_));
    VecTraits<uint16_t>::store(vsxfrac, constants.xfrac);
    constants.src_element_index = static_cast<ptrdiff_t>(sx_base * kChannels);
  }

  void calculate_indices_fractions_base_half(
      HalfVectorInterpolationConstants &constants, int64_t sx_base,
      int64_t sx_fixp, size_t dx) {
    int32_t xfrac0 = static_cast<int32_t>(sx_fixp - (sx_base << kFixpBits));
    int32x4_t vfrac = vdupq_n_s32(xfrac0);
    uint8x16x2_t vsx_delta;
    if constexpr (kRatio == 1) {
      vsx_delta.val[0] = Base::vadd_clamped(vsx0_0_, vfrac);
      vsx_delta.val[1] = Base::vadd_clamped(vsx0_1_, vfrac);
    } else {
      vsx_delta.val[0] = vreinterpretq_u8_s32(vaddq_s32(vsx0_0_, vfrac));
      vsx_delta.val[1] = vreinterpretq_u8_s32(vaddq_s32(vsx0_1_, vfrac));
    }
    uint8x8_t vsx0_idx = vqtbl2_u8(vsx_delta, Base::vsidx_tbl_);
    if constexpr (kChannels > 1) {
      vsx0_idx = vshl_n_u8(vsx0_idx, kChannels == 4 ? 2 : 1);
      vsx0_idx = vadd_u8(
          vsx0_idx, vreinterpret_u8_u32(
                        vdup_n_u32(kChannels == 4 ? 0x03020100U : 0x01000100)));
    }
    vst1(constants.idx, vsx0_idx);
    uint16x8_t vsxfrac =
        vreinterpretq_u16_u8(vqtbl2q_u8(vsx_delta, Base::vsfrac_tbl_));
    VecTraits<uint16_t>::store(vsxfrac, constants.xfrac);
    constants.src_element_index = static_cast<ptrdiff_t>(sx_base * kChannels);
    constants.dst_element_index = static_cast<ptrdiff_t>(dx * kChannels);
  }

  const int32x4_t vsx0_0_;
  const int32x4_t vsx0_1_;
  const int32x4_t vsx0_2_;
  const int32x4_t vsx0_3_;
};

template <int kRatio>
class RowInterpolationConstantsGenerator<kRatio, 3> final
    : RowInterpolationConstantsGeneratorBase<kRatio, 3> {
 public:
  static constexpr size_t kFullVectorSrcReadSize =
      (kRatio == 1) ? sizeof(uint8x16x2_t) : (sizeof(uint8x16_t) * kRatio);
  using Base = RowInterpolationConstantsGeneratorBase<kRatio, 3>;
  RowInterpolationConstantsGenerator(size_t src_width, size_t dst_width)
      : Base{src_width, dst_width},
        sx_fixp_one_dst_pixel_{
            rounding_div<int64_t>(static_cast<int64_t>(src_width << kFixpBits),
                                  static_cast<int64_t>(dst_width))} {}

  std::variant<RowInterpolationConstants, kleidicv_error_t> operator()() {
    VectorPathNums v{Base::calculate_num_of_vector_paths()};
    auto row_interpolation_constants_variant =
        RowInterpolationConstants::create(v);
    if (std::holds_alternative<kleidicv_error_t>(
            row_interpolation_constants_variant)) {
      // Creation failed with some error, return with the variant as it is
      return row_interpolation_constants_variant;
    }
    auto &row_interpolation_constants = *std::get_if<RowInterpolationConstants>(
        &row_interpolation_constants_variant);

    size_t dst_element_index = 0;
    int64_t sx_fixp{};

    // Calculate constants for full vectors

    size_t num_of_full_vector_constants =
        row_interpolation_constants.num_of_vector_paths().two_x * 2;
    if (num_of_full_vector_constants > 0) {
      size_t handled_full_vector_paths = 0;
      size_t first_vector_path_wout_pullback = 0;
      while (first_vector_path_wout_pullback < num_of_full_vector_constants &&
             vector_needs_pullback(first_vector_path_wout_pullback * kStep)) {
        first_vector_path_wout_pullback++;
      }
      size_t last_vector_path_wout_pullback = num_of_full_vector_constants;
      while (
          last_vector_path_wout_pullback > first_vector_path_wout_pullback &&
          vector_needs_pullback((last_vector_path_wout_pullback - 1) * kStep)) {
        last_vector_path_wout_pullback--;
      }

      // The triplet fast path assumes R/G/B-aligned vector groups. Leading
      // or trailing vectors that need pullback are handled scalarly so the
      // vectorized middle range can stay aligned.
      size_t first_triplet_vector_path =
          align_up(first_vector_path_wout_pullback, size_t{3});

      fill_full_constants_scalar_range(
          row_interpolation_constants, handled_full_vector_paths,
          std::min(first_triplet_vector_path, num_of_full_vector_constants),
          dst_element_index, sx_fixp);

      size_t vectorizable_full_vector_paths =
          (last_vector_path_wout_pullback > handled_full_vector_paths)
              ? (last_vector_path_wout_pullback - handled_full_vector_paths)
              : 0;
      size_t vector_path_triplets_wout_pullback =
          vectorizable_full_vector_paths / 3;
      if (vector_path_triplets_wout_pullback > 0) {
        const int32x4x4_t vsx_r = gen_vsx_r();
        const uint8x16_t vsx_idx_diff_r = gen_vsx_idx_diff_r();
        const int32x4x4_t vsx_g = gen_vsx_g();
        const uint8x16_t vsx_idx_diff_g = gen_vsx_idx_diff_g();
        const int32x4x4_t vsx_b = gen_vsx_b();
        const uint8x16_t vsx_idx_diff_b = gen_vsx_idx_diff_b();

        // Difference in source x coordinate for 5 destination pixels
        const int64_t sx_fixp_five_dst_pixel = rounding_div<int64_t>(
            static_cast<int64_t>((Base::src_width_ * 5) << kFixpBits),
            static_cast<int64_t>(Base::dst_width_));
        // Difference in source x coordinate for 6 destination pixels
        const int64_t sx_fixp_six_dst_pixel = rounding_div<int64_t>(
            static_cast<int64_t>((Base::src_width_ * 6) << kFixpBits),
            static_cast<int64_t>(Base::dst_width_));

        sx_fixp = Base::to_src_x(dst_element_index / kChannels);
        unsigned recalibrate_cnt = 0;
        for (size_t i = 0; i < vector_path_triplets_wout_pullback; ++i) {
          // Repeatedly adding sx_fixp_five_dst_pixel and sx_fixp_six_dst_pixel
          // is faster than scaling dx to sx, but it accumulates fixed-point
          // error; periodic recalibration resets it. The maximum per-addition
          // error of these values is 0.5 / (1 << 16). Only the upper 8
          // bits of the 16-bit fractional part are used for interpolation, so
          // once the accumulated error reaches 1 / (1 << 8), it can affect
          // later stages. This corresponds to 512 additions. Since three
          // additions are performed per cycle, we recalibrate every 170 cycles.
          if (recalibrate_cnt == 170) {
            sx_fixp = Base::to_src_x(dst_element_index / kChannels);
            recalibrate_cnt = 0;
          } else {
            recalibrate_cnt++;
          }

          unsigned in_pixel_index = 0;
          fill_full_constants_vectorially(
              row_interpolation_constants
                  .full_vector_constants_array()[handled_full_vector_paths],
              vsx_r, vsx_idx_diff_r, sx_fixp, in_pixel_index);

          sx_fixp += sx_fixp_five_dst_pixel;
          in_pixel_index = 1;
          fill_full_constants_vectorially(
              row_interpolation_constants
                  .full_vector_constants_array()[handled_full_vector_paths + 1],
              vsx_g, vsx_idx_diff_g, sx_fixp, in_pixel_index);

          sx_fixp += sx_fixp_five_dst_pixel;
          in_pixel_index = 2;
          fill_full_constants_vectorially(
              row_interpolation_constants
                  .full_vector_constants_array()[handled_full_vector_paths + 2],
              vsx_b, vsx_idx_diff_b, sx_fixp, in_pixel_index);

          sx_fixp += sx_fixp_six_dst_pixel;
          handled_full_vector_paths += 3;
          dst_element_index += kStep * 3;
        }
      }

      fill_full_constants_scalar_range(
          row_interpolation_constants, handled_full_vector_paths,
          num_of_full_vector_constants, dst_element_index, sx_fixp);
    }

    // Calculate constants for half vectors

    // Maximum source coordinate for half vector path
    size_t half_vector_path_src_read_size =
        kChannels == 3 ? sizeof(uint8x16x2_t)
                       : (sizeof(uint8x16_t) * (kRatio - 1));
    const uint64_t max_src_base_index = saturating_sub(
        Base::src_width_ * kChannels, half_vector_path_src_read_size);
    // Maximum destination coordinate for half vector path
    const uint64_t max_dst_index_half =
        (Base::dst_width_ * kChannels) - kHalfStep;

    for (size_t i = 0;
         i < row_interpolation_constants.num_of_vector_paths().half; ++i) {
      auto &constants =
          row_interpolation_constants.half_vector_constants_array()[i];

      // If (dst index + half vector length) would overrun the buffer, pull it
      // back
      dst_element_index = std::min(dst_element_index, max_dst_index_half);

      size_t dx = dst_element_index / kChannels;
      unsigned in_pixel_index = dst_element_index % kChannels;
      sx_fixp = Base::to_src_x(dx);
      int64_t src_element_index =
          static_cast<int64_t>((sx_fixp >> kFixpBits) * kChannels) +
          in_pixel_index;

      // Pull back src if it would overrun
      int64_t src_element_base = Base::clamp_src_x(
          src_element_index, static_cast<int64_t>(max_src_base_index));

      fill_half_constants_scalarly(constants, dst_element_index, in_pixel_index,
                                   src_element_index, src_element_base,
                                   sx_fixp);

      dst_element_index += kHalfStep;
    }

    return row_interpolation_constants_variant;
  }

 private:
  bool vector_needs_pullback(size_t dst_idx) const {
    unsigned in_pixel_idx = dst_idx % kChannels;
    size_t dx = dst_idx / kChannels;
    int64_t sx_fixp = Base::to_src_x(dx);
    int64_t src_idx =
        static_cast<int64_t>((sx_fixp >> kFixpBits) * kChannels) + in_pixel_idx;

    return (src_idx + kFullVectorSrcReadSize) > Base::src_width_ * kChannels;
  }

  int32x4x4_t gen_vsx_r() {
    return int32x4x4_t{
        Base::scale_x(0), Base::scale_x(0), Base::scale_x(0), Base::scale_x(1),
        Base::scale_x(1), Base::scale_x(1), Base::scale_x(2), Base::scale_x(2),
        Base::scale_x(2), Base::scale_x(3), Base::scale_x(3), Base::scale_x(3),
        Base::scale_x(4), Base::scale_x(4), Base::scale_x(4), Base::scale_x(5)};
  }
  uint8x16_t gen_vsx_idx_diff_r() {
    return uint8x16_t{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
  }

  int32x4x4_t gen_vsx_g() {
    return int32x4x4_t{
        Base::scale_x(0), Base::scale_x(0), Base::scale_x(1), Base::scale_x(1),
        Base::scale_x(1), Base::scale_x(2), Base::scale_x(2), Base::scale_x(2),
        Base::scale_x(3), Base::scale_x(3), Base::scale_x(3), Base::scale_x(4),
        Base::scale_x(4), Base::scale_x(4), Base::scale_x(5), Base::scale_x(5)};
  }
  uint8x16_t gen_vsx_idx_diff_g() {
    if constexpr (kRatio == 1) {
      return uint8x16_t{1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1};
    }
    return uint8x16_t{0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1};
  }

  int32x4x4_t gen_vsx_b() {
    return int32x4x4_t{
        Base::scale_x(0), Base::scale_x(1), Base::scale_x(1), Base::scale_x(1),
        Base::scale_x(2), Base::scale_x(2), Base::scale_x(2), Base::scale_x(3),
        Base::scale_x(3), Base::scale_x(3), Base::scale_x(4), Base::scale_x(4),
        Base::scale_x(4), Base::scale_x(5), Base::scale_x(5), Base::scale_x(5)};
  }
  uint8x16_t gen_vsx_idx_diff_b() {
    if constexpr (kRatio == 1) {
      return uint8x16_t{2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
    }
    return uint8x16_t{0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  }

  void fill_full_constants_vectorially(
      FullVectorInterpolationConstants &constants, int32x4x4_t vsx,
      uint8x16_t vsx_idx_diff, int64_t sx_fixp, unsigned in_pixel_index) {
    const int64_t max_src_base_index =
        std::max<int64_t>(static_cast<int64_t>(Base::src_width_ * kChannels) -
                              static_cast<int64_t>(kFullVectorSrcReadSize),
                          0);
    int64_t src_element_base{};
    int64_t idx_adjust = 0;
    int32_t xfrac0{};
    if constexpr (kRatio == 1) {
      src_element_base = Base::clamp_src_x((sx_fixp >> kFixpBits) * kChannels,
                                           max_src_base_index);
      // The r1 path may need to look back from G/B into the R element of the
      // same source pixel, so keep the table R-aligned.
      xfrac0 = static_cast<int32_t>(
          sx_fixp - ((src_element_base / kChannels) << kFixpBits));
    } else {
      int64_t src_element_index_base =
          static_cast<int64_t>((sx_fixp >> kFixpBits) * kChannels) +
          in_pixel_index;
      src_element_base =
          Base::clamp_src_x(src_element_index_base, max_src_base_index);
      idx_adjust = src_element_index_base - src_element_base;
      xfrac0 = static_cast<int32_t>(sx_fixp & ((1 << kFixpBits) - 1));
    }
    constants.src_element_index = static_cast<ptrdiff_t>(src_element_base);

    // Create x coordinate for all lanes.
    int32x4_t vfrac = vdupq_n_s32(xfrac0);
    uint8x16x2_t vsx_delta_lo, vsx_delta_hi;
    if constexpr (kRatio == 1) {
      vsx_delta_lo.val[0] = Base::vadd_clamped(vsx.val[0], vfrac);
      vsx_delta_lo.val[1] = Base::vadd_clamped(vsx.val[1], vfrac);
      vsx_delta_hi.val[0] = Base::vadd_clamped(vsx.val[2], vfrac);
      vsx_delta_hi.val[1] = Base::vadd_clamped(vsx.val[3], vfrac);
    } else {
      vsx_delta_lo.val[0] = vreinterpretq_u8_s32(vaddq_s32(vsx.val[0], vfrac));
      vsx_delta_lo.val[1] = vreinterpretq_u8_s32(vaddq_s32(vsx.val[1], vfrac));
      vsx_delta_hi.val[0] = vreinterpretq_u8_s32(vaddq_s32(vsx.val[2], vfrac));
      vsx_delta_hi.val[1] = vreinterpretq_u8_s32(vaddq_s32(vsx.val[3], vfrac));
    }
    // Get index from coordinate
    uint8x8_t idx0 = vqtbl2_u8(vsx_delta_lo, Base::vsidx_tbl_);
    uint8x8_t idx1 = vqtbl2_u8(vsx_delta_hi, Base::vsidx_tbl_);
    uint8x16_t vsx0_idx = vcombine_u8(idx0, idx1);
    // One step in x means 3 steps in elements.
    vsx0_idx = vmulq_u8(vsx0_idx, vdupq_n_u8(3));
    if constexpr (kRatio == 1) {
      // The r1 path encodes the channel offsets directly in vsx_idx_diff.
      vsx0_idx = vaddq_u8(vsx0_idx, vsx_idx_diff);
      vst1q(constants.idx, vsx0_idx);
    } else {
      // Align the stepping if the first lane is green or blue.
      vsx0_idx = vqsubq_u8(vsx0_idx, vdupq_n_u8(in_pixel_index));
      // Add in-pixel index.
      vsx0_idx = vaddq_u8(vsx0_idx, vsx_idx_diff);
      vst1q(constants.idx, vsx0_idx);
      if (idx_adjust > 0) {
        for (size_t i = 0; i < kStep; ++i) {
          constants.idx[i] = static_cast<uint8_t>(
              static_cast<int64_t>(constants.idx[i]) + idx_adjust);
        }
      }
    }

    // Get fraction from coordinate
    uint16x8x2_t vsxfrac;
    vsxfrac.val[0] =
        vreinterpretq_u16_u8(vqtbl2q_u8(vsx_delta_lo, Base::vsfrac_tbl_));
    vsxfrac.val[1] =
        vreinterpretq_u16_u8(vqtbl2q_u8(vsx_delta_hi, Base::vsfrac_tbl_));
    VecTraits<uint16_t>::store(vsxfrac, constants.xfrac);
  }

  void fill_full_constants_scalar_range(
      RowInterpolationConstants &row_interpolation_constants,
      size_t &handled_full_vector_paths, size_t end_full_vector_paths,
      size_t &dst_element_index, int64_t &sx_fixp) {
    const int64_t max_src_base_index =
        std::max<int64_t>(static_cast<int64_t>(Base::src_width_ * kChannels) -
                              static_cast<int64_t>(kFullVectorSrcReadSize),
                          0);

    while (handled_full_vector_paths < end_full_vector_paths) {
      auto &constants =
          row_interpolation_constants
              .full_vector_constants_array()[handled_full_vector_paths];
      size_t dx = dst_element_index / kChannels;
      unsigned in_pixel_index = dst_element_index % kChannels;
      sx_fixp = Base::to_src_x(dx);

      int64_t src_element_index =
          static_cast<int64_t>((sx_fixp >> kFixpBits) * kChannels) +
          in_pixel_index;
      int64_t src_element_base =
          Base::clamp_src_x(src_element_index, max_src_base_index);

      fill_full_constants_scalarly(constants, in_pixel_index, src_element_index,
                                   src_element_base, sx_fixp);
      handled_full_vector_paths++;
      dst_element_index += kStep;
    }
  }

  void fill_full_constants_scalarly(FullVectorInterpolationConstants &constants,
                                    unsigned in_pixel_index,
                                    int64_t src_element_index,
                                    int64_t src_element_base, int64_t sx_fixp) {
    constants.src_element_index = static_cast<ptrdiff_t>(src_element_base);

    fill_idx_xfrac(constants, in_pixel_index, src_element_index,
                   src_element_base, sx_fixp);
  }

  void fill_half_constants_scalarly(HalfVectorInterpolationConstants &constants,
                                    size_t dst_element_index,
                                    unsigned in_pixel_index,
                                    int64_t src_element_index,
                                    int64_t src_element_base, int64_t sx_fixp) {
    constants.dst_element_index = static_cast<ptrdiff_t>(dst_element_index);
    constants.src_element_index = static_cast<ptrdiff_t>(src_element_base);

    fill_idx_xfrac(constants, in_pixel_index, src_element_index,
                   src_element_base, sx_fixp);
  }

  template <typename VectorConstants>
  void fill_idx_xfrac(VectorConstants &constants, unsigned in_pixel_index,
                      int64_t src_element_index, int64_t src_element_base,
                      int64_t sx_fixp) {
    // For indexing inside idx and xfrac arrays of
    // the interpolation constants
    unsigned j = 0;
    uint8_t idx = static_cast<uint8_t>(
        std::max<int64_t>(src_element_index - src_element_base, 0));
    uint16_t xfrac =
        (src_element_index < src_element_base)
            ? 0
            : (sx_fixp & ((1 << kFixpBits) - 1)) >> (kFixpBits / 2);

    for (; j < (kChannels - in_pixel_index); ++j) {
      constants.idx[j] = idx + j;
      constants.xfrac[j] = xfrac;
    }

    sx_fixp += sx_fixp_one_dst_pixel_;
    src_element_index =
        static_cast<int64_t>((sx_fixp >> kFixpBits) * kChannels);
    idx = static_cast<uint8_t>(src_element_index - src_element_base);
    xfrac = (sx_fixp & ((1 << kFixpBits) - 1)) >> (kFixpBits / 2);

    constexpr size_t idx_frac_elem_num = sizeof(VectorConstants::idx);

    while (j < idx_frac_elem_num) {
      // k is the index for the elements in one pixel
      for (unsigned k = 0; (j < idx_frac_elem_num) && (k < kChannels);
           ++j, ++k) {
        constants.idx[j] = idx + k;
        constants.xfrac[j] = xfrac;
      }
      sx_fixp += sx_fixp_one_dst_pixel_;
      src_element_index =
          static_cast<int64_t>((sx_fixp >> kFixpBits) * kChannels);
      idx = static_cast<uint8_t>(
          std::max<int64_t>(src_element_index - src_element_base, 0));
      xfrac = (src_element_index < src_element_base)
                  ? 0
                  : (sx_fixp & ((1 << kFixpBits) - 1)) >> (kFixpBits / 2);
    }
  }

  static constexpr size_t kChannels = 3;
  // Difference in source x coordinate for one destination pixel
  const int64_t sx_fixp_one_dst_pixel_;
};

template <int kRatio, int kChannels, bool kSetRightmostLanes = false>
class ResizeGenericU8Operation final {
 public:
  ResizeGenericU8Operation(const uint8_t *src, size_t src_stride,
                           size_t src_width, size_t src_height, size_t y_begin,
                           size_t y_end, uint8_t *dst, size_t dst_stride,
                           size_t dst_height)
      : src_rows_{src, src_stride, kChannels},
        dst_rows_{dst, dst_stride, kChannels},
        src_width_{src_width},
        src_height_{src_height},
        y_begin_{y_begin},
        y_end_{y_end},
        dst_height_{dst_height} {}

  void process_rows(RowInterpolationConstants &row_interpolation_constants) {
    for (size_t dst_y = y_begin_; dst_y < y_end_; ++dst_y) {
      process_row(dst_y, row_interpolation_constants);
    }
  }

 private:
  int64_t to_src_y(size_t dy) const {
    return aligned_scale(static_cast<int64_t>(dy),
                         static_cast<int64_t>(src_height_),
                         static_cast<int64_t>(dst_height_));
  }

  void process_row(size_t dy,
                   RowInterpolationConstants &row_interpolation_constants) {
    VectorPathNums num_of_vector_paths =
        row_interpolation_constants.num_of_vector_paths();
    auto *full_array =
        row_interpolation_constants.full_vector_constants_array();
    auto *half_array =
        row_interpolation_constants.half_vector_constants_array();

    int64_t sy_fixp = to_src_y(dy);
    ptrdiff_t sy = static_cast<ptrdiff_t>(sy_fixp >> kFixpBits);
    const ptrdiff_t max_sy = static_cast<ptrdiff_t>(src_height_ - 1);
    ptrdiff_t sy_top = std::clamp(sy, ptrdiff_t{0}, max_sy);
    ptrdiff_t sy_bottom = std::clamp(sy + 1, ptrdiff_t{0}, max_sy);
    const uint8_t *src_top = &src_rows_.at(sy_top)[0];
    const uint8_t *src_bottom = &src_rows_.at(sy_bottom)[0];
    uint8_t *dst = &dst_rows_.at(static_cast<ptrdiff_t>(dy))[0];
    // Get the highest 8 bits of the fractional part
    // This is a good compromise between accuracy and performance
    // Because the result is 8bits, the error only affects the least
    // significant 1-2 bits, see the accuracy calculation in kleidicv.h
    const int64_t sy_fixp_base = sy * (int64_t{1} << kFixpBits);
    uint16_t yfrac =
        static_cast<uint16_t>((sy_fixp - sy_fixp_base) >> (kFixpBits - 8));

    ptrdiff_t dst_element_index = 0;

    for (size_t i = 0; i < num_of_vector_paths.two_x; i += 1) {
      uint8x16x2_t res{};
      res.val[0] = vector_path(full_array[i * 2], src_top, src_bottom, yfrac);
      res.val[1] =
          vector_path(full_array[(i * 2) + 1], src_top, src_bottom, yfrac);
      VecTraits<uint8_t>::store(res, &dst[dst_element_index]);
      dst_element_index += kStep * 2;
    }

    for (size_t i = 0; i < num_of_vector_paths.half; i += 1) {
      auto res = vector_path_half(half_array[i], yfrac, src_top, src_bottom);
      vst1(&dst[half_array[i].dst_element_index], res);
    }
  }

  uint8x8_t vector_path_half(const HalfVectorInterpolationConstants &constants,
                             uint16_t yfrac, const uint8_t *src_top,
                             const uint8_t *src_bottom) const {
    uint8x8_t vsx0_idx = vld1_u8(constants.idx);
    uint8x8_t vsx1_idx = vadd_u8(vsx0_idx, vdup_n_u8(kChannels));
    ptrdiff_t src_element_index = constants.src_element_index;
    if constexpr (kRatio == 1) {
      uint8_t max_right_idx_data[kHalfStep];
      const ptrdiff_t last_pixel_index =
          static_cast<ptrdiff_t>(src_width_ * kChannels) -
          static_cast<ptrdiff_t>(kChannels);
      const unsigned start_channel =
          static_cast<unsigned>(constants.dst_element_index % kChannels);
      for (size_t i = 0; i < kHalfStep; ++i) {
        const unsigned channel =
            static_cast<unsigned>((start_channel + i) % kChannels);
        max_right_idx_data[i] = static_cast<uint8_t>(std::max<ptrdiff_t>(
            last_pixel_index + static_cast<ptrdiff_t>(channel) -
                src_element_index,
            0));
      }
      vsx1_idx = vmin_u8(vsx1_idx, vld1_u8(max_right_idx_data));
    }
    uint16x8_t vsxfrac;
    VecTraits<uint16_t>::load(constants.xfrac, vsxfrac);

    using SrcVecType =
        std::conditional_t<(kRatio == 1 || kRatio == 2) && kChannels != 3,
                           uint8x16_t, uint8x16x2_t>;
    SrcVecType topsrc, bottomsrc;
    VecTraits<uint8_t>::load(&src_top[src_element_index], topsrc);
    VecTraits<uint8_t>::load(&src_bottom[src_element_index], bottomsrc);

    uint8x8_t a, b, c, d;
    if constexpr ((kRatio == 1 || kRatio == 2) && kChannels != 3) {
      a = vqtbl1_u8(topsrc, vsx0_idx);
      b = vqtbl1_u8(topsrc, vsx1_idx);
      c = vqtbl1_u8(bottomsrc, vsx0_idx);
      d = vqtbl1_u8(bottomsrc, vsx1_idx);
    } else if constexpr (kRatio == 3 || kChannels == 3) {
      a = vqtbl2_u8(topsrc, vsx0_idx);
      b = vqtbl2_u8(topsrc, vsx1_idx);
      c = vqtbl2_u8(bottomsrc, vsx0_idx);
      d = vqtbl2_u8(bottomsrc, vsx1_idx);
    }
    uint8x8_t left =
        vraddhn_u16(vshll_n_u8(a, 8), vmulq_n_u16(vsubl_u8(c, a), yfrac));
    uint8x8_t right =
        vraddhn_u16(vshll_n_u8(b, 8), vmulq_n_u16(vsubl_u8(d, b), yfrac));
    uint8x8_t res = vraddhn_u16(vshll_n_u8(left, 8),
                                vmulq_u16(vsubl_u8(right, left), vsxfrac));
    return res;
  }

  uint8x16_t vector_path(const FullVectorInterpolationConstants &constants,
                         const uint8_t *src_top, const uint8_t *src_bottom,
                         uint16_t yfrac) const {
    uint8x16_t vsx0_idx = vld1q(constants.idx);
    uint8x16_t vsx1_idx = vaddq_u8(vsx0_idx, vdupq_n_u8(kChannels));
    if constexpr (kRatio == 1) {
      if constexpr (kChannels == 1) {
        vsx1_idx = vminq_u8(vsx1_idx, vdupq_n_u8(15));
      } else if constexpr (kChannels == 2) {
        vsx1_idx =
            vminq_u8(vsx1_idx, uint8x16_t{30, 31, 30, 31, 30, 31, 30, 31, 30,
                                          31, 30, 31, 30, 31, 30, 31});
      } else if constexpr (kChannels == 3) {
        vsx1_idx =
            vminq_u8(vsx1_idx, uint8x16_t{29, 30, 31, 29, 30, 31, 29, 30, 31,
                                          29, 30, 31, 29, 30, 31, 29});
      }
    }
    uint16x8x2_t vsxfrac2;
    VecTraits<uint16_t>::load(constants.xfrac, vsxfrac2);
    ptrdiff_t src_element_index = constants.src_element_index;

    using SrcVecType = std::conditional_t<
        kRatio == 1,
        std::conditional_t<kChannels == 1, uint8x16_t, uint8x16x2_t>,
        std::conditional_t<kRatio == 2, uint8x16x2_t, uint8x16x3_t>>;
    SrcVecType topsrc, bottomsrc;
    VecTraits<uint8_t>::load(&src_top[src_element_index], topsrc);
    VecTraits<uint8_t>::load(&src_bottom[src_element_index], bottomsrc);
    uint8x16_t a, b, c, d;
    if constexpr (kRatio == 1) {
      if constexpr (kChannels == 1) {
        a = vqtbl1q_u8(topsrc, vsx0_idx);
        b = vqtbl1q_u8(topsrc, vsx1_idx);
        c = vqtbl1q_u8(bottomsrc, vsx0_idx);
        d = vqtbl1q_u8(bottomsrc, vsx1_idx);
      } else {
        a = vqtbl2q_u8(topsrc, vsx0_idx);
        b = vqtbl2q_u8(topsrc, vsx1_idx);
        c = vqtbl2q_u8(bottomsrc, vsx0_idx);
        d = vqtbl2q_u8(bottomsrc, vsx1_idx);
      }
      static_assert(!kSetRightmostLanes);
    } else if constexpr (kRatio == 2) {
      a = vqtbl2q_u8(topsrc, vsx0_idx);
      b = vqtbl2q_u8(topsrc, vsx1_idx);
      c = vqtbl2q_u8(bottomsrc, vsx0_idx);
      d = vqtbl2q_u8(bottomsrc, vsx1_idx);
      static_assert(!kSetRightmostLanes);
    } else if constexpr (kRatio == 3) {
      a = vqtbl3q_u8(topsrc, vsx0_idx);
      b = vqtbl3q_u8(topsrc, vsx1_idx);
      c = vqtbl3q_u8(bottomsrc, vsx0_idx);
      d = vqtbl3q_u8(bottomsrc, vsx1_idx);
      // table lookup would overindex topsrc and bottomsrc
      if constexpr (kSetRightmostLanes) {
        ptrdiff_t last_right_elem_idx =
            src_element_index + constants.idx[15] + kChannels;
        b = vsetq_lane_u8(src_top[last_right_elem_idx], b, 15);
        d = vsetq_lane_u8(src_bottom[last_right_elem_idx], d, 15);
      }
    }
    uint8x8_t left_lo = lerp_low_half(a, c, yfrac);
    uint8x8_t left_hi = lerp_high_half(a, c, yfrac);
    uint8x8_t right_lo = lerp_low_half(b, d, yfrac);
    uint8x8_t right_hi = lerp_high_half(b, d, yfrac);
    uint8x8_t res_lo = lerp_full(left_lo, right_lo, vsxfrac2.val[0]);
    uint8x8_t res_hi = lerp_full(left_hi, right_hi, vsxfrac2.val[1]);
    return vcombine_u8(res_lo, res_hi);
  }

  static uint8x8_t lerp_low_half(uint8x16_t a, uint8x16_t b, uint16_t w) {
    return vraddhn_u16(
        vshll_n_u8(vget_low_u8(a), 8),
        vmulq_n_u16(vsubl_u8(vget_low_u8(b), vget_low_u8(a)), w));
  }

  static uint8x8_t lerp_high_half(uint8x16_t a, uint8x16_t b, uint16_t w) {
    return vraddhn_u16(vshll_high_n_u8(a, 8),
                       vmulq_n_u16(vsubl_high_u8(b, a), w));
  }

  static uint8x8_t lerp_full(uint8x8_t a, uint8x8_t b, uint16x8_t w) {
    return vraddhn_u16(vshll_n_u8(a, 8), vmulq_u16(vsubl_u8(b, a), w));
  }

  const Rows<const uint8_t> src_rows_;
  const Rows<uint8_t> dst_rows_;
  const size_t src_width_;
  const size_t src_height_;
  const size_t y_begin_;
  const size_t y_end_;
  const size_t dst_height_;
};

}  // namespace kleidicv::neon::resize_linear_generic_u8
