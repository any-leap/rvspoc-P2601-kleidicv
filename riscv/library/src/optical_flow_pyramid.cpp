// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// Pyramid build/release/inspect + coarse-to-fine LK orchestration. Reuses
// upstream's template scaffold from kleidicv/analysis/* — those headers are
// pure C++ and only call public C entry points (kleidicv_blur_and_downsample_u8,
// kleidicv_scharr_interleaved_s16_u8, kleidicv_standalone_lucas_kanade_alg_u8)
// which we provide in this build. The _sme variants forward to the same
// scalar/RVV path because there is no AArch64 backend on RISC-V.

#include <cstddef>
#include <cstdint>

#include "kleidicv/analysis/build_optical_flow_pyr_lk_pyramid.h"
#include "kleidicv/analysis/calc_optical_flow_pyr_lk.h"
#include "kleidicv/kleidicv.h"

namespace kleidicv {

template <bool kUseSME>
static kleidicv_error_t build_optical_flow_pyr_lk_pyramid_impl(
    kleidicv_optical_flow_pyr_lk_pyramid_t** pyramid, const uint8_t* src,
    size_t src_stride, size_t width, size_t height, size_t channels,
    size_t level_count, size_t window_width, size_t window_height) {
  CHECK_POINTERS(pyramid);
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  size_t actual_level_count = 0;
  if (kleidicv_error_t err = validate_and_compute_build_optical_flow_pyr_lk_levels(
          src_stride, width, height, channels, level_count, window_width,
          window_height, &actual_level_count)) {
    return err;
  }

  OpticalFlowLKPyramid::Pointer pyramid_storage;
  if (kleidicv_error_t err = OpticalFlowLKPyramid::allocate(
          pyramid_storage, actual_level_count, width, height, channels,
          window_width, window_height)) {
    return err;
  }
  if (kleidicv_error_t err = pyramid_storage->create<kUseSME>(src, src_stride)) {
    return err;
  }
  *pyramid = reinterpret_cast<kleidicv_optical_flow_pyr_lk_pyramid_t*>(
      pyramid_storage.release());
  return KLEIDICV_OK;
}

}  // namespace kleidicv

extern "C" {

using kleidicv::OpticalFlowLKPyramid;
using kleidicv::OpticalFlowPyrLKCalc;

kleidicv_error_t kleidicv_build_optical_flow_pyr_lk_pyramid(
    kleidicv_optical_flow_pyr_lk_pyramid_t** pyramid, const uint8_t* src,
    size_t src_stride, size_t width, size_t height, size_t channels,
    size_t level_count, size_t window_width, size_t window_height) {
  return kleidicv::build_optical_flow_pyr_lk_pyramid_impl<false>(
      pyramid, src, src_stride, width, height, channels, level_count,
      window_width, window_height);
}

kleidicv_error_t kleidicv_build_optical_flow_pyr_lk_pyramid_sme(
    kleidicv_optical_flow_pyr_lk_pyramid_t** pyramid, const uint8_t* src,
    size_t src_stride, size_t width, size_t height, size_t channels,
    size_t level_count, size_t window_width, size_t window_height) {
  return kleidicv::build_optical_flow_pyr_lk_pyramid_impl<false>(
      pyramid, src, src_stride, width, height, channels, level_count,
      window_width, window_height);
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_release(
    kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid) {
  CHECK_POINTERS(pyramid);
  // NOLINTBEGIN(bugprone-unused-raii)
  OpticalFlowLKPyramid::Pointer{
      reinterpret_cast<OpticalFlowLKPyramid*>(pyramid)};
  // NOLINTEND(bugprone-unused-raii)
  return KLEIDICV_OK;
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_level_count(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid,
    size_t* level_count) {
  CHECK_POINTERS(pyramid, level_count);
  *level_count =
      reinterpret_cast<const OpticalFlowLKPyramid*>(pyramid)->level_count();
  return KLEIDICV_OK;
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_image_level(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid, size_t level,
    const uint8_t** data, size_t* stride, size_t* width, size_t* height) {
  CHECK_POINTERS(pyramid, data, stride, width, height);
  const auto* typed = reinterpret_cast<const OpticalFlowLKPyramid*>(pyramid);
  if (level >= typed->level_count()) return KLEIDICV_ERROR_RANGE;
  const auto& entry = typed->level(level);
  *data = entry.image_data;
  *stride = entry.image_stride;
  *width = entry.width;
  *height = entry.height;
  return KLEIDICV_OK;
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_scharr_level(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* pyramid, size_t level,
    const int16_t** data, size_t* stride, size_t* width, size_t* height) {
  CHECK_POINTERS(pyramid, data, stride, width, height);
  const auto* typed = reinterpret_cast<const OpticalFlowLKPyramid*>(pyramid);
  if (level >= typed->level_count()) return KLEIDICV_ERROR_RANGE;
  const auto& entry = typed->level(level);
  *data = entry.scharr_data;
  *stride = entry.scharr_stride;
  *width = entry.width;
  *height = entry.height;
  return KLEIDICV_OK;
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    const kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    const float* prev_points, float* next_points, size_t point_count,
    uint8_t* status, float* err, kleidicv_optflow_lk_context_t context) {
  return OpticalFlowPyrLKCalc::calc_from_pyramid<false>(
      prev_pyramid, next_pyramid, prev_points, next_points, point_count,
      status, err, context);
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8_from_pyramid_sme(
    const kleidicv_optical_flow_pyr_lk_pyramid_t* prev_pyramid,
    const kleidicv_optical_flow_pyr_lk_pyramid_t* next_pyramid,
    const float* prev_points, float* next_points, size_t point_count,
    uint8_t* status, float* err, kleidicv_optflow_lk_context_t context) {
  return OpticalFlowPyrLKCalc::calc_from_pyramid<false>(
      prev_pyramid, next_pyramid, prev_points, next_points, point_count,
      status, err, context);
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8(
    const uint8_t* prev_img, size_t prev_img_stride, const uint8_t* next_img,
    size_t next_img_stride, size_t width, size_t height, size_t channels,
    const float* prev_points, float* next_points, size_t point_count,
    uint8_t* status, float* err, kleidicv_optflow_lk_context_t context) {
  return OpticalFlowPyrLKCalc::calc_from_images<false>(
      prev_img, prev_img_stride, next_img, next_img_stride, width, height,
      channels, prev_points, next_points, point_count, status, err, context);
}

kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8_sme(
    const uint8_t* prev_img, size_t prev_img_stride, const uint8_t* next_img,
    size_t next_img_stride, size_t width, size_t height, size_t channels,
    const float* prev_points, float* next_points, size_t point_count,
    uint8_t* status, float* err, kleidicv_optflow_lk_context_t context) {
  return OpticalFlowPyrLKCalc::calc_from_images<false>(
      prev_img, prev_img_stride, next_img, next_img_stride, width, height,
      channels, prev_points, next_points, point_count, status, err, context);
}

}  // extern "C"
