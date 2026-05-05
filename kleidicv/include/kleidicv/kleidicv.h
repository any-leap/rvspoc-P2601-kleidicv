// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

/// @mainpage
///
/// For documentation of the KleidiCV C API see @ref kleidicv.h
///
/// Project page: https://gitlab.arm.com/kleidi/kleidicv
///
/// KleidiCV shall use <a href="https://semver.org/spec/v2.0.0.html">Semantic
/// Versioning 2.0.0</a>. The public API is defined according to what is
/// included in the Doxygen-generated documentation as well as the content of
/// the project's Markdown files. Features without such documentation are not
/// part of the public API and should not be relied upon.
///
/// @see <a href="coverage/coverage_report.html">Code Coverage Report</a>

/// @file
/// @brief For an overview of the functions and their supported types see
/// @ref doc/functionality.md.
///
/// Dispatching mechanism:
/// - Each function without the `_sme` postfix uses the Neon or SVE2
///   backends by default, unless the `KLEIDICV_PREFER_SME_BACKEND`
///   environment variable is set to `ON` when the library is loaded.
///   If this variable is set, the function uses the same dispatching
///   strategy as its `_sme` counterpart. If no `_sme` counterpart exists,
///   setting this variable has no effect.
///
/// - Functions with the `_sme` postfix are backend-selection variants of
///   the same operations. They use the SME/SME2 backends if these are
///   available, and fall back to the Neon or SVE2 backends otherwise.
///   The `_sme` postfix does not change algorithm semantics or parameter
///   contracts.
///
/// - The default decision path between Neon and SVE2 checks whether SVE2
///   is available and whether it is expected to provide a benefit over
///   the Neon code path. If so, SVE2 is used. This behavior can be
///   overridden by setting the
///   `KLEIDICV_LIMIT_SVE2_TO_SELECTED_ALGORITHMS` CMake variable to `OFF`,
///   in which case SVE2 is always used when available.
///
/// - The decision path between SME and SME2 is similar: it checks whether
///   SME2 is available and whether it is expected to provide a benefit
///   over the SME code path. If so, SME2 is used. This behavior can be
///   overridden by setting the
///   `KLEIDICV_LIMIT_SME2_TO_SELECTED_ALGORITHMS` CMake variable to `OFF`,
///   in which case SME2 is always used when available.
///
/// Unless stated otherwise, in-place operations are not supported.

/// @defgroup kleidicv_api KleidiCV C API
/// @brief Public C API for KleidiCV.

/// @defgroup kleidicv_analysis Analysis
/// @ingroup kleidicv_api
/// @brief Analysis and statistics operations.

/// @defgroup kleidicv_arithmetics Arithmetics
/// @ingroup kleidicv_api
/// @brief Arithmetic and element-wise operations.

/// @defgroup kleidicv_conversions Conversions
/// @ingroup kleidicv_api
/// @brief Color and type conversion operations.

/// @defgroup kleidicv_filters Filters
/// @ingroup kleidicv_api
/// @brief Filtering operations.

/// @defgroup kleidicv_morphology Morphology
/// @ingroup kleidicv_api
/// @brief Morphological operations.

/// @defgroup kleidicv_resize Resize
/// @ingroup kleidicv_api
/// @brief Resizing operations.

/// @defgroup kleidicv_transform Transform
/// @ingroup kleidicv_api
/// @brief Geometric transform operations.

#ifndef KLEIDICV_H
#define KLEIDICV_H

#ifndef __cplusplus
#include <stdbool.h>
#endif

#include "kleidicv/config.h"
#include "kleidicv/ctypes.h"

#if !defined(__aarch64__) && !(defined(__riscv) && (__riscv_xlen == 64))
#error "KleidiCV is only supported for aarch64 and riscv64"
#endif

#if defined(__aarch64__) || (defined(__riscv) && (__riscv_xlen == 64))
/// Maximum image size in pixels the library accepts.
///
/// In case of AArch64 it is limited to (almost) 256 terapixels. This way 16 bit
/// is left for any arithmetic operations around image size or width or height.
#define KLEIDICV_MAX_IMAGE_PIXELS ((1ULL << 48) - 1)
#endif

/// Size in bytes of the largest possible element type
#define KLEIDICV_MAXIMUM_TYPE_SIZE (8)

/// Maximum number of channels
#define KLEIDICV_MAXIMUM_CHANNEL_COUNT (4)

/// Maximum LK optical-flow window dimension accepted by pyramid builders.
///
/// The limit keeps internal stride/size arithmetic within range for supported
/// image/channel limits on AArch64.
#define KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE (2047)

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef DOXYGEN
#define KLEIDICV_API_DECLARATION(name, ...) kleidicv_error_t name(__VA_ARGS__)
#else
#define KLEIDICV_API_DECLARATION(name, ...) \
  extern kleidicv_error_t (*name)(__VA_ARGS__)
#endif

#define KLEIDICV_BINARY_OP(name, type)                                        \
  KLEIDICV_API_DECLARATION(name, const type *src_a, size_t src_a_stride,      \
                           const type *src_b, size_t src_b_stride, type *dst, \
                           size_t dst_stride, size_t width, size_t height)

#define KLEIDICV_BINARY_OP_SCALE(name, type, scaletype)                       \
  KLEIDICV_API_DECLARATION(name, const type *src_a, size_t src_a_stride,      \
                           const type *src_b, size_t src_b_stride, type *dst, \
                           size_t dst_stride, size_t width, size_t height,    \
                           scaletype scale)

/// @addtogroup kleidicv_analysis
/// @{

/// Counts how many nonzero elements are in the source data. Number of elements
/// is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param count        Pointer to variable to store result. Must be non-null.
///
KLEIDICV_API_DECLARATION(kleidicv_count_nonzeros_u8, const uint8_t *src,
                         size_t src_stride, size_t width, size_t height,
                         size_t *count);

#if KLEIDICV_EXPERIMENTAL_FEATURE_CANNY
/// Canny edge detector for uint8_t grayscale input. Output is also a `uint8_t`
/// grayscale image. Width and height are the same for input and output. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// The steps:
///  - Execute horizontal and vertical Sobel filtering with 3*3 kernels to
///    calculate the gradient approximation in both directions.
///  - Calculate magnitude approximation (by summing horizontal and vertical
///    gradient approximations) and apply lower threshold.
///  - Perform non-maxima suppression and high thresholding.
///  - Perform hysteresis: promote weak edges, which are connected to strong
///    edges, to strong edges.
///  - Suppress remaining weak edges.
///
/// @param src            Pointer to the source data. Must be non-null.
/// @param src_stride     Distance in bytes from the start of one row to the
///                       start of the next row in the source data. Must not be
///                       less than `width`, except for single-row images.
/// @param dst            Pointer to the destination data. Must be non-null.
/// @param dst_stride     Distance in bytes from the start of one row to the
///                       start of the next row in the destination data. Must
///                       not be less than `width`, except for single-row
///                       images.
/// @param width          Number of elements in a row.
/// @param height         Number of rows in the data.
/// @param low_threshold  Low threshold for the edge detector algorithm.
/// @param high_threshold High threshold for the edge detector algorithm.
///
KLEIDICV_API_DECLARATION(kleidicv_canny_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, double low_threshold,
                         double high_threshold);
#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_CANNY

/// Calculates minimum and maximum element value across the source data. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(type)` and no less than `width *
///                     sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row. Must be greater than 0.
/// @param height       Number of rows in the data. Must be greater than 0.
/// @param min_value    Pointer to save result minimum value to, or `nullptr` if
///                     minimum is not to be calculated.
/// @param max_value    Pointer to save result maximum value to, or `nullptr` if
///                     maximum is not to be calculated.
///
KLEIDICV_API_DECLARATION(kleidicv_min_max_u8, const uint8_t *src,
                         size_t src_stride, size_t width, size_t height,
                         uint8_t *min_value, uint8_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_u8_sme, const uint8_t *src,
                         size_t src_stride, size_t width, size_t height,
                         uint8_t *min_value, uint8_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_s8, const int8_t *src,
                         size_t src_stride, size_t width, size_t height,
                         int8_t *min_value, int8_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_s8_sme, const int8_t *src,
                         size_t src_stride, size_t width, size_t height,
                         int8_t *min_value, int8_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_u16, const uint16_t *src,
                         size_t src_stride, size_t width, size_t height,
                         uint16_t *min_value, uint16_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_u16_sme, const uint16_t *src,
                         size_t src_stride, size_t width, size_t height,
                         uint16_t *min_value, uint16_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_s16, const int16_t *src,
                         size_t src_stride, size_t width, size_t height,
                         int16_t *min_value, int16_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_s16_sme, const int16_t *src,
                         size_t src_stride, size_t width, size_t height,
                         int16_t *min_value, int16_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_s32, const int32_t *src,
                         size_t src_stride, size_t width, size_t height,
                         int32_t *min_value, int32_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_s32_sme, const int32_t *src,
                         size_t src_stride, size_t width, size_t height,
                         int32_t *min_value, int32_t *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_f32, const float *src,
                         size_t src_stride, size_t width, size_t height,
                         float *min_value, float *max_value);
/// @copydoc kleidicv_min_max_u8
KLEIDICV_API_DECLARATION(kleidicv_min_max_f32_sme, const float *src,
                         size_t src_stride, size_t width, size_t height,
                         float *min_value, float *max_value);

/// Finds minimum and maximum element value across the source data,
/// and returns their location in the source data as offset in bytes
/// from the source beginning. Number of elements is limited to
/// @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(type)` and no less than `width *
///                     sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row. Must be greater than 0.
/// @param height       Number of rows in the data. Must be greater than 0.
/// @param min_offset   Pointer to save result offset of minimum value to, or
///                     `nullptr` if minimum is not to be calculated.
/// @param max_offset   Pointer to save result offset of maximum value to, or
///                     `nullptr` if maximum is not to be calculated.
///
KLEIDICV_API_DECLARATION(kleidicv_min_max_loc_u8, const uint8_t *src,
                         size_t src_stride, size_t width, size_t height,
                         size_t *min_offset, size_t *max_offset);

/// Returns the sum of element values across the source data.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(type)` and no less than `width *
///                     sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param sum          Pointer to save the sum value to. Must be non-null.
///
KLEIDICV_API_DECLARATION(kleidicv_sum_f32, const float *src, size_t src_stride,
                         size_t width, size_t height, float *sum);
/// @copydoc kleidicv_sum_f32
KLEIDICV_API_DECLARATION(kleidicv_sum_f32_sme, const float *src,
                         size_t src_stride, size_t width, size_t height,
                         float *sum);

/// Builds an opaque handle for Lucas-Kanade optical flow that stores
/// precomputed image and Scharr pyramids.
///
/// The returned handle owns implementation-defined per-level storage used by
/// KleidiCV's LK optical-flow pipeline. Its internal memory layout, padding,
/// and buffer organization are intentionally not part of the public API and
/// may change between releases.
///
/// Public callers should treat the result as an opaque owned handle: create it
/// with @ref kleidicv_build_optical_flow_pyr_lk_pyramid and destroy it with
/// @ref kleidicv_optical_flow_pyr_lk_pyramid_release. Direct inspection or
/// dependence on the internal pyramid representation is not a supported public
/// use case.
///
/// The implementation always uses the following border behavior:
///   - pyramid image generation uses reflect_101
///   - Scharr derivatives are generated from a reflect_101-expanded image and
///     stored as interleaved dx/dy pairs
///
/// The generated Scharr level has the same width and height as its
/// corresponding image level. The Scharr destination channel count is
/// `channels * 2`.
///
/// Level generation stops early if the next level would satisfy:
/// `next_width <= window_width || next_height <= window_height`.
///
/// Internally, Scharr is computed from a reflect_101-bordered image, so level 0
/// only requires non-zero `width` and `height`. Additional levels are built
/// only while the previous level is large enough to be blur-and-downsampled and
/// the next level would remain strictly larger than the LK window in both
/// dimensions.
///
/// Examples:
/// - `level_count = 0` is rejected with @ref KLEIDICV_ERROR_RANGE.
/// - `level_count = 1` builds only level 0 (one image level + one Scharr
///    level), which is useful when only bordered image + Scharr for the source
///    level are needed.
/// - `level_count > 1` requests additional pyramid levels; actual generated
///   level count may still be smaller due to the early-stop rule above.
///
/// @param pyramid       Output pointer for the created pyramid handle.
///                      On success, `*pyramid` is set to the new handle.
///                      On failure, `*pyramid` is left unmodified.
///                      Any existing handle in `*pyramid` must be released by
///                      the caller before reuse.
/// @param src           Pointer to source image data. Must be non-null.
/// @param src_stride    Distance in bytes between source rows. Must not be
///                      less than `width * channels`, except for single-row
///                      images.
/// @param width         Source width in pixels. Must be greater than 0.
/// @param height        Source height in pixels. Must be greater than 0.
///                      The source image size (`width * height`) must not
///                      exceed @ref KLEIDICV_MAX_IMAGE_PIXELS.
/// @param channels      Number of channels in source image.
///                      Maximum supported value is
///                      @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
/// @param level_count   Requested maximum number of pyramid levels.
/// @param window_width  LK window width. Supported range is
///                      [3, @ref KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE].
/// @param window_height LK window height. Supported range is
///                      [3, @ref KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE].
///
kleidicv_error_t kleidicv_build_optical_flow_pyr_lk_pyramid(
    kleidicv_optical_flow_pyr_lk_pyramid_t **pyramid, const uint8_t *src,
    size_t src_stride, size_t width, size_t height, size_t channels,
    size_t level_count, size_t window_width, size_t window_height);

/// @copydoc kleidicv_build_optical_flow_pyr_lk_pyramid
kleidicv_error_t kleidicv_build_optical_flow_pyr_lk_pyramid_sme(
    kleidicv_optical_flow_pyr_lk_pyramid_t **pyramid, const uint8_t *src,
    size_t src_stride, size_t width, size_t height, size_t channels,
    size_t level_count, size_t window_width, size_t window_height);

/// Releases an LK optical-flow pyramid created by
/// @ref kleidicv_build_optical_flow_pyr_lk_pyramid.
///
/// @param pyramid       Pointer to an LK optical-flow pyramid.
///
kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_release(
    kleidicv_optical_flow_pyr_lk_pyramid_t *pyramid);

/// Computes pyramidal LK optical flow using prebuilt pyramids.
///
/// Pyramids must be handles originally created by
/// @ref kleidicv_build_optical_flow_pyr_lk_pyramid. Foreign handles,
/// reinterpreted objects, or pyramid objects modified after creation are not
/// valid inputs.
/// Level 0 dimensions and channel count must match between `prev_pyramid` and
/// `next_pyramid`.
///
/// `prev_points` stores the source point coordinates as interleaved x/y pairs.
/// `next_points` stores the tracked coordinates as interleaved x/y pairs.
/// If @ref KLEIDICV_USE_INITIAL_FLOW is set in `context.flags`, the values in
/// `next_points` are used as initial guesses. Otherwise, `prev_points` are used
/// as initial guesses.
///
/// If `point_count` is 0, the function returns success without accessing point
/// arrays.
///
/// @param prev_pyramid Pointer to the previous-frame LK pyramid. Must be
///                     non-null.
/// @param next_pyramid Pointer to the next-frame LK pyramid. Must be non-null.
/// @param prev_points  Pointer to `2 * point_count` float values containing
///                     x/y coordinates in the previous image. Must be non-null
///                     when `point_count > 0`.
/// @param next_points  Pointer to `2 * point_count` float values containing
///                     initial guesses and receiving tracked coordinates.
///                     Must be non-null when `point_count > 0`.
/// @param point_count  Number of points to track.
/// @param status       Pointer to `point_count` bytes with tracking status per
///                     point. Must be non-null when `point_count > 0`.
/// @param err          Optional pointer to `point_count` float values.
///                     When @ref KLEIDICV_GET_MIN_EIG is set, minimum
///                     eigenvalues are written. Otherwise, `err` receives the
///                     photometric error last produced by the coarse-to-fine
///                     solve. For points that remain valid through level 0,
///                     this is the level-0 photometric error. If a point is
///                     rejected at a finer level after a coarser level has
///                     already produced a photometric error, `err` may retain
///                     that earlier-level value. Callers should therefore use
///                     `status` to decide whether a photometric error value is
///                     meaningful.
/// @param context      LK tracking context. See @ref
///                     kleidicv_optflow_lk_context_t.
kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8_from_pyramid(
    const kleidicv_optical_flow_pyr_lk_pyramid_t *prev_pyramid,
    const kleidicv_optical_flow_pyr_lk_pyramid_t *next_pyramid,
    const float *prev_points, float *next_points, size_t point_count,
    uint8_t *status, float *err, kleidicv_optflow_lk_context_t context);

/// @copydoc kleidicv_optical_flow_pyr_lk_u8_from_pyramid
kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8_from_pyramid_sme(
    const kleidicv_optical_flow_pyr_lk_pyramid_t *prev_pyramid,
    const kleidicv_optical_flow_pyr_lk_pyramid_t *next_pyramid,
    const float *prev_points, float *next_points, size_t point_count,
    uint8_t *status, float *err, kleidicv_optflow_lk_context_t context);

/// Computes pyramidal LK optical flow and builds pyramids internally.
///
/// This function builds LK pyramids for `prev_img` and `next_img` using
/// @ref kleidicv_build_optical_flow_pyr_lk_pyramid, then performs tracking with
/// @ref kleidicv_optical_flow_pyr_lk_u8_from_pyramid.
///
/// The direct path still builds the full opaque pyramid for `next_img`,
/// including Scharr levels, even though the current solver only samples the
/// next-frame image levels. This keeps the prev/next pyramid handles
/// interchangeable, avoids a second pyramid representation with different
/// ownership rules, and helps streaming pipelines reuse one call's `next`
/// pyramid as a later call's `prev` pyramid through the from-pyramid API
/// instead of rebuilding or converting it.
///
/// `prev_points` stores the source point coordinates as interleaved x/y pairs.
/// `next_points` stores the tracked coordinates as interleaved x/y pairs.
/// If @ref KLEIDICV_USE_INITIAL_FLOW is set in `context.flags`, the values in
/// `next_points` are used as initial guesses. Otherwise, `prev_points` are used
/// as initial guesses.
///
/// If `point_count` is 0, the function returns success without accessing point
/// arrays.
///
/// @param prev_img         Pointer to previous image data. Must be non-null.
/// @param prev_img_stride  Distance in bytes between previous-image rows. Must
///                         not be less than `width * channels`, except for
///                         single-row images.
/// @param next_img         Pointer to next image data. Must be non-null.
/// @param next_img_stride  Distance in bytes between next-image rows. Must not
///                         be less than `width * channels`, except for
///                         single-row images.
/// @param width            Image width in pixels. Must be greater than 0.
/// @param height           Image height in pixels. Must be greater than 0.
/// @param channels         Number of image channels. Must be in range
///                         `[1, @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT]`.
/// @param prev_points      Pointer to `2 * point_count` float values containing
///                         x/y coordinates in the previous image. Must be
///                         non-null when `point_count > 0`.
/// @param next_points      Pointer to `2 * point_count` float values containing
///                         initial guesses and receiving tracked coordinates.
///                         Must be non-null when `point_count > 0`.
/// @param point_count      Number of points to track.
/// @param status           Pointer to `point_count` bytes with tracking status
///                         per point. Must be non-null when `point_count > 0`.
/// @param err              Optional pointer to `point_count` float values.
///                         When @ref KLEIDICV_GET_MIN_EIG is set, minimum
///                         eigenvalues are written. Otherwise, `err` receives
///                         the photometric error last produced by the
///                         coarse-to-fine solve. For points that remain valid
///                         through level 0, this is the level-0 photometric
///                         error. If a point is rejected at a finer level
///                         after a coarser level has already produced a
///                         photometric error, `err` may retain that
///                         earlier-level value. Callers should therefore use
///                         `status` to decide whether a photometric error
///                         value is meaningful.
/// @param context          LK tracking context. See
///                         @ref kleidicv_optflow_lk_context_t.
kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8(
    const uint8_t *prev_img, size_t prev_img_stride, const uint8_t *next_img,
    size_t next_img_stride, size_t width, size_t height, size_t channels,
    const float *prev_points, float *next_points, size_t point_count,
    uint8_t *status, float *err, kleidicv_optflow_lk_context_t context);

/// @copydoc kleidicv_optical_flow_pyr_lk_u8
kleidicv_error_t kleidicv_optical_flow_pyr_lk_u8_sme(
    const uint8_t *prev_img, size_t prev_img_stride, const uint8_t *next_img,
    size_t next_img_stride, size_t width, size_t height, size_t channels,
    const float *prev_points, float *next_points, size_t point_count,
    uint8_t *status, float *err, kleidicv_optflow_lk_context_t context);

#ifndef DOXYGEN
/// Internal - not part of the public API and its direct use is not supported.
///
/// Gets number of levels available in an LK optical-flow pyramid.
///
/// @param pyramid       Pointer to an LK optical-flow pyramid.
/// @param level_count   Output pointer for level count.
///
kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_level_count(
    const kleidicv_optical_flow_pyr_lk_pyramid_t *pyramid, size_t *level_count);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Gets an image-pyramid level from an LK optical-flow pyramid.
///
/// @param pyramid       Pointer to an LK optical-flow pyramid.
/// @param level         Zero-based level index.
/// @param data          Output pointer to level pixel data.
/// @param stride        Output row stride in bytes.
/// @param width         Output level width in pixels.
/// @param height        Output level height in pixels.
///
kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_image_level(
    const kleidicv_optical_flow_pyr_lk_pyramid_t *pyramid, size_t level,
    const uint8_t **data, size_t *stride, size_t *width, size_t *height);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Gets a Scharr-pyramid level from an LK optical-flow pyramid.
///
/// Scharr levels are interleaved dx/dy pairs with element type `int16_t`.
///
/// @param pyramid       Pointer to an LK optical-flow pyramid.
/// @param level         Zero-based level index.
/// @param data          Output pointer to Scharr level data.
/// @param stride        Output row stride in bytes.
/// @param width         Output level width in pixels.
/// @param height        Output level height in pixels.
///
kleidicv_error_t kleidicv_optical_flow_pyr_lk_pyramid_get_scharr_level(
    const kleidicv_optical_flow_pyr_lk_pyramid_t *pyramid, size_t level,
    const int16_t **data, size_t *stride, size_t *width, size_t *height);

/// Internal - not part of the public API and its direct use is not supported.
///
/// Tracks feature points between two frames using Lucas-Kanade optical flow
/// solver.
///
/// Gradients for the previous frame must be precomputed with a Scharr operator
/// and stored in `prev_deriv_data` as interleaved x/y derivatives (int16_t) for
/// each channel. `next_points` is both the initial guess and the output for the
/// tracked positions; points that go out of bounds or fail the texture check
/// are marked in `status`. If `get_min_eigen_vals` is true, `err` stores the
/// minimum eigenvalue per point; otherwise it stores the photometric error.
/// When `status` is provided, photometric error is only written for points with
/// non-zero status.
///
/// @param prev_data                 Pointer to the previous frame data. Must be
///                                  non-null.
/// @param prev_data_stride          Distance in bytes from the start of one row
///                                  to the start of the next row in
///                                  `prev_data`. Must be no less than
///                                  `width * channels * sizeof(uint8_t)`.
/// @param prev_deriv_data           Pointer to the precomputed Scharr
///                                  derivatives of the previous frame.
///                                  Derivatives are interleaved as Ix/Iy. Must
///                                  be non-null.
/// @param prev_deriv_stride         Distance in bytes from the start of one
///                                  derivative row to the start of the next
///                                  derivative row. Must be a multiple of
///                                  `sizeof(int16_t)` and no less than
///                                  `2 * width * channels * sizeof(int16_t)`.
/// @param next_data                 Pointer to the next frame data. Must be
///                                  non-null.
/// @param next_data_stride          Distance in bytes from the start of one row
///                                  to the start of the next row in
///                                  `next_data`. Must be no less than
///                                  `width * channels * sizeof(uint8_t)`.
/// @param width                     Number of elements in a row.
/// @param height                    Number of rows in the data.
/// @param channels                  Number of channels. Must be greater than 0
///                                  and no more than
///                                  @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
/// @param prev_points               Pointer to `2 * point_count` floats with
///                                  x/y coordinates in the previous frame. Must
///                                  be non-null when `point_count > 0`.
/// @param next_points               Pointer to `2 * point_count` floats with
///                                  x/y coordinates in the next frame. Serves
///                                  as both the input guess and the output.
///                                  Must be non-null when `point_count > 0`.
/// @param point_count               Number of points to track.
/// @param status                    Optional pointer to `point_count` bytes.
///                                  Each element is set to non-zero on success
///                                  and zero on failure.
/// @param err                       Optional pointer to `point_count` floats.
///                                  If `get_min_eigen_vals` is true, write
///                                  minimum eigenvalues into `err`; otherwise
///                                  write photometric errors. If `status` is
///                                  provided, photometric error is only written
///                                  for points with non-zero status.
/// @param window_width              Width of the integration window. Must be
///                                  no more than
///                                  @ref
///                                  KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE.
/// @param window_height             Height of the integration window. Must be
///                                  no more than
///                                  @ref
///                                  KLEIDICV_MAX_OPTICAL_FLOW_PYR_LK_WINDOW_SIZE.
/// @param termination_count         Maximum number of Newton iterations per
///                                  point.
/// @param termination_epsilon       Early stopping threshold on the squared
///                                  update step.
/// @param get_min_eigen_vals        Whether to write minimum eigenvalues (true)
///                                  or photometric errors (false) into `err`.
/// @param min_eigen_vals_threshold  Points whose minimum eigenvalue is strictly
///                                  less than this threshold are rejected and
///                                  marked as failed in `status`.
KLEIDICV_API_DECLARATION(kleidicv_standalone_lucas_kanade_alg_u8,
                         const uint8_t *prev_data, size_t prev_data_stride,
                         const int16_t *prev_deriv_data,
                         size_t prev_deriv_stride, const uint8_t *next_data,
                         size_t next_data_stride, int width, int height,
                         int channels, const float *prev_points,
                         float *next_points, size_t point_count,
                         uint8_t *status, float *err, int window_width,
                         int window_height, int termination_count,
                         double termination_epsilon, bool get_min_eigen_vals,
                         float min_eigen_vals_threshold);

/// @copydoc kleidicv_standalone_lucas_kanade_alg_u8
KLEIDICV_API_DECLARATION(kleidicv_standalone_lucas_kanade_alg_u8_sme,
                         const uint8_t *prev_data, size_t prev_data_stride,
                         const int16_t *prev_deriv_data,
                         size_t prev_deriv_stride, const uint8_t *next_data,
                         size_t next_data_stride, int width, int height,
                         int channels, const float *prev_points,
                         float *next_points, size_t point_count,
                         uint8_t *status, float *err, int window_width,
                         int window_height, int termination_count,
                         double termination_epsilon, bool get_min_eigen_vals,
                         float min_eigen_vals_threshold);
#endif  // DOXYGEN

/// @}

/// @addtogroup kleidicv_arithmetics
/// @{

/// Adds the values of the corresponding elements in `src_a` and `src_b`, and
/// puts the result into `dst`.
///
/// The addition is saturated, i.e. the result is the largest number of the
/// type of the element if the addition result would overflow. Source data
/// length (in bytes) is `stride * height`. Width and height are the same
/// for the two sources and for the destination. Number of elements is limited
/// to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s8, int8_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s8_sme, int8_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u8, uint8_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u8_sme, uint8_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s16, int16_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s16_sme, int16_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u16, uint16_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u16_sme, uint16_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s32, int32_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s32_sme, int32_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u32, uint32_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u32_sme, uint32_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s64, int64_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_s64_sme, int64_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u64, uint64_t);
/// @copydoc kleidicv_saturating_add_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_add_u64_sme, uint64_t);

/// Subtracts the value of the corresponding element in `src_b` from `src_a`,
/// and puts the result into `dst`.
///
/// The subtraction is saturated, i.e. the result is 0 (unsigned) or the
/// smallest possible value of the type of the element if the subtraction result
/// would underflow. Source data length (in bytes) is `stride * height`.
/// Width and height are the same for the two sources and for the destination.
/// Number of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s8, int8_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s8_sme, int8_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u8, uint8_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u8_sme, uint8_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s16, int16_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s16_sme, int16_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u16, uint16_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u16_sme, uint16_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s32, int32_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s32_sme, int32_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u32, uint32_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u32_sme, uint32_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s64, int64_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_s64_sme, int64_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u64, uint64_t);
/// @copydoc kleidicv_saturating_sub_s8
KLEIDICV_BINARY_OP(kleidicv_saturating_sub_u64_sme, uint64_t);

/// From the corresponding elements in `src_a` and `src_b`, subtracts the lower
/// one from the higher one, and puts the result into `dst`.
///
/// The subtraction is saturated, i.e. the result is the largest number of the
/// type of the element if the result would overflow (it is only possible with
/// signed types). Source data length (in bytes) is `stride * height`. Width
/// and height are the same for the two sources and for the destination. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_u8, uint8_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_u8_sme, uint8_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_s8, int8_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_s8_sme, int8_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_u16, uint16_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_u16_sme, uint16_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_s16, int16_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_s16_sme, int16_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_s32, int32_t);
/// @copydoc kleidicv_saturating_absdiff_u8
KLEIDICV_BINARY_OP(kleidicv_saturating_absdiff_s32_sme, int32_t);

/// Multiplies the values of the corresponding elements in `src_a` and `src_b`,
/// and puts the result into `dst`.
///
/// The multiplication is saturated, i.e. the result is the largest number of
/// the type of the element if the multiplication result would overflow. Source
/// data length (in bytes) is `stride * height`. Width and height are the
/// same for the two sources and for the destination. Number of elements is
/// limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param scale        Currently unused parameter.
///
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_u8, uint8_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_u8_sme, uint8_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_s8, int8_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_s8_sme, int8_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_u16, uint16_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_u16_sme, uint16_t,
                         double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_s16, int16_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_s16_sme, int16_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_s32, int32_t, double);
/// @copydoc kleidicv_saturating_multiply_u8
KLEIDICV_BINARY_OP_SCALE(kleidicv_saturating_multiply_s32_sme, int32_t, double);

/// Adds the absolute values of the corresponding elements in `src_a` and
/// `src_b`. Then, performs a comparison of each element's value in the result
/// with respect to a caller defined threshold. The strictly larger elements
/// remain unchanged and the rest are set to 0.
///
/// The addition is saturated, i.e. the result is the largest number of the
/// type of the element if the addition result would overflow. Source data
/// length (in bytes) is `stride * height`. Width and height are the same
/// for the two sources and for the destination. Number of elements is limited
/// to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param threshold    The value that the elements of the addition result
///                     are compared to.
///
/// In-place operations supported.
///
KLEIDICV_API_DECLARATION(kleidicv_saturating_add_abs_with_threshold_s16,
                         const int16_t *src_a, size_t src_a_stride,
                         const int16_t *src_b, size_t src_b_stride,
                         int16_t *dst, size_t dst_stride, size_t width,
                         size_t height, int16_t threshold);
/// @copydoc kleidicv_saturating_add_abs_with_threshold_s16
KLEIDICV_API_DECLARATION(kleidicv_saturating_add_abs_with_threshold_s16_sme,
                         const int16_t *src_a, size_t src_a_stride,
                         const int16_t *src_b, size_t src_b_stride,
                         int16_t *dst, size_t dst_stride, size_t width,
                         size_t height, int16_t threshold);

/// Bitwise-ands the values of the corresponding elements in `src_a` and
/// `src_b`, and puts the result into `dst`.
///
/// Source data length (in bytes) is `stride * height`. Width and height are
/// the same for the two sources and for the destination. Number of elements is
/// limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_BINARY_OP(kleidicv_bitwise_and, uint8_t);
/// @copydoc kleidicv_bitwise_and
KLEIDICV_BINARY_OP(kleidicv_bitwise_and_sme, uint8_t);

/// Performs a comparison of each element's value in `src` with respect to a
/// caller defined threshold. The strictly larger elements are set to
/// `value` and the rest to 0.
///
/// Width and height are the same for the source and for the destination. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data. Must
///                     not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param dst          Pointer to the first destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must
///                     not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param threshold    The value that the elements of the source data are
///                     compared to.
/// @param value        The value that the larger elements are set to.
///
KLEIDICV_API_DECLARATION(kleidicv_threshold_binary_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, uint8_t threshold,
                         uint8_t value);
/// @copydoc kleidicv_threshold_binary_u8
KLEIDICV_API_DECLARATION(kleidicv_threshold_binary_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, uint8_t threshold,
                         uint8_t value);

/// Performs an 'equal to' comparison of each element's value in `src_a` with
/// respect to the corresponding element's value in `src_b`.
///
/// If the result of the comparison is true then the corresponding element in
/// `dst` is set to 255, otherwise to 0.
///
/// Width and height are the same for the source and for the destination. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the first destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must
///                     not be less than `width * sizeof(type)`.
///                     Must be a multiple of `sizeof(type)`.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_compare_equal_u8, const uint8_t *src_a,
                         size_t src_a_stride, const uint8_t *src_b,
                         size_t src_b_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_compare_equal_u8
KLEIDICV_API_DECLARATION(kleidicv_compare_equal_u8_sme, const uint8_t *src_a,
                         size_t src_a_stride, const uint8_t *src_b,
                         size_t src_b_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Performs a 'strictly greater than' comparison of each element's value in
/// `src_a` with respect to the corresponding element's value in `src_b`.
///
/// If the result of the comparison is true then the corresponding element in
/// `dst` is set to 255, otherwise to 0.
///
/// Width and height are the same for the source and for the destination. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src_a        Pointer to the first source data. Must be non-null.
/// @param src_a_stride Distance in bytes from the start of one row to the
///                     start of the next row for the first source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_b        Pointer to the second source data. Must be non-null.
/// @param src_b_stride Distance in bytes from the start of one row to the
///                     start of the next row for the second source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the first destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_compare_greater_u8, const uint8_t *src_a,
                         size_t src_a_stride, const uint8_t *src_b,
                         size_t src_b_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_compare_greater_u8
KLEIDICV_API_DECLARATION(kleidicv_compare_greater_u8_sme, const uint8_t *src_a,
                         size_t src_a_stride, const uint8_t *src_b,
                         size_t src_b_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Multiplies the elements in `src` by `scale`, then adds `shift` to the
/// result and stores it in `dst`.
///
/// The result is saturated, i.e. it is the smallest/largest number of the
/// type of the element if the result would underflow/overflow. Source data
/// length (in bytes) is `stride * height`. Width and height are the same
/// for the source and destination. Number of elements is limited to
/// @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// Note: Conversion from `uint8_t` to `float16_t` using SVE/SME is
/// significantly faster when `alpha` and `beta` are exact `float16_t` values.
/// This can be forced by explicit casting:  `static_cast<float16_t>(0.3)`.
///
/// In-place operations are supported if the types of `src` and `dst` are the
/// same.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param scale        Value to multiply the input by.
/// @param shift        Value to add to the result.
///
KLEIDICV_API_DECLARATION(kleidicv_scale_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, double scale,
                         double shift);
/// @copydoc kleidicv_scale_u8
KLEIDICV_API_DECLARATION(kleidicv_scale_u8_f16, const uint8_t *src,
                         size_t src_stride, float16_t *dst, size_t dst_stride,
                         size_t width, size_t height, double scale,
                         double shift);
/// @copydoc kleidicv_scale_u8
KLEIDICV_API_DECLARATION(kleidicv_scale_u8_f16_sme, const uint8_t *src,
                         size_t src_stride, float16_t *dst, size_t dst_stride,
                         size_t width, size_t height, double scale,
                         double shift);
/// @copydoc kleidicv_scale_u8
KLEIDICV_API_DECLARATION(kleidicv_scale_f32, const float *src,
                         size_t src_stride, float *dst, size_t dst_stride,
                         size_t width, size_t height, double scale,
                         double shift);
/// @copydoc kleidicv_scale_u8
KLEIDICV_API_DECLARATION(kleidicv_scale_f32_sme, const float *src,
                         size_t src_stride, float *dst, size_t dst_stride,
                         size_t width, size_t height, double scale,
                         double shift);

/// Exponential function, input is the elements in `src`, output is the elements
/// in `dst`.
///
/// In case of `float` type the maximum error is 0.36565+0.5 ULP, or the error
/// of the toolchain's expf implementation, if it is bigger.
///
/// Source and destination data length is `width * height`. Number of elements
/// is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data. Must be a
///                     multiple of `sizeof(type)` and no less than `width *
///                     sizeof(type)`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must be
///                     a multiple of `sizeof(type)` and no less than `width *
///                     sizeof(type)`, except for single-row images.
/// @param width        Number of pixels in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_exp_f32, const float *src, size_t src_stride,
                         float *dst, size_t dst_stride, size_t width,
                         size_t height);
/// @copydoc kleidicv_exp_f32
KLEIDICV_API_DECLARATION(kleidicv_exp_f32_sme, const float *src,
                         size_t src_stride, float *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Performs a per element comparison in `src` with respect to caller defined
/// lower and upper bounds. For the elements exceeding these bounds, the
/// corresponding elements in `dst` are set to 0 and elements within to 255.
/// The bounds are inclusive.
///
/// Width and height are the same for the source and for the destination. Number
/// of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations are supported for the `u8` variant.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data. Must
///                     not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must
///                     not be less than `width * sizeof(type)`, except for
///                     single-row images.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
/// @param lower_bound  The lower bound of the interval.
/// @param upper_bound  The upper bound of the interval.
///
KLEIDICV_API_DECLARATION(kleidicv_in_range_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, uint8_t lower_bound,
                         uint8_t upper_bound);
/// @copydoc kleidicv_in_range_u8
KLEIDICV_API_DECLARATION(kleidicv_in_range_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, uint8_t lower_bound,
                         uint8_t upper_bound);
/// @copydoc kleidicv_in_range_u8
KLEIDICV_API_DECLARATION(kleidicv_in_range_f32, const float *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, float lower_bound,
                         float upper_bound);
/// @copydoc kleidicv_in_range_u8
KLEIDICV_API_DECLARATION(kleidicv_in_range_f32_sme, const float *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, float lower_bound,
                         float upper_bound);
/// @}

/// @addtogroup kleidicv_conversions
/// @{

/// Converts a grayscale image to RGB. All channels are 8-bit wide.
///
/// Destination data is filled as follows: `R = G = B = Gray`
/// resulting in `| R,G,B | R,G,B | R,G,B | ...` image
/// where each letter represents one byte of data, and one pixel is represented
/// by 3 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_gray_to_rgb_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_gray_to_rgb_u8
KLEIDICV_API_DECLARATION(kleidicv_gray_to_rgb_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts a grayscale image to RGBA. All channels are 8-bit wide.
///
/// Destination data is filled as follows: `R = G = B = Gray`, `A = 0xFF`
/// resulting in `| R,G,B,A | R,G,B,A | R,G,B,A | ...` image
/// where each letter represents one byte of data, and one pixel is represented
/// by 4 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_gray_to_rgba_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_gray_to_rgba_u8
KLEIDICV_API_DECLARATION(kleidicv_gray_to_rgba_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGB image to BGR. All channels are 8-bit wide.
///
/// Destination data is filled as follows:
/// `| B,G,R | B,G,R | B,G,R | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 3 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_bgr_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_rgb_to_bgr_u8
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_bgr_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Copies a source RGB image to destination buffer.
/// All channels are 8-bit wide.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_rgb_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGBA image to BGRA. All channels are 8-bit wide.
///
/// Destination data is filled as follows:
/// `| B,G,R,A | B,G,R,A | B,G,R,A | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 4 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_bgra_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_rgba_to_bgra_u8
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_bgra_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Copies a source RGBA image to destination buffer.
/// All channels are 8-bit wide.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_rgba_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGB image to BGRA. All channels are 8-bit wide.
///
/// Corresponding colors are set while Alpha channel is set to 0xFF.
/// Destination data is filled as follows:
/// `| B,G,R,A | B,G,R,A | B,G,R,A | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 4 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_bgra_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_rgb_to_bgra_u8
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_bgra_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGB image to RGBA. All channels are 8-bit wide.
///
/// Corresponding colors are set while Alpha channel is set to 0xFF.
/// Destination data is filled as follows:
/// `| R,G,B,A | R,G,B,A | R,G,B,A | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 4 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_rgba_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_rgb_to_rgba_u8
KLEIDICV_API_DECLARATION(kleidicv_rgb_to_rgba_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGBA image to BGR. All channels are 8-bit wide.
///
/// Corresponding colors are set while Alpha channel is discarded.
/// Destination data is filled as follows:
/// `| B,G,R | B,G,R | B,G,R | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 3 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_bgr_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_rgba_to_bgr_u8
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_bgr_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts an RGBA image to RGB. All channels are 8-bit wide.
///
/// Corresponding colors are set while Alpha channel is discarded.
/// Destination data is filled as follows:
/// `| R,G,B | R,G,B | R,G,B | ...`
/// Each letter represents one byte of data, and one pixel is represented
/// by 3 bytes. There is no padding between the pixels.
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src         Pointer to the source data. Must be non-null.
/// @param src_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the source data.
///                    Must not be less than `4 * width`, except for single-row
///                    images.
/// @param dst         Pointer to the destination data. Must be non-null.
/// @param dst_stride  Distance in bytes from the start of one row to the
///                    start of the next row for the destination data.
///                    Must not be less than `3 * width`, except for single-row
///                    images.
/// @param width       Number of pixels in a row.
/// @param height      Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_rgb_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_rgba_to_rgb_u8
KLEIDICV_API_DECLARATION(kleidicv_rgba_to_rgb_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts a YUV image (planar or interleaved) to RGB, RGBA, BGR, or BGRA
/// format. All channels are 8-bit wide. If the output format includes an
/// alpha channel, the alpha value is set to 0xFF.
///
/// Source formats:
///
/// - Planar YUV420 (I420 or YV12 layout)
///
///   The input buffer consists of three planes stored sequentially in memory:
///   - Y plane: full resolution, `size = width x height`
///   - U plane: quarter resolution, `size = (width / 2) x (height / 2)`
///   - V plane: quarter resolution, `size = (width / 2) x (height / 2)`
///
///   The layout order (Y + U + V or Y + V + U) is determined by `color_format`.
///
/// - Interleaved YUV444
///
///   The input data contains three 8-bit channels per pixel, stored
///   consecutively in memory as `| Y, U, V | Y, U, V | ... |`.
///
///   Each pixel occupies 3 bytes. There is no padding between pixels.
///
/// - Interleaved YUV422
///
///   The input buffer consists of interleaved Y, U, and V samples, stored
///   in one of several layouts depending on `color_format`:
///
///   - YUYV: [Y0 U0 Y1 V0] [Y2 U2 Y3 V2] ...
///   - UYVY: [U0 Y0 V0 Y1] [U2 Y2 V2 Y3] ...
///   - YVYU: [Y0 V0 Y1 U0] [Y2 V2 Y3 U2] ...
///
///   Each pair of pixels shares one U and one V sample. Because pixels are
///   processed in pairs (`Y0, U, Y1, V`), the image width must be even and at
///   least 2 pixels wide.
///
/// Destination format:
///
/// - Interleaved RGB
///
///   The destination buffer uses an interleaved pixel layout with 3 or 4
///   channels per pixel:
///   - R, G, B
///   - B, G, R
///   - R, G, B, Alpha
///   - B, G, R, Alpha
///
///   One pixel occupies 3 or 4 bytes, depending on the format.
///
/// Width and height refer to the logical image dimensions, i.e. the number of
/// pixels per row and the number of rows. The total number of pixels must not
/// exceed @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source buffer containing the YUV data.
///                     Must be non-null.
/// @param src_stride   Byte offset between the start of one row and the next
///                     in the source data.
///                     For planar YUV420, this is the stride of the Y plane,
///                     and the U and V planes follow sequentially in memory.
///                     For interleaved YUV444, this must be at least
///                     `3 * width`, unless the image has only one row.
///                     For interleaved YUV422, Must be at least `width * 2`
/// @param dst          Pointer to the destination buffer. Must be non-null.
/// @param dst_stride   Byte offset between the start of one destination row
///                     and the next. Must be at least
///                     `(destination channel count) * width`, unless the image
///                     has only one row.
/// @param width        Number of pixels in a row.
/// @param height       Number of rows in the data.
/// @param color_format Specifies the color conversion type, defining both the
///                     source YUV layout (e.g., I420, YV12, YUV444) and the
///                     destination RGB(A)/BGR(A) format.
///                     Must be one of @ref kleidicv_color_conversion_t.
kleidicv_error_t kleidicv_yuv_to_rgb_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format);
/// @copydoc kleidicv_yuv_to_rgb_u8
kleidicv_error_t kleidicv_yuv_to_rgb_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format);

/// Converts a semi-planar YUV image (NV12 or NV21 layout) to RGB, RGBA, BGR, or
/// BGRA format. All channels are 8-bit wide. If the output format includes an
/// alpha channel, the alpha value is set to 0xFF.
///
/// Source format:
///
/// - Semi-Planar YUV420
///
///   The input consists of two planes:
///   - Y plane:  full resolution, `size = width x height`
///   - UV plane: half resolution, `size = (width / 2) x (height / 2)`
///
///   The UV plane contains interleaved chroma samples:
///   - NV12: UVUVUV...
///   - NV21: VUVUVU...
///
/// Destination format:
///
/// - Interleaved RGB
///
///   The destination buffer uses an interleaved pixel layout with 3 or 4
///   channels per pixel:
///   - R, G, B
///   - B, G, R
///   - R, G, B, Alpha
///   - B, G, R, Alpha
///
///   One pixel occupies 3 or 4 bytes, depending on the format.
///
/// Width and height refer to the logical image dimensions, i.e. the number of
/// pixels per row and the number of rows. The total number of pixels must not
/// exceed @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src_y         Pointer to the input Y component. Must be non-null.
/// @param src_y_stride  Byte offset between the start of one Y row and the
///                      next.
///                      Must be at least `width`, except for single-row images.
/// @param src_uv        Pointer to the interleaved UV (or VU) component. Must
///                      be non-null. The layout order depends on
///                      `color_format`.
/// @param src_uv_stride Byte offset between the start of one UV row and the
///                      next. Must be at least `width`, except for single-row
///                      images.
/// @param dst           Pointer to the destination buffer. Must be non-null.
/// @param dst_stride    Byte offset between the start of one destination row
///                      and the next. Must be at least `(destination channel
///                      count) * width`, unless the image has only one row.
/// @param width         Number of pixels in a row.
/// @param height        Number of rows in the data.
/// @param color_format  Specifies the color conversion type, defining both the
///                      source YUV layout (e.g., I420, YV12, YUV444) and the
///                      destination RGB(A)/BGR(A) format.
///                      Must be one of @ref kleidicv_color_conversion_t.
KLEIDICV_API_DECLARATION(kleidicv_yuv_semiplanar_to_rgb_u8,
                         const uint8_t *src_y, size_t src_y_stride,
                         const uint8_t *src_uv, size_t src_uv_stride,
                         uint8_t *dst, size_t dst_stride, size_t width,
                         size_t height,
                         kleidicv_color_conversion_t color_format);
/// @copydoc kleidicv_yuv_semiplanar_to_rgb_u8
KLEIDICV_API_DECLARATION(kleidicv_yuv_semiplanar_to_rgb_u8_sme,
                         const uint8_t *src_y, size_t src_y_stride,
                         const uint8_t *src_uv, size_t src_uv_stride,
                         uint8_t *dst, size_t dst_stride, size_t width,
                         size_t height,
                         kleidicv_color_conversion_t color_format);

/// Converts an interleaved RGB-like image to YUV image (planar or interleaved).
/// All channels are 8-bit wide. If the source format includes an alpha channel,
/// it is ignored.
///
/// Source format:
///
/// - Interleaved RGB
///
///   The source buffer uses an interleaved pixel layout with 3 or 4 channels
///   per pixel:
///   - R, G, B
///   - B, G, R
///   - R, G, B, Alpha
///   - B, G, R, Alpha
///
///   One pixel occupies 3 or 4 bytes, depending on the format. If present, the
///   alpha channel is ignored during conversion.
///
/// Destination format:
///
/// - Planar YUV420 (I420 or YV12 layout)
///
///   The destination buffer consists of three planes stored sequentially
///   in memory:
///   - Y plane: full resolution, `size = width x height`
///   - U plane: quarter resolution, `size = (width / 2) x (height / 2)`
///   - V plane: quarter resolution, `size = (width / 2) x (height / 2)`
///
///   The layout order (Y + U + V or Y + V + U) is determined by `color_format`.
///
///   Chroma (U and V) are subsampled by a factor of 2 in both horizontal and
///   vertical directions. Only even rows and even columns from the RGB image
///   are used to compute U and V values. If the input image has an odd width or
///   height, the chroma resolution is rounded up. For example, a height of 9
///   will result in 5 chroma rows (since 9 / 2 = 4.5 -> rounded up). The same
///   applies to the width. This behavior matches OpenCV's implementation.
///
/// - Interleaved YUV444
///
///   The destination data contains three 8-bit channels per pixel, stored
///   consecutively in memory as `| Y, U, V | Y, U, V | ... |`.
///   Each pixel occupies 3 bytes. There is no padding between pixels.
///
/// - Interleaved YUV422
///
///   The output buffer consists of interleaved Y, U, and V samples stored in
///   one of several layouts depending on `color_format`:
///
///   - YUYV: [Y0 U0 Y1 V0] [Y2 U2 Y3 V2] ...
///   - UYVY: [U0 Y0 V0 Y1] [U2 Y2 V2 Y3] ...
///   - YVYU: [Y0 V0 Y1 U0] [Y2 V2 Y3 U2] ...
///
///   Each pair of output pixels shares one U and one V sample, computed as the
///   average of the two corresponding source pixels' chroma values. Because
///   pixels are processed in pairs, the image width must be at least 2 and kept
///   even for all YUV422 conversions.
///
/// Width and height refer to the logical image dimensions, i.e., the number of
/// pixels per row and the number of rows. The total number of pixels must not
/// exceed @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source buffer containing interleaved
///                     RGBX/BGRX data. Must be non-null.
/// @param src_stride   Byte offset between the start of one source row and the
///                     next. Must be at least `(source channel count) * width`,
///                     unless the image has only one row.
/// @param dst          Pointer to the destination buffer. Must be non-null.
/// @param dst_stride   Byte offset between the start of one destination row and
///                     the next.
///                     For planar YUV420:
///                     This is the stride of the Y plane (the byte offset
///                     between the start of one row and the next).
///                     The U and V planes are stored sequentially in memory.
///                     Their row increments can be computed as:
///                     `uvsteps[2] = { width / 2, dst_stride - width / 2 }`.
///                     For interleaved YUV444:
///                     This is simply the byte offset between the start of one
///                     row and the next.
///                     For interleaved YUV422:
///                     Must be at least `width * 2`
/// @param width        Number of pixels in a row.
/// @param height       Number of rows in the image.
/// @param color_format Specifies the color conversion type, defining both the
///                     source YUV layout and the destination RGB(A)/BGR(A)
///                     format. Must be one of @ref
///                     kleidicv_color_conversion_t.
kleidicv_error_t kleidicv_rgb_to_yuv_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format);
/// @copydoc kleidicv_rgb_to_yuv_u8
kleidicv_error_t kleidicv_rgb_to_yuv_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format);

/// Converts an interleaved RGB-like image to a semi-planar YUV420 format
/// (NV12 or NV21). All channels are 8-bit wide. If the source format includes
/// an alpha channel, it is ignored.
///
/// Source format:
///
/// - Interleaved RGB
///
///   The source buffer uses an interleaved pixel layout with 3 or 4 channels
///   per pixel:
///   - R, G, B
///   - B, G, R
///   - R, G, B, Alpha
///   - B, G, R, Alpha
///
///   One pixel occupies 3 or 4 bytes, depending on the format. If present, the
///   alpha channel is ignored.
///
/// Destination format:
///
/// - Semi-Planar YUV420
///
///    The destination consists of two planes:
///    - Y plane:  full resolution, `size = width x height`
///    - UV plane: half resolution, `size = (width / 2) x (height / 2) x 2`
///      bytes per chroma sample pair
///
///    The UV plane contains interleaved chroma samples:
///    - NV12: UVUVUV...
///    - NV21: VUVUVU...
///
/// Chroma (U and V) are subsampled by a factor of 2 in both horizontal and
/// vertical directions. Only even rows and even columns from the RGB image
/// are used to compute U and V values. If the input image has an odd width or
/// height, the chroma resolution is rounded up. For example, a height of 9
/// will result in 5 chroma rows (since 9 / 2 = 4.5 -> rounded up). The same
/// applies to the width. This behavior matches OpenCV's implementation.
///
/// Width and height refer to the logical image dimensions, i.e. the number of
/// pixels per row and the number of rows. The total number of pixels must not
/// exceed @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported.
///
/// @param src           Pointer to the source buffer containing interleaved
///                      RGBX/BGRX data. Must be non-null.
/// @param src_stride    Byte offset between the start of one source row and the
///                      next. Must be at least `(source channel count) *
///                      width`, unless the image has only one row.
/// @param y_dst         Pointer to the destination Y plane. Must be non-null.
/// @param y_stride      Byte offset between the start of one row in the Y plane
///                      and the next. Must be at least `width`.
/// @param uv_dst        Pointer to the destination UV plane (interleaved).
///                      Must be non-null.
/// @param uv_stride     Byte offset between the start of one row in the UV
///                      plane and the next. Must be at least
///                      `__builtin_align_up(width, 2)`.
/// @param width         Number of pixels per row in the image.
/// @param height        Number of rows in the image.
/// @param color_format  Specifies the color conversion type, defining both the
///                      source YUV layout and the destination RGB(A)/BGR(A)
///                      format. Must be one of @ref
///                      kleidicv_color_conversion_t.
kleidicv_error_t kleidicv_rgb_to_yuv_semiplanar_u8(
    const uint8_t *src, size_t src_stride, uint8_t *y_dst, size_t y_stride,
    uint8_t *uv_dst, size_t uv_stride, size_t width, size_t height,
    kleidicv_color_conversion_t color_format);
/// @copydoc kleidicv_rgb_to_yuv_semiplanar_u8
kleidicv_error_t kleidicv_rgb_to_yuv_semiplanar_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *y_dst, size_t y_stride,
    uint8_t *uv_dst, size_t uv_stride, size_t width, size_t height,
    kleidicv_color_conversion_t color_format);

/// Splits a multi channel source stream into separate 1-channel streams. Width
/// and height are the same for the source stream and for all the destination
/// streams. Number of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operation is not supported.
///
/// @param src_data     Pointer to the source data. Must be non-null.
///                     Must be aligned to `element_size`.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `element_size` and no less than `width *
///                     element_size * channels`, except for single-row images.
/// @param dst_data     A C style array of pointers to the destination data.
///                     Number of pointers in the array must be the same as the
///                     channel number. All pointers must be non-null.
///                     All pointers must be aligned to `element_size`.
/// @param dst_strides  A C style array of stride values for the destination
///                     streams. A stride value represents the distance in
///                     bytes from the start of one row to the start of the
///                     next row in the given destination stream. Number of
///                     stride values in the array must be the same as the
///                     channel number. All stride values must be a multiple of
///                     `element_size` and no less than `width * element_size`,
///                     except for single-row images.
/// @param width        Number of pixels in one row of the source data. (One
///                     pixel consists of `channels` number of elements.)
/// @param height       Number of rows in the source data.
/// @param channels     Number of channels in the source data. Must be 2, 3 or
///                     4.
/// @param element_size Size of one element in bytes. Must be 1, 2, 4 or 8.
///
KLEIDICV_API_DECLARATION(kleidicv_split, const void *src_data,
                         size_t src_stride, void **dst_data,
                         const size_t *dst_strides, size_t width, size_t height,
                         size_t channels, size_t element_size);

/// Merges separate 1-channel source streams to one multi channel stream. Width
/// and height are the same for all the source streams and for the destination.
/// Number of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param srcs         A C style array of pointers to the source data.
///                     Number of pointers in the array must be the same as the
///                     channel number. All pointers must be non-null.
///                     All pointers must be aligned to `element_size`.
/// @param src_strides  A C style array of stride values for the source
///                     streams. A stride value represents the distance in
///                     bytes from the start of one row to the start of the
///                     next row in the given source stream. Number of
///                     stride values in the array must be the same as the
///                     channel number. All stride values must be a multiple of
///                     `element_size` and no less than `width * element_size`,
///                     except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
///                     Must be aligned to element_size.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the destination data. Must be a
///                     multiple of `element_size` and no less than `width *
///                     element_size * channels`, except for single-row images.
/// @param width        Number of elements in a row for the source streams,
///                     number of pixels in a row for the destination data.
/// @param height       Number of rows in the data.
/// @param channels     Number of channels in the destination data. Must be 2,
///                     3 or 4.
/// @param element_size Size of one element in bytes. Must be 1, 2, 4 or 8.
///
KLEIDICV_API_DECLARATION(kleidicv_merge, const void **srcs,
                         const size_t *src_strides, void *dst,
                         size_t dst_stride, size_t width, size_t height,
                         size_t channels, size_t element_size);

/// Converts the elements in `src` from a floating-point type to an integer
/// type, then stores the result in `dst`.
///
/// Each resulting element is saturated, i.e. it is the smallest/largest
/// number of the type of the element if the `src` data type cannot be
/// represented as the `dst` type. In case of some special values, such as the
/// different variations of `NaN`, the result is `0`. Source and destination
/// data length is `width * height`. Number of elements is limited to @ref
/// KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must not be less than `width * sizeof(type)`.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must not be less than `width * sizeof(type)`.
/// @param width        Number of elements in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_f32_to_s8, const float *src,
                         size_t src_stride, int8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_f32_to_s8
KLEIDICV_API_DECLARATION(kleidicv_f32_to_s8_sme, const float *src,
                         size_t src_stride, int8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_f32_to_s8
KLEIDICV_API_DECLARATION(kleidicv_f32_to_u8, const float *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_f32_to_s8
KLEIDICV_API_DECLARATION(kleidicv_f32_to_u8_sme, const float *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height);

/// Converts the elements in `src` from an integer type to a floating-point
/// type, then stores the result in `dst`.
///
/// Each resulting element is saturated, i.e. it is the smallest/largest
/// number of the type of the element if the `src` data type cannot be
/// represented as the `dst` type. Source and destination data length is
/// `width * height`. Number of elements is limited to @ref
/// KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data. Must
///                     not be less than `width * sizeof(type)`.
///                     Must be a multiple of `sizeof(type)`.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data. Must
///                     not be less than `width * sizeof(type)`.
///                     Must be a multiple of `sizeof(type)`.
/// @param width        Number of pixels in a row.
/// @param height       Number of rows in the data.
///
KLEIDICV_API_DECLARATION(kleidicv_s8_to_f32, const int8_t *src,
                         size_t src_stride, float *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_s8_to_f32
KLEIDICV_API_DECLARATION(kleidicv_s8_to_f32_sme, const int8_t *src,
                         size_t src_stride, float *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_s8_to_f32
KLEIDICV_API_DECLARATION(kleidicv_u8_to_f32, const uint8_t *src,
                         size_t src_stride, float *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @copydoc kleidicv_s8_to_f32
KLEIDICV_API_DECLARATION(kleidicv_u8_to_f32_sme, const uint8_t *src,
                         size_t src_stride, float *dst, size_t dst_stride,
                         size_t width, size_t height);
/// @}

/// @addtogroup kleidicv_filters
/// @{

/// Calculates vertical derivative approximation with Sobel filter.
///
/// The used convolution kernel is:
/// ```
/// [  1  2  1 ]
/// [  0  0  0 ]
/// [ -1 -2 -1 ]
/// ```
/// Note, that the kernel is mirrored both vertically and horizontally during
/// the convolution.
///
/// The only supported border type is @ref KLEIDICV_BORDER_TYPE_REPLICATE
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(src type)` and no less than `width *
///                     sizeof(src type) * channels`, except for single-row
///                     images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the destination data. Must be a
///                     multiple of `sizeof(dst type)` and no less than `width *
///                     sizeof(dst type) * channels`, except for single-row
///                     images.
/// @param width        Number of columns in the data. (One column consists of
///                     `channels` number of elements.) Must be greater than or
///                     equal to kernel size (== 3) - 1.
/// @param height       Number of rows in the data. Must be greater than or
///                     equal to kernel size (== 3) - 1.
/// @param channels     Number of channels in the data. Must be not more than
///                     @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
///
kleidicv_error_t kleidicv_sobel_3x3_vertical_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels);
/// @copydoc kleidicv_sobel_3x3_vertical_s16_u8
kleidicv_error_t kleidicv_sobel_3x3_vertical_s16_u8_sme(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels);

/// Calculates horizontal derivative approximation with Sobel filter.
///
/// The used convolution kernel is:
/// ```
/// [  1  0 -1 ]
/// [  2  0 -2 ]
/// [  1  0 -1 ]
/// ```
/// Note, that the kernel is mirrored both vertically and horizontally during
/// the convolution.
///
/// The only supported border type is @ref KLEIDICV_BORDER_TYPE_REPLICATE
///
/// Width and height are the same for the source and for the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(src type)` and no less than `width *
///                     sizeof(src type) * channels`, except for single-row
///                     images.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the destination data. Must be a
///                     multiple of `sizeof(dst type)` and no less than `width *
///                     sizeof(dst type) * channels`, except for single-row
///                     images.
/// @param width        Number of columns in the data. (One column consists of
///                     `channels` number of elements.) Must be greater than or
///                     equal to kernel size (== 3) - 1.
/// @param height       Number of rows in the data. Must be greater than or
///                     equal to kernel size (== 3) - 1.
/// @param channels     Number of channels in the data. Must be not more than
///                     @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
///
kleidicv_error_t kleidicv_sobel_3x3_horizontal_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels);
/// @copydoc kleidicv_sobel_3x3_horizontal_s16_u8
kleidicv_error_t kleidicv_sobel_3x3_horizontal_s16_u8_sme(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels);

/// Applies a two-dimensional separable filter to the source image using the
/// specified parameters. In-place filtering is not supported.
///
/// Width and height are assumed to be the same for the source and for the
/// destination. The number of elements is limited to @ref
/// KLEIDICV_MAX_IMAGE_PIXELS.
///
/// Note, from the border types only KLEIDICV_BORDER_TYPE_REPLICATE is
/// supported.
///
/// @param src           Pointer to the source data. Must be non-null.
/// @param src_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the source data. Must be a
///                      multiple of `sizeof(type)` and no less than `width *
///                      sizeof(type) * channels`, except for single-row images.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the destination data. Must be
///                      a multiple of `sizeof(type)` and no less than `width *
///                      sizeof(type) * channels`, except for single-row images.
/// @param width         Number of columns in the data. (One column consists of
///                      `channels` number of elements.) Must be greater than
///                      or equal to `kernel_width - 1`.
/// @param height        Number of rows in the data. Must be greater than
///                      or equal to `kernel_height - 1`.
/// @param channels      Number of channels in the data. Must be not more than
///                      @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
/// @param kernel_x      Pointer to the horizontal 2D kernel values.
/// @param kernel_width  Size of the horizontal 2D kernel.
/// @param kernel_y      Pointer to the vertical 2D kernel values.
/// @param kernel_height Size of the vertical 2D kernel.
/// @param border_type   Way of handling the border.
///
kleidicv_error_t kleidicv_separable_filter_2d_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint8_t *kernel_x,
    size_t kernel_width, const uint8_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type);
/// @copydoc kleidicv_separable_filter_2d_u8
kleidicv_error_t kleidicv_separable_filter_2d_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint8_t *kernel_x,
    size_t kernel_width, const uint8_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type);
/// @copydoc kleidicv_separable_filter_2d_u8
kleidicv_error_t kleidicv_separable_filter_2d_u16(
    const uint16_t *src, size_t src_stride, uint16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint16_t *kernel_x,
    size_t kernel_width, const uint16_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type);
/// @copydoc kleidicv_separable_filter_2d_u8
kleidicv_error_t kleidicv_separable_filter_2d_u16_sme(
    const uint16_t *src, size_t src_stride, uint16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, const uint16_t *kernel_x,
    size_t kernel_width, const uint16_t *kernel_y, size_t kernel_height,
    kleidicv_border_type_t border_type);

/// Applies Gaussian blur to the source image using the specified parameters.
/// In-place filtering is not supported.
///
/// Width and height are assumed to be the same for the source and for the
/// destination. The number of elements is limited to @ref
/// KLEIDICV_MAX_IMAGE_PIXELS.
///
/// Note, from the border types only these are supported:
///                       - @ref KLEIDICV_BORDER_TYPE_REPLICATE
///                       - @ref KLEIDICV_BORDER_TYPE_REFLECT
///                       - @ref KLEIDICV_BORDER_TYPE_WRAP
///                       - @ref KLEIDICV_BORDER_TYPE_REVERSE
///
/// @param src           Pointer to the source data. Must be non-null.
/// @param src_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the source data. Must be a
///                      multiple of `sizeof(type)` and no less than `width *
///                      sizeof(type) * channels`, except for single-row images.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the destination data. Must be
///                      a multiple of `sizeof(type)` and no less than `width *
///                      sizeof(type) * channels`, except for single-row images.
/// @param width         Number of columns in the data. (One column consists of
///                      `channels` number of elements.) Must be greater than
///                      or equal to `kernel_width - 1` (if kernel_width is
///                      3,5,7,15,21), or `(kernel_width/2) rounded up to 8,
///                      plus kernel_width/2` (for other kernel sizes).
/// @param height        Number of rows in the data. Must be greater than
///                      or equal to `kernel_height - 1`.
/// @param channels      Number of channels in the data. Must be not more than
///                      @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
/// @param kernel_width  Width of the Gaussian kernel.
/// @param kernel_height Height of the Gaussian kernel.
/// @param sigma_x       Horizontal sigma (standard deviation) value. If equal
///                      to 0.0, Gaussian filter is approximated by the
///                      probability mass function of the binomial distribution
///                      in the horizontal direction.
/// @param sigma_y       Vertical sigma (standard deviation) value. If equal
///                      to 0.0, Gaussian filter is approximated by the
///                      probability mass function of the binomial distribution
///                      in the vertical direction.
/// @param border_type   Way of handling the border.
///
kleidicv_error_t kleidicv_gaussian_blur_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, float sigma_x, float sigma_y,
    kleidicv_border_type_t border_type);
/// @copydoc kleidicv_gaussian_blur_u8
kleidicv_error_t kleidicv_gaussian_blur_u8_sme(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t channels, size_t kernel_width,
    size_t kernel_height, float sigma_x, float sigma_y,
    kleidicv_border_type_t border_type);

#ifndef DOXYGEN
/// Internal - not part of the public API and its direct use is not supported.
///
/// Applies 5x5 binomial Gaussian blur to the source image and downsamples the
/// result by keeping odd rows and columns only.
/// This function can be used to generate an Image Pyramid.
/// In-place operation is not supported.
///
/// The number of elements in the source is limited to @ref
/// KLEIDICV_MAX_IMAGE_PIXELS.
///
/// Width and height of the destination is calculated as:
///   - `dst_width = (src_width + 1) / 2`
///   - `dst_height = (src_height + 1) / 2`
///
/// Note, from the border types only these are supported:
///                       - @ref KLEIDICV_BORDER_TYPE_REPLICATE
///                       - @ref KLEIDICV_BORDER_TYPE_REFLECT
///                       - @ref KLEIDICV_BORDER_TYPE_WRAP
///                       - @ref KLEIDICV_BORDER_TYPE_REVERSE
///
/// @param src           Pointer to the source data. Must be non-null.
/// @param src_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the source data. Must be a
///                      multiple of `sizeof(type)` and no less than `src_width
///                      * sizeof(type) * channels`, except for single-row
///                      images.
/// @param src_width     Number of columns in the source data. (One column
///                      consists of `channels` number of elements.)
///                      Must be greater than or equal to kernel size (== 5)
///                      - 1.
/// @param src_height    Number of rows in the source data.
///                      Must be greater than or equal to kernel size (== 5)
///                      - 1.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row in the destination data. Must be
///                      a multiple of `sizeof(type)` and no less than
///                      `dst_width * sizeof(type) * channels`, except for
///                      single-row images.
/// @param channels      Number of channels in the data. Must be equal to 1.
/// @param border_type   Way of handling the border.
///
kleidicv_error_t kleidicv_blur_and_downsample_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels,
    kleidicv_border_type_t border_type);
/// @copydoc kleidicv_blur_and_downsample_u8
kleidicv_error_t kleidicv_blur_and_downsample_u8_sme(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t channels,
    kleidicv_border_type_t border_type);
#endif

#define KLEIDICV_FILTER_OP_MEDIAN(name, type)                            \
  kleidicv_error_t name(                                                 \
      const type *src, size_t src_stride, type *dst, size_t dst_stride,  \
      size_t width, size_t height, size_t channels, size_t kernel_width, \
      size_t kernel_height, kleidicv_border_type_t border_type)

/// Reduces noise by applying a median filter.
///
/// Width and height are assumed to be the same for the source and for the
/// destination. The total number of elements is limited to
/// @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src            Pointer to the source data. Must be non-null.
/// @param src_stride     Distance in bytes from the start of one row to the
///                       start of the next row in the source data. Must be a
///                       multiple of `sizeof(type)` and no less than
///                       `width * sizeof(type) * channels`, except for
///                       single-row images.
/// @param dst            Pointer to the destination data. Must be non-null.
/// @param dst_stride     Distance in bytes from the start of one row to the
///                       start of the next row in the destination data. Must be
///                       a multiple of `sizeof(type)` and no less than
///                       `width * sizeof(type) * channels`, except for
///                       single-row images.
/// @param width          Number of columns in the data. (One column consists of
///                       `channels` number of elements.) Must be greater than
///                       or equal to `kernel_width - 1`.
/// @param height         Number of rows in the data. Must be greater than or
///                       equal to `kernel_height - 1`.
/// @param channels       Number of channels in the data. Must not be more than
///                       @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
/// @param kernel_width   Width of the Median kernel. Must be odd and equal to
///                      `kernel_height`.
///                       For `uint8_t`, values 3 to 255 are supported. For
///                       other types, only 3, 5 and 7.
/// @param kernel_height  Height of the Median kernel. Must be odd and equal to
///                       `kernel_width`.
///                       For `uint8_t`, values 3 to 255 are supported. For
///                       other types, only 3, 5 and 7.
/// @param border_type    Way of handling the border. The supported border types
///                       are: \n
///                         - @ref KLEIDICV_BORDER_TYPE_REPLICATE \n
///                         - @ref KLEIDICV_BORDER_TYPE_REFLECT \n
///                         - @ref KLEIDICV_BORDER_TYPE_WRAP \n
///                         - @ref KLEIDICV_BORDER_TYPE_REVERSE
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_u8, uint8_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_u8_sme, uint8_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_s16, int16_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_s16_sme, int16_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_u16, uint16_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_u16_sme, uint16_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_f32, float);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_f32_sme, float);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_s8, int8_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_s8_sme, int8_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_s32, int32_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_s32_sme, int32_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_u32, uint32_t);
/// @copydoc kleidicv_median_blur_u8
KLEIDICV_FILTER_OP_MEDIAN(kleidicv_median_blur_u32_sme, uint32_t);
/// @}

/// @addtogroup kleidicv_morphology
/// @{

/// Calculates maximum (dilate) or minimum (erode) element value of `src`
/// values using a given kernel which has a rectangular shape, and puts the
/// result into `dst`.
///
/// Width and height are the same for the source and the destination. Number
/// of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// The kernel has an anchor point, it is usually the center of the kernel.
/// The algorithm takes a rectangle from the source data using the kernel and
/// the anchor, and calculates the max/min value in that rectangle.
///
/// Example for dilate:
/// ```
///    [ 2, 8, 9, 3, 6 ]  kernel: (3, 3)        [ 8, 9, 9, 9, 6 ]
///    [ 7, 2, 4, 1, 5 ]  anchor: (1, 1)     -> [ 8, 9, 9, 9, 8 ]
///    [ 4, 3, 6, 8, 1 ]  border: replicate     [ 7, 7, 8, 8, 8 ]
///    [ 1, 2, 5, 3, 7 ]                        [ 4, 6, 8, 8, 8 ]
/// ```
///
/// @param src           Pointer to the source data. Must be non-null.
/// @param src_stride    Distance in bytes from the start of one row to the
///                      start of the next row for the source data.
///                      Must not be less than `width * channels *
///                      sizeof(uint8)`, except for single-row images.
/// @param dst           Pointer to the destination data. Must be non-null.
/// @param dst_stride    Distance in bytes from the start of one row to the
///                      start of the next row for the destination data. Must
///                      not be less than `width * channels * sizeof(uint8)`,
///                      except for single-row images.
/// @param width         Number of columns in the data. (One column consists of
///                      `channels` number of elements.) Must be greater than
///                      or equal to `kernel - 1`.
/// @param height        Number of rows in the data. Must be greater than
///                      or equal to `kernel - 1`.
/// @param channels      Number of channels in the data. Must not be more than
///                      @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
/// @param kernel_width  Width of the kernel. The product of kernel_width and
///                      kernel_height must not be more than
///                      @ref KLEIDICV_MAX_IMAGE_PIXELS.
/// @param kernel_height Height of the kernel. See kernel_width for the kernel
///                      size limit.
/// @param anchor_x      X coordinate in the kernel aligned to the source point.
///                      Must not point out of the kernel.
/// @param anchor_y      Y coordinate in the kernel aligned to the source point.
///                      Must not point out of the kernel.
/// @param border_type   Way of handling the border. The supported border types
///                      are: \n
///                         - @ref KLEIDICV_BORDER_TYPE_CONSTANT \n
///                         - @ref KLEIDICV_BORDER_TYPE_REPLICATE
/// @param border_value  Border value if the border_type is
///                      @ref KLEIDICV_BORDER_TYPE_CONSTANT.
/// @param iterations    Number of times to do the morphology operation.
///
KLEIDICV_API_DECLARATION(kleidicv_dilate_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         size_t anchor_x, size_t anchor_y,
                         kleidicv_border_type_t border_type,
                         const uint8_t *border_value, size_t iterations);
/// @copydoc kleidicv_dilate_u8
KLEIDICV_API_DECLARATION(kleidicv_dilate_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         size_t anchor_x, size_t anchor_y,
                         kleidicv_border_type_t border_type,
                         const uint8_t *border_value, size_t iterations);

/// @copydoc kleidicv_dilate_u8
///
KLEIDICV_API_DECLARATION(kleidicv_erode_u8, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         size_t anchor_x, size_t anchor_y,
                         kleidicv_border_type_t border_type,
                         const uint8_t *border_value, size_t iterations);
/// @copydoc kleidicv_dilate_u8
KLEIDICV_API_DECLARATION(kleidicv_erode_u8_sme, const uint8_t *src,
                         size_t src_stride, uint8_t *dst, size_t dst_stride,
                         size_t width, size_t height, size_t channels,
                         size_t kernel_width, size_t kernel_height,
                         size_t anchor_x, size_t anchor_y,
                         kleidicv_border_type_t border_type,
                         const uint8_t *border_value, size_t iterations);
/// @}

/// @addtogroup kleidicv_resize
/// @{

/// Resize image using linear interpolation.
/// In-place operation not supported.
///
/// Supported ratios:
/// - 2x2 and 4x4 upsizing, and 8x8 for float data.
/// - downsizing to any width ratio between 0.33 and 1.0 for uint8 data,
///   height ratio can be anything between 0 and 1.0
/// For other ratios KLEIDICV_ERROR_NOT_IMPLEMENTED is returned.
///
/// Supported channels:
/// - 1 channel for float data.
/// - 1 channel for upsizing uint8 data.
/// - 1, 2, or 3 channels for downsizing uint8 data between any width ratio of
///   0.33 and 1.0, height ratio can be anything between 0 and 1.0.
///
/// Width and height of source and destination images must be less than
/// (1 << 24), i.e. 16,777,216.
///
/// \par Generic downsizing algorithm accuracy for uint8 data:
/// For the best performance, 2-D linear interpolation uses 8-bit weights.
/// Its maximum error from the exact value can be calculated as below.<br>
/// The weights are rounded to 8-bit integers, this leads to this error:
/// > `E = 1 / 2`
///
/// The 8-bit weight is therefore the exact weight `We` plus the error:
/// > `W = We + E`
///
/// 1-D interpolation with 8-bit weights is done according to this formula:
/// > `R = A + ((B - A) * W) / 256`
///
/// The maximum error happens at `A = 0` and `B = 255`, substituting these the
/// error is the difference between the result `R` and the exact result `Re`,
/// plus a rounding error of `1 / 2`:
/// > `E1D = Re - R = (255 * E) / 256 + 1 / 2 = 511 / 512`
///
/// For two dimensions (i.e. doing the horizontal interpolation after the
/// vertical one), the formula is the same, but here `A` and `B` are the results
/// of the 1D calculation above, so they also have some error:
/// > `A = Ae + E1D`<br>
/// > `B = Be + E1D`<br>
/// > `W = We + E`<br>
///
/// Calculating the error from
/// > `R = A + ((B - A) * W) / 256`
///
/// The total error comes from the following addends
/// (note that the error is always added, not allowed to subtract it):
/// > `A ---> E1D`<br>
/// > `(((Be - Ae) + E1D + E1D) * (We + E)) / 256
/// >   ---> ((Be - Ae) * E) / 256 + (2 * E1D * We) / 256`
/// >   (as `E1D*E` is very small and it can be ignored)<br>
/// > `1/2` (rounding error)
///
/// From these, `(Be - Ae) <= 255` and `We <= 255`, so the total theoretical
/// error is:
///  > `E2D = E1D + (2 * E1D) + (E * 255) / 256 + 1 / 2 = 4 * E1D = 2044 / 512`
///
/// But this theoretical error cannot be triggered, as the error components are
/// not independent. So the biggest difference compared to a perfect 8-bit
/// result can be `2`.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param src_width    Number of elements in the source row. For downsizing,
///                     `src_width * channels` must be at least 16 if the ratio
///                     is between 1/2 and 1, and at least 32 if the ratio is
///                     between 1/3 and 1/2. Must be less than (1 << 24).
/// @param src_height   Number of rows in the source data.
///                     Must be less than (1 << 24).
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst_width    Number of elements in the destination row.
///                     For downsizing, `dst_width * channels` must be at
///                     least 8. For upsizing, it must be inline with the chosen
///                     operation, for example `src_width * 2` in case of 2x2.
///                     Must be less than (1 << 24).
/// @param dst_height   Number of rows in the destination data.
///                     For upsizing, it must be inline with the chosen
///                     operation, for example `src_height * 2` in case of 2x2.
///                     Must be less than (1 << 24).
/// @param channels     Number of channels in the data. Must be no more than
///                     @ref KLEIDICV_MAXIMUM_CHANNEL_COUNT.
///
kleidicv_error_t kleidicv_resize_linear_u8(const uint8_t *src,
                                           size_t src_stride, size_t src_width,
                                           size_t src_height, uint8_t *dst,
                                           size_t dst_stride, size_t dst_width,
                                           size_t dst_height, size_t channels);
/// @copydoc kleidicv_resize_linear_u8
kleidicv_error_t kleidicv_resize_linear_u8_sme(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels);

/// @copydoc kleidicv_resize_linear_u8
kleidicv_error_t kleidicv_resize_linear_f32(const float *src, size_t src_stride,
                                            size_t src_width, size_t src_height,
                                            float *dst, size_t dst_stride,
                                            size_t dst_width, size_t dst_height,
                                            size_t channels);
/// @copydoc kleidicv_resize_linear_u8
kleidicv_error_t kleidicv_resize_linear_f32_sme(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    float *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels);
/// @}

/// @addtogroup kleidicv_transform
/// @{

/// Matrix transpose operation.
/// In-place transpose (`src == dst`) is only supported for
/// square matrixes (`src_width == src_height`).
///
/// Example for `src[4,3]` to `dst[3,4]`:
/// ```
/// | 0 | 2 | 2 | 2 |    | 0 | 1 | 1 |
/// | 1 | 0 | 2 | 2 | -> | 2 | 0 | 1 |
/// | 1 | 1 | 0 | 0 |    | 2 | 2 | 0 |
///                      | 2 | 2 | 2 |
/// ```
///
/// Number of pixels is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// In-place operations supported if `src_width` equals `src_height`.
///
/// @param src          Pointer to the source data. Must be non-null.
///                     Must be aligned to `pixel_size`.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must be a multiple of `pixel_size` and no less than
///                     `width * pixel_size`, except for single-row images.
/// @param dst          Pointer to the destination data. Must be non-null.
///                     Must be aligned to `pixel_size`.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `pixel_size` and no less than
///                     `height * pixel_size`, except for single-column
///                     images.
/// @param src_width    Number of pixels in a row.
/// @param src_height   Number of rows in the data.
/// @param pixel_size   Size of one pixel in bytes. Must be 1, 2, 3, 4, 6 or 8.
///
KLEIDICV_API_DECLARATION(kleidicv_transpose, const void *src, size_t src_stride,
                         void *dst, size_t dst_stride, size_t src_width,
                         size_t src_height, size_t pixel_size);

/// Matrix rotate operation.
/// In-place operation is not supported.
/// Supports 90-degree clockwise and 90-degree counter-clockwise rotation.
/// Example (90-degree clockwise) for `src[3,2]` to `dst[2,3]`:
/// ```
/// | 0 | 1 | 2 |    | 4 | 0 |
/// | 4 | 5 | 6 | -> | 5 | 1 |
///                  | 6 | 2 |
/// ```
/// Number of elements is limited to @ref KLEIDICV_MAX_IMAGE_PIXELS.
/// @param src          Pointer to the source data. Must be non-null.
///                     Must be aligned to `pixel_size`.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data.
///                     Must be a multiple of `pixel_size` and no less than
///                     `width * pixel_size`, except for single-row images.
/// @param width        Number of columns in the source data.
/// @param height       Number of rows in the source data.
/// @param dst          Pointer to the destination data. Must be non-null.
///                     Must be aligned to `pixel_size`.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `pixel_size` and no less than
///                     `height * pixel_size`, except for single-column
///                     images.
/// @param angle        Degrees to rotate clockwise. Must be 90, -90, or 270.
/// @param pixel_size   Size of one pixel in bytes. Must be 1, 2, 3, 4, 6 or 8.
///
KLEIDICV_API_DECLARATION(kleidicv_rotate, const void *src, size_t src_stride,
                         size_t width, size_t height, void *dst,
                         size_t dst_stride, int angle, size_t pixel_size);

/// Transforms the `src` image by taking the pixels specified by the coordinates
/// from the `mapxy` image.
///
/// Width and height must be the same for `mapxy` and for `dst`. `src`
/// dimensions may be different. Coordinates outside of `src` dimensions are
/// considered border.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the source data. Must
///                     not be less than `width * sizeof(type)`, except for
///                     single-row images. Must be less than `2^16 *
///                     sizeof(type)`.
/// @param src_width    Number of elements in the source row. Must not be bigger
///                     than 2^15.
/// @param src_height   Number of rows in the source data. Must not be bigger
///                     than 2^15.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(type)` and no less than
///                     `width * sizeof(type)`, except for single-row images.
/// @param dst_width    Number of elements in the destination row. Must be at
///                     least 8.
/// @param dst_height   Number of rows in the destination data.
/// @param mapxy        Pointer to the mapping data. Must be non-null.
/// @param mapxy_stride Distance in bytes from the start of one row to the
///                     start of the next row for the destination data.
///                     Must be a multiple of `sizeof(int16_t)` and no less than
///                     `width * sizeof(int16_t)`, except for single-row images.
/// @param channels     Number of channels in the data. Must be 1.
/// @param border_type  Way of handling the border. The supported border types
///                     are: \n
///                         - @ref KLEIDICV_BORDER_TYPE_CONSTANT
///                         - @ref KLEIDICV_BORDER_TYPE_REPLICATE
/// @param border_value Border value if the border_type is
///                      @ref KLEIDICV_BORDER_TYPE_CONSTANT.
KLEIDICV_API_DECLARATION(kleidicv_remap_s16_u8, const uint8_t *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         uint8_t *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height, size_t channels,
                         const int16_t *mapxy, size_t mapxy_stride,
                         kleidicv_border_type_t border_type,
                         const uint8_t *border_value);

/// @copydoc kleidicv_remap_s16_u8
KLEIDICV_API_DECLARATION(kleidicv_remap_s16_u16, const uint16_t *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         uint16_t *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height, size_t channels,
                         const int16_t *mapxy, size_t mapxy_stride,
                         kleidicv_border_type_t border_type,
                         const uint16_t *border_value);

#ifndef DOXYGEN
/// Internal - not part of the public API and its direct use is not supported.
/// Functionality is similar to @ref kleidicv_remap_s16_u8 , the difference is
/// in the data format: it contains a fractional part with 5+5 bits (`mapfrac`).
/// Other difference:
/// @param channels     Number of channels in the data.
///                     - Supported values:  1 and 4.
KLEIDICV_API_DECLARATION(kleidicv_remap_s16point5_u8, const uint8_t *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         uint8_t *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height, size_t channels,
                         const int16_t *mapxy, size_t mapxy_stride,
                         const uint16_t *mapfrac, size_t mapfrac_stride,
                         kleidicv_border_type_t border_type,
                         const uint8_t *border_value);

/// @copydoc kleidicv_remap_s16point5_u8
KLEIDICV_API_DECLARATION(kleidicv_remap_s16point5_u16, const uint16_t *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         uint16_t *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height, size_t channels,
                         const int16_t *mapxy, size_t mapxy_stride,
                         const uint16_t *mapfrac, size_t mapfrac_stride,
                         kleidicv_border_type_t border_type,
                         const uint16_t *border_value);
#endif  // DOXYGEN

/// Transforms the `src` image by taking the pixels specified by the coordinates
/// from the `mapxy` image.
///
/// Width and height are the same for `mapx`, `mapy` and for `dst`. `src`
/// dimensions may be different, but due to the limits of 32-bit float format,
/// its width and height must be less than 2^24. Coordinates outside of `src`
/// dimensions are considered border. Zero width or height `src` is not
/// supported.
///
/// @param src            Pointer to the source data. Must be non-null.
/// @param src_stride     Distance in bytes from the start of one row to the
///                       start of the next row for the source data. Must
///                       not be less than `width * sizeof(type)`, except for
///                       single-row images. Must be less than 2^32.
/// @param src_width      Number of elements in the source row. Must be less
///                       than 2^24.
/// @param src_height     Number of rows in the source data. Must be less than
///                       2^24.
/// @param dst            Pointer to the destination data. Must be non-null.
/// @param dst_stride     Distance in bytes from the start of one row to the
///                       start of the next row for the destination data.
///                       Must be a multiple of `sizeof(type)` and no less than
///                       `width * sizeof(type)`, except for single-row images.
/// @param dst_width      Number of elements in a destination row. Must be at
///                       least 4.
/// @param dst_height     Number of rows in the destination data.
/// @param channels       Number of channels in the (source and destination)
///                       data. Can be 1 or 2.
/// @param mapx           Pointer to the x coordinates' data. Must be non-null.
/// @param mapx_stride    Distance in bytes from the start of one row to the
///                       start of the next row for `mapx`. Must be a multiple
///                       of `sizeof(float)` and no less than `width *
///                       sizeof(float)`, except for single-row images.
/// @param mapy           Pointer to the y coordinates' data. Must be non-null.
/// @param mapy_stride    Distance in bytes from the start of one row to the
///                       start of the next row for `mapy`. Must be a multiple
///                       of `sizeof(float)` and no less than `width *
///                       sizeof(float)`, except for single-row images.
/// @param interpolation  Interpolation algorithm. Supported types: \n
///                         - @ref KLEIDICV_INTERPOLATION_LINEAR
///                         - @ref KLEIDICV_INTERPOLATION_NEAREST
/// @param border_type    Way of handling the border. The supported border types
///                       are: \n
///                         - @ref KLEIDICV_BORDER_TYPE_REPLICATE
///                         - @ref KLEIDICV_BORDER_TYPE_CONSTANT
/// @param border_value   Border values if the border_type is
///                       @ref KLEIDICV_BORDER_TYPE_CONSTANT.
KLEIDICV_API_DECLARATION(kleidicv_remap_f32_u8, const uint8_t *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         uint8_t *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height, size_t channels, const float *mapx,
                         size_t mapx_stride, const float *mapy,
                         size_t mapy_stride,
                         kleidicv_interpolation_type_t interpolation,
                         kleidicv_border_type_t border_type,
                         const uint8_t *border_value);

/// @copydoc kleidicv_remap_f32_u8
KLEIDICV_API_DECLARATION(kleidicv_remap_f32_u16, const uint16_t *src,
                         size_t src_stride, size_t src_width, size_t src_height,
                         uint16_t *dst, size_t dst_stride, size_t dst_width,
                         size_t dst_height, size_t channels, const float *mapx,
                         size_t mapx_stride, const float *mapy,
                         size_t mapy_stride,
                         kleidicv_interpolation_type_t interpolation,
                         kleidicv_border_type_t border_type,
                         const uint16_t *border_value);

#ifndef DOXYGEN
/// Internal - not part of the public API and its direct use is not supported.
///
/// Calculates horizontal and vertical derivative approximation with Scharr
/// filter and store the results interleaved.
///
/// The horizontal convolution kernel is:
/// ```
/// [  3   0  -3 ]
/// [ 10   0 -10 ]
/// [  3   0  -3 ]
/// ```
///
/// The vertical convolution kernel is:
/// ```
/// [  3  10   3 ]
/// [  0   0   0 ]
/// [ -3 -10  -3 ]
/// ```
///
/// Note, that the kernels are mirrored both vertically and horizontally during
/// the convolution.
///
/// This API does not handle borders, so the result's width and height is `width
/// - 2` and `height - 2`, respectively. Number of pixels in the source is
/// limited to @ref KLEIDICV_MAX_IMAGE_PIXELS. Result's channel count is the
/// double of the source' channel count, as the calculated derivative
/// approximations are stored interleaved:
/// ```
/// | dx,dy | dx,dy | dx,dy | ...
/// ```
/// Where `dx` is the horizontal derivative approximation and `dy` is the
/// vertical derivative approximation.
///
/// @param src          Pointer to the source data. Must be non-null.
/// @param src_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the source data. Must be a
///                     multiple of `sizeof(src type)` and no less than `width *
///                     sizeof(src type) * channels`.
/// @param src_width    Number of columns in the source. Must be more than 2.
///                     (One column consists of `channels` number of elements.)
/// @param src_height   Number of rows in the source. Must be more than 2.
/// @param src_channels Number of channels in the source data. Must be equal
///                     to 1.
/// @param dst          Pointer to the destination data. Must be non-null.
/// @param dst_stride   Distance in bytes from the start of one row to the
///                     start of the next row in the destination data. Must be a
///                     multiple of `sizeof(dst type)` and no less than `(width
///                     - 2) * sizeof(dst type) * channels`.
///
kleidicv_error_t kleidicv_scharr_interleaved_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride);
/// @copydoc kleidicv_scharr_interleaved_s16_u8
kleidicv_error_t kleidicv_scharr_interleaved_s16_u8_sme(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride);
#endif  // DOXYGEN

/// Performs a perspective transformation on an image.
/// For each pixel in `dst` take a pixel from `src` specified by
/// the transformed x and y coordinates, and optionally doing a bilinear
/// interpolation.
///
/// `src` and `dst` dimensions may be different. Number of elements is limited
/// to @ref KLEIDICV_MAX_IMAGE_PIXELS.
///
/// @param src            Pointer to the source data. Must be non-null.
/// @param src_stride     Distance in bytes from the start of one row to the
///                       start of the next row for the source data. Must
///                       not be less than `width * sizeof(type)`, except for
///                       single-row images. Must be less than 2^32.
/// @param src_width      Number of elements in the source row. Must be less
///                       than 2^24.
/// @param src_height     Number of rows in the source data. Must be less than
///                       2^24.
/// @param dst            Pointer to the destination data. Must be non-null.
/// @param dst_stride     Distance in bytes from the start of one row to the
///                       start of the next row for the destination data.
///                       Must be a multiple of `sizeof(type)` and no less than
///                       `width * sizeof(type)`, except for single-row images.
/// @param dst_width      Number of elements in the destination row. Must be at
///                       least 8. Must be less than 2^24.
/// @param dst_height     Number of rows in the destination data. Must be less
///                       than 2^24.
/// @param transformation Pointer to the transformation matrix of 9 values.
/// @param channels       Number of channels in the data. Must be 1.
/// @param interpolation  Interpolation algorithm. Supported types: \n
///                         - @ref KLEIDICV_INTERPOLATION_NEAREST
///                         - @ref KLEIDICV_INTERPOLATION_LINEAR
/// @param border_type    Way of handling the border. The supported border types
///                       are: \n
///                         - @ref KLEIDICV_BORDER_TYPE_REPLICATE
///                         - @ref KLEIDICV_BORDER_TYPE_CONSTANT
/// @param border_value   Border value if the border_type is
///                       @ref KLEIDICV_BORDER_TYPE_CONSTANT.
kleidicv_error_t kleidicv_warp_perspective_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    const float transformation[9], size_t channels,
    kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, const uint8_t *border_value);
/// @}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // KLEIDICV_H
