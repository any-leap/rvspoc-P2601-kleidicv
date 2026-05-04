// SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

/// @file ctypes.h
/// @brief Helper type definitions

#ifndef KLEIDICV_CTYPES_H
#define KLEIDICV_CTYPES_H

#ifdef __cplusplus
#include <cinttypes>
#include <cstddef>
#else  // __cplusplus
#include "inttypes.h"
#include "stddef.h"
#endif  // __cplusplus

// This is defined in arm_neon.h or arm_sve.h, but we need it before including
// those.
#if defined(__aarch64__)
typedef __fp16 float16_t;
#elif defined(__FLT16_MANT_DIG__)
// Other architectures (e.g. RISC-V) reach the public header through the same
// include path; expose IEEE-754 binary16 as the storage type so signatures
// parse. Real fp16 arithmetic on RISC-V requires the Zfh extension.
typedef _Float16 float16_t;
#else
// Last-resort opaque storage type — algorithms touching fp16 will not build,
// but headers still parse so unrelated operators can be used.
typedef struct { uint16_t bits; } float16_t;
#endif

#include "kleidicv/config.h"

/// Error values reported by KleidiCV
typedef enum KLEIDICV_NODISCARD {
  /// Success.
  KLEIDICV_OK = 0,
  /// Requested operation is not implemented.
  KLEIDICV_ERROR_NOT_IMPLEMENTED,
  /// Null pointer was passed as an argument.
  KLEIDICV_ERROR_NULL_POINTER,
  /// A value was encountered outside the representable or valid range.
  KLEIDICV_ERROR_RANGE,
  /// Could not allocate memory.
  KLEIDICV_ERROR_ALLOCATION,
  /// A value did not meet alignment requirements.
  KLEIDICV_ERROR_ALIGNMENT,
  /// The provided context is not compatible with the operation.
  KLEIDICV_ERROR_CONTEXT_MISMATCH,
} kleidicv_error_t;

/// Struct to represent a point
typedef struct {
  /// x coordinate
  size_t x;
  /// y coordinate
  size_t y;
} kleidicv_point_t;

/// Struct to represent a rectangle
typedef struct {
  /// Width of the rectangle
  size_t width;
  /// Height of the rectangle
  size_t height;
} kleidicv_rectangle_t;

/// KleidiCV border types
typedef enum {
  /// The border is a constant value.
  KLEIDICV_BORDER_TYPE_CONSTANT,
  /// The border is the value of the first/last element.
  KLEIDICV_BORDER_TYPE_REPLICATE,
  /// The border is the mirrored value of the first/last elements.
  KLEIDICV_BORDER_TYPE_REFLECT,
  /// The border simply acts as a "wrap around" to the beginning/end.
  KLEIDICV_BORDER_TYPE_WRAP,
  /// Like KLEIDICV_BORDER_TYPE_REFLECT, but the first/last elements are
  /// ignored.
  KLEIDICV_BORDER_TYPE_REVERSE,
  /// The border is the "continuation" of the input rows. It is the caller's
  /// responsibility to provide the input data (and an appropriate stride value)
  /// in a way that the rows can be under and over read. E.g. can be used when
  /// executing an operation on a region of a picture.
  KLEIDICV_BORDER_TYPE_TRANSPARENT,
  /// The border is a hard border, there are no additional values to use.
  KLEIDICV_BORDER_TYPE_NONE,
} kleidicv_border_type_t;

/// KleidiCV interpolation types
typedef enum {
  /** Nearest neighbour interpolation */
  KLEIDICV_INTERPOLATION_NEAREST,
  /** Bilinear interpolation */
  KLEIDICV_INTERPOLATION_LINEAR,
} kleidicv_interpolation_type_t;

/// Internal structure where optical flow LK pyramid stores its state
typedef struct kleidicv_optical_flow_pyr_lk_pyramid_t_
    kleidicv_optical_flow_pyr_lk_pyramid_t;

/// Flags for pyramidal LK optical flow.
enum {
  /// @brief `next_points` provides the initial flow estimate.
  ///
  /// The input guess is scaled to the coarsest used pyramid level and then
  /// refined down to level 0.
  KLEIDICV_USE_INITIAL_FLOW = 1 << 0,
  /// `err` stores minimum eigenvalues. When unset, `err` stores photometric
  /// error.
  KLEIDICV_GET_MIN_EIG = 1 << 1
};

/// Controls for pyramidal Lucas-Kanade optical flow.
typedef struct {
  /// Width of the window, must be greater than 2.
  size_t window_width;
  /// Width of the window, must be greater than 2.
  size_t window_height;
  /// @brief Maximum pyramid level index to use during tracking.
  ///
  /// `max_level = N` uses levels `N` down to 0 (inclusive).
  ///
  /// Examples:
  /// * `max_level = 0` uses level 0 only.
  /// * `max_level = 1` uses levels 1 and 0.
  /// * `max_level = 2` uses levels 2, 1, and 0.
  ///
  /// The effective top level may be lower if the input images are too small
  /// for that many levels or if prebuilt pyramids contain fewer levels.
  size_t max_level;
  /// @brief Maximum number of LK refinement iterations per point at each
  /// pyramid level.
  ///
  /// Must be greater than 0.
  size_t termination_count;
  /// @brief Early-stopping tolerance for the per-level LK update step.
  ///
  /// Uses the same unsquared epsilon semantics as OpenCV's pyramidal LK API.
  /// Must be non-negative.
  ///
  /// Internally, the single-level LK solver compares the squared update length
  /// (`dx*dx + dy*dy`) against the squared epsilon threshold.
  float termination_epsilon;
  /// @brief Minimum eigenvalue threshold used to reject weak features.
  ///
  /// Points whose local structure tensor is too weak are marked as not tracked.
  /// Must be non-negative.
  float min_eig_threshold;
  /// @brief More control flags.
  ///
  /// Can contain @ref KLEIDICV_USE_INITIAL_FLOW and/or
  /// @ref KLEIDICV_GET_MIN_EIG.
  int flags;
} kleidicv_optflow_lk_context_t;

/// Supported color conversions and base formats/modifier flags.
typedef enum {
  // Base formats:
  /// Base YUV444 format (interleaved, full resolution)
  KLEIDICV_COLOR_CONVERSION_FMT_YUV444 = 0x01,
  /// Base YUV420 semi-planar format (NV12/NV21)
  KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP = 0x02,
  /// Base YUV420 planar format (I420/IYUV or YV12)
  KLEIDICV_COLOR_CONVERSION_FMT_YUV420P = 0x03,
  /// Base YUV422 format (interleaved 4:2:2)
  KLEIDICV_COLOR_CONVERSION_FMT_YUV422 = 0x04,
  /// Mask to extract the base YUV format (lower 4 bits) from a color conversion
  /// value. Use this to identify whether the base format is YUV444, YUV420SP,
  /// YUV420P, or YUV422.
  KLEIDICV_COLOR_CONVERSION_YUV_FMT_MASK = 0x0F,

  // Modifier flags:
  /// Indicates VU chroma order (V before U)
  KLEIDICV_COLOR_CONVERSION_FLAG_VU = 0x10,
  /// Indicates image data is in BGR format (instead of RGB)
  KLEIDICV_COLOR_CONVERSION_FLAG_BGR = 0x20,
  /// Indicates that alpha channel is present (RGBA or BGRA)
  KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA = 0x40,
  /// Indicates chroma byte precedes luma (Y) byte in interleaved 4:2:2 data
  KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST = 0x80,

  // YUV444 conversions:
  /// Convert YUV444 to RGB
  KLEIDICV_YUV444_TO_RGB = KLEIDICV_COLOR_CONVERSION_FMT_YUV444,
  /// Convert RGB to YUV444
  KLEIDICV_RGB_TO_YUV444 = KLEIDICV_YUV444_TO_RGB,
  /// Convert YUV444 to BGR
  KLEIDICV_YUV444_TO_BGR =
      KLEIDICV_COLOR_CONVERSION_FMT_YUV444 | KLEIDICV_COLOR_CONVERSION_FLAG_BGR,
  /// Convert BGR to YUV444
  KLEIDICV_BGR_TO_YUV444 = KLEIDICV_YUV444_TO_BGR,
  /// Convert YUV444 to RGBA
  KLEIDICV_YUV444_TO_RGBA = KLEIDICV_COLOR_CONVERSION_FMT_YUV444 |
                            KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert RGBA to YUV444
  KLEIDICV_RGBA_TO_YUV444 = KLEIDICV_YUV444_TO_RGBA,
  /// Convert YUV444 to BGRA
  KLEIDICV_YUV444_TO_BGRA = KLEIDICV_COLOR_CONVERSION_FMT_YUV444 |
                            KLEIDICV_COLOR_CONVERSION_FLAG_BGR |
                            KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert BGRA to YUV444
  KLEIDICV_BGRA_TO_YUV444 = KLEIDICV_YUV444_TO_BGRA,

  // YUV420 semi-planar (NV12/NV21) conversions:
  /// Convert NV12 (Y + interleaved UV) to RGB
  KLEIDICV_NV12_TO_RGB = KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP,
  /// Convert RGB to NV12 (Y + interleaved UV)
  KLEIDICV_RGB_TO_NV12 = KLEIDICV_NV12_TO_RGB,
  /// Convert NV12 (Y + interleaved UV) to BGR
  KLEIDICV_NV12_TO_BGR = KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP |
                         KLEIDICV_COLOR_CONVERSION_FLAG_BGR,
  /// Convert BGR to NV12 (Y + interleaved UV)
  KLEIDICV_BGR_TO_NV12 = KLEIDICV_NV12_TO_BGR,
  /// Convert NV12 (Y + interleaved UV) to RGBA
  KLEIDICV_NV12_TO_RGBA = KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert RGBA to NV12 (Y + interleaved UV)
  KLEIDICV_RGBA_TO_NV12 = KLEIDICV_NV12_TO_RGBA,
  /// Convert NV12 (Y + interleaved UV) to BGRA
  KLEIDICV_NV12_TO_BGRA = KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP |
                          KLEIDICV_COLOR_CONVERSION_FLAG_BGR |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert BGRA to NV12 (Y + interleaved UV)
  KLEIDICV_BGRA_TO_NV12 = KLEIDICV_NV12_TO_BGRA,

  /// Convert NV21 (Y + interleaved VU) to RGB
  KLEIDICV_NV21_TO_RGB = KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP |
                         KLEIDICV_COLOR_CONVERSION_FLAG_VU,
  /// Convert RGB to NV21 (Y + interleaved VU)
  KLEIDICV_RGB_TO_NV21 = KLEIDICV_NV21_TO_RGB,
  /// Convert NV21 (Y + interleaved VU) to BGR
  KLEIDICV_NV21_TO_BGR = KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP |
                         KLEIDICV_COLOR_CONVERSION_FLAG_VU |
                         KLEIDICV_COLOR_CONVERSION_FLAG_BGR,
  /// Convert BGR to NV21 (Y + interleaved VU)
  KLEIDICV_BGR_TO_NV21 = KLEIDICV_NV21_TO_BGR,
  /// Convert NV21 (Y + interleaved VU) to RGBA
  KLEIDICV_NV21_TO_RGBA = KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP |
                          KLEIDICV_COLOR_CONVERSION_FLAG_VU |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert RGBA to NV21 (Y + interleaved VU)
  KLEIDICV_RGBA_TO_NV21 = KLEIDICV_NV21_TO_RGBA,
  /// Convert NV21 (Y + interleaved VU) to BGRA
  KLEIDICV_NV21_TO_BGRA = KLEIDICV_COLOR_CONVERSION_FMT_YUV420SP |
                          KLEIDICV_COLOR_CONVERSION_FLAG_VU |
                          KLEIDICV_COLOR_CONVERSION_FLAG_BGR |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert BGRA to NV21 (Y + interleaved VU)
  KLEIDICV_BGRA_TO_NV21 = KLEIDICV_NV21_TO_BGRA,

  // YUV420 planar (I420/IYUV/YV12) conversions:
  /// Convert I420/IYUV (Y, U, V planes) to RGB
  KLEIDICV_IYUV_TO_RGB = KLEIDICV_COLOR_CONVERSION_FMT_YUV420P,
  /// Convert RGB to I420/IYUV (Y, U, V planes)
  KLEIDICV_RGB_TO_IYUV = KLEIDICV_IYUV_TO_RGB,
  /// Convert I420/IYUV (Y, U, V planes) to BGR
  KLEIDICV_IYUV_TO_BGR = KLEIDICV_COLOR_CONVERSION_FMT_YUV420P |
                         KLEIDICV_COLOR_CONVERSION_FLAG_BGR,
  /// Convert BGR to I420/IYUV (Y, U, V planes)
  KLEIDICV_BGR_TO_IYUV = KLEIDICV_IYUV_TO_BGR,
  /// Convert I420/IYUV (Y, U, V planes) to RGBA
  KLEIDICV_IYUV_TO_RGBA = KLEIDICV_COLOR_CONVERSION_FMT_YUV420P |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert RGBA to I420/IYUV (Y, U, V planes)
  KLEIDICV_RGBA_TO_IYUV = KLEIDICV_IYUV_TO_RGBA,
  /// Convert I420/IYUV (Y, U, V planes) to BGRA
  KLEIDICV_IYUV_TO_BGRA = KLEIDICV_COLOR_CONVERSION_FMT_YUV420P |
                          KLEIDICV_COLOR_CONVERSION_FLAG_BGR |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert BGRA to I420/IYUV (Y, U, V planes)
  KLEIDICV_BGRA_TO_IYUV = KLEIDICV_IYUV_TO_BGRA,

  /// Convert YV12 (Y, V, U planes) to RGB
  KLEIDICV_YV12_TO_RGB =
      KLEIDICV_COLOR_CONVERSION_FMT_YUV420P | KLEIDICV_COLOR_CONVERSION_FLAG_VU,
  /// Convert RGB to YV12 (Y, V, U planes)
  KLEIDICV_RGB_TO_YV12 = KLEIDICV_YV12_TO_RGB,
  /// Convert YV12 (Y, V, U planes) to BGR
  KLEIDICV_YV12_TO_BGR = KLEIDICV_COLOR_CONVERSION_FMT_YUV420P |
                         KLEIDICV_COLOR_CONVERSION_FLAG_VU |
                         KLEIDICV_COLOR_CONVERSION_FLAG_BGR,
  /// Convert BGR to YV12 (Y, V, U planes)
  KLEIDICV_BGR_TO_YV12 = KLEIDICV_YV12_TO_BGR,
  /// Convert YV12 (Y, V, U planes) to RGBA
  KLEIDICV_YV12_TO_RGBA = KLEIDICV_COLOR_CONVERSION_FMT_YUV420P |
                          KLEIDICV_COLOR_CONVERSION_FLAG_VU |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert RGBA to YV12 (Y, V, U planes)
  KLEIDICV_RGBA_TO_YV12 = KLEIDICV_YV12_TO_RGBA,
  /// Convert YV12 (Y, V, U planes) to BGRA
  KLEIDICV_YV12_TO_BGRA = KLEIDICV_COLOR_CONVERSION_FMT_YUV420P |
                          KLEIDICV_COLOR_CONVERSION_FLAG_VU |
                          KLEIDICV_COLOR_CONVERSION_FLAG_BGR |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert BGRA to YV12 (Y, V, U planes)
  KLEIDICV_BGRA_TO_YV12 = KLEIDICV_YV12_TO_BGRA,

  // YUV422 interleaved conversions:
  /// Convert YUYV (Y, U, Y, V order) to RGB
  KLEIDICV_YUYV_TO_RGB = KLEIDICV_COLOR_CONVERSION_FMT_YUV422,
  /// Convert RGB to YUYV (Y, U, Y, V order)
  KLEIDICV_RGB_TO_YUYV = KLEIDICV_YUYV_TO_RGB,
  /// Convert YUYV (Y, U, Y, V order) to BGR
  KLEIDICV_YUYV_TO_BGR =
      KLEIDICV_COLOR_CONVERSION_FMT_YUV422 | KLEIDICV_COLOR_CONVERSION_FLAG_BGR,
  /// Convert BGR to YUYV (Y, U, Y, V order)
  KLEIDICV_BGR_TO_YUYV = KLEIDICV_YUYV_TO_BGR,
  /// Convert YUYV (Y, U, Y, V order) to RGBA
  KLEIDICV_YUYV_TO_RGBA = KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert RGBA to YUYV (Y, U, Y, V order)
  KLEIDICV_RGBA_TO_YUYV = KLEIDICV_YUYV_TO_RGBA,
  /// Convert YUYV (Y, U, Y, V order) to BGRA
  KLEIDICV_YUYV_TO_BGRA = KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                          KLEIDICV_COLOR_CONVERSION_FLAG_BGR |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert BGRA to YUYV (Y, U, Y, V order)
  KLEIDICV_BGRA_TO_YUYV = KLEIDICV_YUYV_TO_BGRA,

  /// Convert YVYU (Y, V, Y, U order) to RGB
  KLEIDICV_YVYU_TO_RGB =
      KLEIDICV_COLOR_CONVERSION_FMT_YUV422 | KLEIDICV_COLOR_CONVERSION_FLAG_VU,
  /// Convert RGB to YVYU (Y, V, Y, U order)
  KLEIDICV_RGB_TO_YVYU = KLEIDICV_YVYU_TO_RGB,
  /// Convert YVYU (Y, V, Y, U order) to BGR
  KLEIDICV_YVYU_TO_BGR = KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                         KLEIDICV_COLOR_CONVERSION_FLAG_VU |
                         KLEIDICV_COLOR_CONVERSION_FLAG_BGR,
  /// Convert BGR to YVYU (Y, V, Y, U order)
  KLEIDICV_BGR_TO_YVYU = KLEIDICV_YVYU_TO_BGR,
  /// Convert YVYU (Y, V, Y, U order) to RGBA
  KLEIDICV_YVYU_TO_RGBA = KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                          KLEIDICV_COLOR_CONVERSION_FLAG_VU |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert RGBA to YVYU (Y, V, Y, U order)
  KLEIDICV_RGBA_TO_YVYU = KLEIDICV_YVYU_TO_RGBA,
  /// Convert YVYU (Y, V, Y, U order) to BGRA
  KLEIDICV_YVYU_TO_BGRA =
      KLEIDICV_COLOR_CONVERSION_FMT_YUV422 | KLEIDICV_COLOR_CONVERSION_FLAG_VU |
      KLEIDICV_COLOR_CONVERSION_FLAG_BGR | KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert BGRA to YVYU (Y, V, Y, U order)
  KLEIDICV_BGRA_TO_YVYU = KLEIDICV_YVYU_TO_BGRA,

  /// Convert UYVY (U, Y, V, Y order) to RGB
  KLEIDICV_UYVY_TO_RGB = KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                         KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST,
  /// Convert RGB to UYVY (U, Y, V, Y order)
  KLEIDICV_RGB_TO_UYVY = KLEIDICV_UYVY_TO_RGB,
  /// Convert UYVY (U, Y, V, Y order) to BGR
  KLEIDICV_UYVY_TO_BGR = KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                         KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST |
                         KLEIDICV_COLOR_CONVERSION_FLAG_BGR,
  /// Convert BGR to UYVY (U, Y, V, Y order)
  KLEIDICV_BGR_TO_UYVY = KLEIDICV_UYVY_TO_BGR,
  /// Convert UYVY (U, Y, V, Y order) to RGBA
  KLEIDICV_UYVY_TO_RGBA = KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                          KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert RGBA to UYVY (U, Y, V, Y order)
  KLEIDICV_RGBA_TO_UYVY = KLEIDICV_UYVY_TO_RGBA,
  /// Convert UYVY (U, Y, V, Y order) to BGRA
  KLEIDICV_UYVY_TO_BGRA = KLEIDICV_COLOR_CONVERSION_FMT_YUV422 |
                          KLEIDICV_COLOR_CONVERSION_FLAG_CHROMA_FIRST |
                          KLEIDICV_COLOR_CONVERSION_FLAG_BGR |
                          KLEIDICV_COLOR_CONVERSION_FLAG_ALPHA,
  /// Convert BGRA to UYVY (U, Y, V, Y order)
  KLEIDICV_BGRA_TO_UYVY = KLEIDICV_UYVY_TO_BGRA,
} kleidicv_color_conversion_t;

#endif  // KLEIDICV_CTYPES_H
