<!--
SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Changelog

This file documents significant changes between KleidiCV releases.

KleidiCV uses [Calendar Versioning](https://calver.org/) with the format of 0Y.0M.

This changelog aims to follow the guiding principles of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 26.06 - not yet released

### Added
- Rotate +-90 and transpose support for 3-channel images.
- Neon implementation of add padding by copy.

## 26.03 - 2026-03-26

### Added
- Support for OpenCV 4.13.
- Generic Linear Resize for downscaling with factors from 1/3 to 1 (1-, 2- or 3-channel images).
- Gaussian Blur 9x9 kernel support for NEON and SVE/SME fixed paths.
- Scharr interleaved multi-channel support.
- Blur and downsample multi-channel support.
- Standalone Lucas-Kanade algorithm.
- Rotate support for -90 degrees (90 degrees counter-clockwise).
- Multithreaded transpose for out-of-place calls (in-place remains single-threaded).
- LK optical-flow pyramid builder.
- Pyramidal LK optical-flow calculation API.

### Changed
- sepFilter2D is enabled in the OpenCV HAL.
- Switch project versioning from Semantic Versioning to Calendar Versioning using 0Y.0M.
- API functions without the `_sme` postfix prefer the Neon/SVE2 backends,
  unless the `KLEIDICV_PREFER_SME_BACKEND` environment variable is set to
  `ON` when the library is loaded. Functions with the `_sme` postfix prefer
  the SME/SME2 backends when available, and otherwise fall back to the
  Neon/SVE2 backends.

### Fixed
- Return `CV_HAL_ERROR_NOT_IMPLEMENTED` in the OpenCV HAL for Median Blur when called in-place,
  because the aliasing of the input and the output might yield invalid results.
- Fixed in_range_u8 SVE/SSVE variant, it was wrong in cases where upper_limit - x >= 128.

### Removed
- Support for OpenCV 4.12.
- Context creation and release functions for filters and morphology. All filter and morphology functions were
  made stateless.
- `kleidicv_resize_to_quarter_u8` API function. `kleidicv_resize_linear_u8` covers the same functionality.

## 0.7.0 - 2025-12-17

### Added
- Scale function supporting uint8_t input and float16 output.
- SME2 version of many simple operations with multivector loads and stores, for some performance uplift.
- Conversion from RGB/BGR/RGBA/BGRA to YUV 4:2:2.
- Conversion from YUV 4:2:2 to RGB/BGR/RGBA/BGRA.

### Changed
- AArch64 Ubuntu promoted from Tier 2 to Tier 1 on Arm Neoverse N1.

### Fixed
- Multithreaded performance in the OpenCV HAL. (By changing the batch size.)
- SME and SME2 builds with Android NDK r29.

## 0.6.0 - 2025-09-25

### Added
- Median Blur for 3x3 kernels.
- Median Blur for generic kernels (odd-sized only, max kernel size 255x255), Neon backend only.
- Gaussian Blur for any odd kernel size (up to 255x255) with replicated borders, Neon backend only.
- Conversion from packed YUV 4:4:4 (interleaved and non-subsampled) to RGBA/BGRA.
- SME2 version of saturating add with multivector loads and stores. It is marked experimental as it is not covered by CI as of now.
- Conversion from YUV 4:2:0 planar (IYUV/YV12) to RGBA/BGRA.
- Conversion from RGBA/BGRA to YUV 4:2:0 (planar/semi-planar).
- Multi-threaded implementation of InRange.
- Support for OpenCV 4.12.
- Dynamic dispatcher for Mac computers with Apple silicon.
- Windows 11 Arm-based PCs to Tier 3 platforms.

### Changed
- Performance of Gaussian Blur is greatly improved in return for some accuracy.
  (Except for binomial variants up to 7x7 kernel size.)
- Separate SME only implementations from ones using SME2 and add the build options
  `KLEIDICV_ENABLE_SME` and `KLEIDICV_LIMIT_SME_TO_SELECTED_ALGORITHMS`.

### Fixed
- OpenCV patch for adding InRange HAL.

### Removed
- Support for OpenCV 4.11.

## 0.5.0 - 2025-06-10

### Added
- Median Blur for 5x5 and 7x7 kernels.
- Gaussian Blur for 21x21 kernels.
- Support for GCC 8.4.
- C example.

### Changed
- Multithreaded dispatching in the OpenCV HAL of convertScale.

### Fixed
- Inplace operation of kleidicv_scale_f32.

## 0.4.0 - 2025-03-25

### Added
- Implementation of Rotate 90 degrees clockwise.
- Remap implementations for u8 and u16 images, Replicated and Constant borders:
  - Integer coordinates with Nearest Neighbour method
    - 1 channel only
  - Fixed-point coordinates with Linear interpolation
    - 1 and 4 channels
  - Floating-point coordinates with Nearest Neighbour and Linear interpolation
    - 1 and 2 channels
- WarpPerspective implementation for 1-channel u8 input:
  - Nearest Neighbour and Linear interpolation methods
  - Replicated and Constant borders
- Support for OpenCV 4.11.

### Changed
- Increased precision of sum for 32 bit floats and expose it to OpenCV HAL.

### Fixed
- Handling of cv::erode and cv::dilate non-default constant borders.

### Removed
- Support for OpenCV 4.10.

## 0.3.0 - 2024-12-12

### Added
- Implementation of cv::pyrDown in the OpenCV HAL.
- Implementation of cv::buildOpticalFlowPyramid in the OpenCV HAL.
- Sum implementation for 1-channel f32 input (not exposed to OpenCV).

### Changed
- Build options `KLEIDICV_ENABLE_SVE2` and `KLEIDICV_ENABLE_SME2` take effect directly.
  Previously the build scripts had additional checks that attempted to identify whether the compiler supported SVE2/SME2 - these checks have been removed.
- The default setting for `KLEIDICV_ENABLE_SVE2` is on for some popular compilers known to support SVE2, otherwise off.
- `KLEIDICV_ENABLE_SME2` defaults to off. This is because the ACLE SME specification has not yet been finalized.
- In the OpenCV HAL, cvtColor for gray-RGBA & BGRA-RGBA are multithreaded.
- Improved performance of 8-bit int to 32-bit float conversion.
- Renamed functions:
  - kleidicv_float_conversion_f32_s8 to kleidicv_f32_to_s8
  - kleidicv_float_conversion_f32_u8 to kleidicv_f32_to_u8
  - kleidicv_float_conversion_s8_f32 to kleidicv_s8_to_f32
  - kleidicv_float_conversion_u8_f32 to kleidicv_u8_to_f32

## 0.2.0 - 2024-09-30

### Added
- Exponential function for float.
- Bitwise and.
- Gaussian Blur for 7x7 kernels.
- Gaussian Blur for 15x15 kernels.
- Separable Filter 2D for 5x5 kernels (not exposed to OpenCV).
- Enable specifying standard deviation for Gaussian blur.
- Scale function for float.
- Add, subtract, multiply & absdiff enabled in OpenCV HAL.
- MinMax enabled in OpenCV HAL, float version added.
- Resize 4x4 for float.
- Resize 8x8 for float.
- Resize 0.5x0.5 for uint8_t.
- Conversion from float to (u)int8_t and vice versa.
- KLEIDICV_LIMIT_SME2_TO_SELECTED_ALGORITHMS configuration option.
- Conversion from non-subsampled, interleaved YUV to RGB/BGR.
- InRange single channel, scalar bounds for uint8_t and float.

### Changed
- Filter context creation API specification.
- Gaussian Blur API specification.
- In the OpenCV HAL, the following operations are multithreaded:
  * cvtColor
  * convertTo
  * exp
  * minMaxIdx
  * GaussianBlur
  * Sobel
  * resize

### Removed
- Support for OpenCV 4.9.
- threshold from OpenCV HAL.

## 0.1.0 - 2024-05-21

Initial release.
