<!--
SPDX-FileCopyrightText: 2023 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# OpenCV hardware acceleration layer

[OpenCV](https://github.com/opencv/opencv) can be built with KleidiCV
acceleration via KleidiCV's OpenCV Hardware Acceleration Layer (HAL).
For details of building OpenCV with KleidiCV see [the build documentation](build.md).

In addition, KleidiCV is enabled in certain official OpenCV builds. For example,
the [OpenCV maven package](https://mvnrepository.com/artifact/org.opencv/opencv)
from version 4.11 includes KleidiCV acceleration already, and it can be easily
used in Android applications. (See this
[learning path](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/android_opencv_kleidicv/)
for more information.) Please check OpenCV's documentation, change log or build
configuration whether KleidiCV is enabled in a given OpenCV version for a given
platform, and which version of KleidiCV is integrated.

## Functionality in KleidiCV OpenCV HAL

This section summarizes OpenCV APIs accelerated through the KleidiCV HAL and
their supported parameter constraints.

| Module      | OpenCV APIs covered |
|-------------|---------------------|
| Analysis    | `cv::minMaxIdx`, `cv::calcOpticalFlowPyrLK` |
| Arithmetics | `cv::add`, `cv::subtract`, `cv::absdiff`, `cv::multiply`, `cv::bitwise_and`, `cv::exp`, `cv::inRange` |
| Conversions | `cv::cvtColor`, `cv::Mat::convertTo` |
| Filters     | `cv::GaussianBlur`, `cv::Sobel`, `cv::medianBlur`, `cv::sepFilter2D` |
| Morphology  | `cv::dilate`, `cv::erode` |
| Resize      | `cv::resize`, `cv::pyrDown`, `cv::buildOpticalFlowPyramid` |
| Transform   | `cv::transpose`, `cv::rotate`, `cv::remap`, `cv::warpPerspective` |

Detailed per-API behavior and parameter constraints are documented in
the Modules section below.

## Modules

This section groups OpenCV HAL coverage by the same module structure used by
the KleidiCV C API documentation.


### Analysis

Statistics and extrema operations in the OpenCV HAL.

#### [`cv::minMaxIdx()`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#ga7622c466c628a75d9ed008b42250a73f)
Finds the minimum and maximum element values and their positions.

Notes on parameters:

* `src.depth()` - supported element sizes (without specifying index) are:

  - `CV_8S`
  - `CV_8U`
  - `CV_16S`
  - `CV_16U`
  - `CV_32S`
  - `CV_32F`
* `minIdx`,`maxIdx` - are only supported for `src.depth() == CV_8U`.

#### [`cv::calcOpticalFlowPyrLK()`](https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323)
Finds optical flow of points from one image to another.

Notes on parameters:

* image inputs only support `CV_8U` depth.
* image inputs support 1 to 4 channels.
* `prevPts` and `nextPts` are handled as point arrays of interleaved x/y coordinates.
* `status` is written per point; `err` contains either photometric error or minimum eigenvalues depending on flags.


### Arithmetics

Element-wise arithmetic and logical operations in the OpenCV HAL.

#### [`cv::add()`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6), [`cv::subtract()`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#gaa0f00d98b4b5edeaeb7b8333b2de353b), [`cv::absdiff()`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#ga6fef31bc8c4071cbc114a758a2b79c14)
Element-wise addition, subtraction and absolute difference.

Notes on parameters:

* `src1.depth() == src2.depth()` - `CV_8U`, `CV_8S`, `CV_16U` & `CV_16S` are supported.\
  `CV_32S` is not supported as KleidiCV does not currently provide the
  non-saturating implementation required by OpenCV for this type.

#### [`cv::multiply()`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#ga979d898a58d7f61c53003e162e7ad89f)
Element-wise multiplication.

Notes on parameters:

* `src1.depth() == src2.depth()` - `CV_8U`, `CV_8S`, `CV_16U` & `CV_16S` are supported.\
  `CV_32S` is not supported as KleidiCV does not currently provide the
  non-saturating implementation required by OpenCV for this type.
* `scale` - only a value of 1.0 is supported.

#### [`cv::bitwise_and()`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14)
Bitwise conjunction of two arrays.

#### [`cv::exp()`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#ga3e10108e2162c338f1b848af619f39e5)
Warning: The operation is not bit exact with OpenCV because OpenCV uses a
less precise algorithm.

Exponential function. Currently only `CV_32F` type is supported.

#### [`cv::inRange()`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981)

Checks whether array elements fall between the lower and upper bounds set by the user.\
Currently only scalar bounds are supported.

Notes on parameters:

* `src.type()` - only supports `CV_8UC1` and `CV_32FC1`.
* `src`, `lowerb` and `upperb` need to have the same type.


### Conversions

Color and data type conversion operations in the OpenCV HAL.

#### [`cv::cvtColor()`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab)
Converts the color space of an image.

Notes on parameters:

* `code` - supported [`OpenCV color conversion codes`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#ga4e0972be5de079fed4e3a10e24ef5ef0) are:

##### [`COLOR_GRAY2RGB`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a164663625b9c473f69dd47cdd7a31acc),[`COLOR_GRAY2RGBA`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0aecc1b054bb042ee91478e591cf1f19b8)
Converts grayscale images to RGB or RGBA.

Notes on parameters:

* `src.depth()` - only supports `CV_8U` depth.
* `dstCn` - supports 3 for RGB, 4 for RGBA.

##### [`COLOR_RGB2BGR`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a01a06e50cd3689f5e34e26daf3faaa39), [`COLOR_BGR2RGB`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ad3db9ff253b87d02efe4887b2f5d77ee),[`COLOR_RGBA2BGRA`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a488919a903d6434cacc6a47cf058d803), [`COLOR_BGRA2RGBA`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a06d2c61b23b77b9a3075388146ba5180)
RGB to RGB/RGBA image conversion. All supported permutations are listed in the table below.

|     | RGB | BGR | RGBA | BGRA |
|-----|-----|-----|------|------|
| RGB |     |  x  |      |      |
| BGR |  x  |     |      |      |
| RGBA|     |     |      |   x  |
| BGRA|     |     |   x  |      |

Notes on parameters:

* `src.depth()` - only supports `CV_8U` depth.
* `src.channels()` - supports 3 for RGB and 4 for RGBA.
* `dst.channels()` - supports 3 for RGB and 4 for RGBA.

##### [`COLOR_YUV2RGB_I420`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a35687717fabb536c1e1ec0857714aaf9:~:text=V%2C%20see%20color_convert_rgb_yuv_42x-,COLOR_YUV2RGB_I420,-Python%3A%20cv.COLOR_YUV2RGB_I420), [`COLOR_YUV2RGB_NV12`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a35687717fabb536c1e1ec0857714aaf9:~:text=Python%3A%20cv.COLOR_YUV2RGB-,COLOR_YUV2RGB_NV12,-Python%3A%20cv.COLOR_YUV2RGB_NV12)
YUV420 to RGB/RGBA image conversion supporting **both planar (I420/YV12)** and **semi-planar (NV12/NV21)** layouts.
All supported permutations are shown below:

| YUV Layout    | RGB | BGR | RGBA | BGRA |
|---------------|-----|-----|------|------|
| Planar        |  x  |  x  |  x   |  x   |
| Semi-planar   |  x  |  x  |  x   |  x   |

Notes on parameters:

* `dst.channels()` - supports 3 for RGB and 4 for RGBA.

##### [`COLOR_RGB2YUV_YUY2`](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html), [`COLOR_RGB2YUV_UYVY`](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html), [`COLOR_RGB2YUV_YVYU`](https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html)

RGB (and BGR/RGBA/BGRA) to **YUV422 interleaved** image conversion supporting the common interleaved layouts **YUYV/YUY2** (aliases: **YUYV**, **YUNV**), **UYVY** (aliases: **Y422**, **UYNV**), and **YVYU**.
All supported permutations are shown below:

|      | YUYV (YUY2) | UYVY | YVYU |
|------|-------------|------|------|
| RGB  |      x      |  x   |  x   |
| BGR  |      x      |  x   |  x   |
| RGBA |      x      |  x   |  x   |
| BGRA |      x      |  x   |  x   |

Notes on parameters:

* `src.channels()` - supports 3 for RGB and 4 for RGBA.

##### [`COLOR_YUV2RGB_YUY2`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a6b1ceb56c6a2e8dc80ec1d3b77e0bb0c:~:text=synonym%20to%20UYVY-,COLOR_YUV2RGB_YUY2,-Python%3A%20cv.COLOR_YUV2RGB_YUY2), [`COLOR_YUV2RGB_UYVY`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a6b1ceb56c6a2e8dc80ec1d3b77e0bb0c:~:text=synonym%20to%20COLOR_YUV2GRAY_420-,COLOR_YUV2RGB_UYVY,-Python%3A%20cv.COLOR_YUV2RGB_UYVY), [`COLOR_YUV2RGB_YVYU`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a6b1ceb56c6a2e8dc80ec1d3b77e0bb0c:~:text=V%2C%20see%20color_convert_rgb_yuv_42x-,COLOR_YUV2RGB_YVYU,-Python%3A%20cv.COLOR_YUV2RGB_YVYU)

YUV422 (interleaved) to RGB image conversion supporting the common interleaved layouts **YUYV/YUY2**, **UYVY**, and **YVYU**.
All supported permutations are shown below:

|               | RGB | BGR | RGBA | BGRA |
|---------------|-----|-----|------|------|
| YUYV (YUY2)   |  x  |  x  |   x  |   x  |
| UYVY          |  x  |  x  |   x  |   x  |
| YVYU          |  x  |  x  |   x  |   x  |

Notes on parameters:

* `dst.channels()` - supports 3 for RGB and 4 for RGBA.

##### [`COLOR_YUV2RGB`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ab09d8186a9e5aaac83acd157a1be43b0),[`COLOR_YUV2BGR`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ab053f0cf23ae1b0bfee1964fd9a182c9)
YUV to RGB image conversion, 3 channels to 3 or 4 channels, no subsampling.\
All supported permutations are listed in the table below.

|   | RGB | BGR | RGBA | BGRA |
|---|-----|-----|------|------|
|YUV|  x  |  x  |  x   |  x   |

##### [`COLOR_RGB2YUV`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0adc0f8a1354c98d1701caad4b384e0d18),[`COLOR_BGR2YUV`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0a611d58d4a431fdbc294b4c79701f3d1a)
RGB/RGBA to YUV image conversion, 3 or 4 channels to 3 channels, no subsampling.\
All supported permutations are listed in the table below.

|    | YUV |
|----|-----|
|RGB |  x  |
|RGBA|  x  |
|BGR |  x  |
|BGRA|  x  |

Notes on parameters:

* `src.depth()` - only supports `CV_8U` depth.
* `src.channels()` - supports 3 for RGB/BGR and 4 for RGBA/BGRA.


##### [`COLOR_RGB2YUV_IYUV`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ab91f1a5041e1d4b7b0c1f4d0b69479e5), [`COLOR_BGR2YUV_IYUV`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ab91f1a5041e1d4b7b0c1f4d0b69479e5), [`COLOR_RGBA2YUV_IYUV`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ab91f1a5041e1d4b7b0c1f4d0b69479e5), [`COLOR_BGRA2YUV_IYUV`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ab91f1a5041e1d4b7b0c1f4d0b69479e5), [`COLOR_RGB2YUV_YV12`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ab91f1a5041e1d4b7b0c1f4d0b69479e5), [`COLOR_BGR2YUV_YV12`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ab91f1a5041e1d4b7b0c1f4d0b69479e5), [`COLOR_RGBA2YUV_YV12`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ab91f1a5041e1d4b7b0c1f4d0b69479e5), [`COLOR_BGRA2YUV_YV12`](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html#gga4e0972be5de079fed4e3a10e24ef5ef0ab91f1a5041e1d4b7b0c1f4d0b69479e5)

RGB/BGR to YUV420 planar (IYUV or YV12) conversion.
This transformation outputs three separate planes: Y, U, and V, where chroma channels (U and V) are subsampled by 2 in both horizontal and vertical directions.
The difference between the two formats is the layout order of U and V:

- IYUV: Y is followed by U, then V
- YV12: Y is followed by V, then U

Notes on parameters:

- `src.depth()` - only supports `CV_8U` depth.
- `src.channels()` - supports 3 (RGB/BGR), 4 (RGBA/BGRA). For RGBA/BGRA, the A channel is ignored.
- `dst` - output is a single image containing the Y plane followed by U and V planes (IYUV layout) or V and U planes (YV12 layout).


#### [`cv::Mat::convertTo()`](https://docs.opencv.org/4.10.0/d3/d63/classcv_1_1Mat.html#adf88c60c5b4980e05bb556080916978b)
Warning: The operation is not bit exact with OpenCV due to rounding differences.

For the following source/destination pairs (`rtype` parameter of `-1` means they are the same), scale and offset values using `alpha` and `beta`:

* `CV_8U` to `CV_8U`
* `CV_32F` to `CV_32F`
* `CV_8U` to `CV_16F`
Note: Conversion to 16-bit floating point using SVE/SME is significantly
faster when `alpha` and `beta` are exact `float16_t` values. This can be
forced by explicit casting: `static_cast<float16_t>(0.3)`.

Otherwise, if `alpha` and `beta` are 1 and 0 respectively, conversion between data types is supported as follows:

|`depth()` | `rtype`  |
|----------|----------|
| `CV_32F` | `CV_8S`  |
| `CV_32F` | `CV_8U`  |
| `CV_8S`  | `CV_32F` |
| `CV_8U`  | `CV_32F` |


### Filters

Image filtering operations in the OpenCV HAL.

#### [`cv::GaussianBlur()`](https://docs.opencv.org/4.11.0/d4/d86/group__imgproc__filter.html#gae8bdcd9154ed5ca3cbc1766d960f45c1)
Warning: The operation is not bit exact with OpenCV due to rounding differences
even if ALGO_HINT_ACCURATE is used as the hint parameter.

Blurs an image using a Gaussian filter.\
The filter's standard deviation must be the same in horizontal and vertical directions (`sigmaX == sigmaY`).\
In-place filtering is not supported i.e. `src` and `dst` must be different (non-overlapping) images.

If `src` is a submatrix, the operation is not supported unless `cv::BORDER_ISOLATED` is OR-ed into `borderType`.

Notes on parameters:

* `src.depth()` - only supports `CV_8U` depth.
* `src.cols`,`src.rows` - image width and height must be greater than or equal to `ksize - 1`
* `ksize` - supported kernel sizes are 3x3, 5x5, 7x7, 9x9, 15x15 and 21x21. Odd kernel sizes between 11 and 255 only supported with `cv::BORDER_REPLICATE`.
* `sigmaX`, `sigmaY` - If set to 0, it is automatically calculated. Up to 7x7 kernel size optimal performance is achieved this way.
* `borderType` - supported [OpenCV border types](https://docs.opencv.org/4.11.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5) are:

  - `cv::BORDER_REPLICATE`
  - `cv::BORDER_REFLECT`
  - `cv::BORDER_WRAP`
  - `cv::BORDER_REFLECT_101`
* `hint` - must be [`ALGO_HINT_APPROX`](https://docs.opencv.org/4.11.0/db/de0/group__core__utils.html#gafeab8763db9cdb68a6c20353efe0e9de) if `sigmaX` or `sigmaY` are not 0.

#### [`cv::Sobel()`](https://docs.opencv.org/4.10.0/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d)
Applies Sobel gradient filter to a given image.

If `src` is a submatrix, the operation is not supported unless `cv::BORDER_ISOLATED` is OR-ed into `borderType`.

Notes on parameters:

* `src.depth()` - only supports `CV_8U` depth.
* `src.cols`,`src.rows` - image width and height must be greater than or equal to `ksize - 1`
* `ddepth` - only supports `CV_16S` depth.
* `dx`,`dy` - either vertical `{dx,dy} == {0,1}` or horizontal `{dx,dy} == {1,0}` operation is supported.
* `ksize` - only supports 3x3 kernel size.
* `scale` - scaling is not supported (`1.0`).
* `delta` - delta value is not supported.
* `borderType` - only supports [`cv::BORDER_REPLICATE`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#gga209f2f4869e304c82d07739337eae7c5aa1de4cff95e3377d6d0cbe7569bd4e9f).

#### [`cv::medianBlur()`](https://docs.opencv.org/4.11.0/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)
Applies median filter to a given image.

In-place filtering is not supported i.e. `src` and `dst` must be different (non-overlapping) images.

Notes on parameters:

* `src.cols`,`src.rows` - image width and height must be greater than or equal to `ksize - 1` (e.g., `>= 4` for 5x5, `>= 14` for 15x15).
* `ksize` - for `CV_8U`, supported kernel sizes are 3x3 to 255x255.\
  For other types, only 3x3, 5x5 and 7x7 are supported.

#### [`cv::sepFilter2D()`](https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html#ga910e29ff7d7b105057d1625a4bf6318d)
Applies a separable filter to a given image.

In-place filtering is not supported i.e. `src` and `dst` must be different (non-overlapping) images.

If `src` is a submatrix, the operation is not supported unless `cv::BORDER_ISOLATED` is OR-ed into `borderType`.

Notes on parameters:

* `src.depth()` - only supports `CV_8U`, `CV_16U` and `CV_16S`.
* `src.type()`, `dst.type()`, `kernelX.type()` and `kernelY.type()` must be the same.
* `ddepth` must be `-1`.
* `anchor` must be `(-1, -1)`.
* `delta` must be `0.0`.
* `borderType` - supported [OpenCV border types](https://docs.opencv.org/4.13.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5) are:

  - `cv::BORDER_REPLICATE`
  - `cv::BORDER_REFLECT`
  - `cv::BORDER_WRAP`
  - `cv::BORDER_REFLECT_101`


### Morphology

Morphological operations in the OpenCV HAL.

#### [`cv::dilate()`](https://docs.opencv.org/4.10.0/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c)
If `src` is a submatrix, the operation is not supported unless `cv::BORDER_ISOLATED` is OR-ed into `borderType`.

Notes on parameters:

* `src.depth()`,`dst.depth()`,`kernel.depth()` - only support `CV_8U` depth.
* `src.rows` - image height must be greater than or equal to `kernel.rows - 1`
* `src.cols` - image width must be greater than or equal to `kernel.cols - 1`
* `kernel` - only support kernels where all values equal to 1, and having width and height of at least 5.
* `borderType` - only supports [`BORDER_CONSTANT`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#gga209f2f4869e304c82d07739337eae7c5aed2e4346047e265c8c5a6d0276dcd838).

#### [`cv::erode()`](https://docs.opencv.org/4.10.0/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb)
If `src` is a submatrix, the operation is not supported unless `cv::BORDER_ISOLATED` is OR-ed into `borderType`.

Notes on parameters:

* `src.depth()`,`dst.depth()`,`kernel.depth()` - only support `CV_8U` depth.
* `src.rows` - image height must be greater than or equal to `kernel.rows - 1`
* `src.cols` - image width must be greater than or equal to `kernel.cols - 1`
* `kernel` - only support kernels where all values equal to 1, and having width and height of at least 5.
* `borderType` - only supports [`BORDER_CONSTANT`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#gga209f2f4869e304c82d07739337eae7c5aed2e4346047e265c8c5a6d0276dcd838).


### Resize

Image scaling and pyramid operations in the OpenCV HAL.

#### [`cv::resize()`](https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d)
Warning: The operation is not bit exact with OpenCV due to rounding differences.

Notes on parameters:

* `src.type()` - only supports certain values in combination with other parameters as shown in the table below.
* `dst.cols`,`dst.rows` - must be a multiple of `src.cols` and `src.rows` respectively, in combination with other parameters as shown in the table below.
* `fx`,`fy` - must be 0 or `dst.cols / src.cols`.
* `interpolation` - supported [`InterpolationFlags`](https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121) are as shown in the table below.

|`src.type()`|`interpolation`|src:dst dimensions ratio|
|------------|---------------|------------------------|
| `CV_8UC1`  | `INTER_AREA`  |        0.5x0.5         |
| `CV_8UC2`  | `INTER_AREA`  |        0.5x0.5         |
| `CV_8UC3`  | `INTER_AREA`  |        0.5x0.5         |
| `CV_8UC4`  | `INTER_AREA`  |        0.5x0.5         |
| `CV_8UC1`  |`INTER_LINEAR` |  0.33 - 1.0, 2x2, 4x4  |
| `CV_8UC2`  |`INTER_LINEAR` |      0.33 - 1.0        |
| `CV_8UC3`  |`INTER_LINEAR` |      0.33 - 1.0        |
| `CV_8UC4`  |`INTER_LINEAR` |        0.5x0.5         |
| `CV_32FC1` |`INTER_LINEAR` |     2x2, 4x4, 8x8      |

Note: For the generic linear resize, width ratio is limited to 0.33 - 1.0, but
height ratio can be anything between 0.0 and 1.0.

#### [`cv::pyrDown()`](https://docs.opencv.org/4.10.0/d4/d86/group__imgproc__filter.html#gaf9bba239dfca11654cb7f50f889fc2ff)
Blurs and downsamples an image.

Notes on parameters:

* `src.type()` - only supports `CV_8UC1`.
* `src.cols`,`src.rows` - image width and height must be greater than or equal to `kernel size (== 5) - 1`
* if `dstsize` is specified it must be equal to `Size((src.cols + 1) / 2, (src.rows + 1) / 2)`

#### [`cv::buildOpticalFlowPyramid()`](https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html#ga86640c1c470f87b2660c096d2b22b2ce)
Constructs an image pyramid which can be passed to `cv::calcOpticalFlowPyrLK`.

Notes on parameters:

* input images only support `CV_8U` depth.
* input images support 1 to 4 channels.
* `winSize.width`,`winSize.height` must be in the range `[3, 2047]`.
* `maxLevel` requests the maximum level count, but pyramid generation may stop earlier if the next level would be too small for the LK window.


### Transform

Geometric transform operations in the OpenCV HAL.

#### [`cv::transpose()`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#ga46630ed6c0ea6254a35f447289bd7404)
Transposes a matrix.

Notes on parameters:
* The size of each matrix element must be either 1, 2, 3, or 6 bytes, for example `CV_8UC1`, `CV_8SC2`, `CV_8UC3`, or `CV_16SC3`.
* In-place transpose is supported only for square matrices.

#### [`cv::rotate()`](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html#ga4ad01c0978b0ce64baa246811deeac24)
Rotates a 2D array in multiples of 90 degrees.

Notes on parameters:

* In-place `rotate` is not supported i.e. `src` and `dst` must be different (non-overlapping) images.
* `rotateCode` - supports `ROTATE_90_CLOCKWISE` and `ROTATE_90_COUNTERCLOCKWISE`.
* The size of each matrix element must be either 1, 2, 3, or 6 bytes, for example `CV_8UC1`, `CV_8SC2`, `CV_8UC3`, or `CV_16SC3`.

#### [`cv::remap()`](https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4)
Geometrically transforms the `src` image by taking the pixels specified by the coordinates from the `map` image.

Notes on parameters:

* `src.step` - must be less than 65536 * element size.
* `src.width`, `src_height` - must not be greater than 32768.
* `src.type()` - supports `CV_8UC1` and `CV_16UC1` with all map configs
  * additionally, with `CV_32FC1` map config, it supports `CV_8UC2` and `CV_16UC2` as well.
  * additionally, with `CV_16SC2` plus `CV_16UC1` map config, it supports `CV_8UC4` and `CV_16UC4`
* `dst.cols` - must be at least 4 (32FC1-type maps) or 8 (16SC2-type maps)
* `borderMode` - supports `BORDER_REPLICATE` and `BORDER_CONSTANT`.

Supported map configurations:

* `map1.type()` is `CV_16SC2` and `map2` is empty:

  * channel #1 is x coordinates (column)
  * channel #2 is y coordinates (row)
  * supported `interpolation`: `INTER_NEAREST` only
* `map1.type()` is `CV_16SC2` and `map2.type()` is `CV_16UC1` - fixed-point representation as generated by [`cv::convertMaps`](https://docs.opencv.org/4.11.0/da/d54/group__imgproc__transform.html#ga9156732fa8f01be9ebd1a194f2728b7f):

  * Warning: The operation is not bit exact with OpenCV due to rounding differences.
  * supported `interpolation`: `INTER_LINEAR` only
* `map1` is 32FC1 and `map2` is 32FC1:

  * Warning: The operation is not bit exact with OpenCV because OpenCV uses a less precise algorithm.
  * `map1` is x coordinates (column)
  * `map2` is y coordinates (row)
  * supported `interpolation`: `INTER_NEAREST` and `INTER_LINEAR`

#### [`cv::warpPerspective()`](https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87)
Warning: The operation is not bit exact with OpenCV due to rounding differences.

Performs a perspective transformation on an image.

Notes on parameters:

* `src.type()` - only supports `CV_8UC1`.
* `src.cols`, `src.rows`, `dst.cols`, `dst.rows` - must be less than 16777216
* `src.step` - must be less than 4294967296
* `dst.cols` - must be at least 8
* `borderMode` - supports `BORDER_REPLICATE` and `BORDER_CONSTANT`
* `interpolation` - supports `INTER_NEAREST` and `INTER_LINEAR`
