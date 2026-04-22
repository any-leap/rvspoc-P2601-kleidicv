#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# OpenCV unit tests (imgproc/core/video) without conformity checks.

set -exu

SCRIPT_PATH="$(realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

# Ensure we're at the root of the repo.
cd "${SCRIPT_PATH}/.."

if [[ $(dpkg --print-architecture) = arm64 ]]; then
  : "${OPENCV_VERSION:=4.13.0}"
  : "${OPENCV_URL:=/opt/opencv-${OPENCV_VERSION}.tar.gz}"

  # Build OpenCV with the same patch set used by the conformity checks so
  # the CI path exercises the same KleidiCV/OpenCV integration surface.
  OPENCV_PATCH_VERSION="${OPENCV_VERSION%.*}"
  OPENCV_BUILD_ID="ci/opencv-patched"
  OPENCV_SRC_ROOT="build/ci/opencv-patched-src"
  OPENCV_SRC_PATH="$(pwd)/${OPENCV_SRC_ROOT}/opencv-${OPENCV_VERSION}"

  rm -rf build/ci/opencv-patched*
  mkdir -p "${OPENCV_SRC_ROOT}"
  tar -xzf "${OPENCV_URL}" -C "${OPENCV_SRC_ROOT}"

  OPENCV_PATCH_FILES=(
    "$(pwd)/adapters/opencv/opencv-${OPENCV_PATCH_VERSION}.patch"
    "$(pwd)/conformity/opencv/opencv-${OPENCV_PATCH_VERSION}-copymakeborder.patch"
  )

  pushd "${OPENCV_SRC_PATH}"
  for patch_file in "${OPENCV_PATCH_FILES[@]}"; do
    if [[ -f "${patch_file}" ]] && grep -qvE '^(//|$)' "${patch_file}"; then
      patch -p1 -i "${patch_file}"
    fi
  done
  popd

  BUILD_ID="${OPENCV_BUILD_ID}" \
  OPENCV_PATH="${OPENCV_SRC_PATH}" \
  CMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -fuse-ld=lld" \
  EXTRA_CMAKE_ARGS="\
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_KLEIDICV=ON \
    -DKLEIDICV_SOURCE_PATH=$(pwd) \
    -DKLEIDICV_ENABLE_SME=ON \
    -DBUILD_ANDROID_EXAMPLES=OFF \
    -DBUILD_ANDROID_PROJECTS=OFF \
    -DCV_TRACE=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DBUILD_JAVA=OFF \
    -DWITH_QT=OFF \
    -DBUILD_OPENCV_PYTHON=NO \
    -DBUILD_OPENCV_PYTHON2=NO \
    -DBUILD_OPENCV_PYTHON3=NO \
    -DWITH_VTK=OFF \
    -DWITH_JASPER=OFF \
    -DWITH_OPENJPEG=OFF \
    -DWITH_JPEG=OFF \
    -DWITH_WEBP=OFF \
    -DWITH_TIFF=OFF \
    -DWITH_V4L=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_FLATBUFFERS=OFF \
    -DWITH_PROTOBUF=OFF \
    -DWITH_IMGCODEC_HDR=OFF \
    -DWITH_IMGCODEC_SUNRASTER=OFF \
    -DWITH_IMGCODEC_PXM=OFF \
    -DWITH_IMGCODEC_PFM=OFF \
    -DWITH_ADE=OFF \
    -DWITH_LAPACK=OFF \
    -DOPENCV_PYTHON_SKIP_DETECTION=ON \
    -DOPENCV_ALGO_HINT_DEFAULT=ALGO_HINT_APPROX \
    -DCMAKE_COMPILE_WARNING_AS_ERROR=ON" \
  ./scripts/build-opencv.sh

  BUILD_PATH="build/ci/conformity" \
  CLEAN="ON" \
    ./scripts/opencv-conformity-build.sh

  ninja -C build/ci/conformity/opencv_kleidicv \
    opencv_test_imgproc \
    opencv_test_core \
    opencv_test_video

  rm -rf build/ci/opencv_extra*
  tar xf "/opt/opencv-extra-${OPENCV_VERSION}.tar.gz" -C build/ci
  mv "build/ci/opencv_extra-${OPENCV_VERSION}" build/ci/opencv_extra

  pushd build/ci/opencv_extra/testdata/cv

  join_strings_with_colon() {
    local array="${1}"
    # shellcheck disable=SC2068
    # Here array should be re-splitted.
    array="$(printf ":%s" ${array[@]})"
    echo "${array:1}"
  }

  IMGPROC_TEST_PATTERNS=(
    '*Imgproc_ColorGray*'
    '*Imgproc_ColorRGB*'
    '*Imgproc_ColorYUV*'
    '*Imgproc_cvtColor_BE*'
    '*Imgproc_Threshold*'
    '*Imgproc_Morphology*'
    '*Imgproc_GaussianBlur*'
    '*Imgproc_Sobel*'
    '*Imgproc_Resize*'
    '*Imgproc_Dilate*'
    '*Imgproc_Erode*'
    '*Imgproc_PyramidDown*'
    '*Imgproc_Remap*'
    '*Imgproc_MedianBlur*'
    '*Imgproc_Warp*'
    '*Imgproc_CopyMakeBorder*'
  )
  IMGPROC_TEST_PATTERNS_STR="$(join_strings_with_colon "${IMGPROC_TEST_PATTERNS[*]}")"
  ../../../conformity/opencv_kleidicv/bin/opencv_test_imgproc \
    --gtest_filter="${IMGPROC_TEST_PATTERNS_STR}"

  CORE_TEST_PATTERNS=(
    '*Core_AbsDiff/*'
    '*Core_Add/*'
    '*Core_And/*'
    '*Core_Mul/*'
    '*Core_Sub/*'
    '*Core_Rotate/*'
    '*Core_Transpose/*'
    '*MinMaxLoc*'
    '*Core_ConvertScale/*'
    '*Core_Exp/*'
    '*Core_Sum/*'
    '*Core_MinMaxIdx*'
    '*Core_minMaxIdx*'
    '*Core_Array*'
    'Compare*'
    '*Core_InRangeS/*'
    '*Core_Mat.copyMakeBoderUndefinedBehavior*'
  )
  CORE_TEST_PATTERNS_STR="$(join_strings_with_colon "${CORE_TEST_PATTERNS[*]}")"
  ../../../conformity/opencv_kleidicv/bin/opencv_test_core \
    --gtest_filter="${CORE_TEST_PATTERNS_STR}"

  VIDEO_TEST_PATTERNS=(
    'Video_OpticalFlowPyrLK.accuracy'
  )
  VIDEO_TEST_PATTERNS_STR="$(join_strings_with_colon "${VIDEO_TEST_PATTERNS[*]}")"
  ../../../conformity/opencv_kleidicv/bin/opencv_test_video \
    --gtest_filter="${VIDEO_TEST_PATTERNS_STR}"

  popd
else
  echo "Skipping OpenCV tests on non-arm64 runner."
fi
