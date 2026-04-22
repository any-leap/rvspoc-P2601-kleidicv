# SPDX-FileCopyrightText: 2024 - 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

project("OpenCV Conformity" CXX)

set(CMAKE_CXX_STANDARD 17)

# Please update SECURITY.md if adding, removing or changing the version of third party content.
set(OPENCV_VERSION "4.13.0" CACHE STRING "Version of OpenCV")
set(OPENCV_URL "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" CACHE STRING "URL of OpenCV")

# Remove the last part of the version number (e.g., 4.9.0 -> 4.9)
string(REGEX REPLACE "\.[^\.]+$" "" OPENCV_PATCH_VERSION ${OPENCV_VERSION})

# List of patch files to apply. Keep the base adapter patch explicit and
# collect any version-specific conformity patches automatically.
set(OPENCV_PATCH_FILES
    "${CMAKE_CURRENT_LIST_DIR}/../../adapters/opencv/opencv-${OPENCV_PATCH_VERSION}.patch")

file(GLOB OPENCV_VERSIONED_PATCH_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_LIST_DIR}/opencv-${OPENCV_PATCH_VERSION}-*.patch")
list(SORT OPENCV_VERSIONED_PATCH_FILES)
list(APPEND OPENCV_PATCH_FILES ${OPENCV_VERSIONED_PATCH_FILES})

# Check if a patch file has non-comment content.
function(patch_has_content patch_file result_var)
  set(has_content FALSE)
  if (EXISTS "${patch_file}")
    file(STRINGS "${patch_file}" patch_lines)
    foreach(line IN LISTS patch_lines)
      if (NOT line MATCHES "^(//|$)")
        set(has_content TRUE)
        break()
      endif()
    endforeach()
  endif()
  set(${result_var} ${has_content} PARENT_SCOPE)
endfunction()

# Build patch command from all valid patches.
set(OPENCV_PATCH_COMMAND)
foreach(patch_file IN LISTS OPENCV_PATCH_FILES)
  patch_has_content("${patch_file}" has_content)
  if (has_content)
    list(APPEND OPENCV_PATCH_COMMAND COMMAND patch -p1 -i "${patch_file}")
    message(STATUS "Will apply OpenCV patch: ${patch_file}")
  else()
    message(STATUS "Skipping empty/missing patch: ${patch_file}")
  endif()
endforeach()

if (NOT OPENCV_PATCH_COMMAND)
  set(OPENCV_PATCH_COMMAND COMMAND ${CMAKE_COMMAND} -E true)
endif()

include(FetchContent)

FetchContent_Declare(
  OpenCV
  URL ${OPENCV_URL}
  PATCH_COMMAND ${OPENCV_PATCH_COMMAND}
)

FetchContent_MakeAvailable(OpenCV)

file(GLOB conformity_test_sources CONFIGURE_DEPENDS "test_*.cpp")

# Manager
add_executable(
  manager
  manager.cpp
  tests.cpp
  ${conformity_test_sources}
)

target_link_libraries(
  manager
  opencv_core
  opencv_imgproc
  opencv_video
)

target_include_directories(
  manager
  PRIVATE
    ${CMAKE_BINARY_DIR}
    ${OpenCV_SOURCE_DIR}/modules/core/include
    ${OpenCV_SOURCE_DIR}/modules/imgproc/include
    ${OpenCV_SOURCE_DIR}/modules/video/include
)

target_compile_definitions(
  manager
  PRIVATE MANAGER=1
)

target_compile_options(
  manager
  PRIVATE
    "-Werror"
    "-Wall"
    "-Wextra"
)

# Subordinate
add_executable(
  subordinate
  subordinate.cpp
  tests.cpp
  ${conformity_test_sources}
)

target_link_libraries(
  subordinate
  opencv_core
  opencv_imgproc
  opencv_video
)

target_include_directories(
  subordinate
  PRIVATE
    ${CMAKE_BINARY_DIR}
    ${OpenCV_SOURCE_DIR}/modules/core/include
    ${OpenCV_SOURCE_DIR}/modules/imgproc/include
    ${OpenCV_SOURCE_DIR}/modules/video/include
)

target_compile_definitions(
  subordinate
  PRIVATE MANAGER=0
)

target_compile_options(
  subordinate
  PRIVATE
    "-Werror"
    "-Wall"
    "-Wextra"
)
