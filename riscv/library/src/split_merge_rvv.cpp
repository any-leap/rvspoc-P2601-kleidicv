// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// RVV split/merge specialised for channels∈{2,3,4} × element_size∈{1,2,4,8}
// via vlsegN/vssegN. Anything outside that grid falls back to the scalar
// implementation in split_merge_scalar.cpp.

#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

#include "kleidicv/kleidicv.h"
#include "split_merge_decls.h"

namespace kleidicv::scalar {
kleidicv_error_t split(const void *, size_t, void **, const size_t *, size_t,
                       size_t, size_t, size_t);
kleidicv_error_t merge(const void **, const size_t *, void *, size_t, size_t,
                       size_t, size_t, size_t);
}  // namespace kleidicv::scalar

namespace kleidicv::rvv {

namespace {

// Per-cell SPLIT specialisation.
#define SPLIT_FN(NAME, NF, SEW, T, TUPLE_T, BASE)                              \
  static kleidicv_error_t NAME(const void *src, size_t src_stride,             \
                               void **dsts, const size_t *dst_strides,        \
                               size_t width, size_t height) {                  \
    for (size_t y = 0; y < height; ++y) {                                      \
      const T *rs = reinterpret_cast<const T *>(                               \
          static_cast<const uint8_t *>(src) + y * src_stride);                 \
      T *rd[NF];                                                               \
      for (size_t c = 0; c < NF; ++c)                                          \
        rd[c] = reinterpret_cast<T *>(static_cast<uint8_t *>(dsts[c]) +        \
                                      y * dst_strides[c]);                    \
      size_t vl;                                                               \
      for (size_t x = 0; x < width; x += vl) {                                 \
        vl = __riscv_vsetvl_e##SEW##m1(width - x);                             \
        TUPLE_T t = __riscv_vlseg##NF##e##SEW##_v_##BASE##x##NF(rs + NF * x, vl); \
        SPLIT_STORE_##NF##_(SEW, BASE, t, rd, x, vl);                          \
      }                                                                        \
    }                                                                          \
    return KLEIDICV_OK;                                                        \
  }

#define SPLIT_STORE_2_(SEW, BASE, t, rd, x, vl)                                  \
  __riscv_vse##SEW##_v_##BASE(rd[0] + (x),                                       \
                              __riscv_vget_v_##BASE##x2_##BASE(t, 0), vl);       \
  __riscv_vse##SEW##_v_##BASE(rd[1] + (x),                                       \
                              __riscv_vget_v_##BASE##x2_##BASE(t, 1), vl);

#define SPLIT_STORE_3_(SEW, BASE, t, rd, x, vl)                                  \
  __riscv_vse##SEW##_v_##BASE(rd[0] + (x),                                       \
                              __riscv_vget_v_##BASE##x3_##BASE(t, 0), vl);       \
  __riscv_vse##SEW##_v_##BASE(rd[1] + (x),                                       \
                              __riscv_vget_v_##BASE##x3_##BASE(t, 1), vl);       \
  __riscv_vse##SEW##_v_##BASE(rd[2] + (x),                                       \
                              __riscv_vget_v_##BASE##x3_##BASE(t, 2), vl);

#define SPLIT_STORE_4_(SEW, BASE, t, rd, x, vl)                                  \
  __riscv_vse##SEW##_v_##BASE(rd[0] + (x),                                       \
                              __riscv_vget_v_##BASE##x4_##BASE(t, 0), vl);       \
  __riscv_vse##SEW##_v_##BASE(rd[1] + (x),                                       \
                              __riscv_vget_v_##BASE##x4_##BASE(t, 1), vl);       \
  __riscv_vse##SEW##_v_##BASE(rd[2] + (x),                                       \
                              __riscv_vget_v_##BASE##x4_##BASE(t, 2), vl);       \
  __riscv_vse##SEW##_v_##BASE(rd[3] + (x),                                       \
                              __riscv_vget_v_##BASE##x4_##BASE(t, 3), vl);

// (channels, sew) cells we specialise. Element types: u8/u16/u32/u64.
SPLIT_FN(split_2_u8,  2,  8, uint8_t,  vuint8m1x2_t,  u8m1)
SPLIT_FN(split_3_u8,  3,  8, uint8_t,  vuint8m1x3_t,  u8m1)
SPLIT_FN(split_4_u8,  4,  8, uint8_t,  vuint8m1x4_t,  u8m1)
SPLIT_FN(split_2_u16, 2, 16, uint16_t, vuint16m1x2_t, u16m1)
SPLIT_FN(split_3_u16, 3, 16, uint16_t, vuint16m1x3_t, u16m1)
SPLIT_FN(split_4_u16, 4, 16, uint16_t, vuint16m1x4_t, u16m1)
SPLIT_FN(split_2_u32, 2, 32, uint32_t, vuint32m1x2_t, u32m1)
SPLIT_FN(split_3_u32, 3, 32, uint32_t, vuint32m1x3_t, u32m1)
SPLIT_FN(split_4_u32, 4, 32, uint32_t, vuint32m1x4_t, u32m1)
SPLIT_FN(split_2_u64, 2, 64, uint64_t, vuint64m1x2_t, u64m1)
SPLIT_FN(split_3_u64, 3, 64, uint64_t, vuint64m1x3_t, u64m1)
SPLIT_FN(split_4_u64, 4, 64, uint64_t, vuint64m1x4_t, u64m1)

// MERGE: load N planar streams via vleN, vcreate the tuple, vssegN store.
#define MERGE_FN(NAME, NF, SEW, T, TUPLE_T, BASE)                              \
  static kleidicv_error_t NAME(const void **srcs, const size_t *src_strides,  \
                               void *dst, size_t dst_stride, size_t width,    \
                               size_t height) {                                \
    for (size_t y = 0; y < height; ++y) {                                      \
      const T *rs[NF];                                                         \
      for (size_t c = 0; c < NF; ++c)                                          \
        rs[c] = reinterpret_cast<const T *>(                                   \
            static_cast<const uint8_t *>(srcs[c]) + y * src_strides[c]);       \
      T *rd =                                                                  \
          reinterpret_cast<T *>(static_cast<uint8_t *>(dst) + y * dst_stride); \
      size_t vl;                                                               \
      for (size_t x = 0; x < width; x += vl) {                                 \
        vl = __riscv_vsetvl_e##SEW##m1(width - x);                             \
        MERGE_LOAD_##NF##_(SEW, BASE, rs, x, vl, t)                            \
        __riscv_vsseg##NF##e##SEW##_v_##BASE##x##NF(rd + NF * x, t, vl);       \
      }                                                                        \
    }                                                                          \
    return KLEIDICV_OK;                                                        \
  }

#define MERGE_LOAD_2_(SEW, BASE, rs, x, vl, t)                                  \
  auto v0 = __riscv_vle##SEW##_v_##BASE(rs[0] + (x), vl);                       \
  auto v1 = __riscv_vle##SEW##_v_##BASE(rs[1] + (x), vl);                       \
  auto t = __riscv_vcreate_v_##BASE##x2(v0, v1);

#define MERGE_LOAD_3_(SEW, BASE, rs, x, vl, t)                                  \
  auto v0 = __riscv_vle##SEW##_v_##BASE(rs[0] + (x), vl);                       \
  auto v1 = __riscv_vle##SEW##_v_##BASE(rs[1] + (x), vl);                       \
  auto v2 = __riscv_vle##SEW##_v_##BASE(rs[2] + (x), vl);                       \
  auto t = __riscv_vcreate_v_##BASE##x3(v0, v1, v2);

#define MERGE_LOAD_4_(SEW, BASE, rs, x, vl, t)                                  \
  auto v0 = __riscv_vle##SEW##_v_##BASE(rs[0] + (x), vl);                       \
  auto v1 = __riscv_vle##SEW##_v_##BASE(rs[1] + (x), vl);                       \
  auto v2 = __riscv_vle##SEW##_v_##BASE(rs[2] + (x), vl);                       \
  auto v3 = __riscv_vle##SEW##_v_##BASE(rs[3] + (x), vl);                       \
  auto t = __riscv_vcreate_v_##BASE##x4(v0, v1, v2, v3);

MERGE_FN(merge_2_u8,  2,  8, uint8_t,  vuint8m1x2_t,  u8m1)
MERGE_FN(merge_3_u8,  3,  8, uint8_t,  vuint8m1x3_t,  u8m1)
MERGE_FN(merge_4_u8,  4,  8, uint8_t,  vuint8m1x4_t,  u8m1)
MERGE_FN(merge_2_u16, 2, 16, uint16_t, vuint16m1x2_t, u16m1)
MERGE_FN(merge_3_u16, 3, 16, uint16_t, vuint16m1x3_t, u16m1)
MERGE_FN(merge_4_u16, 4, 16, uint16_t, vuint16m1x4_t, u16m1)
MERGE_FN(merge_2_u32, 2, 32, uint32_t, vuint32m1x2_t, u32m1)
MERGE_FN(merge_3_u32, 3, 32, uint32_t, vuint32m1x3_t, u32m1)
MERGE_FN(merge_4_u32, 4, 32, uint32_t, vuint32m1x4_t, u32m1)
MERGE_FN(merge_2_u64, 2, 64, uint64_t, vuint64m1x2_t, u64m1)
MERGE_FN(merge_3_u64, 3, 64, uint64_t, vuint64m1x3_t, u64m1)
MERGE_FN(merge_4_u64, 4, 64, uint64_t, vuint64m1x4_t, u64m1)

}  // namespace

kleidicv_error_t split(const void *src, size_t src_stride, void **dsts,
                       const size_t *dst_strides, size_t width, size_t height,
                       size_t channels, size_t element_size) {
  if (!src || !dsts || !dst_strides) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels == 0 || element_size == 0) return KLEIDICV_ERROR_RANGE;
  for (size_t c = 0; c < channels; ++c)
    if (!dsts[c]) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  using SplitFn = kleidicv_error_t (*)(const void *, size_t, void **,
                                       const size_t *, size_t, size_t);
  SplitFn fn = nullptr;
  if (channels == 2) {
    if (element_size == 1) fn = split_2_u8;
    else if (element_size == 2) fn = split_2_u16;
    else if (element_size == 4) fn = split_2_u32;
    else if (element_size == 8) fn = split_2_u64;
  } else if (channels == 3) {
    if (element_size == 1) fn = split_3_u8;
    else if (element_size == 2) fn = split_3_u16;
    else if (element_size == 4) fn = split_3_u32;
    else if (element_size == 8) fn = split_3_u64;
  } else if (channels == 4) {
    if (element_size == 1) fn = split_4_u8;
    else if (element_size == 2) fn = split_4_u16;
    else if (element_size == 4) fn = split_4_u32;
    else if (element_size == 8) fn = split_4_u64;
  }
  if (!fn) {
    return kleidicv::scalar::split(src, src_stride, dsts, dst_strides, width,
                                   height, channels, element_size);
  }
  return fn(src, src_stride, dsts, dst_strides, width, height);
}

kleidicv_error_t merge(const void **srcs, const size_t *src_strides, void *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t element_size) {
  if (!srcs || !src_strides || !dst) return KLEIDICV_ERROR_NULL_POINTER;
  if (channels == 0 || element_size == 0) return KLEIDICV_ERROR_RANGE;
  for (size_t c = 0; c < channels; ++c)
    if (!srcs[c]) return KLEIDICV_ERROR_NULL_POINTER;
  if (width == 0 || height == 0) return KLEIDICV_OK;

  using MergeFn = kleidicv_error_t (*)(const void **, const size_t *, void *,
                                       size_t, size_t, size_t);
  MergeFn fn = nullptr;
  if (channels == 2) {
    if (element_size == 1) fn = merge_2_u8;
    else if (element_size == 2) fn = merge_2_u16;
    else if (element_size == 4) fn = merge_2_u32;
    else if (element_size == 8) fn = merge_2_u64;
  } else if (channels == 3) {
    if (element_size == 1) fn = merge_3_u8;
    else if (element_size == 2) fn = merge_3_u16;
    else if (element_size == 4) fn = merge_3_u32;
    else if (element_size == 8) fn = merge_3_u64;
  } else if (channels == 4) {
    if (element_size == 1) fn = merge_4_u8;
    else if (element_size == 2) fn = merge_4_u16;
    else if (element_size == 4) fn = merge_4_u32;
    else if (element_size == 8) fn = merge_4_u64;
  }
  if (!fn) {
    return kleidicv::scalar::merge(srcs, src_strides, dst, dst_stride, width,
                                   height, channels, element_size);
  }
  return fn(srcs, src_strides, dst, dst_stride, width, height);
}

}  // namespace kleidicv::rvv
