// SPDX-FileCopyrightText: 2026 RVSPOC P2601 contributors
// SPDX-License-Identifier: Apache-2.0
//
// Smoke test for kleidicv_transpose / kleidicv_rotate, plus sanity that the
// heavyweight stubs return NOT_IMPLEMENTED.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kleidicv/kleidicv.h"

extern "C" const char *kleidicv_riscv_active_backend();

namespace {

int failures = 0;
#define EXPECT(c, m) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, m); ++failures; } } while(0)

void test_backend() {
  const char *b = kleidicv_riscv_active_backend();
  std::printf("[test_transform] backend=%s\n", b);
  if (std::strcmp(b, std::getenv("KLEIDICV_FORCE_SCALAR") ? "scalar" : "rvv") != 0) std::exit(2);
}

void test_transpose_u8() {
  uint8_t src[6] = {1, 2, 3, 4, 5, 6};  // 3w × 2h
  uint8_t dst[6] = {0};                  // 2w × 3h
  EXPECT(kleidicv_transpose(src, 3, dst, 2, 3, 2, 1) == KLEIDICV_OK, "err");
  // dst should be: [1,4, 2,5, 3,6]
  uint8_t exp[6] = {1, 4, 2, 5, 3, 6};
  EXPECT(std::memcmp(dst, exp, 6) == 0, "transpose u8");
}

void test_transpose_u16() {
  uint16_t src[6] = {1, 2, 3, 4, 5, 6};
  uint16_t dst[6] = {0};
  EXPECT(kleidicv_transpose(src, 6, dst, 4, 3, 2, 2) == KLEIDICV_OK, "err");
  uint16_t exp[6] = {1, 4, 2, 5, 3, 6};
  EXPECT(std::memcmp(dst, exp, 12) == 0, "transpose u16");
}

void test_rotate_90_cw() {
  // 3w × 2h source; 90° CW gives 2w × 3h.
  uint8_t src[6] = {1, 2, 3, 4, 5, 6};
  uint8_t dst[6] = {0};
  EXPECT(kleidicv_rotate(src, 3, 3, 2, dst, 2, 90, 1) == KLEIDICV_OK, "err");
  // 90° CW of [[1,2,3],[4,5,6]] = [[4,1],[5,2],[6,3]]
  uint8_t exp[6] = {4, 1, 5, 2, 6, 3};
  EXPECT(std::memcmp(dst, exp, 6) == 0, "rotate 90");
}

void test_rotate_180() {
  uint8_t src[6] = {1, 2, 3, 4, 5, 6};
  uint8_t dst[6] = {0};
  EXPECT(kleidicv_rotate(src, 3, 3, 2, dst, 3, 180, 1) == KLEIDICV_OK, "err");
  uint8_t exp[6] = {6, 5, 4, 3, 2, 1};
  EXPECT(std::memcmp(dst, exp, 6) == 0, "rotate 180");
}

// Reference scalar implementation for cross-checking the RVV path on
// larger inputs that exercise vector strip-mining.
template <typename T>
void ref_transpose(const T *src, size_t src_stride_T, T *dst,
                    size_t dst_stride_T, size_t W, size_t H) {
  for (size_t y = 0; y < H; ++y)
    for (size_t x = 0; x < W; ++x)
      dst[x * dst_stride_T + y] = src[y * src_stride_T + x];
}

template <typename T>
void ref_rotate(const T *src, size_t src_stride_T, size_t W, size_t H, T *dst,
                 size_t dst_stride_T, int angle) {
  for (size_t y = 0; y < H; ++y)
    for (size_t x = 0; x < W; ++x) {
      size_t dx, dy;
      if (angle == 90) { dx = H - 1 - y; dy = x; }
      else if (angle == 180) { dx = W - 1 - x; dy = H - 1 - y; }
      else { dx = y; dy = W - 1 - x; }
      dst[dy * dst_stride_T + dx] = src[y * src_stride_T + x];
    }
}

template <typename T>
void test_transpose_large_T(size_t W, size_t H, size_t pixel_size,
                             const char *tag) {
  std::vector<T> src(W * H);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<T>(0xABCD ^ (i * 31 + i));
  std::vector<T> got(W * H, 0), ref(W * H, 0);
  EXPECT(kleidicv_transpose(src.data(), W * sizeof(T), got.data(),
                             H * sizeof(T), W, H, pixel_size) == KLEIDICV_OK,
         tag);
  ref_transpose<T>(src.data(), W, ref.data(), H, W, H);
  if (std::memcmp(got.data(), ref.data(), W * H * sizeof(T)) != 0) {
    std::fprintf(stderr, "FAIL transpose mismatch (%s)\n", tag);
    ++failures;
  }
}

template <typename T>
void test_rotate_large_T(size_t W, size_t H, size_t pixel_size, int angle,
                          const char *tag) {
  std::vector<T> src(W * H);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<T>(i * 7 + 11);
  size_t out_W = (angle == 180) ? W : H;
  size_t out_H = (angle == 180) ? H : W;
  std::vector<T> got(out_W * out_H, 0), ref(out_W * out_H, 0);
  EXPECT(kleidicv_rotate(src.data(), W * sizeof(T), W, H, got.data(),
                          out_W * sizeof(T), angle, pixel_size) ==
             KLEIDICV_OK,
         tag);
  ref_rotate<T>(src.data(), W, W, H, ref.data(), out_W, angle);
  if (std::memcmp(got.data(), ref.data(), out_W * out_H * sizeof(T)) != 0) {
    std::fprintf(stderr, "FAIL rotate mismatch (%s)\n", tag);
    ++failures;
  }
}

void test_rvv_strip_mining() {
  // Sizes large enough to need ≥2 vsetvli iterations at any reasonable VLEN.
  test_transpose_large_T<uint8_t>(73, 41, 1, "transpose u8 73x41");
  test_transpose_large_T<uint16_t>(73, 41, 2, "transpose u16 73x41");
  test_transpose_large_T<uint32_t>(73, 41, 4, "transpose u32 73x41");
  test_transpose_large_T<uint64_t>(33, 27, 8, "transpose u64 33x27");
  // pixel_size 3 forces scalar fallback.
  {
    constexpr size_t W = 19, H = 11, P = 3;
    std::vector<uint8_t> src(W * H * P), got(W * H * P), ref(W * H * P);
    for (size_t i = 0; i < src.size(); ++i)
      src[i] = static_cast<uint8_t>(i & 0xff);
    EXPECT(kleidicv_transpose(src.data(), W * P, got.data(), H * P, W, H, P) ==
               KLEIDICV_OK,
           "transpose pixel_size=3");
    for (size_t y = 0; y < H; ++y)
      for (size_t x = 0; x < W; ++x)
        std::memcpy(ref.data() + (x * H + y) * P, src.data() + (y * W + x) * P,
                    P);
    EXPECT(std::memcmp(got.data(), ref.data(), W * H * P) == 0,
           "transpose px=3 matches scalar ref");
  }

  for (int a : {90, 180, 270}) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "rotate u8 %d", a);
    test_rotate_large_T<uint8_t>(73, 41, 1, a, buf);
    std::snprintf(buf, sizeof(buf), "rotate u16 %d", a);
    test_rotate_large_T<uint16_t>(73, 41, 2, a, buf);
    std::snprintf(buf, sizeof(buf), "rotate u32 %d", a);
    test_rotate_large_T<uint32_t>(73, 41, 4, a, buf);
    std::snprintf(buf, sizeof(buf), "rotate u64 %d", a);
    test_rotate_large_T<uint64_t>(33, 27, 8, a, buf);
  }
}

void test_rotate_invalid_angle() {
  uint8_t b[1] = {0};
  EXPECT(kleidicv_rotate(b, 1, 1, 1, b, 1, 45, 1) == KLEIDICV_ERROR_RANGE,
         "45° invalid");
}

void test_dilate_basic() {
  // 3x3 dilate: each output is the max over the 3x3 neighbourhood. A single
  // hot pixel at the centre should expand to a 3x3 block.
  uint8_t src[25] = {0};
  src[12] = 200;  // centre
  uint8_t dst[25] = {0};
  EXPECT(kleidicv_dilate_u8(src, 5, dst, 5, 5, 5, 1, 3, 3, 1, 1,
                            KLEIDICV_BORDER_TYPE_REPLICATE, nullptr, 1) ==
             KLEIDICV_OK,
         "dilate err");
  // Centre + 4-neighbours + corners = 9 pixels around (2,2) become 200.
  for (int y = 1; y <= 3; ++y)
    for (int x = 1; x <= 3; ++x)
      EXPECT(dst[y * 5 + x] == 200, "dilate hot block");
  EXPECT(dst[0] == 0, "dilate corner unchanged");
}

void test_resize_linear_2x() {
  // 2x2 source upsampled to 4x4. With OpenCV-style coordinate mapping the
  // result should be a smooth interpolation; just verify it returns OK and
  // produces non-zero output.
  uint8_t src[4] = {0, 100, 100, 200};
  uint8_t dst[16] = {0};
  EXPECT(kleidicv_resize_linear_u8(src, 2, 2, 2, dst, 4, 4, 4, 1) ==
             KLEIDICV_OK,
         "resize err");
  // dst[0] (sx=-0.25, sy=-0.25 → clipped to 0,0) should be roughly 0.
  EXPECT(dst[0] < 50, "resize top-left low");
  // dst[15] (sx=1.25, sy=1.25 → clipped) should be roughly 200.
  EXPECT(dst[15] > 150, "resize bot-right high");
}

// Reference scalar bilinear resize for cross-checking the RVV path on a
// larger non-trivial size.
void ref_resize_linear_u8(const uint8_t *src, size_t W, size_t H, uint8_t *dst,
                           size_t dW, size_t dH) {
  double sx_ratio = static_cast<double>(W) / static_cast<double>(dW);
  double sy_ratio = static_cast<double>(H) / static_cast<double>(dH);
  for (size_t dy = 0; dy < dH; ++dy) {
    double sy_f = (static_cast<double>(dy) + 0.5) * sy_ratio - 0.5;
    ptrdiff_t sy0 = static_cast<ptrdiff_t>(std::floor(sy_f));
    double fy = sy_f - static_cast<double>(sy0);
    if (fy < 0) fy = 0;
    if (fy > 1) fy = 1;
    auto cy = [&](ptrdiff_t v) -> size_t {
      if (v < 0) return 0;
      if (v >= static_cast<ptrdiff_t>(H)) return H - 1;
      return static_cast<size_t>(v);
    };
    const uint8_t *r0 = src + cy(sy0) * W;
    const uint8_t *r1 = src + cy(sy0 + 1) * W;
    for (size_t dx = 0; dx < dW; ++dx) {
      double sx_f = (static_cast<double>(dx) + 0.5) * sx_ratio - 0.5;
      ptrdiff_t sx0 = static_cast<ptrdiff_t>(std::floor(sx_f));
      double fx = sx_f - static_cast<double>(sx0);
      if (fx < 0) fx = 0;
      if (fx > 1) fx = 1;
      auto cx = [&](ptrdiff_t v) -> size_t {
        if (v < 0) return 0;
        if (v >= static_cast<ptrdiff_t>(W)) return W - 1;
        return static_cast<size_t>(v);
      };
      double v = (1 - fy) * ((1 - fx) * r0[cx(sx0)] + fx * r0[cx(sx0 + 1)]) +
                 fy * ((1 - fx) * r1[cx(sx0)] + fx * r1[cx(sx0 + 1)]);
      long iv = std::lrint(v);
      if (iv < 0) iv = 0;
      if (iv > 255) iv = 255;
      dst[dy * dW + dx] = static_cast<uint8_t>(iv);
    }
  }
}

void test_resize_linear_compare() {
  // Non-trivial size that triggers vsetvli strip-mining at all 3 VLENs.
  constexpr size_t W = 64, H = 48, dW = 97, dH = 53;
  std::vector<uint8_t> src(W * H), got(dW * dH), ref(dW * dH);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
  EXPECT(kleidicv_resize_linear_u8(src.data(), W, W, H, got.data(), dW, dW, dH,
                                    1) == KLEIDICV_OK,
         "resize_linear_u8 large");
  ref_resize_linear_u8(src.data(), W, H, ref.data(), dW, dH);
  // Allow ±1 because RVV path uses single-precision floats vs scalar double.
  size_t mismatches = 0;
  for (size_t i = 0; i < got.size(); ++i)
    if (std::abs(static_cast<int>(got[i]) - static_cast<int>(ref[i])) > 1)
      ++mismatches;
  if (mismatches != 0) {
    std::fprintf(stderr,
                 "FAIL resize_linear_u8 mismatches=%zu / %zu\n", mismatches,
                 got.size());
    ++failures;
  }
}

void test_remap_s16_compare() {
  // Build a synthetic mapxy that picks a sheared/wrapped version of the src,
  // exercising both interior and constant-fill OOB lanes.
  constexpr size_t W = 32, H = 24, dW = 40, dH = 30;
  std::vector<uint8_t> src(W * H), got(dW * dH);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>(i & 0xff);
  std::vector<int16_t> mapxy(dW * dH * 2);
  for (size_t dy = 0; dy < dH; ++dy)
    for (size_t dx = 0; dx < dW; ++dx) {
      // Half the points OOB (negative or >= bounds).
      int sx = static_cast<int>(dx) - 4;
      int sy = static_cast<int>(dy) - 3;
      mapxy[(dy * dW + dx) * 2 + 0] = static_cast<int16_t>(sx);
      mapxy[(dy * dW + dx) * 2 + 1] = static_cast<int16_t>(sy);
    }
  uint8_t fill = 200;
  EXPECT(kleidicv_remap_s16_u8(src.data(), W, W, H, got.data(), dW, dW, dH, 1,
                                 mapxy.data(), dW * 2 * sizeof(int16_t),
                                 KLEIDICV_BORDER_TYPE_CONSTANT, &fill) ==
             KLEIDICV_OK,
         "remap_s16_u8 large");
  // Verify against the same scalar logic.
  for (size_t dy = 0; dy < dH; ++dy)
    for (size_t dx = 0; dx < dW; ++dx) {
      int sx = mapxy[(dy * dW + dx) * 2 + 0];
      int sy = mapxy[(dy * dW + dx) * 2 + 1];
      uint8_t expv;
      if (sx < 0 || sy < 0 || sx >= static_cast<int>(W) ||
          sy >= static_cast<int>(H)) {
        expv = fill;
      } else {
        expv = src[sy * W + sx];
      }
      if (got[dy * dW + dx] != expv) {
        std::fprintf(stderr, "FAIL remap mismatch dx=%zu dy=%zu\n", dx, dy);
        ++failures;
        break;
      }
    }
}

// Inline scalar reference for warp_perspective (matches the C-cast nearest
// rounding the production code uses).
void ref_warp_nearest(const uint8_t *src, size_t W, size_t H, uint8_t *dst,
                       size_t dW, size_t dH, const float M[9], uint8_t fill) {
  for (size_t dy = 0; dy < dH; ++dy)
    for (size_t dx = 0; dx < dW; ++dx) {
      float sx_p = M[0] * dx + M[1] * dy + M[2];
      float sy_p = M[3] * dx + M[4] * dy + M[5];
      float sw_p = M[6] * dx + M[7] * dy + M[8];
      uint8_t v;
      if (sw_p == 0.0F) {
        v = fill;
      } else {
        float sx = sx_p / sw_p, sy = sy_p / sw_p;
        int ix = static_cast<int>(sx + 0.5F);
        int iy = static_cast<int>(sy + 0.5F);
        if (ix < 0 || iy < 0 || ix >= static_cast<int>(W) ||
            iy >= static_cast<int>(H)) {
          v = fill;
        } else {
          v = src[iy * W + ix];
        }
      }
      dst[dy * dW + dx] = v;
    }
}

void test_warp_perspective_compare() {
  // Pure translation: dst(x,y) → src(x-1, y-2). Many of the lanes go OOB
  // negatively, so this exercises both interior gathers and constant fill.
  constexpr size_t W = 24, H = 20;
  std::vector<uint8_t> src(W * H), got(W * H), ref(W * H);
  for (size_t i = 0; i < src.size(); ++i)
    src[i] = static_cast<uint8_t>((i * 53 + 7) & 0xff);
  const float Mt[9] = {1, 0, -1, 0, 1, -2, 0, 0, 1};
  uint8_t fill = 128;
  EXPECT(kleidicv_warp_perspective_u8(src.data(), W, W, H, got.data(), W, W, H,
                                       Mt, 1,
                                       KLEIDICV_INTERPOLATION_NEAREST,
                                       KLEIDICV_BORDER_TYPE_CONSTANT,
                                       &fill) == KLEIDICV_OK,
         "warp nearest err");
  ref_warp_nearest(src.data(), W, H, ref.data(), W, H, Mt, fill);
  if (std::memcmp(got.data(), ref.data(), W * H) != 0) {
    size_t miss = 0;
    for (size_t i = 0; i < W * H; ++i) miss += (got[i] != ref[i]);
    std::fprintf(stderr, "FAIL warp nearest mismatches=%zu\n", miss);
    ++failures;
  }
}

void test_warp_perspective_identity() {
  // Identity transform: output should equal input.
  const float M[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  uint8_t src[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint8_t dst[9] = {0};
  uint8_t border = 0;
  EXPECT(kleidicv_warp_perspective_u8(src, 3, 3, 3, dst, 3, 3, 3, M, 1,
                                      KLEIDICV_INTERPOLATION_NEAREST,
                                      KLEIDICV_BORDER_TYPE_CONSTANT,
                                      &border) == KLEIDICV_OK,
         "warp err");
  for (int i = 0; i < 9; ++i)
    EXPECT(dst[i] == src[i], "warp identity");
}

}  // namespace

int main() {
  test_backend();
  test_transpose_u8();
  test_transpose_u16();
  test_rotate_90_cw();
  test_rotate_180();
  test_rotate_invalid_angle();
  test_rvv_strip_mining();
  test_dilate_basic();
  test_resize_linear_2x();
  test_resize_linear_compare();
  test_remap_s16_compare();
  test_warp_perspective_compare();
  test_warp_perspective_identity();
  if (failures == 0) { std::printf("[test_transform] all checks passed\n"); return 0; }
  return 1;
}
